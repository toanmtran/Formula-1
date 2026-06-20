import os
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from scipy.stats import linregress

VALID_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

EXPECTED_TYRE_LIFE = {
    "SOFT": 18,
    "MEDIUM": 28,
    "HARD": 40,
}


def clean_laps_dataframe(df):
    df = df.dropna(subset=["Compound"]).copy()
    df["Compound"] = df["Compound"].astype(str).str.upper()
    df = df[df["Compound"].isin(VALID_COMPOUNDS)].copy()
    return df


def convert_time_columns(df):
    time_columns = [
        'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time',
        'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime',
        'SessionTime', 'Time', 'LapStartTime',
    ]

    for col in time_columns:
        if col in df.columns:
            df[f"{col}_Seconds"] = pd.to_timedelta(df[col]).dt.total_seconds()

    return df


def map_telemetry_to_laps(laps_df, telemetry_df):
    telemetry_df = telemetry_df.copy()
    telemetry_df = convert_time_columns(telemetry_df)

    telemetry_df['DriverNumber'] = telemetry_df['Driver'].astype(str)
    laps_df['DriverNumber'] = laps_df['DriverNumber'].astype(str)
    telemetry_df['LapNumber'] = np.nan

    for driver in laps_df['DriverNumber'].unique():
        driver_laps = (
            laps_df[laps_df['DriverNumber'] == driver]
            .sort_values('LapNumber')
            .copy()
        )
        driver_tel = telemetry_df[telemetry_df['DriverNumber'] == driver].copy()

        if len(driver_tel) == 0:
            continue

        lap_intervals = []
        for i in range(len(driver_laps)):
            current_lap = driver_laps.iloc[i]
            lap_num = current_lap['LapNumber']
            start_time = current_lap['LapStartTime_Seconds']

            if i < len(driver_laps) - 1:
                end_time = driver_laps.iloc[i + 1]['LapStartTime_Seconds']
            else:
                end_time = start_time + current_lap['LapTime_Seconds']

            lap_intervals.append((lap_num, start_time, end_time))

        for lap_num, start, end in lap_intervals:
            mask = (
                (telemetry_df['DriverNumber'] == driver)
                & (telemetry_df['SessionTime_Seconds'] >= start)
                & (telemetry_df['SessionTime_Seconds'] < end)
            )
            telemetry_df.loc[mask, 'LapNumber'] = lap_num

    telemetry_df = telemetry_df.dropna(subset=['LapNumber'])
    telemetry_df['LapNumber'] = telemetry_df['LapNumber'].astype(int)
    return telemetry_df


def aggregate_telemetry(laps_df, telemetry_df):
    telemetry_df = map_telemetry_to_laps(laps_df, telemetry_df)

    telemetry_features = telemetry_df.groupby(['DriverNumber', 'LapNumber']).agg({
        'Speed':                 ['mean', 'max', 'std'],
        'RPM':                   ['mean', 'max'],
        'Throttle':              ['mean', 'std'],
        'Brake':                 ['mean', 'sum'],
        'DRS':                   ['mean', 'sum'],
        'DistanceToDriverAhead': ['mean', 'min'],
        'nGear':                 ['mean', 'max'],
    })

    telemetry_features.columns = ["_".join(col) for col in telemetry_features.columns]
    telemetry_features = telemetry_features.reset_index()
    return telemetry_features


def merge_telemetry_features(laps_df, telemetry_df):
    telemetry_features = aggregate_telemetry(laps_df, telemetry_df)
    laps_df['DriverNumber'] = laps_df['DriverNumber'].astype(str)
    telemetry_features['DriverNumber'] = telemetry_features['DriverNumber'].astype(str)
    return laps_df.merge(telemetry_features, on=['DriverNumber', 'LapNumber'], how='left')


def add_core_features(df):
    df['HasPitStop'] = df['PitInTime'].notna().astype(int)

    df['delta_laptime'] = (
        df.groupby('Driver')['LapTime_Seconds'].diff().fillna(0)
    )

    df['CumulativeTimeStint'] = (
        df.groupby(['Driver', 'Stint'])['LapTime_Seconds'].cumsum()
    )

    max_laps = df['LapNumber'].max()
    df['race_progress_fraction'] = df['LapNumber'] / max_laps

    df['expected_tyre_life'] = df['Compound'].map(EXPECTED_TYRE_LIFE)
    df['relative_tire_age'] = df['TyreLife'] / df['expected_tyre_life']
    df['tire_compound_age_delta'] = df['TyreLife'] - df['expected_tyre_life']

    return df


def add_tire_decay_features(df):
    best_lap = (
        df.groupby(['Driver', 'Stint'])['LapTime_Seconds'].transform('min')
    )
    df['tire_performance_decay'] = df['LapTime_Seconds'] - best_lap

    df['rolling_pace_mean_5'] = (
        df.groupby(['Driver', 'Stint'])['LapTime_Seconds']
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )
    df['pace_std_5'] = (
        df.groupby(['Driver', 'Stint'])['LapTime_Seconds']
        .transform(lambda x: x.rolling(5, min_periods=1).std())
    )

    return df


def compute_pace_degradation(group, window=5):
    group = group.copy()
    slopes = []
    lap_times = group['LapTime_Seconds'].values

    for i in range(len(group)):
        if i < window:
            slopes.append(0)
            continue
        y = lap_times[i - window:i]
        x = np.arange(window)
        slope, _, _, _, _ = linregress(x, y)
        slopes.append(slope)

    group['pace_degradation_slope'] = slopes
    return group


def add_historical_pit_features(df):
    pit_laps = df[df['HasPitStop'] == 1]
    historical = (
        pit_laps.groupby(['Compound', 'Team'])['LapNumber']
        .mean()
        .reset_index()
    )
    historical.columns = ['Compound', 'Team', 'historical_pit_lap']
    df = df.merge(historical, on=['Compound', 'Team'], how='left')
    df['pit_window_delta'] = df['LapNumber'] - df['historical_pit_lap']
    return df


def add_telemetry_features(df):
    df['traffic_pressure'] = 1 / (df['DistanceToDriverAhead_mean'] + 1)
    df['drs_dependency']   = df['DRS_sum'] / (df['Speed_mean'] + 1)
    df['thermal_stress_proxy'] = df['Throttle_mean'] * df['Brake_sum'] * df['Speed_std']
    df['high_speed_stress']    = df['Speed_mean'] * df['Throttle_mean']
    df['brake_aggression']     = df['Brake_sum'] * df['Speed_mean']
    df['pace_vs_ahead']        = df['Speed_mean'] / (df['DistanceToDriverAhead_mean'] + 1)
    return df


def add_stochastic_features(df):
    df = df.sort_values(by=['Driver', 'Stint', 'LapNumber']).copy()

    df['TrackStatus_str']  = df['TrackStatus'].astype(str)
    df['Prev_TrackStatus'] = (
        df.groupby('Driver')['TrackStatus_str'].shift(1).fillna('1')
    )

    # TrackStatus codes 4 (SC) and 6 (VSC) — flag the lap a deployment starts.
    vsc_sc_codes = ['4', '6']
    df['Stochastic_Shock_VSC_SC'] = (
        (df['Prev_TrackStatus'] == '1')
        & (df['TrackStatus_str'].isin(vsc_sc_codes))
    ).astype(int)

    df = df.drop(columns=['TrackStatus_str', 'Prev_TrackStatus'])
    return df


def preprocess_race_weekend(laps_csv, telemetry_csv):
    laps_df = pd.read_csv(laps_csv)
    telemetry_df = pd.read_csv(telemetry_csv)

    laps_df = clean_laps_dataframe(laps_df)
    laps_df = convert_time_columns(laps_df)

    df = merge_telemetry_features(laps_df, telemetry_df)
    df = add_core_features(df)
    df = add_tire_decay_features(df)
    df = (
        df.groupby(['Driver', 'Stint'], group_keys=False)
        .apply(compute_pace_degradation)
    )
    df = add_historical_pit_features(df)
    df = add_telemetry_features(df)
    df = add_stochastic_features(df)

    mode_series = df['TrackStatus'].mode()
    track_status_mode = mode_series.iloc[0] if not mode_series.empty else 1
    df['TrackStatus'] = df['TrackStatus'].fillna(track_status_mode)

    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


def process_all_races(laps_root, telemetry_root, output_root):
    os.makedirs(output_root, exist_ok=True)

    for year in os.listdir(laps_root):
        year_path = os.path.join(laps_root, year)
        if not os.path.isdir(year_path):
            continue

        output_year = os.path.join(output_root, year)
        os.makedirs(output_year, exist_ok=True)

        for file in os.listdir(year_path):
            if not file.endswith("_Laps.csv"):
                continue

            race_name = file.replace("_Laps.csv", "")
            laps_csv = os.path.join(year_path, file)
            telemetry_csv = os.path.join(
                telemetry_root, year, f"{race_name}_Telemetry.csv"
            )

            if not os.path.exists(telemetry_csv):
                print(f"Telemetry missing: {race_name} {telemetry_csv}")
                continue

            print(f"Processing {year} {race_name}")

            try:
                processed_df = preprocess_race_weekend(laps_csv, telemetry_csv)
                output_csv = os.path.join(output_year, f"{race_name}_processed.csv")
                processed_df.to_csv(output_csv, index=False)
                print(f"Saved: {output_csv}")
            except Exception as e:
                print(f"FAILED {race_name}: {e}")


if __name__ == "__main__":
    from src.pit_stop_strategy.paths import RAW_FASTF1_DIR, PROCESSED_DIR

    process_all_races(
        str(RAW_FASTF1_DIR),
        str(RAW_FASTF1_DIR),
        str(PROCESSED_DIR),
    )

    print("DONE")
