import os
import glob
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf
from scipy import stats
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.pit_stop_strategy.paths import AGGREGATED_CSV, PLOTS_DIR

try:
    from lifelines import CoxPHFitter
except ImportError:
    CoxPHFitter = None


def load_data(data_path=str(AGGREGATED_CSV)):
    if os.path.isfile(data_path):
        df = pd.read_csv(data_path)
        df['File_Path'] = data_path
        return df

    df_list = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith("_processed.csv"):
                file_path = os.path.join(root, file)
                df_temp = pd.read_csv(file_path)
                if 'Driver' in df_temp.columns and 'Stint' in df_temp.columns:
                    df_temp['File_Path'] = file_path
                    df_list.append(df_temp)
    if not df_list:
        return pd.DataFrame()
    return pd.concat(df_list, ignore_index=True)


def apply_pacf_and_filter(df):
    print("\n--- 1. Time Series Analysis (PACF) ---")

    df = df.sort_values(by=['Driver', 'Stint', 'LapNumber']).copy()
    df['Stationary_PaceDecay'] = np.nan
    pacf_plotted = False

    for (driver, stint), group in df.groupby(['Driver', 'Stint']):
        if len(group) > 5:
            lap_times = group['LapTime_Seconds'].values
            try:
                pacf_vals = pacf(lap_times, nlags=min(10, len(lap_times) // 2 - 1))
                # AR(1) first-difference filter: ΔX_t = X_t − X_{t−1}
                decay = group['LapTime_Seconds'].diff()
                df.loc[group.index, 'Stationary_PaceDecay'] = decay

                if not pacf_plotted:
                    fig = plot_pacf(
                        lap_times,
                        lags=min(15, len(lap_times) // 2 - 1),
                        title=f"PACF of Lap Times (Driver {driver}, Stint {stint})",
                    )
                    plt.savefig(str(PLOTS_DIR / "PACF_Sample_Stint.png"))
                    plt.close(fig)
                    pacf_plotted = True
            except Exception:
                pass

    print("Applied AR(1) first-difference filter to isolate PaceDecay.")
    return df


def analyze_tyrelife_variance(df):
    print("\n--- 3. Random Variables & Target Modeling ---")

    pit_stops = df[df['HasPitStop'] == 1].copy()

    if len(pit_stops) == 0:
        print("No pit stops found in dataset to calculate variance.")
        return

    team_stats = pit_stops.groupby('Team')['TyreLife'].agg(['mean', 'var', 'count']).dropna()
    team_stats.columns = ['E[X] (Mean)', 'Var(X) (Variance)', 'Sample Size']
    team_stats = team_stats.sort_values(by='Var(X) (Variance)', ascending=True)

    print("\nTyrelife Distribution by Team:")
    print(team_stats.to_string())
    print("\nLow Variance = Highly deterministic, safety-first strategy")
    print("High Variance = Risk-tolerant, opportunistic strategic protocol")

    plt.figure(figsize=(12, 6))
    order = team_stats.index
    sns.boxplot(data=pit_stops, x='Team', y='TyreLife', order=order,
                hue='Team', legend=False, palette='viridis')
    plt.title('Tyrelife Distribution by Team (Ordered by Variance)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(str(PLOTS_DIR / 'Tyrelife_Variance_by_Team.png'))
    plt.close()

    return team_stats


def build_hazard_model(df):
    print("\n--- 2. Stochastic Processes & Hazard Modeling ---")
    if CoxPHFitter is None:
        print("lifelines package missing. Install via `pip install lifelines` for Cox Hazard modeling.")
        df['Pit_Probability'] = 0.0
        return df

    subset_cols = ['TyreLife', 'HasPitStop', 'Stationary_PaceDecay', 'traffic_pressure']
    surv_df = df.dropna(subset=[c for c in subset_cols if c in df.columns]).copy()

    # One row per stint (the last lap) for the static survival fit
    stint_ends = surv_df.groupby(['Driver', 'Stint']).tail(1).copy()

    features = ['TyreLife', 'HasPitStop', 'Stationary_PaceDecay', 'traffic_pressure']
    cox_data = stint_ends[features]

    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(cox_data, duration_col='TyreLife', event_col='HasPitStop')
        print("\nCox Proportional Hazards Model Summary:")
        cph.print_summary()

        plt.figure(figsize=(10, 6))
        cph.plot()
        plt.title('Cox Proportional Hazards - Coefficient Effects')
        plt.tight_layout()
        plt.savefig(str(PLOTS_DIR / 'Cox_Hazard_Coefficients.png'))
        plt.close()

        predict_df = df[features].copy()
        predict_df['Stationary_PaceDecay'] = predict_df['Stationary_PaceDecay'].fillna(0)
        predict_df['traffic_pressure'] = predict_df['traffic_pressure'].fillna(
            predict_df['traffic_pressure'].median()
        )
        partial_hazard = cph.predict_partial_hazard(predict_df)
        df['Pit_Probability'] = (
            (partial_hazard - partial_hazard.min())
            / (partial_hazard.max() - partial_hazard.min() + 1e-9)
        )
        return df
    except Exception as e:
        print(f"Hazard model fitting failed: {e}")
        df['Pit_Probability'] = 0.0
        return df


def evaluate_errors(df):
    print("\n--- 4. Parameter Estimation & Error Evaluation ---")

    high_traffic = df[df['traffic_pressure'] > df['traffic_pressure'].median()]['Stationary_PaceDecay'].dropna()
    low_traffic  = df[df['traffic_pressure'] <= df['traffic_pressure'].median()]['Stationary_PaceDecay'].dropna()

    t_stat, p_val = stats.ttest_ind(high_traffic, low_traffic, equal_var=False)
    print(f"\nWelch's T-Test (Pace Decay: High vs Low Traffic Pressure):")
    print(f"t-statistic: {t_stat:.4f}, p-value: {p_val:.4e}")
    if p_val < 0.05:
        print("Result: Statistically significant difference in pace decay under traffic.")

    plt.figure(figsize=(8, 6))
    plot_data = pd.DataFrame({
        'Stationary_PaceDecay': np.concatenate([high_traffic, low_traffic]),
        'Traffic Pressure': ['High'] * len(high_traffic) + ['Low'] * len(low_traffic),
    })
    sns.boxplot(data=plot_data, x='Traffic Pressure', y='Stationary_PaceDecay',
                hue='Traffic Pressure', legend=False, palette='coolwarm')
    plt.title("Pace Decay: High vs Low Traffic Pressure")
    plt.tight_layout()
    plt.savefig(str(PLOTS_DIR / 'Pace_Decay_vs_Traffic.png'))
    plt.close()

    if 'expected_tyre_life' in df.columns:
        y_true = df[df['HasPitStop'] == 1]['TyreLife'].dropna()
        y_pred = df[df['HasPitStop'] == 1]['expected_tyre_life'].dropna()

        if len(y_true) > 0 and len(y_true) == len(y_pred):
            mse = mean_squared_error(y_true, y_pred)
            bias = np.mean(y_pred - y_true)
            variance = np.var(y_pred)
            noise = mse - (bias ** 2) - variance

            print("\nMSE Bias-Variance Decomposition (Target: Tyrelife vs Expected Tyrelife):")
            print(f"Total MSE: {mse:.4f}")
            print(f"Bias^2:    {bias**2:.4f}")
            print(f"Variance:  {variance:.4f}")
            print(f"Noise:     {noise:.4f}")


if __name__ == "__main__":
    data_df = load_data()
    if not data_df.empty:
        print(f"Loaded {len(data_df)} records.")
        data_df = apply_pacf_and_filter(data_df)
        data_df = build_hazard_model(data_df)
        analyze_tyrelife_variance(data_df)
        evaluate_errors(data_df)

        print("\nSaving data with Pit_Probability back to processed files...")
        for path, group in data_df.groupby('File_Path'):
            save_df = group.drop(columns=['File_Path'])
            save_df.to_csv(path, index=False)
        print("Done.")
    else:
        print("No processed data found.")
