import pandas as pd


def engineer_home_race(df: pd.DataFrame) -> pd.DataFrame:
    df['home_race'] = (df['en_short_name'] == df['country']).astype(int)
    return df


def engineer_teammate_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Era-adjusted teammate delta: normalises points by the race winner's total."""
    df['max_race_points'] = df.groupby('raceId')['points'].transform('max')
    # Pre-1958 races and a handful of edge cases award zero max points; guard the division.
    df['max_race_points'] = df['max_race_points'].replace(0, 1)

    df['normalized_points']   = df['points'] / df['max_race_points']
    df['team_avg_norm_points'] = df.groupby(['raceId', 'constructorId'])['normalized_points'].transform('mean')
    df['teammate_delta']      = df['normalized_points'] - df['team_avg_norm_points']

    return df


def engineer_crash_risk(df: pd.DataFrame) -> pd.DataFrame:
    # statusId 3 = Accident, 4 = Collision
    df['is_crash'] = df['statusId'].isin([3, 4]).astype(int)
    df['circuit_crash_rate']    = df.groupby('circuitId')['is_crash'].transform('mean')
    df['normalized_crash_risk'] = df['is_crash'] - df['circuit_crash_rate']
    return df


def engineer_experience(df: pd.DataFrame) -> pd.DataFrame:
    driver_debuts = df.groupby('driverId')['year'].min().reset_index()
    driver_debuts.rename(columns={'year': 'debut_year'}, inplace=True)
    df = df.merge(driver_debuts, on='driverId')
    df['experience_years'] = df['year'] - df['debut_year']
    return df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Engineering features...")
    df = engineer_home_race(df)
    df = engineer_teammate_delta(df)
    df = engineer_crash_risk(df)
    df = engineer_experience(df)
    return df
