import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf


def run_paired_ttest(df: pd.DataFrame):
    print("\n--- Running Paired T-Test ---")
    drivers_with_home_races = df[df['home_race'] == 1]['driverId'].unique()
    test_df = df[df['driverId'].isin(drivers_with_home_races)]

    home_perf = test_df[test_df['home_race'] == 1].groupby('driverId')['teammate_delta'].mean()
    away_perf = test_df[test_df['home_race'] == 0].groupby('driverId')['teammate_delta'].mean()

    aligned_home, aligned_away = home_perf.align(away_perf, join='inner')
    t_stat, p_value = stats.ttest_rel(aligned_home, aligned_away)

    print(f"Paired t-test results: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")


def run_mixed_effects_model(df: pd.DataFrame):
    print("\n--- Running Mixed-Effects Model ---")
    clean_df = df.dropna(subset=['teammate_delta', 'home_race', 'experience_years'])

    model = smf.mixedlm(
        "teammate_delta ~ home_race * experience_years",
        clean_df,
        groups=clean_df["driverId"],
        re_formula="~1",
    )
    result = model.fit()
    print(result.summary())
