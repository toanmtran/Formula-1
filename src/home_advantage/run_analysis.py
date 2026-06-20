"""End-to-end pipeline: figures, statistical tests, model, and tables."""

import json
import os
import warnings
from io import StringIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import shap
import statsmodels.formula.api as smf
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from .data_loader import load_and_merge_data
from .features import build_all_features
from .config import OUTPUT_DIR, FIG_DIR, TAB_DIR

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="talk")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

RESULTS = {}


def savefig(name, fig=None, dpi=160):
    path = os.path.join(FIG_DIR, name)
    (fig or plt.gcf()).savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close("all")
    print(f"  saved {path}")


df = load_and_merge_data()
df = build_all_features(df)
df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
RESULTS["n_rows"]     = int(len(df))
RESULTS["n_drivers"]  = int(df["driverId"].nunique())
RESULTS["n_races"]    = int(df["raceId"].nunique())
RESULTS["n_circuits"] = int(df["circuitId"].nunique())
RESULTS["year_min"]   = int(df["year"].min())
RESULTS["year_max"]   = int(df["year"].max())
RESULTS["n_home"]     = int((df["home_race"] == 1).sum())
RESULTS["n_away"]     = int((df["home_race"] == 0).sum())
print(
    f"Loaded {RESULTS['n_rows']:,} driver-races, "
    f"{RESULTS['n_drivers']} drivers, "
    f"{RESULTS['n_races']} races, "
    f"{RESULTS['n_circuits']} circuits, "
    f"years {RESULTS['year_min']}-{RESULTS['year_max']}."
)


print("\n[1/6] EDA figures")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["teammate_delta"].dropna(), bins=60, color="#4C72B0", edgecolor="white")
ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.set_xlabel(r"Era-adjusted teammate delta $\Delta_{i,r}$")
ax.set_ylabel("Driver-race count")
ax.set_title("Distribution of teammate delta")
savefig("01_teammate_delta_hist.png")

fig, ax = plt.subplots(figsize=(7, 5))
plot_df_box = df.copy()
plot_df_box["Race location"] = plot_df_box["home_race"].map({0: "Away", 1: "Home"})
sns.boxplot(
    data=plot_df_box,
    x="Race location",
    y="teammate_delta",
    order=["Away", "Home"],
    ax=ax,
    palette={"Away": "#888", "Home": "#C44E52"},
    showfliers=False,
)
ax.set_xlabel("")
ax.set_ylabel(r"Teammate delta $\Delta_{i,r}$")
ax.set_title("Teammate delta: Home vs Away (raw distribution)")
savefig("02_home_vs_away_box.png")

home_mean = df.loc[df["home_race"] == 1, "teammate_delta"].mean()
away_mean = df.loc[df["home_race"] == 0, "teammate_delta"].mean()
home_sem  = df.loc[df["home_race"] == 1, "teammate_delta"].sem()
away_sem  = df.loc[df["home_race"] == 0, "teammate_delta"].sem()
fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(
    ["Away", "Home"],
    [away_mean, home_mean],
    yerr=[1.96 * away_sem, 1.96 * home_sem],
    capsize=8,
    color=["#888", "#C44E52"],
)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel(r"Mean teammate delta $\bar{\Delta}$")
ax.set_title("Mean teammate delta with 95% CI")
savefig("03_home_vs_away_meanci.png")
RESULTS["home_mean"] = float(home_mean)
RESULTS["away_mean"] = float(away_mean)

nat_counts = (
    df.dropna(subset=["en_short_name"])
    .groupby("en_short_name")["driverId"]
    .nunique()
    .sort_values(ascending=False)
    .head(12)
)
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(x=nat_counts.values, y=nat_counts.index, ax=ax, color="#4C72B0")
ax.set_xlabel("Distinct drivers")
ax.set_ylabel("")
ax.set_title("Top 12 nationalities by distinct drivers")
savefig("04_top_nationalities.png")

circ = (
    df.groupby(["circuitRef"])
    .agg(starts=("is_crash", "size"), crash_rate=("is_crash", "mean"))
    .query("starts >= 100")
    .sort_values("crash_rate", ascending=False)
    .head(15)
)
fig, ax = plt.subplots(figsize=(9, 6))
sns.barplot(x="crash_rate", y=circ.index, data=circ.reset_index(), ax=ax, color="#C44E52")
ax.set_xlabel("Empirical crash/accident rate")
ax.set_ylabel("")
ax.set_title("Most dangerous circuits (>=100 starts)")
savefig("05_circuit_crash_rates.png")


print("\n[2/6] Paired t-test")

drivers_home = df.loc[df["home_race"] == 1, "driverId"].unique()
test_df = df[df["driverId"].isin(drivers_home)]
home_perf = test_df[test_df["home_race"] == 1].groupby("driverId")["teammate_delta"].mean()
away_perf = test_df[test_df["home_race"] == 0].groupby("driverId")["teammate_delta"].mean()
aligned_home, aligned_away = home_perf.align(away_perf, join="inner")

t_stat, p_value = stats.ttest_rel(aligned_home, aligned_away)
diff = aligned_home - aligned_away
mean_diff = float(diff.mean())
sd_diff   = float(diff.std(ddof=1))
n_pairs   = int(len(diff))
sem_diff  = sd_diff / np.sqrt(n_pairs)
ci_low    = mean_diff - 1.96 * sem_diff
ci_high   = mean_diff + 1.96 * sem_diff
cohens_d  = mean_diff / sd_diff if sd_diff > 0 else float("nan")

RESULTS["ttest"] = {
    "n_pairs": n_pairs,
    "mean_diff": mean_diff,
    "sd_diff": sd_diff,
    "sem_diff": float(sem_diff),
    "ci_low": float(ci_low),
    "ci_high": float(ci_high),
    "t_stat": float(t_stat),
    "p_value": float(p_value),
    "cohens_d": float(cohens_d),
}
print(json.dumps(RESULTS["ttest"], indent=2))

counts = test_df.groupby("driverId").size()
top_ids = counts[counts.index.isin(diff.index)].sort_values(ascending=False).head(25).index
plot_df = (
    pd.DataFrame({"home": aligned_home, "away": aligned_away, "diff": diff})
    .loc[top_ids]
    .merge(
        df[["driverId", "driverRef"]].drop_duplicates(),
        left_index=True,
        right_on="driverId",
    )
    .sort_values("diff")
)
fig, ax = plt.subplots(figsize=(8, 8))
colors = ["#C44E52" if d > 0 else "#4C72B0" for d in plot_df["diff"]]
ax.barh(plot_df["driverRef"], plot_df["diff"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel(r"Mean home delta $-$ mean away delta")
ax.set_title("Per-driver home vs away effect (top 25 by starts)")
savefig("06_per_driver_effect.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(diff, bins=30, color="#4C72B0", edgecolor="white")
ax.axvline(0, color="black", linestyle="--")
ax.axvline(mean_diff, color="#C44E52", linewidth=2, label=f"mean = {mean_diff:.4f}")
ax.set_xlabel(r"$\bar{\Delta}_{i,\text{home}} - \bar{\Delta}_{i,\text{away}}$")
ax.set_ylabel("Number of drivers")
ax.set_title(f"Paired-difference distribution (n={n_pairs} drivers)")
ax.legend()
savefig("07_paired_diff_hist.png")


print("\n[3/6] Linear mixed-effects model")

clean_df = df.dropna(subset=["teammate_delta", "home_race", "experience_years"]).copy()
me_model = smf.mixedlm(
    "teammate_delta ~ home_race * experience_years",
    clean_df,
    groups=clean_df["driverId"],
    re_formula="~1",
)
me_result = me_model.fit()
summary_text = me_result.summary().as_text()
print(summary_text)
with open(os.path.join(TAB_DIR, "mixedlm_summary.txt"), "w") as f:
    f.write(summary_text)

params = me_result.params.to_dict()
ses    = me_result.bse.to_dict()
pvals  = me_result.pvalues.to_dict()
mlm = {}
for k in ["Intercept", "home_race", "experience_years", "home_race:experience_years"]:
    mlm[k] = {
        "coef": float(params.get(k, np.nan)),
        "se":   float(ses.get(k, np.nan)),
        "p":    float(pvals.get(k, np.nan)),
    }
mlm["group_var"] = float(me_result.cov_re.iloc[0, 0])
mlm["resid_var"] = float(me_result.scale)
mlm["n_obs"]     = int(me_result.nobs)
mlm["n_groups"]  = int(me_result.n_groups if hasattr(me_result, "n_groups") else clean_df["driverId"].nunique())
RESULTS["mlm"] = mlm

exp_grid = np.linspace(0, max(20, clean_df["experience_years"].quantile(0.99)), 100)
b0 = mlm["Intercept"]["coef"]
b1 = mlm["home_race"]["coef"]
b2 = mlm["experience_years"]["coef"]
b3 = mlm["home_race:experience_years"]["coef"]
away_pred = b0 + b2 * exp_grid
home_pred = b0 + b1 + (b2 + b3) * exp_grid
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(exp_grid, away_pred, label="Away", color="#4C72B0", linewidth=2)
ax.plot(exp_grid, home_pred, label="Home", color="#C44E52", linewidth=2)
ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
ax.set_xlabel("Experience (years since debut)")
ax.set_ylabel(r"Predicted teammate delta $\hat{\Delta}$")
ax.set_title("Marginal effect of experience: Home vs Away")
ax.legend()
savefig("08_mixedlm_marginal.png")


print("\n[4/6] XGBoost + SHAP")

features = ["experience_years", "home_race", "grid", "circuit_crash_rate"]
target = "teammate_delta"
model_df = df.dropna(subset=features + [target]).copy()
X = model_df[features]
y = model_df[target]

xgb_model = xgb.XGBRegressor(
    n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42,
)
xgb_model.fit(X, y)

importances = pd.Series(xgb_model.feature_importances_, index=features).sort_values()
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(x=importances.values, y=importances.index, ax=ax, color="#4C72B0")
ax.set_xlabel("Gain importance")
ax.set_title("XGBoost feature importance")
savefig("09_xgb_importance.png")

explainer = shap.TreeExplainer(xgb_model)
sample_idx = np.random.RandomState(42).choice(len(X), size=min(5000, len(X)), replace=False)
X_sample = X.iloc[sample_idx]
shap_values = explainer.shap_values(X_sample)

plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
savefig("10_shap_beeswarm.png")

plt.figure()
shap.dependence_plot(
    "experience_years",
    shap_values,
    X_sample,
    interaction_index="home_race",
    show=False,
)
savefig("11_shap_dependence_exp_home.png")

plt.figure()
shap.dependence_plot("grid", shap_values, X_sample, interaction_index="experience_years", show=False)
savefig("12_shap_dependence_grid.png")


print("\n[5/6] Time-series backtest")

model_df = df.dropna(subset=features + [target, "year"]).copy()
model_df = model_df.sort_values(by=["year", "raceId"]).reset_index(drop=True)
X = model_df[features]
y = model_df[target]

tscv = TimeSeriesSplit(n_splits=5)
fold_records = []
rmse_scores, mae_scores = [], []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    train_years = model_df.iloc[train_idx]["year"]
    test_years  = model_df.iloc[test_idx]["year"]
    m = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    mae  = float(mean_absolute_error(y_te, pred))
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    fold_records.append({
        "fold": fold,
        "train_years": f"{int(train_years.min())}-{int(train_years.max())}",
        "test_years":  f"{int(test_years.min())}-{int(test_years.max())}",
        "n_train": int(len(train_idx)),
        "n_test":  int(len(test_idx)),
        "rmse": rmse,
        "mae":  mae,
    })

fold_df = pd.DataFrame(fold_records)
fold_df.to_csv(os.path.join(TAB_DIR, "backtest_folds.csv"), index=False)
RESULTS["backtest_folds"]    = fold_records
RESULTS["backtest_mean_rmse"] = float(np.mean(rmse_scores))
RESULTS["backtest_sd_rmse"]   = float(np.std(rmse_scores))
RESULTS["backtest_mean_mae"]  = float(np.mean(mae_scores))
RESULTS["backtest_sd_mae"]    = float(np.std(mae_scores))

fig, ax = plt.subplots(figsize=(8, 5))
xpos = np.arange(len(fold_df))
ax.bar(xpos - 0.2, fold_df["rmse"], width=0.4, label="RMSE", color="#4C72B0")
ax.bar(xpos + 0.2, fold_df["mae"],  width=0.4, label="MAE",  color="#C44E52")
ax.set_xticks(xpos)
ax.set_xticklabels(
    [f"F{r.fold}\n{r.test_years}" for r in fold_df.itertuples()], fontsize=10,
)
ax.set_ylabel("Normalized error")
ax.set_title("Time-series backtest (5 chronological folds)")
ax.legend()
savefig("13_backtest_folds.png")


print("\n[6/6] Saving results")
with open(os.path.join(TAB_DIR, "results.json"), "w") as f:
    json.dump(RESULTS, f, indent=2)
print("DONE.")
