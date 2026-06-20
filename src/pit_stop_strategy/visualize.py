"""Generate all analytical plots for the pit-stop strategy report.

White-print theme, one chart per PNG (matching the splitter naming so
existing report .tex references stay valid).  Sized for ~0.78-0.85
\\textwidth inclusion on A4 with Palatino.
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# White-print palette
# ---------------------------------------------------------------------------
PAGE_BG   = "#FFFFFF"
TEXT      = "#1F1F1F"
MUTED     = "#555555"
GRID_C    = "#D7D7DC"
SOFT_GREY = "#9F9FA3"

F1_RED    = "#C42021"
BLUE      = "#2E5BA8"
TEAL      = "#0F9E8C"
ORANGE    = "#D9650A"
PURPLE    = "#6A3D9A"

SOFT_C    = "#C42021"   # red
MEDIUM_C  = "#B8860B"   # dark gold (visible on white)
HARD_C    = "#444444"   # dark grey
COMPOUND_COLORS = {"SOFT": SOFT_C, "MEDIUM": MEDIUM_C, "HARD": HARD_C}
COMPOUND_ORDER  = ["SOFT", "MEDIUM", "HARD"]

# Team colours: bumped saturation a notch for visibility against white
TEAM_COLORS = {
    "Mercedes":         "#00A19A",
    "Red Bull Racing":  "#1530A4",
    "Ferrari":          "#C00000",
    "McLaren":          "#E5701F",
    "Alpine":           "#0072C6",
    "Aston Martin":     "#1F6F5C",
    "Williams":         "#0046A0",
    "AlphaTauri":       "#3D6BAC",
    "Alfa Romeo Racing":"#7A1A1A",
    "Haas F1 Team":     "#7A7A7A",
    "Racing Point":     "#D070A6",
    "Renault":          "#D9B500",
    "Toro Rosso":       "#3E73C6",
    "Force India":      "#D070A6",
    "Sauber":           "#7A1A1A",
    "Alfa Romeo":       "#A22A40",
    "RB":               "#1A4FA8",
    "Racing Bulls":     "#4376C4",
    "Kick Sauber":      "#2EA02E",
}


def apply_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor":     PAGE_BG,
        "figure.edgecolor":     PAGE_BG,
        "savefig.facecolor":    PAGE_BG,
        "savefig.edgecolor":    PAGE_BG,
        "axes.facecolor":       PAGE_BG,
        "axes.edgecolor":       MUTED,
        "axes.labelcolor":      TEXT,
        "axes.titlecolor":      TEXT,
        "axes.titlesize":       14,
        "axes.titleweight":     "bold",
        "axes.labelsize":       12,
        "axes.linewidth":       0.9,
        "axes.titlepad":        10,
        "axes.grid":            True,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "grid.color":           GRID_C,
        "grid.linewidth":       0.7,
        "grid.alpha":           0.8,
        "xtick.color":          TEXT,
        "ytick.color":          TEXT,
        "xtick.labelsize":      11,
        "ytick.labelsize":      11,
        "legend.facecolor":     PAGE_BG,
        "legend.edgecolor":     GRID_C,
        "legend.labelcolor":    TEXT,
        "legend.fontsize":      11,
        "legend.title_fontsize": 11,
        "lines.linewidth":      2.0,
        "font.family":          "serif",
        "font.serif":           ["Palatino Linotype", "Palatino", "DejaVu Serif"],
        "font.size":            12,
        "text.color":           TEXT,
        "savefig.dpi":          200,
        "savefig.bbox":         "tight",
    })


FIG_SOLO    = (7.5, 4.8)
FIG_SQUARE  = (6.8, 6.0)
FIG_WIDE    = (8.5, 4.8)
FIG_HEATMAP = (7.5, 6.5)


def _solo(figsize=FIG_SOLO, title=""):
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        ax.set_title(title)
    return fig, ax


def save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [saved] {path}")


def load_data(csv_path: str) -> pd.DataFrame:
    print(f"\nLoading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")

    if "LapTime_Seconds" in df.columns:
        lt = df["LapTime_Seconds"]
        df = df[(lt > 55) | (lt.isna())]
    if "TyreLife" in df.columns:
        df = df[df["TyreLife"] <= 70]
    if "tire_performance_decay" in df.columns:
        df["tire_performance_decay"] = df["tire_performance_decay"].clip(0, 60)

    print(f"  After cleaning: {len(df):,} rows")
    if "HasPitStop" in df.columns:
        print(f"  Pit stop rate : {df['HasPitStop'].mean() * 100:.2f}%")
    return df


# =========================================================================
# 01  Dataset overview  — 4 charts
# =========================================================================
def plot_01a_lap_time_by_compound(df, out):
    fig, ax = _solo(title="Lap-time distribution by compound")
    lt_all = df["LapTime_Seconds"].dropna()
    lt_all = lt_all[(lt_all > 65) & (lt_all < 175)]
    for comp in COMPOUND_ORDER:
        sub = df[df["Compound"] == comp]["LapTime_Seconds"].dropna()
        sub = sub[(sub > 65) & (sub < 175)]
        if len(sub) == 0:
            continue
        ax.hist(sub, bins=60, alpha=0.55, color=COMPOUND_COLORS[comp],
                label=comp, density=True, edgecolor="white", linewidth=0.3)
    ax.axvline(lt_all.mean(), color=TEXT, ls="--", lw=1.5,
               label=f"Overall mean: {lt_all.mean():.1f} s")
    ax.set_xlabel("Lap time (seconds)")
    ax.set_ylabel("Density")
    ax.legend()
    save(fig, f"{out}/01a_laptime_compound.png")


def plot_01b_tyre_life_dist(df, out):
    fig, ax = _solo(title="Tyre-life distribution")
    tl = df["TyreLife"].dropna()
    ax.hist(tl, bins=50, color=TEAL, alpha=0.78, edgecolor="white", linewidth=0.3)
    for exp, comp in [(18, "SOFT"), (28, "MEDIUM"), (40, "HARD")]:
        ax.axvline(exp, color=COMPOUND_COLORS[comp], ls="--", lw=1.6,
                   label=f"{comp} expected ({exp} laps)")
    ax.set_xlabel("Tyre life (laps)")
    ax.set_ylabel("Count")
    ax.legend()
    save(fig, f"{out}/01b_tyre_life_dist.png")


def plot_01c_pitstop_lap_dist(df, out):
    fig, ax = _solo(title="Pit-stop lap distribution")
    pit_laps = df[df["HasPitStop"] == 1]["LapNumber"].dropna()
    ax.hist(pit_laps, bins=55, color=ORANGE, alpha=0.78,
            edgecolor="white", linewidth=0.3, density=True, label="Pit stops")
    x_kde = np.linspace(0, pit_laps.max(), 300)
    kde = stats.gaussian_kde(pit_laps, bw_method=0.2)
    ax.plot(x_kde, kde(x_kde), color=TEXT, lw=2.2, label="KDE")
    ax.axvline(pit_laps.mean(), color=F1_RED, ls="--", lw=1.5,
               label=f"Mean: lap {pit_laps.mean():.1f}")
    ax.set_xlabel("Lap number")
    ax.set_ylabel("Density")
    ax.legend()
    save(fig, f"{out}/01c_pitstop_lap_dist.png")


def plot_01d_pit_rate_season(df, out):
    """If a single season is available, plot pit-stop rate by race
    (more informative); else fall back to the seasonal view."""
    years = sorted(df["Year"].unique())
    if len(years) >= 3:
        fig, ax = _solo(figsize=FIG_WIDE, title="Pit-stop rate by season")
        rates = [df[df["Year"] == y]["HasPitStop"].mean() * 100 for y in years]
        colors = [F1_RED if r == max(rates) else BLUE for r in rates]
        bars = ax.bar(years, rates, color=colors, alpha=0.85, width=0.7,
                      edgecolor="white", linewidth=0.5)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.03,
                    f"{rate:.1f}%", ha="center", va="bottom",
                    fontsize=11, color=TEXT)
        ax.set_xlabel("Season")
        ax.set_ylabel("Pit-stop rate (%)")
        ax.set_xticks(years)
    else:
        # Per-race breakdown
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        ax.set_title("Pit-stop rate by race  ({})"
                     .format("/".join(str(y) for y in years)))
        race_rate = (df.groupby("RaceID")["HasPitStop"]
                     .mean().sort_values() * 100)
        race_short = [r.replace(" Grand Prix", "") for r in race_rate.index]
        max_r = race_rate.max()
        colors = [F1_RED if r == max_r else BLUE for r in race_rate.values]
        ax.barh(race_short, race_rate.values, color=colors, alpha=0.85,
                edgecolor="white", linewidth=0.4)
        for i, r in enumerate(race_rate.values):
            ax.text(r + max_r * 0.01, i, f"{r:.1f}%",
                    va="center", fontsize=10, color=TEXT)
        ax.set_xlabel("Pit-stop rate (% of laps)")
        ax.set_xlim(right=max_r * 1.12)
    save(fig, f"{out}/01d_pit_rate_season.png")


def plot_dataset_overview(df, out):
    plot_01a_lap_time_by_compound(df, out)
    plot_01b_tyre_life_dist(df, out)
    plot_01c_pitstop_lap_dist(df, out)
    plot_01d_pit_rate_season(df, out)


# =========================================================================
# 02  Tyre analysis  — 4 charts
# =========================================================================
def plot_02a_tyre_decay_compound(df, out):
    fig, ax = _solo(title="Tyre performance decay by compound  (mean ± 1σ)")
    for comp in COMPOUND_ORDER:
        sub = df[df["Compound"] == comp].copy()
        sub = sub[sub["TyreLife"] <= 52]
        grp = (sub.groupby("TyreLife")["tire_performance_decay"]
               .agg(["mean", "std"]).reset_index())
        grp = grp[grp["TyreLife"] > 0]
        col = COMPOUND_COLORS[comp]
        ax.plot(grp["TyreLife"], grp["mean"], color=col, lw=2.5, label=comp)
        ax.fill_between(grp["TyreLife"],
                        grp["mean"] - grp["std"], grp["mean"] + grp["std"],
                        color=col, alpha=0.15)
    ax.set_xlabel("Tyre life (laps)")
    ax.set_ylabel("Performance decay (seconds vs. best lap)")
    ax.legend(title="Compound")
    save(fig, f"{out}/02a_tyre_decay_compound.png")


def plot_02b_tyre_life_at_pit(df, out):
    fig, ax = _solo(title="Tyre life at pit stop by compound")
    pit_df = df[df["HasPitStop"] == 1]
    data_by_comp = [pit_df[pit_df["Compound"] == c]["TyreLife"].dropna().values
                    for c in COMPOUND_ORDER]
    bp = ax.boxplot(data_by_comp, patch_artist=True, notch=False,
                    medianprops=dict(color=TEXT, linewidth=2),
                    whiskerprops=dict(color=MUTED),
                    capprops=dict(color=MUTED),
                    flierprops=dict(marker="o", markersize=3,
                                   markerfacecolor=MUTED, alpha=0.3))
    for patch, col in zip(bp["boxes"],
                          [COMPOUND_COLORS[c] for c in COMPOUND_ORDER]):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
    ax.set_xticklabels(COMPOUND_ORDER)
    ax.set_xlabel("Compound")
    ax.set_ylabel("Tyre life at pit (laps)")
    for i, d in enumerate(data_by_comp):
        if len(d):
            med = np.median(d)
            ax.text(i + 1, med + 0.6, f"{med:.0f}", ha="center",
                    fontsize=11, color=TEXT, fontweight="bold")
    save(fig, f"{out}/02b_tyre_life_at_pit.png")


def plot_02c_pitlap_heatmap(df, out):
    fig, ax = plt.subplots(figsize=FIG_HEATMAP)
    pit_sub = df[df["HasPitStop"] == 1].copy()
    pit_sub["Stint_int"] = pit_sub["Stint"].astype(int).clip(1, 4)
    hm_data = (pit_sub.groupby(["Stint_int", "Compound"])["LapNumber"]
               .mean().unstack(fill_value=0))
    hm_data = hm_data.reindex(columns=COMPOUND_ORDER, fill_value=0)
    sns.heatmap(hm_data, ax=ax, cmap="YlOrRd", annot=True, fmt=".1f",
                linewidths=0.5, cbar_kws={"label": "Mean pit lap"},
                annot_kws={"size": 12, "color": TEXT})
    ax.set_title("Mean pit lap by compound × stint number")
    ax.set_xlabel("Compound")
    ax.set_ylabel("Stint number")
    save(fig, f"{out}/02c_pitlap_heatmap.png")


def plot_02d_stint_length(df, out):
    fig, ax = _solo(title="Stint-length distribution by compound")
    stint_len = (df.groupby(["Year", "RaceID", "Driver", "Stint"])
                 ["LapNumber"].count().reset_index(name="StintLen"))
    comp_per_stint = (df.groupby(["Year", "RaceID", "Driver", "Stint"])
                      ["Compound"].agg(lambda x: x.mode()[0]).reset_index())
    stint_len = stint_len.merge(
        comp_per_stint, on=["Year", "RaceID", "Driver", "Stint"], how="left")
    for comp in COMPOUND_ORDER:
        sub = stint_len[stint_len["Compound"] == comp]["StintLen"]
        ax.hist(sub, bins=35, alpha=0.55, color=COMPOUND_COLORS[comp],
                label=comp, density=True, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Stint length (laps)")
    ax.set_ylabel("Density")
    ax.legend(title="Compound")
    save(fig, f"{out}/02d_stint_length.png")


def plot_tyre_analysis(df, out):
    plot_02a_tyre_decay_compound(df, out)
    plot_02b_tyre_life_at_pit(df, out)
    plot_02c_pitlap_heatmap(df, out)
    plot_02d_stint_length(df, out)


# =========================================================================
# 03  BiLSTM model outputs  — 4 charts
# =========================================================================
def plot_03a_pred_prob_dist(df, out, y_true, y_prob):
    fig, ax = _solo(title="Predicted-probability distribution")
    bins = np.linspace(0, 1, 50)
    ax.hist(y_prob[y_true == 0], bins=bins, density=True, alpha=0.7,
            color=TEAL, label="No pit (0)", edgecolor="white", linewidth=0.3)
    ax.hist(y_prob[y_true == 1], bins=bins, density=True, alpha=0.85,
            color=F1_RED, label="Pit stop (1)", edgecolor="white", linewidth=0.3)
    ax.axvline(0.5, color=TEXT, ls="--", lw=1.5, label="Threshold = 0.5")
    ax.set_xlabel("Predicted pit-stop probability")
    ax.set_ylabel("Density")
    ax.legend(loc="upper center")
    ax.text(0.97, 0.95,
            f"Pit laps: {int(y_true.sum()):,}\n"
            f"Non-pit laps: {int((y_true == 0).sum()):,}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, color=MUTED,
            bbox=dict(fc=PAGE_BG, ec=GRID_C, boxstyle="round,pad=0.4"))
    save(fig, f"{out}/03a_pred_prob_dist.png")


def plot_03b_roc(df, out, y_true, y_prob):
    fig, ax = _solo(title="ROC curve")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=ORANGE, lw=2.5, label=f"BiLSTM  AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1.2, label="Random baseline")
    ax.fill_between(fpr, tpr, alpha=0.10, color=ORANGE)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    save(fig, f"{out}/03b_roc_curve.png")


def plot_03c_pr(df, out, y_true, y_prob):
    fig, ax = _solo(title="Precision–recall curve")
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    ax.plot(recall, precision, color=BLUE, lw=2.5,
            label=f"BiLSTM  AP = {ap:.4f}")
    baseline = float(y_true.mean())
    ax.axhline(baseline, color=MUTED, ls="--", lw=1.2,
               label=f"Baseline (class freq = {baseline:.3f})")
    ax.fill_between(recall, precision, alpha=0.12, color=BLUE)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    save(fig, f"{out}/03c_pr_curve.png")


def plot_03d_calibration(df, out, y_true, y_prob):
    fig, ax = _solo(title="Calibration (reliability diagram)")
    n_bins = 15
    bin_edges = np.linspace(0, 1, n_bins + 1)
    frac_pos, mean_pred = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() > 20:
            frac_pos.append(float(y_true[mask].mean()))
            mean_pred.append(float(y_prob[mask].mean()))
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1.4, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color=PURPLE, lw=2, ms=6, label="BiLSTM")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, f"{out}/03d_calibration.png")


def plot_model_outputs(df, out):
    if "PitStopProbability" not in df.columns:
        print("  [skip] no PitStopProbability column")
        return
    scored = df[df["PitStopProbability"] > 0].copy()
    if len(scored) == 0 or scored["HasPitStop"].nunique() < 2:
        print("  [skip] no model predictions available "
              "(run train.py + predict_pit_probabilities.py first)")
        return
    y_true = scored["HasPitStop"].values
    y_prob = scored["PitStopProbability"].values
    print(f"  BiLSTM eval subset: {len(scored):,} laps  "
          f"({int(y_true.sum()):,} pits, {y_true.mean() * 100:.2f}%)")
    for name, fn in [
        ("03a", plot_03a_pred_prob_dist),
        ("03b", plot_03b_roc),
        ("03c", plot_03c_pr),
        ("03d", plot_03d_calibration),
    ]:
        try:
            fn(df, out, y_true, y_prob)
        except Exception as e:
            print(f"  [ERR] {name}: {e}")


# =========================================================================
# 04  Team strategy  — 3 charts
# =========================================================================
def plot_04a_risk_scatter(df, out):
    pit_df = df[df["HasPitStop"] == 1].copy()
    team_stats = (pit_df.groupby("Team")["TyreLife"]
                  .agg(["mean", "var", "count"]).reset_index())
    team_stats.columns = ["Team", "Mean", "Variance", "Count"]
    team_stats = team_stats[team_stats["Count"] >= 30].copy()

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.set_title("Strategy risk scatter:  E[X] vs Var(X)")

    for _, row in team_stats.iterrows():
        col = TEAM_COLORS.get(row["Team"], SOFT_GREY)
        ax.scatter(row["Mean"], row["Variance"], s=160, color=col,
                   edgecolor=TEXT, linewidth=0.6, zorder=3)
        ax.annotate(row["Team"], (row["Mean"], row["Variance"]),
                    textcoords="offset points", xytext=(7, 4),
                    fontsize=10, color=TEXT)
    ax.axhline(team_stats["Variance"].mean(), color=MUTED, ls=":", lw=1)
    ax.axvline(team_stats["Mean"].mean(),    color=MUTED, ls=":", lw=1)

    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    pad_x = (xlim[1] - xlim[0]) * 0.02
    pad_y = (ylim[1] - ylim[0]) * 0.02
    ax.text(xlim[0] + pad_x, ylim[1] - pad_y, "Early & reactive",
            fontsize=10, color=MUTED, ha="left", va="top", style="italic")
    ax.text(xlim[1] - pad_x, ylim[1] - pad_y, "Late & reactive",
            fontsize=10, color=MUTED, ha="right", va="top", style="italic")
    ax.text(xlim[0] + pad_x, ylim[0] + pad_y, "Early & deterministic",
            fontsize=10, color=MUTED, ha="left", va="bottom", style="italic")
    ax.text(xlim[1] - pad_x, ylim[0] + pad_y, "Late & deterministic",
            fontsize=10, color=MUTED, ha="right", va="bottom", style="italic")

    ax.set_xlabel("Mean tyre life at pit  E[X]  (laps)")
    ax.set_ylabel("Variance of tyre life  Var(X)  (laps²)")
    save(fig, f"{out}/04a_risk_scatter.png")


def plot_04b_team_tyrelife_box(df, out):
    pit_df = df[df["HasPitStop"] == 1].copy()
    team_order = (pit_df.groupby("Team")["TyreLife"].median()
                  .sort_values().index.tolist())
    team_order = [t for t in team_order
                  if pit_df[pit_df["Team"] == t].shape[0] >= 30]
    data_list   = [pit_df[pit_df["Team"] == t]["TyreLife"].values
                   for t in team_order]
    colors_list = [TEAM_COLORS.get(t, SOFT_GREY) for t in team_order]

    h = max(4.5, 0.32 * len(team_order))
    fig, ax = plt.subplots(figsize=(8.5, h))
    ax.set_title("Tyre life at pit stop by team  (ordered by median)")

    bp = ax.boxplot(data_list, vert=False, patch_artist=True,
                    medianprops=dict(color=TEXT, linewidth=2),
                    whiskerprops=dict(color=MUTED),
                    capprops=dict(color=MUTED),
                    flierprops=dict(marker="o", markersize=2.5,
                                   markerfacecolor=MUTED, alpha=0.35))
    for patch, col in zip(bp["boxes"], colors_list):
        patch.set_facecolor(col)
        patch.set_alpha(0.72)
    ax.set_yticks(range(1, len(team_order) + 1))
    ax.set_yticklabels(team_order, fontsize=11)
    ax.set_xlabel("Tyre life (laps)")
    save(fig, f"{out}/04b_team_tyrelife_box.png")


def plot_04c_compound_mix_heatmap(df, out):
    pit_df = df[df["HasPitStop"] == 1].copy()
    team_order = (pit_df.groupby("Team")["TyreLife"].median()
                  .sort_values().index.tolist())
    team_order = [t for t in team_order
                  if pit_df[pit_df["Team"] == t].shape[0] >= 30]
    comp_heat = (pit_df.groupby(["Team", "Compound"])
                 .size().unstack(fill_value=0))
    comp_heat_pct = comp_heat.div(comp_heat.sum(axis=1), axis=0) * 100
    comp_heat_pct = comp_heat_pct.reindex(columns=COMPOUND_ORDER, fill_value=0)
    comp_heat_pct = comp_heat_pct.reindex(
        [t for t in team_order if t in comp_heat_pct.index])

    h = max(5.0, 0.32 * len(team_order))
    fig, ax = plt.subplots(figsize=(7.5, h))
    sns.heatmap(comp_heat_pct, ax=ax, cmap="RdYlGn", center=33,
                annot=True, fmt=".0f", linewidths=0.4,
                cbar_kws={"label": "% of pit stops"},
                annot_kws={"size": 11, "color": TEXT})
    ax.set_title("Compound mix at pit stop by team  (%)")
    ax.set_xlabel("Compound")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    save(fig, f"{out}/04c_compound_mix_heatmap.png")


def plot_team_strategy(df, out):
    for name, fn in [
        ("04a", plot_04a_risk_scatter),
        ("04b", plot_04b_team_tyrelife_box),
        ("04c", plot_04c_compound_mix_heatmap),
    ]:
        try:
            fn(df, out)
        except Exception as e:
            print(f"  [ERR] {name}: {e}")


# =========================================================================
# 05  Race dynamics  — 4 charts
# =========================================================================
def plot_05a_position_changes(df, out):
    fig, ax = _solo(figsize=FIG_WIDE, title="Total position changes by lap")
    d = df.sort_values(["Year", "RaceID", "Driver", "LapNumber"])
    d["PosChange"] = (d.groupby(["Year", "RaceID", "Driver"])["Position"]
                      .diff().abs().fillna(0))
    changes = (d.groupby("LapNumber")["PosChange"].sum().reset_index())
    changes = changes[changes["LapNumber"] <= 70]
    phase_colors = np.where(changes["LapNumber"] <= 15, F1_RED,
                   np.where(changes["LapNumber"] <= 40, ORANGE, TEAL))
    ax.bar(changes["LapNumber"], changes["PosChange"],
           color=phase_colors, width=1.0, edgecolor="none", alpha=0.88)
    smooth = gaussian_filter1d(changes["PosChange"].values, sigma=2)
    ax.plot(changes["LapNumber"], smooth, color=TEXT, lw=1.8, zorder=5,
            label="Smoothed trend")

    handles = [
        mpatches.Patch(color=F1_RED, label="Early (laps 1–15)"),
        mpatches.Patch(color=ORANGE, label="Mid   (laps 16–40)"),
        mpatches.Patch(color=TEAL,   label="Late  (laps 41+)"),
        plt.Line2D([0], [0], color=TEXT, lw=1.8, label="Smoothed trend"),
    ]
    ax.legend(handles=handles, loc="upper right")
    ax.set_xlabel("Lap number")
    ax.set_ylabel("Sum of |position changes|")
    save(fig, f"{out}/05a_position_changes.png")


def plot_05b_pit_prob_progress(df, out):
    fig, ax = _solo(title="Mean pit probability over race progress")
    scored = df[df["PitStopProbability"] > 0].copy()
    scored["progress_bin"] = pd.cut(scored["race_progress_fraction"],
                                    bins=30, labels=False)
    prob_trace = (scored.groupby("progress_bin")["PitStopProbability"]
                  .mean().reset_index())
    prob_pit   = (scored[scored["HasPitStop"] == 1]
                  .groupby("progress_bin")["PitStopProbability"]
                  .mean().reset_index())
    ax.fill_between(prob_trace["progress_bin"],
                    prob_trace["PitStopProbability"],
                    alpha=0.22, color=BLUE)
    ax.plot(prob_trace["progress_bin"], prob_trace["PitStopProbability"],
            color=BLUE, lw=2.2, label="Mean prob. (all laps)")
    ax.scatter(prob_pit["progress_bin"], prob_pit["PitStopProbability"],
               color=F1_RED, s=22, zorder=5, label="Mean prob. (pit laps)")
    ax.set_xlabel("Race-progress bin  (0 = start → 29 = end)")
    ax.set_ylabel("Mean PitStopProbability")
    ax.legend(loc="upper left")
    save(fig, f"{out}/05b_pit_prob_progress.png")


def plot_05c_pit_compound_season(df, out):
    """If a single season is available, plot per-race; otherwise per-season."""
    years = sorted(df["Year"].unique())
    pit_sub = df[df["HasPitStop"] == 1]

    if len(years) >= 3:
        fig, ax = _solo(figsize=FIG_WIDE,
                        title="Pit stops by compound per season")
        comp_year = (pit_sub.groupby(["Year", "Compound"]).size()
                     .unstack(fill_value=0)
                     .reindex(columns=COMPOUND_ORDER, fill_value=0))
        bottom = np.zeros(len(comp_year))
        xs = comp_year.index.tolist()
        for comp in COMPOUND_ORDER:
            vals = comp_year[comp].values
            ax.bar(xs, vals, bottom=bottom,
                   color=COMPOUND_COLORS[comp], label=comp,
                   alpha=0.85, edgecolor="white", linewidth=0.5)
            bottom += vals
        ax.set_xlabel("Season")
        ax.set_xticks(xs)
    else:
        fig, ax = plt.subplots(figsize=(10.5, 5.5))
        ax.set_title("Pit stops by compound per race  ({})"
                     .format("/".join(str(y) for y in years)))
        comp_race = (pit_sub.groupby(["RaceID", "Compound"]).size()
                     .unstack(fill_value=0)
                     .reindex(columns=COMPOUND_ORDER, fill_value=0))
        # Order races by total pit-stop count (descending) for readability
        comp_race["__total"] = comp_race.sum(axis=1)
        comp_race = comp_race.sort_values("__total", ascending=True)
        race_short = [r.replace(" Grand Prix", "") for r in comp_race.index]

        bottom = np.zeros(len(comp_race))
        for comp in COMPOUND_ORDER:
            vals = comp_race[comp].values
            ax.barh(race_short, vals, left=bottom,
                    color=COMPOUND_COLORS[comp], label=comp,
                    alpha=0.85, edgecolor="white", linewidth=0.4)
            bottom += vals
        ax.set_xlabel("Number of pit stops")

    ax.set_ylabel("" if len(years) < 3 else "Number of pit stops")
    ax.legend(title="Compound", loc="lower right")
    save(fig, f"{out}/05c_pit_compound_season.png")


def plot_05d_pit_prob_heatmap(df, out):
    fig, ax = plt.subplots(figsize=FIG_HEATMAP)
    scored2 = df[(df["PitStopProbability"] > 0) & (df["TyreLife"] <= 50)].copy()
    scored2["TyreLife_int"] = scored2["TyreLife"].astype(int)
    hm = (scored2.groupby(["TyreLife_int", "Compound"])["PitStopProbability"]
          .mean().unstack(fill_value=0)
          .reindex(columns=COMPOUND_ORDER, fill_value=0))
    sns.heatmap(hm.T, ax=ax, cmap="YlOrRd",
                cbar_kws={"label": "Mean PitStopProbability"},
                linewidths=0)
    ax.set_title("Mean pit probability — tyre life × compound")
    ax.set_xlabel("Tyre life (laps)")
    ax.set_ylabel("Compound")
    ax.tick_params(axis="y", rotation=0)
    step = max(1, len(hm.index) // 12)
    ax.set_xticks(range(0, len(hm.index), step))
    ax.set_xticklabels(hm.index[::step], rotation=0)
    save(fig, f"{out}/05d_pit_prob_heatmap.png")


def plot_race_dynamics(df, out):
    plot_05a_position_changes(df, out)
    plot_05c_pit_compound_season(df, out)
    if "PitStopProbability" in df.columns and (df["PitStopProbability"] > 0).any():
        for name, fn in [
            ("05b", plot_05b_pit_prob_progress),
            ("05d", plot_05d_pit_prob_heatmap),
        ]:
            try:
                fn(df, out)
            except Exception as e:
                print(f"  [ERR] {name}: {e}")
    else:
        print("  [skip] 05b/05d need model predictions")


# =========================================================================
# 06  Feature analysis  — 4 charts
# =========================================================================
def plot_06a_corr_matrix(df, out):
    """Tight 8-feature correlation matrix — large readable cells."""
    CORE_FEATURES = [
        "TyreLife", "tire_performance_decay", "pace_degradation_slope",
        "relative_tire_age", "pit_window_delta", "traffic_pressure",
        "race_progress_fraction", "HasPitStop",
    ]
    DISPLAY_NAMES = {
        "TyreLife":               "Tyre life",
        "tire_performance_decay": "Perf. decay",
        "pace_degradation_slope": "Pace slope",
        "relative_tire_age":      "Rel. tyre age",
        "pit_window_delta":       "Pit-window Δ",
        "traffic_pressure":       "Traffic pressure",
        "race_progress_fraction": "Race progress",
        "HasPitStop":             "Pit stop",
    }
    available = [c for c in CORE_FEATURES if c in df.columns]
    core_df = df[available].dropna()
    corr = core_df.corr()
    corr.index = [DISPLAY_NAMES.get(c, c) for c in corr.index]
    corr.columns = [DISPLAY_NAMES.get(c, c) for c in corr.columns]

    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    ax.set_title("Feature correlation matrix")
    # Mask the upper triangle AND the trivial diagonal
    mask = np.triu(np.ones_like(corr, dtype=bool), k=0)
    cmap = LinearSegmentedColormap.from_list(
        "f1div", [BLUE, "#FFFFFF", F1_RED], N=256)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True,
                vmin=-0.7, vmax=0.7, linewidths=0.6, annot=True, fmt=".2f",
                annot_kws={"size": 13, "color": TEXT},
                cbar_kws={"shrink": 0.70, "label": "Pearson r"}, ax=ax)
    ax.tick_params(axis="x", rotation=35, labelsize=12)
    ax.tick_params(axis="y", rotation=0,  labelsize=12)
    plt.setp(ax.get_xticklabels(), ha="right")
    save(fig, f"{out}/06a_corr_matrix.png")


def plot_06b_feature_violin(df, out):
    fig, ax = _solo(figsize=FIG_WIDE,
                    title="Feature distribution — pit vs. non-pit  (normalised 0–1)")
    VIOL_FEATURES = [
        "TyreLife", "tire_performance_decay",
        "traffic_pressure", "pace_degradation_slope",
        "relative_tire_age", "pit_window_delta",
    ]
    available = [c for c in VIOL_FEATURES if c in df.columns]
    sample = df.sample(min(20000, len(df)),
                       random_state=42)[available + ["HasPitStop"]].dropna().copy()
    for f in available:
        rng = sample[f].max() - sample[f].min()
        if rng > 0:
            sample[f] = (sample[f] - sample[f].min()) / rng
    melted = sample.melt(id_vars="HasPitStop", value_vars=available,
                         var_name="Feature", value_name="Scaled value")
    melted["Pit stop"] = melted["HasPitStop"].map({0: "No pit", 1: "Pit"})
    sns.violinplot(data=melted, x="Feature", y="Scaled value",
                   hue="Pit stop",
                   palette={"No pit": TEAL, "Pit": F1_RED},
                   split=True, inner="quart", linewidth=0.7,
                   ax=ax, legend=True)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    save(fig, f"{out}/06b_feature_violin.png")


def plot_06c_traffic_pit_prob(df, out):
    fig, ax = _solo(title="Traffic pressure vs. predicted pit probability")
    scored = df[(df["PitStopProbability"] > 0.001) &
                (df["traffic_pressure"] < df["traffic_pressure"].quantile(0.97))].copy()
    scored["tp_bin"] = pd.qcut(scored["traffic_pressure"], 30,
                                duplicates="drop", labels=False)
    tp_agg = (scored.groupby("tp_bin")
              .agg(tp_mean=("traffic_pressure", "mean"),
                   prob_mean=("PitStopProbability", "mean"),
                   prob_std=("PitStopProbability", "std"))
              .reset_index())
    ax.plot(tp_agg["tp_mean"], tp_agg["prob_mean"],
            color=PURPLE, lw=2.6, label="Mean")
    ax.fill_between(tp_agg["tp_mean"],
                    tp_agg["prob_mean"] - tp_agg["prob_std"],
                    tp_agg["prob_mean"] + tp_agg["prob_std"],
                    alpha=0.20, color=PURPLE, label="±1σ")
    ax.set_xlabel("Traffic pressure  (1 / DistanceToDriverAhead)")
    ax.set_ylabel("Mean PitStopProbability")
    ax.legend()
    save(fig, f"{out}/06c_traffic_pit_prob.png")


def plot_06d_pace_decay_compound(df, out):
    fig, ax = _solo(title="Pace-degradation slope by compound  (AR(1)-filtered)")

    # Trim to the central distribution so violins are readable.
    # Pace slope is in seconds/lap; values beyond ±1.5 are pit-affected or SC.
    slope_data, ylim_max = [], 0.0
    for comp in COMPOUND_ORDER:
        vals = (df[(df["Compound"] == comp) &
                   (df["pace_degradation_slope"].between(-1.5, 1.5))]
                ["pace_degradation_slope"].values)
        slope_data.append(vals)
        if len(vals):
            ylim_max = max(ylim_max,
                           float(np.percentile(np.abs(vals), 98)))

    vp = ax.violinplot(slope_data, positions=[1, 2, 3],
                       showmedians=True, showextrema=False,
                       widths=0.85)
    for body, col in zip(vp["bodies"],
                         [COMPOUND_COLORS[c] for c in COMPOUND_ORDER]):
        body.set_facecolor(col)
        body.set_alpha(0.70)
        body.set_edgecolor(TEXT)
        body.set_linewidth(0.6)
    vp["cmedians"].set_color(TEXT)
    vp["cmedians"].set_linewidth(1.8)

    # Annotate each violin with its median and IQR width
    for i, vals in enumerate(slope_data):
        if len(vals):
            med = float(np.median(vals))
            q1, q3 = np.percentile(vals, [25, 75])
            ax.text(i + 1, ylim_max * 1.05,
                    f"med={med:.3f}\nIQR={q3 - q1:.3f}",
                    ha="center", va="bottom", fontsize=10, color=TEXT)

    ax.axhline(0, color=MUTED, ls="--", lw=1.2)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(COMPOUND_ORDER)
    ax.set_xlabel("Compound")
    ax.set_ylabel("Pace-degradation slope (s/lap)")
    # Tight y limits so the violins fill the panel
    ax.set_ylim(-ylim_max * 1.10, ylim_max * 1.35)
    save(fig, f"{out}/06d_pace_decay_compound.png")


def plot_feature_analysis(df, out):
    plot_06a_corr_matrix(df, out)
    plot_06b_feature_violin(df, out)
    plot_06d_pace_decay_compound(df, out)
    if "PitStopProbability" in df.columns and (df["PitStopProbability"] > 0).any():
        try:
            plot_06c_traffic_pit_prob(df, out)
        except Exception as e:
            print(f"  [ERR] 06c: {e}")
    else:
        print("  [skip] 06c needs model predictions")


# =========================================================================
# 07  Stochastic & survival  — 4 charts
# =========================================================================
def plot_07a_hazard_rate(df, out):
    fig, ax = _solo(title="Empirical hazard rate  h(t)  by compound")
    for comp in COMPOUND_ORDER:
        sub = df[df["Compound"] == comp].copy()
        grp = (sub.groupby("TyreLife")["HasPitStop"]
               .agg(["sum", "count"]).reset_index())
        grp = grp[grp["TyreLife"] <= 52]
        grp["hazard"] = grp["sum"] / grp["count"].clip(lower=1)
        smooth_h = gaussian_filter1d(grp["hazard"].values, sigma=1.5)
        ax.plot(grp["TyreLife"], smooth_h,
                color=COMPOUND_COLORS[comp], lw=2.5, label=comp)
        ax.fill_between(grp["TyreLife"], smooth_h, alpha=0.10,
                        color=COMPOUND_COLORS[comp])
    ax.set_xlabel("Tyre life (laps)")
    ax.set_ylabel("P(pit | tyre life = t)")
    ax.legend(title="Compound")
    save(fig, f"{out}/07a_hazard_rate.png")


def plot_07b_pace_decay_hist(df, out):
    fig, ax = _solo(title="Stationary pace decay  Δt = Xₜ − Xₜ₋₁")
    decay = df["Stationary_PaceDecay"].dropna()
    decay = decay[decay.between(-8, 8)]
    ax.hist(decay, bins=80, color=ORANGE, alpha=0.78, density=True,
            edgecolor="white", linewidth=0.3, label="AR(1) ΔLapTime")
    x_r = np.linspace(-8, 8, 300)
    mu, sigma = stats.norm.fit(decay)
    ax.plot(x_r, stats.norm.pdf(x_r, mu, sigma),
            color=TEXT, lw=2, label=f"Normal  μ={mu:.3f}, σ={sigma:.3f}")
    ax.axvline(0, color=MUTED, ls="--", lw=1)
    ax.set_xlabel("Pace decay (seconds)")
    ax.set_ylabel("Density")
    ax.legend()
    save(fig, f"{out}/07b_pace_decay_hist.png")


def plot_07c_cox_vs_bilstm(df, out):
    fig, ax = _solo(title="Cox Pit_Probability vs. BiLSTM PitStopProbability")
    both = df[(df["Pit_Probability"] > 0) &
              (df["PitStopProbability"] > 0)].copy()
    sample = both.sample(min(8000, len(both)), random_state=42)
    col_by_pit = np.where(sample["HasPitStop"] == 1, F1_RED, TEAL)
    ax.scatter(sample["Pit_Probability"], sample["PitStopProbability"],
               c=col_by_pit, s=10, alpha=0.4, linewidths=0)
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1.4, label="Identity")
    both["cox_bin"] = pd.qcut(both["Pit_Probability"], 30,
                               duplicates="drop", labels=False)
    trend = (both.groupby("cox_bin")
             .agg(cox_mean=("Pit_Probability", "mean"),
                  bi_mean=("PitStopProbability", "mean"))
             .reset_index())
    ax.plot(trend["cox_mean"], trend["bi_mean"],
            color=PURPLE, lw=2.2, label="Binned trend")
    handles = [
        mpatches.Patch(color=F1_RED, label="Pit lap"),
        mpatches.Patch(color=TEAL,   label="Non-pit lap"),
        plt.Line2D([0], [0], color=PURPLE, lw=2.2, label="Binned trend"),
        plt.Line2D([0], [0], color=MUTED, ls="--", label="Identity"),
    ]
    ax.legend(handles=handles, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Cox proportional hazard  Pit_Probability")
    ax.set_ylabel("BiLSTM  PitStopProbability")
    save(fig, f"{out}/07c_cox_vs_bilstm.png")


def plot_07d_vsc_pit_prob(df, out):
    fig, ax = _solo(title="Mean pit probability around VSC events  (±8 laps)")

    def is_vsc(ts):
        s = str(int(ts)) if not np.isnan(ts) else ""
        return "4" in s or "6" in s

    df = df.copy()
    df["IsVSC"] = df["TrackStatus"].apply(
        lambda x: is_vsc(x) if pd.notna(x) else False)

    scored = df[df["PitStopProbability"] > 0].copy()
    scored["LapNumber_int"] = scored["LapNumber"].astype(int)

    window = 8
    vsc_laps = scored[scored["IsVSC"]]["LapNumber_int"].values
    offsets = list(range(-window, window + 1))
    prob_at_offset = {o: [] for o in offsets}
    for vsc_lap in vsc_laps:
        for offset in offsets:
            target = vsc_lap + offset
            row_probs = scored[scored["LapNumber_int"] == target][
                "PitStopProbability"].values
            if len(row_probs):
                prob_at_offset[offset].extend(row_probs.tolist())

    means = [np.mean(prob_at_offset[o]) if prob_at_offset[o] else 0
             for o in offsets]
    stds  = [np.std(prob_at_offset[o])  if prob_at_offset[o] else 0
             for o in offsets]

    ax.bar(offsets, means,
           color=[F1_RED if o >= 0 else BLUE for o in offsets],
           alpha=0.78, edgecolor="white", linewidth=0.4)
    ax.errorbar(offsets, means, yerr=stds, fmt="none",
                color=TEXT, capsize=3, linewidth=0.8)
    ax.axvline(0, color=ORANGE, ls="--", lw=1.5, label="VSC deployment")
    ax.set_xlabel("Laps relative to VSC deployment  (0 = deployment lap)")
    ax.set_ylabel("Mean PitStopProbability")
    ax.legend(loc="upper left")
    save(fig, f"{out}/07d_vsc_pit_prob.png")


def plot_stochastic(df, out):
    plot_07a_hazard_rate(df, out)
    if "Stationary_PaceDecay" in df.columns and df["Stationary_PaceDecay"].notna().any():
        plot_07b_pace_decay_hist(df, out)
    has_predictions = ("PitStopProbability" in df.columns
                       and (df["PitStopProbability"] > 0).any())
    if "Pit_Probability" in df.columns and (df["Pit_Probability"] > 0).any() and has_predictions:
        try:
            plot_07c_cox_vs_bilstm(df, out)
        except Exception as e:
            print(f"  [ERR] 07c: {e}")
    else:
        print("  [skip] 07c needs BiLSTM + Cox predictions")
    if "TrackStatus" in df.columns and has_predictions:
        try:
            plot_07d_vsc_pit_prob(df, out)
        except Exception as e:
            print(f"  [ERR] 07d: {e}")
    else:
        print("  [skip] 07d needs model predictions")


# =========================================================================
# Section dispatcher
# =========================================================================
SECTION_MAP = {
    "dataset":    plot_dataset_overview,
    "tyres":      plot_tyre_analysis,
    "model":      plot_model_outputs,
    "teams":      plot_team_strategy,
    "race":       plot_race_dynamics,
    "features":   plot_feature_analysis,
    "stochastic": plot_stochastic,
}


def parse_args() -> argparse.Namespace:
    from src.pit_stop_strategy.paths import AGGREGATED_CSV, PLOTS_DIR
    parser = argparse.ArgumentParser(
        description="Generate F1 strategy plots (one PNG per chart).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data",   default=str(AGGREGATED_CSV),
                        help=f"Aggregated CSV (default: {AGGREGATED_CSV})")
    parser.add_argument("--output", default=str(PLOTS_DIR),
                        help=f"Output directory (default: {PLOTS_DIR})")
    parser.add_argument("--section",
                        choices=list(SECTION_MAP.keys()) + ["all"],
                        default="all",
                        help="Which section to render (default: all)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_theme()
    df = load_data(args.data)

    sections = (list(SECTION_MAP.items()) if args.section == "all"
                else [(args.section, SECTION_MAP[args.section])])

    print(f"\nGenerating {len(sections)} section(s) -> {args.output}/\n")
    for name, fn in sections:
        print(f"-- {name.upper()}")
        try:
            fn(df, args.output)
        except KeyError as e:
            print(f"  [skip] {name}: required column missing {e}")
        except Exception as e:
            print(f"  [ERR ] {name}: {e}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
