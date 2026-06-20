"""
visualize.py — F1 Pit Stop Strategy Intelligence System
========================================================
Generates all analytical plots for the project report.

Usage:
    python visualize.py                          # all plots
    python visualize.py --section model          # one section only
    python visualize.py --output ./my_plots      # custom output dir
    python visualize.py --data path/to/data.csv  # custom data path

Sections:
    dataset     — Dataset overview & distributions
    tyres       — Tyre degradation & compound analysis
    model       — BiLSTM model outputs & calibration
    teams       — Team strategy risk profiles
    race        — Race dynamics & position changes
    features    — Feature correlations & importance
    stochastic  — Stochastic / survival modelling

Output:
    PNG files saved to ./plots/ (or --output directory)
    Summary stats printed to stdout
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
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────

# Palette: F1 dark-mode with compound-accurate colours
BG        = "#0F0F1A"   # page background
PANEL     = "#1A1A2E"   # axes background
GRID      = "#252540"   # gridlines
BORDER    = "#333355"   # spine / tick colour
TEXT      = "#E8E8F0"   # primary text
MUTED     = "#8888AA"   # captions, axis labels

F1_RED    = "#E8002D"   # Ferrari / danger
F1_WHITE  = "#F5F5F5"
ORANGE    = "#FF6B00"   # McLaren / warning
BLUE      = "#1A56FF"   # Williams / info
TEAL      = "#00B89C"   # positive / Mercedes
PURPLE    = "#9B59B6"   # highlight

# Tyre compound colours (official F1)
SOFT_C    = "#FF3333"
MEDIUM_C  = "#FFD700"
HARD_C    = "#C8C8C8"

COMPOUND_COLORS = {"SOFT": SOFT_C, "MEDIUM": MEDIUM_C, "HARD": HARD_C}
COMPOUND_ORDER  = ["SOFT", "MEDIUM", "HARD"]

# Team palette (19 teams across 2018-2025)
TEAM_COLORS = {
    "Mercedes":         "#00D2BE",
    "Red Bull Racing":  "#0600EF",
    "Ferrari":          "#DC0000",
    "McLaren":          "#FF8000",
    "Alpine":           "#0090FF",
    "Aston Martin":     "#006F62",
    "Williams":         "#005AFF",
    "AlphaTauri":       "#2B4562",
    "Alfa Romeo Racing":"#900000",
    "Haas F1 Team":     "#FFFFFF",
    "Racing Point":     "#F596C8",
    "Renault":          "#FFF500",
    "Toro Rosso":       "#469BFF",
    "Force India":      "#F595C8",
    "Sauber":           "#9B0000",
    "Alfa Romeo":       "#C92D4B",
    "RB":               "#1E5BC6",
    "Racing Bulls":     "#6692FF",
    "AlphaTauri":       "#4A99D1",
    "Kick Sauber":      "#52E252",
}


def apply_theme() -> None:
    """Apply the dark F1 theme to all matplotlib elements."""
    plt.rcParams.update({
        # Figure
        "figure.facecolor":     BG,
        "figure.edgecolor":     BG,
        "savefig.facecolor":    BG,
        "savefig.edgecolor":    BG,
        # Axes
        "axes.facecolor":       PANEL,
        "axes.edgecolor":       BORDER,
        "axes.labelcolor":      MUTED,
        "axes.titlecolor":      TEXT,
        "axes.titlesize":       13,
        "axes.titleweight":     "bold",
        "axes.labelsize":       11,
        "axes.grid":            True,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        # Grid
        "grid.color":           GRID,
        "grid.linewidth":       0.7,
        "grid.alpha":           0.6,
        # Ticks
        "xtick.color":          MUTED,
        "ytick.color":          MUTED,
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        # Legend
        "legend.facecolor":     "#22224A",
        "legend.edgecolor":     BORDER,
        "legend.labelcolor":    TEXT,
        "legend.fontsize":      9,
        "legend.title_fontsize": 9,
        # Lines
        "lines.linewidth":      2.0,
        # Font
        "font.family":          "DejaVu Sans",
        "text.color":           TEXT,
    })


def make_fig(nrows: int = 1, ncols: int = 1, figsize: tuple = (14, 8),
             title: str = "") -> tuple:
    """Create a styled figure + axes array (always 2-D)."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=BG,
                             squeeze=False)
    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold", color=TEXT,
                     y=1.01)
    for ax in axes.flat:
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
    return fig, axes


def save(fig: plt.Figure, path: str) -> None:
    """Save figure tightly, then close it."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  [saved] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    """Load the aggregated training CSV with basic sanity checks."""
    print(f"\nLoading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")

    # Lap-time outlier guard (pit laps can be very slow)
    lt = df["LapTime_Seconds"]
    df = df[(lt > 55) | (lt.isna())]

    # Tyre life outlier guard
    df = df[df["TyreLife"] <= 70]

    # Clip extreme tire decay (safety car artefacts)
    df["tire_performance_decay"] = df["tire_performance_decay"].clip(0, 60)

    print(f"  After cleaning: {len(df):,} rows")
    print(f"  Pit stop rate : {df['HasPitStop'].mean()*100:.2f}%")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATASET OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

def plot_dataset_overview(df: pd.DataFrame, out: str) -> None:
    """
    4-panel figure:
      (A) Lap time density by compound
      (B) Tyre life histogram with expected-life markers
      (C) Pit stop lap distribution (histogram + KDE)
      (D) Yearly pit stop rate (bar chart)
    """
    fig, axes = make_fig(2, 2, (15, 10),
                         "Dataset Overview — F1 Pit Stop Intelligence System (2018–2025)")

    # ── A: Lap time by compound ──
    ax = axes[0, 0]
    lt_all = df["LapTime_Seconds"].dropna()
    lt_all = lt_all[(lt_all > 65) & (lt_all < 175)]
    for comp in COMPOUND_ORDER:
        sub = df[df["Compound"] == comp]["LapTime_Seconds"].dropna()
        sub = sub[(sub > 65) & (sub < 175)]
        if len(sub) == 0:
            continue
        ax.hist(sub, bins=60, alpha=0.55, color=COMPOUND_COLORS[comp],
                label=comp, density=True)
    ax.axvline(lt_all.mean(), color=F1_RED, ls="--", lw=1.5,
               label=f"Overall mean: {lt_all.mean():.1f} s")
    ax.set_title("(A) Lap Time Distribution by Compound")
    ax.set_xlabel("Lap Time (seconds)")
    ax.set_ylabel("Density")
    ax.legend()

    # ── B: Tyre life ──
    ax = axes[0, 1]
    tl = df["TyreLife"].dropna()
    ax.hist(tl, bins=50, color=TEAL, alpha=0.75, edgecolor="none")
    for exp, comp, col in [(18, "SOFT", SOFT_C), (28, "MEDIUM", MEDIUM_C),
                            (40, "HARD", HARD_C)]:
        ax.axvline(exp, color=col, ls="--", lw=1.8,
                   label=f"{comp} expected ({exp} laps)")
    ax.set_title("(B) Tyre Life Distribution")
    ax.set_xlabel("Tyre Life (laps)")
    ax.set_ylabel("Count")
    ax.legend()

    # ── C: Pit lap distribution ──
    ax = axes[1, 0]
    pit_laps = df[df["HasPitStop"] == 1]["LapNumber"].dropna()
    ax.hist(pit_laps, bins=55, color=ORANGE, alpha=0.80, edgecolor="none",
            density=True, label="Pit stops")
    # Smooth KDE overlay
    x_kde = np.linspace(0, pit_laps.max(), 300)
    kde = stats.gaussian_kde(pit_laps, bw_method=0.2)
    ax.plot(x_kde, kde(x_kde), color=F1_WHITE, lw=2, label="KDE")
    ax.axvline(pit_laps.mean(), color=F1_RED, ls="--", lw=1.5,
               label=f"Mean: lap {pit_laps.mean():.1f}")
    ax.set_title("(C) Pit Stop Lap Distribution")
    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Density")
    ax.legend()

    # ── D: Yearly pit stop rate ──
    ax = axes[1, 1]
    years  = sorted(df["Year"].unique())
    rates  = [df[df["Year"] == y]["HasPitStop"].mean() * 100 for y in years]
    colors = [F1_RED if r == max(rates) else BLUE for r in rates]
    bars   = ax.bar(years, rates, color=colors, alpha=0.85, width=0.7,
                    edgecolor="none")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f"{rate:.1f}%", ha="center", va="bottom",
                fontsize=8.5, color=TEXT)
    ax.set_title("(D) Pit Stop Rate by Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Pit Stop Rate (%)")
    ax.set_xticks(years)

    plt.tight_layout()
    save(fig, f"{out}/01_dataset_overview.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — TYRE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def plot_tyre_analysis(df: pd.DataFrame, out: str) -> None:
    """
    4-panel figure:
      (A) Degradation curves by compound (mean ± 1-sigma)
      (B) Box plot of tyre life at pit stop by compound
      (C) Pit window heatmap: compound × stint number
      (D) Rolling pace mean by compound (heat map over tyre age)
    """
    fig, axes = make_fig(2, 2, (15, 10),
                         "Tyre Strategy Analysis — Degradation, Pit Windows & Stint Patterns")

    # ── A: Degradation curves ──
    ax = axes[0, 0]
    for comp in COMPOUND_ORDER:
        sub = df[df["Compound"] == comp].copy()
        sub = sub[sub["TyreLife"] <= 52]
        grp = sub.groupby("TyreLife")["tire_performance_decay"].agg(
            ["mean", "std"]).reset_index()
        grp = grp[grp["TyreLife"] > 0]
        col = COMPOUND_COLORS[comp]
        ax.plot(grp["TyreLife"], grp["mean"], color=col, lw=2.5, label=comp)
        ax.fill_between(grp["TyreLife"],
                        grp["mean"] - grp["std"],
                        grp["mean"] + grp["std"],
                        color=col, alpha=0.15)
    ax.set_title("(A) Tyre Performance Decay by Compound  (mean ± 1σ)")
    ax.set_xlabel("Tyre Life (laps)")
    ax.set_ylabel("Performance Decay (seconds vs best lap)")
    ax.legend(title="Compound")

    # ── B: Box plot of tyre life at pit ──
    ax = axes[0, 1]
    pit_df = df[df["HasPitStop"] == 1]
    data_by_comp = [pit_df[pit_df["Compound"] == c]["TyreLife"].dropna().values
                    for c in COMPOUND_ORDER]
    bp = ax.boxplot(data_by_comp, patch_artist=True, notch=False,
                    medianprops=dict(color=BG, linewidth=2),
                    whiskerprops=dict(color=MUTED),
                    capprops=dict(color=MUTED),
                    flierprops=dict(marker="o", markersize=2,
                                   markerfacecolor=MUTED, alpha=0.3))
    for patch, col in zip(bp["boxes"], [COMPOUND_COLORS[c] for c in COMPOUND_ORDER]):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    ax.set_xticklabels(COMPOUND_ORDER)
    ax.set_title("(B) Tyre Life at Pit Stop by Compound")
    ax.set_xlabel("Compound")
    ax.set_ylabel("Tyre Life at Pit (laps)")

    # Annotate medians
    for i, d in enumerate(data_by_comp):
        med = np.median(d)
        ax.text(i + 1, med + 0.5, f"{med:.0f}", ha="center",
                fontsize=9, color=BG, fontweight="bold")

    # ── C: Pit window heatmap (compound × stint) ──
    ax = axes[1, 0]
    pit_sub = df[df["HasPitStop"] == 1].copy()
    pit_sub["Stint_int"] = pit_sub["Stint"].astype(int).clip(1, 4)
    hm_data = (pit_sub.groupby(["Stint_int", "Compound"])["LapNumber"]
               .mean().unstack(fill_value=0))
    # Reorder columns
    hm_data = hm_data.reindex(columns=COMPOUND_ORDER, fill_value=0)
    sns.heatmap(hm_data, ax=ax, cmap="YlOrRd", annot=True, fmt=".1f",
                linewidths=0.5, cbar_kws={"label": "Mean Pit Lap"},
                annot_kws={"size": 10, "color": BG})
    ax.set_title("(C) Mean Pit Lap — Compound × Stint Number")
    ax.set_xlabel("Compound")
    ax.set_ylabel("Stint Number")
    ax.tick_params(axis="x", rotation=0)

    # ── D: Stint length distribution ──
    ax = axes[1, 1]
    stint_len = (df.groupby(["Year", "RaceID", "Driver", "Stint"])
                 ["LapNumber"].count().reset_index(name="StintLen"))
    # Merge compound (majority compound in stint)
    comp_per_stint = (df.groupby(["Year", "RaceID", "Driver", "Stint"])
                      ["Compound"].agg(lambda x: x.mode()[0]).reset_index())
    stint_len = stint_len.merge(comp_per_stint,
                                on=["Year", "RaceID", "Driver", "Stint"],
                                how="left")
    for comp in COMPOUND_ORDER:
        sub = stint_len[stint_len["Compound"] == comp]["StintLen"]
        ax.hist(sub, bins=35, alpha=0.6, color=COMPOUND_COLORS[comp],
                label=comp, density=True)
    ax.set_title("(D) Stint Length Distribution by Compound")
    ax.set_xlabel("Stint Length (laps)")
    ax.set_ylabel("Density")
    ax.legend(title="Compound")

    plt.tight_layout()
    save(fig, f"{out}/02_tyre_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MODEL OUTPUTS & CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_outputs(df: pd.DataFrame, out: str) -> None:
    """
    4-panel figure:
      (A) Pit probability histogram: pit vs non-pit laps
      (B) ROC curve
      (C) Precision-Recall curve
      (D) Calibration curve (reliability diagram)
    """
    # Filter: only rows where the model produced a real prediction (SEQ_LENGTH window filled)
    scored = df[df["PitStopProbability"] > 0].copy()
    y_true = scored["HasPitStop"].values
    y_prob = scored["PitStopProbability"].values

    print(f"\n  Model evaluation subset: {len(scored):,} laps "
          f"({y_true.sum():,} pit events, "
          f"{y_true.mean()*100:.2f}% rate)")

    fig, axes = make_fig(2, 2, (14, 10),
                         "BiLSTM Model — Outputs, ROC, Precision-Recall & Calibration")

    # ── A: Probability distribution ──
    ax = axes[0, 0]
    bins = np.linspace(0, 1, 50)
    ax.hist(y_prob[y_true == 0], bins=bins, density=True, alpha=0.65,
            color=TEAL,   label="No Pit (0)", edgecolor="none")
    ax.hist(y_prob[y_true == 1], bins=bins, density=True, alpha=0.80,
            color=F1_RED, label="Pit Stop (1)", edgecolor="none")
    ax.axvline(0.5, color=F1_WHITE, ls="--", lw=1.5, label="Threshold = 0.5")
    ax.set_title("(A) Predicted Probability Distribution")
    ax.set_xlabel("PitStopProbability")
    ax.set_ylabel("Density")
    ax.legend()
    # Inset text
    ax.text(0.97, 0.95,
            f"Pit laps: {y_true.sum():,}\nNon-pit laps: {(y_true==0).sum():,}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color=MUTED,
            bbox=dict(fc=PANEL, ec=BORDER, boxstyle="round,pad=0.3"))

    # ── B: ROC curve ──
    ax = axes[0, 1]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=ORANGE, lw=2.5,
            label=f"BiLSTM  AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1.2, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.08, color=ORANGE)
    ax.set_title("(B) ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    # ── C: Precision-Recall curve ──
    ax = axes[1, 0]
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    ax.plot(recall, precision, color=BLUE, lw=2.5,
            label=f"BiLSTM  AP = {ap:.4f}")
    baseline = y_true.mean()
    ax.axhline(baseline, color=MUTED, ls="--", lw=1.2,
               label=f"Baseline (class freq = {baseline:.3f})")
    ax.fill_between(recall, precision, alpha=0.10, color=BLUE)
    ax.set_title("(C) Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    # ── D: Calibration curve ──
    ax = axes[1, 1]
    n_bins = 15
    bin_edges = np.linspace(0, 1, n_bins + 1)
    frac_pos, mean_pred = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() > 20:
            frac_pos.append(y_true[mask].mean())
            mean_pred.append(y_prob[mask].mean())
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1.5, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color=PURPLE, lw=2, ms=5,
            label="BiLSTM")
    ax.set_title("(D) Calibration (Reliability Diagram)")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    save(fig, f"{out}/03_model_outputs.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — TEAM STRATEGY PROFILES
# ─────────────────────────────────────────────────────────────────────────────

def plot_team_strategy(df: pd.DataFrame, out: str) -> None:
    """
    3-panel figure:
      (A) Strategy risk scatter: E[X] vs Var(X) of TyreLife at pit
      (B) Box plot of tyre life at pit by team (ordered by median)
      (C) Pit compound preference heat map per team
    """
    pit_df = df[df["HasPitStop"] == 1].copy()
    team_stats = (pit_df.groupby("Team")["TyreLife"]
                  .agg(["mean", "var", "count"])
                  .reset_index())
    team_stats.columns = ["Team", "Mean", "Variance", "Count"]
    team_stats = team_stats[team_stats["Count"] >= 30].copy()

    fig, axes = make_fig(1, 3, (18, 7),
                         "Team Strategy Profiles — Risk Tolerance & Compound Preference (2018-2025)")

    # ── A: Risk scatter (E[X] vs Var(X)) ──
    ax = axes[0, 0]
    for _, row in team_stats.iterrows():
        col = TEAM_COLORS.get(row["Team"], MUTED)
        ax.scatter(row["Mean"], row["Variance"], s=120, color=col,
                   edgecolor=F1_WHITE, linewidth=0.6, zorder=3)
        ax.annotate(row["Team"],
                    (row["Mean"], row["Variance"]),
                    textcoords="offset points", xytext=(6, 3),
                    fontsize=7.5, color=TEXT)
    ax.axhline(team_stats["Variance"].mean(), color=MUTED, ls=":", lw=1,
               label="Mean variance")
    ax.axvline(team_stats["Mean"].mean(),    color=MUTED, ls=":", lw=1,
               label="Mean tyre age")
    # Quadrant labels
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    ax.text(xlim[0] + 0.5, ylim[1] * 0.96, "Early & Reactive",
            fontsize=8, color=MUTED, ha="left", va="top")
    ax.text(xlim[1] - 0.5, ylim[1] * 0.96, "Late & Reactive",
            fontsize=8, color=MUTED, ha="right", va="top")
    ax.text(xlim[0] + 0.5, ylim[0] * 1.02, "Early & Deterministic",
            fontsize=8, color=MUTED, ha="left", va="bottom")
    ax.text(xlim[1] - 0.5, ylim[0] * 1.02, "Late & Deterministic",
            fontsize=8, color=MUTED, ha="right", va="bottom")
    ax.set_title("(A) Strategy Risk Scatter:  E[X] vs Var(X)")
    ax.set_xlabel("Mean TyreLife at Pit — E[X]  (laps)")
    ax.set_ylabel("Variance of TyreLife — Var(X)  (laps²)")
    ax.legend(fontsize=8)

    # ── B: Team box plot ordered by median ──
    ax = axes[0, 1]
    team_order = (pit_df.groupby("Team")["TyreLife"].median()
                  .sort_values().index.tolist())
    team_order = [t for t in team_order if
                  pit_df[pit_df["Team"] == t].shape[0] >= 30]
    data_list   = [pit_df[pit_df["Team"] == t]["TyreLife"].values
                   for t in team_order]
    colors_list = [TEAM_COLORS.get(t, MUTED) for t in team_order]

    bp = ax.boxplot(data_list, vert=False, patch_artist=True,
                    medianprops=dict(color=BG, linewidth=2),
                    whiskerprops=dict(color=MUTED),
                    capprops=dict(color=MUTED),
                    flierprops=dict(marker="o", markersize=2,
                                   markerfacecolor=MUTED, alpha=0.3))
    for patch, col in zip(bp["boxes"], colors_list):
        patch.set_facecolor(col)
        patch.set_alpha(0.80)
    ax.set_yticks(range(1, len(team_order) + 1))
    ax.set_yticklabels(team_order, fontsize=8)
    ax.set_title("(B) TyreLife at Pit Stop by Team  (ordered by median)")
    ax.set_xlabel("Tyre Life (laps)")

    # ── C: Compound heat map ──
    ax = axes[0, 2]
    comp_heat = (pit_df.groupby(["Team", "Compound"])
                 .size().unstack(fill_value=0))
    # Normalise to row %
    comp_heat_pct = comp_heat.div(comp_heat.sum(axis=1), axis=0) * 100
    comp_heat_pct = comp_heat_pct.reindex(columns=COMPOUND_ORDER, fill_value=0)
    comp_heat_pct = comp_heat_pct.reindex(
        [t for t in team_order if t in comp_heat_pct.index])
    sns.heatmap(comp_heat_pct, ax=ax, cmap="RdYlGn",
                annot=True, fmt=".0f", linewidths=0.4,
                cbar_kws={"label": "% of Pit Stops"},
                annot_kws={"size": 9, "color": BG})
    ax.set_title("(C) Compound Mix at Pit Stop by Team  (%)")
    ax.set_xlabel("Compound")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    save(fig, f"{out}/04_team_strategy.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — RACE DYNAMICS
# ─────────────────────────────────────────────────────────────────────────────

def plot_race_dynamics(df: pd.DataFrame, out: str) -> None:
    """
    4-panel figure:
      (A) Cumulative position changes by lap phase
      (B) Probability trajectory: mean model output over race progress
      (C) Compound usage per season (stacked bar)
      (D) Predicted pit probability vs tyre life (heatmap)
    """
    fig, axes = make_fig(2, 2, (15, 10),
                         "Race Dynamics — Position Changes, Strategy Windows & Pit Probability")

    # ── A: Position changes by lap ──
    ax = axes[0, 0]
    d = df.sort_values(["Year", "RaceID", "Driver", "LapNumber"])
    d["PosChange"] = (d.groupby(["Year", "RaceID", "Driver"])["Position"]
                      .diff().abs().fillna(0))
    changes = (d.groupby("LapNumber")["PosChange"].sum()
               .reset_index())
    changes = changes[changes["LapNumber"] <= 70]

    phase_colors = np.where(changes["LapNumber"] <= 15, F1_RED,
                   np.where(changes["LapNumber"] <= 40, ORANGE, TEAL))
    ax.bar(changes["LapNumber"], changes["PosChange"],
           color=phase_colors, width=1.0, edgecolor="none", alpha=0.85)
    # Smooth trend
    smooth = gaussian_filter1d(changes["PosChange"].values, sigma=2)
    ax.plot(changes["LapNumber"], smooth, color=F1_WHITE, lw=1.8, zorder=5)

    patches = [mpatches.Patch(color=F1_RED,  label="Early  (laps 1-15)"),
               mpatches.Patch(color=ORANGE,  label="Mid    (laps 16-40)"),
               mpatches.Patch(color=TEAL,    label="Late   (laps 41+)")]
    ax.legend(handles=patches, fontsize=8)
    ax.set_title("(A) Total Position Changes by Lap")
    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Sum of |Position Changes|")

    # ── B: Mean pit probability over race progress ──
    ax = axes[0, 1]
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
                    alpha=0.25, color=BLUE)
    ax.plot(prob_trace["progress_bin"],
            prob_trace["PitStopProbability"],
            color=BLUE, lw=2, label="Mean probability (all laps)")
    ax.scatter(prob_pit["progress_bin"],
               prob_pit["PitStopProbability"],
               color=F1_RED, s=18, zorder=5, label="Mean probability (pit laps)")
    ax.set_title("(B) Mean Pit Probability Over Race Progress")
    ax.set_xlabel("Race Progress (bin, 0 = start → 29 = end)")
    ax.set_ylabel("Mean PitStopProbability")
    ax.legend()

    # ── C: Stacked compound usage per year ──
    ax = axes[1, 0]
    pit_sub = df[df["HasPitStop"] == 1]
    comp_year = (pit_sub.groupby(["Year", "Compound"]).size()
                 .unstack(fill_value=0)
                 .reindex(columns=COMPOUND_ORDER, fill_value=0))
    bottom = np.zeros(len(comp_year))
    years  = comp_year.index.tolist()
    for comp in COMPOUND_ORDER:
        vals = comp_year[comp].values
        ax.bar(years, vals, bottom=bottom,
               color=COMPOUND_COLORS[comp], label=comp,
               alpha=0.85, edgecolor=BG, linewidth=0.4)
        bottom += vals
    ax.set_title("(C) Pit Stops by Compound per Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Number of Pit Stops")
    ax.set_xticks(years)
    ax.legend(title="Compound")

    # ── D: Pit probability vs tyre life heat map ──
    ax = axes[1, 1]
    scored2 = df[(df["PitStopProbability"] > 0) & (df["TyreLife"] <= 50)].copy()
    scored2["TyreLife_int"] = scored2["TyreLife"].astype(int)
    hm = (scored2.groupby(["TyreLife_int", "Compound"])["PitStopProbability"]
          .mean().unstack(fill_value=0)
          .reindex(columns=COMPOUND_ORDER, fill_value=0))
    sns.heatmap(hm.T, ax=ax, cmap="inferno",
                cbar_kws={"label": "Mean PitStopProbability"},
                linewidths=0)
    ax.set_title("(D) Mean Pit Probability — Tyre Life × Compound")
    ax.set_xlabel("Tyre Life (laps)")
    ax.set_ylabel("Compound")
    ax.tick_params(axis="y", rotation=0)
    # Thin out x-tick labels
    step = max(1, len(hm.index) // 10)
    ax.set_xticks(range(0, len(hm.index), step))
    ax.set_xticklabels(hm.index[::step], rotation=0)

    plt.tight_layout()
    save(fig, f"{out}/05_race_dynamics.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — FEATURE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_analysis(df: pd.DataFrame, out: str) -> None:
    """
    4-panel figure:
      (A) Correlation heat map (key features vs HasPitStop)
      (B) Feature distribution: pit vs non-pit (violin, 6 features)
      (C) Traffic pressure vs pit probability scatter
      (D) Pace degradation slope by compound (violin)
    """
    fig, axes = make_fig(2, 2, (15, 11),
                         "Feature Analysis — Correlations, Distributions & Pit Drivers")

    CORE_FEATURES = [
        "TyreLife", "tire_performance_decay", "traffic_pressure",
        "pace_degradation_slope", "relative_tire_age", "pit_window_delta",
        "Speed_mean", "Throttle_mean", "Brake_mean", "DRS_mean",
        "race_progress_fraction", "delta_laptime", "HasPitStop",
    ]
    core_df = df[CORE_FEATURES].dropna()

    # ── A: Correlation heat map ──
    ax = axes[0, 0]
    corr = core_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = LinearSegmentedColormap.from_list(
        "f1div", [BLUE, PANEL, F1_RED], N=256)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True,
                linewidths=0.4, annot=True, fmt=".2f",
                annot_kws={"size": 7, "color": TEXT},
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("(A) Feature Correlation Matrix")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    # ── B: Violin — pit vs non-pit for 6 key features ──
    ax = axes[0, 1]
    VIOL_FEATURES = [
        "TyreLife", "tire_performance_decay",
        "traffic_pressure", "pace_degradation_slope",
        "relative_tire_age", "pit_window_delta",
    ]
    # Normalise each feature to [0, 1] for joint plot
    sample = df.sample(min(20000, len(df)), random_state=42)[
        VIOL_FEATURES + ["HasPitStop"]].dropna().copy()
    for f in VIOL_FEATURES:
        rng = sample[f].max() - sample[f].min()
        if rng > 0:
            sample[f] = (sample[f] - sample[f].min()) / rng
    melted = sample.melt(id_vars="HasPitStop", value_vars=VIOL_FEATURES,
                         var_name="Feature", value_name="Scaled Value")
    melted["Pit Stop"] = melted["HasPitStop"].map({0: "No Pit", 1: "Pit"})
    sns.violinplot(data=melted, x="Feature", y="Scaled Value",
                   hue="Pit Stop",
                   palette={"No Pit": TEAL, "Pit": F1_RED},
                   split=True, inner="quart", linewidth=0.7,
                   ax=ax, legend=True)
    ax.set_title("(B) Feature Distribution: Pit vs Non-Pit  (normalised 0-1)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.get_legend().set_title("Outcome")

    # ── C: Traffic pressure vs pit probability ──
    ax = axes[1, 0]
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
            color=PURPLE, lw=2.5)
    ax.fill_between(tp_agg["tp_mean"],
                    tp_agg["prob_mean"] - tp_agg["prob_std"],
                    tp_agg["prob_mean"] + tp_agg["prob_std"],
                    alpha=0.2, color=PURPLE)
    ax.set_title("(C) Traffic Pressure vs Predicted Pit Probability")
    ax.set_xlabel("Traffic Pressure  (1 / DistanceToDriverAhead)")
    ax.set_ylabel("Mean PitStopProbability")

    # ── D: Pace degradation slope by compound ──
    ax = axes[1, 1]
    slope_data = []
    for comp in COMPOUND_ORDER:
        vals = (df[(df["Compound"] == comp) &
                   (df["pace_degradation_slope"].between(-5, 5))]
                ["pace_degradation_slope"].values)
        slope_data.append(vals)

    vp = ax.violinplot(slope_data, positions=[1, 2, 3],
                       showmedians=True, showextrema=False)
    for body, col in zip(vp["bodies"],
                         [COMPOUND_COLORS[c] for c in COMPOUND_ORDER]):
        body.set_facecolor(col)
        body.set_alpha(0.70)
    vp["cmedians"].set_color(BG)
    vp["cmedians"].set_linewidth(2)
    ax.axhline(0, color=MUTED, ls="--", lw=1.2)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(COMPOUND_ORDER)
    ax.set_title("(D) Pace Degradation Slope by Compound  (AR1-filtered)")
    ax.set_xlabel("Compound")
    ax.set_ylabel("Pace Degradation Slope (s/lap)")

    plt.tight_layout()
    save(fig, f"{out}/06_feature_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — STOCHASTIC / SURVIVAL MODELLING
# ─────────────────────────────────────────────────────────────────────────────

def plot_stochastic(df: pd.DataFrame, out: str) -> None:
    """
    4-panel figure:
      (A) Empirical hazard rate h(t) — pit probability vs tyre age
      (B) Stationary pace decay distribution (AR1 first-difference)
      (C) Pit_Probability (Cox baseline) vs PitStopProbability (BiLSTM)
      (D) TrackStatus / VSC event — pit probability spike
    """
    fig, axes = make_fig(2, 2, (15, 10),
                         "Stochastic & Survival Modelling — Hazard Rate, Pace Decay & VSC Impact")

    # ── A: Empirical hazard h(t) per compound ──
    ax = axes[0, 0]
    for comp in COMPOUND_ORDER:
        sub = df[df["Compound"] == comp].copy()
        grp = (sub.groupby("TyreLife")["HasPitStop"]
               .agg(["sum", "count"])
               .reset_index())
        grp = grp[grp["TyreLife"] <= 52]
        grp["hazard"] = grp["sum"] / grp["count"].clip(lower=1)
        smooth_h = gaussian_filter1d(grp["hazard"].values, sigma=1.5)
        ax.plot(grp["TyreLife"], smooth_h, color=COMPOUND_COLORS[comp],
                lw=2.5, label=comp)
        ax.fill_between(grp["TyreLife"], smooth_h, alpha=0.10,
                        color=COMPOUND_COLORS[comp])
    ax.set_title("(A) Empirical Hazard Rate  h(t) by Compound")
    ax.set_xlabel("Tyre Life (laps)")
    ax.set_ylabel("P(Pit | TyreLife = t)")
    ax.legend(title="Compound")

    # ── B: Stationary pace decay (AR1) ──
    ax = axes[1, 0]
    decay = df["Stationary_PaceDecay"].dropna()
    decay = decay[decay.between(-8, 8)]
    ax.hist(decay, bins=80, color=ORANGE, alpha=0.75, density=True,
            edgecolor="none", label="AR(1) ΔLapTime")
    x_r = np.linspace(-8, 8, 300)
    mu, sigma = stats.norm.fit(decay)
    ax.plot(x_r, stats.norm.pdf(x_r, mu, sigma),
            color=F1_WHITE, lw=2, label=f"Normal  μ={mu:.3f}, σ={sigma:.3f}")
    ax.axvline(0, color=MUTED, ls="--", lw=1)
    ax.set_title("(B) Stationary Pace Decay  Δt = Xₜ − Xₜ₋₁")
    ax.set_xlabel("Pace Decay (seconds)")
    ax.set_ylabel("Density")
    ax.legend()

    # ── C: Cox Pit_Probability vs BiLSTM PitStopProbability ──
    ax = axes[0, 1]
    both = df[(df["Pit_Probability"] > 0) &
              (df["PitStopProbability"] > 0)].copy()
    # Sample for scatter (avoid overplotting)
    sample = both.sample(min(8000, len(both)), random_state=42)
    col_by_pit = np.where(sample["HasPitStop"] == 1, F1_RED, TEAL)
    ax.scatter(sample["Pit_Probability"], sample["PitStopProbability"],
               c=col_by_pit, s=8, alpha=0.35, linewidths=0)
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1.2,
            label="Identity line")
    # Bins trend
    both["cox_bin"] = pd.qcut(both["Pit_Probability"], 30,
                               duplicates="drop", labels=False)
    trend = (both.groupby("cox_bin")
             .agg(cox_mean=("Pit_Probability", "mean"),
                  bi_mean=("PitStopProbability", "mean"))
             .reset_index())
    ax.plot(trend["cox_mean"], trend["bi_mean"],
            color=PURPLE, lw=2, label="Binned trend")
    legend_patches = [
        mpatches.Patch(color=F1_RED, label="Pit lap"),
        mpatches.Patch(color=TEAL,   label="Non-pit lap"),
        plt.Line2D([0], [0], color=PURPLE, lw=2, label="Binned trend"),
        plt.Line2D([0], [0], color=MUTED, ls="--", label="Identity"),
    ]
    ax.legend(handles=legend_patches, fontsize=8)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.set_title("(C) Cox Pit_Probability vs BiLSTM PitStopProbability")
    ax.set_xlabel("Cox Proportional Hazard  Pit_Probability")
    ax.set_ylabel("BiLSTM  PitStopProbability")

    # ── D: Pit probability spike at VSC events ──
    ax = axes[1, 1]
    # Flag VSC/SC laps (TrackStatus digits containing 4 or 6)
    def is_vsc(ts):
        s = str(int(ts)) if not np.isnan(ts) else ""
        return "4" in s or "6" in s

    df["IsVSC"] = df["TrackStatus"].apply(
        lambda x: is_vsc(x) if pd.notna(x) else False)

    scored = df[df["PitStopProbability"] > 0].copy()
    scored["LapNumber_int"] = scored["LapNumber"].astype(int)

    # Average probability n laps before and after a VSC deployment
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

    means  = [np.mean(prob_at_offset[o]) if prob_at_offset[o] else 0
               for o in offsets]
    stds   = [np.std(prob_at_offset[o])  if prob_at_offset[o] else 0
               for o in offsets]

    ax.bar(offsets, means, color=[F1_RED if o >= 0 else BLUE for o in offsets],
           alpha=0.75, edgecolor="none")
    ax.errorbar(offsets, means, yerr=stds, fmt="none",
                color=F1_WHITE, capsize=3, linewidth=0.8)
    ax.axvline(0, color=ORANGE, ls="--", lw=1.5, label="VSC deployment")
    ax.set_title("(D) Mean Pit Probability Around VSC Events  (±8 laps)")
    ax.set_xlabel("Laps Relative to VSC Deployment  (0 = deployment lap)")
    ax.set_ylabel("Mean PitStopProbability")
    ax.legend()

    plt.tight_layout()
    save(fig, f"{out}/07_stochastic_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

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
    parser = argparse.ArgumentParser(
        description="Generate all F1 strategy plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data",
        default="all_training_data.csv",
        help="Path to the aggregated CSV file (default: all_training_data.csv)",
    )
    parser.add_argument(
        "--output",
        default="./plots",
        help="Output directory for PNG files (default: ./plots)",
    )
    parser.add_argument(
        "--section",
        choices=list(SECTION_MAP.keys()) + ["all"],
        default="all",
        help="Which section of plots to generate (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    apply_theme()
    df = load_data(args.data)

    sections = (list(SECTION_MAP.items()) if args.section == "all"
                else [(args.section, SECTION_MAP[args.section])])

    print(f"\nGenerating {len(sections)} section(s) → {args.output}/\n")
    for name, fn in sections:
        print(f"── {name.upper()}")
        fn(df, args.output)

    print(f"\nDone. {len(sections)} figure(s) saved to {args.output}/")


if __name__ == "__main__":
    main()
