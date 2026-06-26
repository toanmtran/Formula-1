"""Lap-time distribution analysis — one chart per PNG, white print theme.

Generates individual figures for inclusion in a LaTeX/Palatino paper.
Each plotting function produces a single PNG sized for ~0.78\\textwidth
inclusion (typeset legibly at A4 with 1in margins).
"""

import sys
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT  = Path(__file__).resolve().parents[2]
DATA_DIR   = REPO_ROOT / "data" / "raw"
OUTPUT_DIR = REPO_ROOT / "outputs" / "lap_time_distributions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Theme — calm, print-friendly white background
# ---------------------------------------------------------------------------
DIST_COLORS = {
    "Normal":     "#C42021",   # red
    "Log-Normal": "#2E5BA8",   # blue
    "Gamma":      "#0F9E8C",   # teal
}

OBS_FILL    = "#B5C9E2"   # soft blue histogram fill
TEXT        = "#1F1F1F"
MUTED       = "#555555"
GRID_C      = "#D7D7DC"
HIGHLIGHT   = "#B8141A"

plt.rcParams.update({
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
    "axes.edgecolor":      MUTED,
    "axes.linewidth":      0.9,
    "axes.labelcolor":     TEXT,
    "axes.labelsize":      14,
    "axes.titlesize":      16,
    "axes.titleweight":    "bold",
    "axes.titlecolor":     TEXT,
    "axes.titlepad":       12,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.color":          GRID_C,
    "grid.alpha":          0.85,
    "grid.linewidth":      0.7,
    "text.color":          TEXT,
    "xtick.color":         TEXT,
    "ytick.color":         TEXT,
    "xtick.labelsize":     12,
    "ytick.labelsize":     12,
    "legend.facecolor":    "white",
    "legend.edgecolor":    "#BBBBBB",
    "legend.fontsize":     12,
    "legend.framealpha":   0.95,
    "font.family":         "serif",
    "font.serif":          ["Palatino Linotype", "Palatino", "DejaVu Serif"],
    "font.size":           12,
    "savefig.dpi":         220,
    "savefig.bbox":        "tight",
    "savefig.facecolor":   "white",
})

FIG_SOLO   = (7.5, 4.8)
FIG_BAR    = (9.0, 5.0)
FIG_PIE    = (6.0, 5.5)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data():
    print("Loading data ...")
    laps     = pd.read_csv(DATA_DIR / "lap_times.csv", na_values="\\N")
    races    = pd.read_csv(DATA_DIR / "races.csv",     na_values="\\N")
    circuits = pd.read_csv(DATA_DIR / "circuits.csv",  na_values="\\N")
    results  = pd.read_csv(DATA_DIR / "results.csv",   na_values="\\N")

    laps = laps.merge(
        races[["raceId", "year", "round", "circuitId", "name"]],
        on="raceId", how="left",
    )
    laps = laps.merge(
        circuits[["circuitId", "circuitRef", "name", "country"]],
        on="circuitId", how="left", suffixes=("_race", "_circuit"),
    )
    print(f"  {len(laps):,} lap-time rows  |  "
          f"{laps['raceId'].nunique()} races  |  "
          f"{laps['year'].nunique()} seasons")
    return laps, races, circuits, results


def find_race_id(races, circuits, circuit_ref, year):
    cr = circuits[circuits["circuitRef"] == circuit_ref]
    if cr.empty:
        return None
    cid = cr.iloc[0]["circuitId"]
    rr = races[(races["circuitId"] == cid) & (races["year"] == year)]
    if rr.empty:
        return None
    return rr.iloc[0]["raceId"]


def clean_race_laps(laps_df, race_id):
    rl = laps_df[laps_df["raceId"] == race_id].copy()
    rl = rl[rl["lap"] > 1]
    rl["time_seconds"] = rl["milliseconds"] / 1000.0

    q1, q3 = rl["time_seconds"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 2.0 * iqr, q3 + 2.0 * iqr

    n_before = len(rl)
    rl = rl[(rl["time_seconds"] >= lower) & (rl["time_seconds"] <= upper)]
    return rl, n_before, len(rl)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
class DistributionAnalysis:
    DISTRIBUTIONS = {
        "Normal":     stats.norm,
        "Log-Normal": stats.lognorm,
        "Gamma":      stats.gamma,
    }

    def __init__(self, data, label=""):
        self.data = data
        self.label = label
        self.n = len(data)
        self.fits = {}
        self.results = {}

    def fit_all(self):
        for name, dist in self.DISTRIBUTIONS.items():
            params = dist.fit(self.data)
            ll = float(np.sum(dist.logpdf(self.data, *params)))
            k = len(params)
            self.fits[name] = (dist, params)
            self.results[name] = {
                "params":         params,
                "log_likelihood": ll,
                "k":              k,
                "aic":            2 * k - 2 * ll,
                "bic":            k * np.log(self.n) - 2 * ll,
            }

    def get_best(self, metric="aic"):
        return min(self.results, key=lambda n: self.results[n][metric])


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _save(fig, filename):
    out = OUTPUT_DIR / filename
    fig.savefig(out)
    plt.close(fig)
    print(f"  [SAVED] {filename}")


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------
def plot_histogram_with_fits(analysis, race_label, filename):
    fig, ax = plt.subplots(figsize=FIG_SOLO)
    data = analysis.data
    x_range = np.linspace(data.min() - 1, data.max() + 1, 500)

    ax.hist(data, bins=60, density=True, alpha=0.65, color=OBS_FILL,
            edgecolor="white", linewidth=0.4, label="Observed", zorder=2)
    for name, (dist, params) in analysis.fits.items():
        ax.plot(x_range, dist.pdf(x_range, *params),
                linewidth=2.4, label=name, color=DIST_COLORS[name], zorder=3)

    ax.set_xlabel("Lap time (seconds)")
    ax.set_ylabel("Probability density")
    ax.set_title(f"Histogram with fitted distributions — {race_label}")
    ax.legend(loc="upper right", framealpha=0.95)

    _save(fig, filename)


def plot_aic_bic(analysis, race_label, filename):
    fig, ax = plt.subplots(figsize=FIG_SOLO)
    names = list(analysis.DISTRIBUTIONS.keys())
    aic = [analysis.results[n]["aic"] for n in names]
    bic = [analysis.results[n]["bic"] for n in names]
    x = np.arange(len(names))
    w = 0.36

    bars_a = ax.bar(x - w / 2, aic, w, color="#2E5BA8", alpha=0.88, label="AIC")
    bars_b = ax.bar(x + w / 2, bic, w, color="#D9650A", alpha=0.88, label="BIC")

    best_i = int(np.argmin(aic))
    bars_a[best_i].set_edgecolor(HIGHLIGHT)
    bars_a[best_i].set_linewidth(2.2)

    for i, (a, b) in enumerate(zip(aic, bic)):
        ax.text(i - w / 2, a, f"{a:.0f}", ha="center", va="bottom",
                fontsize=10, color=TEXT)
        ax.text(i + w / 2, b, f"{b:.0f}", ha="center", va="bottom",
                fontsize=10, color=TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Information criterion (lower is better)")
    ax.set_title(f"AIC and BIC comparison — {race_label}")
    ax.legend(loc="upper left", framealpha=0.95)

    # Zoom the y-axis so differences between candidates are visible
    ymax = max(max(aic), max(bic))
    ymin = min(min(aic), min(bic))
    span = ymax - ymin
    ax.set_ylim(bottom=ymin - 0.25 * span, top=ymax + 0.20 * span)

    _save(fig, filename)


def plot_cdf(analysis, race_label, filename):
    fig, ax = plt.subplots(figsize=FIG_SOLO)
    data = analysis.data
    sorted_data = np.sort(data)
    n = len(sorted_data)
    ecdf = np.arange(1, n + 1) / n
    x_range = np.linspace(data.min() - 1, data.max() + 1, 500)

    ax.step(sorted_data, ecdf, where="post", linewidth=2.4,
            color=TEXT, label="Empirical CDF", alpha=0.92, zorder=3)
    for name, (dist, params) in analysis.fits.items():
        ax.plot(x_range, dist.cdf(x_range, *params),
                linewidth=1.7, color=DIST_COLORS[name], label=name,
                alpha=0.9, zorder=2)

    ax.set_xlabel("Lap time (seconds)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title(f"Empirical vs. theoretical CDF — {race_label}")
    ax.legend(loc="lower right", framealpha=0.95, ncol=2)

    _save(fig, filename)


def plot_single_distribution(analysis, label, sublabel, filename):
    fig, ax = plt.subplots(figsize=FIG_SOLO)
    data = analysis.data
    best_name = analysis.get_best("aic")
    best_dist, best_params = analysis.fits[best_name]
    x_range = np.linspace(data.min() - 0.5, data.max() + 0.5, 300)

    ax.hist(data, bins=50, density=True, alpha=0.6, color=OBS_FILL,
            edgecolor="white", linewidth=0.4, label="Observed", zorder=2)
    ax.plot(x_range, best_dist.pdf(x_range, *best_params),
            linewidth=2.7, color=DIST_COLORS[best_name],
            label=f"{best_name} fit", zorder=3)

    mu  = float(np.mean(data))
    sig = float(np.std(data))
    sk  = float(stats.skew(data))
    stats_text = (
        f"Best fit: {best_name}\n"
        f"μ = {mu:.2f} s\n"
        f"σ = {sig:.2f} s\n"
        f"Skewness = {sk:.3f}\n"
        f"n = {len(data):,}"
    )
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            fontsize=12, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="#BBBBBB", alpha=0.95, linewidth=1))

    ax.set_xlabel("Lap time (seconds)")
    ax.set_ylabel("Probability density")
    ax.set_title(f"{label} — {sublabel}")
    ax.legend(loc="upper left", framealpha=0.95)

    _save(fig, filename)


def plot_skewness_summary(all_analyses, all_labels, filename):
    fig, ax = plt.subplots(figsize=FIG_BAR)
    sk_vals = [float(stats.skew(a.data)) for a in all_analyses]
    best_fits = [a.get_best("aic") for a in all_analyses]
    colors = [DIST_COLORS[bf] for bf in best_fits]

    y = np.arange(len(all_labels))
    ax.barh(y, sk_vals, color=colors, alpha=0.88, edgecolor="white",
            height=0.65, zorder=3)

    ax.axvline(x=0, color=MUTED, linestyle="--", linewidth=1, zorder=2)

    for i, (sk, bf) in enumerate(zip(sk_vals, best_fits)):
        ax.text(sk + 0.02, i, f"  {sk:.3f}  ({bf})",
                va="center", fontsize=12, color=TEXT)

    ax.set_yticks(y)
    ax.set_yticklabels(all_labels, fontsize=12)
    ax.set_xlabel("Skewness  (positive = right-skewed)")
    ax.set_title("Skewness of lap-time distributions across races")
    ax.invert_yaxis()
    ax.set_xlim(left=0, right=max(sk_vals) * 1.28)

    _save(fig, filename)


def plot_best_fit_pie(all_analyses, filename):
    fig, ax = plt.subplots(figsize=FIG_PIE)
    best_fits = [a.get_best("aic") for a in all_analyses]
    counts = Counter(best_fits)
    names = list(counts.keys())
    sizes = list(counts.values())
    colors = [DIST_COLORS[n] for n in names]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=names, colors=colors, autopct="%1.0f%%",
        startangle=90, pctdistance=0.72,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops=dict(color=TEXT, fontsize=13),
    )
    for at in autotexts:
        at.set_fontsize(13)
        at.set_fontweight("bold")
        at.set_color("white")
    ax.set_title("Best-fit family across all races analysed",
                 fontsize=14, fontweight="bold", pad=15)
    _save(fig, filename)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def _purge_old_combined_files():
    """Remove the old multi-panel PNGs left over from the previous theme."""
    legacy = [
        "fig1_distribution_overview.png",
        "fig2_qq_all_distributions.png",
        "fig3_gof_summary_table.png",
        "fig4_cross_circuit.png",
        "fig5_cross_era.png",
        "fig6_skewness_synthesis.png",
    ]
    for f in legacy:
        p = OUTPUT_DIR / f
        if p.exists():
            p.unlink()
            print(f"  [removed legacy] {f}")


def run_analysis():
    _purge_old_combined_files()
    laps, races, circuits, results = load_data()

    print("\n[Part A] Deep dive — 2023 Italian Grand Prix")
    primary_id = find_race_id(races, circuits, "monza", 2023)
    clean_laps, nb, na = clean_race_laps(laps, primary_id)
    rinfo = races[races["raceId"] == primary_id].iloc[0]
    race_label = f"{int(rinfo['year'])} Italian Grand Prix"
    print(f"  {race_label}: {nb} → {na} laps after cleaning")

    data = clean_laps["time_seconds"].values
    A = DistributionAnalysis(data, label=race_label)
    A.fit_all()

    plot_histogram_with_fits(A, race_label, "01_overview_histogram.png")
    plot_aic_bic           (A, race_label, "02_overview_aic_bic.png")
    plot_cdf               (A, race_label, "04_overview_cdf.png")

    print("\n[Part B] Cross-circuit comparison (2023 season)")
    circuit_configs = [
        ("monza",       "Monza 2023",       "High-Speed Permanent",  "11_circuit_monza.png"),
        ("monaco",      "Monaco 2023",      "Narrow Street",         "12_circuit_monaco.png"),
        ("silverstone", "Silverstone 2023", "Mixed / High-Downforce","13_circuit_silverstone.png"),
    ]
    circuit_analyses = []
    for cref, label, ctype, fname in circuit_configs:
        rid = find_race_id(races, circuits, cref, 2023)
        if rid is None:
            continue
        clean, _, _ = clean_race_laps(laps, rid)
        if len(clean) < 50:
            continue
        a = DistributionAnalysis(clean["time_seconds"].values, label=label)
        a.fit_all()
        plot_single_distribution(a, label, ctype, fname)
        circuit_analyses.append((label, a))

    print("\n[Part C] Cross-era at Monza")
    era_configs = [
        ("monza", 2005, "Monza 2005", "V10 era — 3.0L naturally aspirated",  "14_era_v10.png"),
        ("monza", 2013, "Monza 2013", "V8 era — 2.4L naturally aspirated",   "15_era_v8.png"),
        ("monza", 2019, "Monza 2019", "Turbo-hybrid era — 1.6L V6 hybrid",   "16_era_hybrid.png"),
        ("monza", 2023, "Monza 2023", "Ground-effect era — new aero rules",  "17_era_ground.png"),
    ]
    era_analyses = []
    for cref, year, label, descr, fname in era_configs:
        rid = find_race_id(races, circuits, cref, year)
        if rid is None:
            continue
        clean, _, _ = clean_race_laps(laps, rid)
        if len(clean) < 50:
            continue
        a = DistributionAnalysis(clean["time_seconds"].values, label=label)
        a.fit_all()
        plot_single_distribution(a, label, descr, fname)
        era_analyses.append((label, a))

    print("\n[Part D] Synthesis")
    all_analyses = ([a for _, a in circuit_analyses]
                    + [a for _, a in era_analyses])
    all_labels   = ([l for l, _ in circuit_analyses]
                    + [l for l, _ in era_analyses])
    plot_skewness_summary(all_analyses, all_labels, "18_skewness_summary.png")
    plot_best_fit_pie    (all_analyses,            "19_best_fit_pie.png")

    print(f"\nAll figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    run_analysis()
