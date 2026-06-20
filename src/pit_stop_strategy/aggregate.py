"""Aggregate per-race processed CSVs into the single training/visualisation
file used by `visualize.py` (and by `train.py` / `predict_pit_probabilities.py`).

Also adds the columns the downstream pipeline expects:
  - Year, RaceID    derived from the source directory and file name
  - Stationary_PaceDecay  first-difference of LapTime_Seconds per (Driver, Stint)

If the trained BiLSTM weights and matching feature columns are available, a
separate step (`predict_pit_probabilities.py`) appends `PitStopProbability`;
without it, downstream model-dependent plots are skipped gracefully by the
new `visualize.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.pit_stop_strategy.paths import (
    PROCESSED_DIR, AGGREGATED_CSV, PIPELINE_DATA_DIR,
)


def gather() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for year_dir in sorted(PROCESSED_DIR.glob("*")):
        if not year_dir.is_dir():
            continue
        try:
            year = int(year_dir.name)
        except ValueError:
            continue

        for csv in sorted(year_dir.glob("*_processed.csv")):
            try:
                df = pd.read_csv(csv, low_memory=False)
            except Exception as e:
                print(f"  [skip] {csv.name}: {e}")
                continue
            race_name = csv.stem.replace("_processed", "")
            df["Year"] = year
            df["RaceID"] = race_name
            df["SourceFile"] = csv.name
            parts.append(df)
            print(f"  [read] {year}/{race_name}  rows={len(df):,}")

    if not parts:
        raise SystemExit("No processed CSVs found.")
    return pd.concat(parts, ignore_index=True)


def add_stationary_pacedecay(df: pd.DataFrame) -> pd.DataFrame:
    if "Stationary_PaceDecay" in df.columns:
        return df
    df = df.sort_values(["Year", "RaceID", "Driver", "Stint", "LapNumber"]).copy()
    df["Stationary_PaceDecay"] = (
        df.groupby(["Year", "RaceID", "Driver", "Stint"])["LapTime_Seconds"]
          .diff()
    )
    return df


def main() -> None:
    print(f"Scanning processed CSVs under: {PROCESSED_DIR}")
    df = gather()
    print(f"\nTotal rows: {len(df):,}  |  columns: {df.shape[1]}")

    df = add_stationary_pacedecay(df)

    # Default placeholders so visualize.py can skip cleanly
    if "PitStopProbability" not in df.columns:
        df["PitStopProbability"] = 0.0
    if "Pit_Probability" not in df.columns:
        df["Pit_Probability"] = 0.0

    AGGREGATED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(AGGREGATED_CSV, index=False)
    print(f"\nWrote {AGGREGATED_CSV} "
          f"({AGGREGATED_CSV.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
