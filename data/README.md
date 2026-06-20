# Data

## `raw/` — Kaggle Ergast Formula 1 World Championship CSVs

This folder contains the single canonical copy of the Kaggle Ergast
historical Formula 1 dataset (seasons 1950–2024, with the 2025 schedule).
Three of the four subprojects (`lap_time_distributions`,
`starting_grid_advantage`, and `home_advantage`) read exclusively from this
location. The `pit_stop_strategy` subproject additionally consults these
CSVs in its `strategic_regression.py` module for the historical
pit-duration analysis.

Source:
<https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020>

| File | Rows | Description |
|---|---|---|
| `results.csv` | 27,304 | One row per driver per race: grid position, finishing position, points, status. |
| `lap_times.csv` | 618,766 | Every lap of every modern race. |
| `pit_stops.csv` | 22,193 | Every recorded pit stop. |
| `qualifying.csv` | — | Q1, Q2, and Q3 lap times. |
| `races.csv` | 1,171 | One row per Grand Prix. |
| `circuits.csv` | 78 | Circuit metadata: location, country, latitude, longitude. |
| `drivers.csv` | — | Driver metadata. |
| `constructors.csv` | — | Team (constructor) metadata. |
| `seasons.csv` | — | Season-level metadata. |
| `sprint_results.csv` | 502 | Sprint-race results (2021 onwards). |
| `status.csv` | 140 | DNF and classification reasons. |
| `driver_standings.csv` | 35,427 | Cumulative driver standings after each race. |
| `constructor_standings.csv` | 13,664 | Cumulative constructor standings. |
| `constructor_results.csv` | 12,898 | Per-race constructor points. |
| `countries.csv` | — | Auxiliary nationality-to-country mapping used by the `home_advantage` subproject. |

See [`../docs/f1_data_explainer.md`](../docs/f1_data_explainer.md) for the
shared primer on the relationships among these files.

## `pit_stop_pipeline/` — FastF1 telemetry dataset (not distributed)

The `pit_stop_strategy` subproject additionally requires a per-lap
telemetry dataset constructed from the FastF1 API. To regenerate this
dataset locally:

```bash
python -m src.pit_stop_strategy.download      # Slow: several hours via the FastF1 API.
python -m src.pit_stop_strategy.preprocess    # Produces the per-race _processed.csv files.
python -m src.pit_stop_strategy.analyse       # Aggregates into all_training_data.csv.
```

Full instructions are provided in
[`../src/pit_stop_strategy/README.md`](../src/pit_stop_strategy/README.md).
