# Pit-Stop Strategy: BiLSTM Prediction and Survival Analysis

**Topics:** Sequence modelling, survival analysis (Cox proportional
hazards), AR(1) time-series analysis, hypothesis testing, and regression.

## Research questions

1. Can a sequence model predict imminent pit stops from on-track telemetry?
2. Which stochastic process best characterises the hazard of pitting as
   tyre age increases?
3. Is there a statistically significant relationship between pit-crew speed
   and constructor championship performance?
4. Does a driver's reaction time to virtual safety-car (VSC) events predict
   whether they finish above their qualifying grid position?

## Data sources

This subproject draws on two distinct data sources.

| Source | Contents | Location |
|---|---|---|
| Ergast (Kaggle) CSVs | Historical pit-stop durations, constructor identifiers, and championship points. | `data/raw/` (shared with the other subprojects). |
| FastF1 telemetry | Per-lap speed, RPM, throttle, brake, DRS, gear, and tyre-age measurements. | `data/pit_stop_pipeline/` (regenerated locally; see below). |

The FastF1 dataset is not included in the repository because its retrieval
through the FastF1 API requires several hours. The trained model weights
and the final report figures are included so that the report can be
reproduced without re-running the data-acquisition pipeline.

## Execution

All commands are executed from the repository root
(`applied_stats_project/`).

### Reproducing the report figures (requires `all_training_data.csv`)

If the aggregated dataset is present at
`data/pit_stop_pipeline/all_training_data.csv`:

```bash
python -m src.pit_stop_strategy.visualize
```

### Full pipeline (requires FastF1 API access)

```bash
# 1. Download race and telemetry CSVs (slow; several hours per season).
python -m src.pit_stop_strategy.download

# 2. Preprocess each race into a per-race feature CSV.
python -m src.pit_stop_strategy.preprocess

# 3. Aggregate the per-race CSVs and produce descriptive statistics.
python -m src.pit_stop_strategy.analyse

# 4. Train the BiLSTM pit-stop predictor.
python -m src.pit_stop_strategy.train

# 5. Score every lap with the trained model and append pit-stop probabilities.
python -m src.pit_stop_strategy.predict_pit_probabilities

# 6. Cox proportional-hazards survival model and AR(1) pace-decay analysis.
python -m src.pit_stop_strategy.stochastic_modeling

# 7. Strategic regression: pit duration vs championship points; VSC reaction
#    vs finishing position.
python -m src.pit_stop_strategy.strategic_regression

# 8. Overtake classification (strategic versus on-track).
python -m src.pit_stop_strategy.overtake_analysis

# 9. Generate the final report figures.
python -m src.pit_stop_strategy.visualize
```

## Outputs

| Path | Contents |
|---|---|
| `outputs/pit_stop_strategy/plots/` | All figures (numbered 01–07 from `visualize.py`, plus diagnostic plots from `analyse.py` and `stochastic_modeling.py`). |
| `outputs/pit_stop_strategy/models/` | Pre-trained BiLSTM weights: `best_f1_model.pt` and `f1_bilstm_model.pt`. |

The full written report is available at
[`docs/pit_stop_strategy_report.md`](../../docs/pit_stop_strategy_report.md).
