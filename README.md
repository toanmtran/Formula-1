# Applied Statistics on Formula 1

## Overview

This project investigates four distinct statistical research questions on a
shared corpus of historical Formula 1 data (1950–2025). The work is
organised as four self-contained subprojects that share a common dataset, a
unified directory layout, and a consistent convention for producing
reproducible outputs (figures, metrics, and trained models written to disk
under `outputs/`).

## Subprojects

| Subproject | Research question | Statistical methods |
|---|---|---|
| [`src/lap_time_distributions/`](src/lap_time_distributions/) | Which probability distribution best describes lap times within a single race, and does the answer depend on the circuit or the regulation era? | Maximum-likelihood fits of Normal, Log-Normal, Gamma, Weibull, and Gumbel distributions; Kolmogorov–Smirnov and chi-squared goodness-of-fit tests; AIC and BIC model selection; quantile–quantile diagnostics. |
| [`src/pit_stop_strategy/`](src/pit_stop_strategy/) | Can pit stops be predicted from on-track telemetry? Which stochastic process best models the pit-stop hazard as a function of tyre age? Does pit-crew speed correlate with constructor performance? | Three-layer bidirectional LSTM sequence model; Cox proportional hazards survival model; AR(1) pace-decay filtering; Welch's t-test; random-forest classification of tactical agility. |
| [`src/starting_grid_advantage/`](src/starting_grid_advantage/) | What is the probability of victory given pole position? How does podium probability decay with starting grid position? Does the relationship between starting and finishing position depend on circuit type? | Descriptive probability estimation; ordinary least squares with interaction terms; logistic regression; random forest; LSTM regression. |
| [`src/home_advantage/`](src/home_advantage/) | Do drivers perform better, relative to their teammate, at their home Grand Prix than at away races, and does any such effect interact with experience? | Era-adjusted teammate delta; paired t-test; linear mixed-effects model with random intercept per driver; XGBoost regression with SHAP interpretation; time-series cross-validation. |

Each subproject contains its own `README.md` documenting the research
hypotheses, the implementation details, the expected outputs, and the
commands required to reproduce them.

## Repository layout

```
applied_stats_project/
├── README.md                 — project overview (this file)
├── requirements.txt          — consolidated pip dependencies
├── environment.yml           — Conda environment specification
│
├── data/
│   └── raw/                  — Kaggle Ergast CSVs and countries.csv (shared)
│
├── src/
│   ├── lap_time_distributions/
│   ├── pit_stop_strategy/        — FastF1 ingest and BiLSTM pipeline
│   ├── starting_grid_advantage/
│   └── home_advantage/
│
├── outputs/                  — generated artefacts, one folder per subproject
│   ├── lap_time_distributions/
│   ├── pit_stop_strategy/{plots,models}/
│   ├── starting_grid_advantage/{plots,metrics,models}/
│   └── home_advantage/{figures,tables}/
│
└── docs/                     — shared documentation and per-subproject reports
    ├── f1_data_explainer.md
    ├── home_advantage_report.pdf
    ├── home_advantage_report.tex
    ├── pit_stop_strategy_report.md
    └── starting_grid_advantage_report.pdf
```

## Reproducing the results

All commands are executed from the repository root
(`applied_stats_project/`). Each subproject is organised as a Python package
under `src/` and is invoked with `python -m`.

```bash
# Lap-time distributions
python -m src.lap_time_distributions.lap_distributions

# Starting-grid advantage
python -m src.starting_grid_advantage.main

# Home-race advantage (full report pipeline)
python -m src.home_advantage.run_analysis

# Pit-stop strategy (requires all_training_data.csv in data/pit_stop_pipeline/)
python -m src.pit_stop_strategy.visualize
```

The pit-stop subproject involves a multi-stage pipeline (FastF1 download,
preprocessing, BiLSTM training, inference, survival modelling, and
visualisation). The full sequence of commands is documented in
[`src/pit_stop_strategy/README.md`](src/pit_stop_strategy/README.md).

## Installation

The project targets Python 3.11. A fresh virtual environment is
recommended.

```bash
pip install -r requirements.txt
```

Conda users may instead create the environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate applied_stats
```

### Notes on optional dependencies

- The `pit_stop_strategy` subproject requires `torch` and, only for
  regenerating its source dataset, `fastf1`. The `starting_grid_advantage`
  subproject requires `tensorflow` for its LSTM model. Both `torch` and
  `tensorflow` are heavyweight installations; they may be omitted if the
  corresponding subprojects are not being executed.
- Pre-trained BiLSTM weights (`best_f1_model.pt`, `f1_bilstm_model.pt`) are
  included under `outputs/pit_stop_strategy/models/`, so the pit-stop report
  figures may be reproduced without retraining the network.

## Dataset

`data/raw/` contains the Kaggle Ergast Formula 1 World Championship CSVs
(seasons 1950–2024, with the 2025 schedule) together with the auxiliary
`countries.csv` nationality-to-country mapping used by the home-advantage
subproject. A complete data dictionary is provided in
[`docs/f1_data_explainer.md`](docs/f1_data_explainer.md).

The pit-stop subproject additionally requires per-lap telemetry data
obtained from the FastF1 API for the 2018–2025 seasons. These files are
not distributed with the repository owing to their size (several gigabytes)
and the time required to retrieve them. The trained model weights and the
final report figures are included so that the report can be reproduced
without re-running the data-acquisition pipeline.
