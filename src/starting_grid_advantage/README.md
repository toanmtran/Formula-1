# Starting-Grid Advantage: Analysis and Prediction

**Topics:** Probability, descriptive statistics, ordinary least squares,
classification, and regression (logistic regression, random forest, LSTM).

## Research questions

1. What is the historical probability of victory given pole position,
   P(Win | Pole)?
2. How do the top five starting positions (P1–P5) compare with the midfield
   (P6–P10) in the modern era (≥ 2010), in terms of mean finishing
   position and points scored?
3. How does the Pearson correlation between starting and finishing position
   differ between street circuits (e.g. Monaco) and high-speed circuits
   (e.g. Monza)?
4. How do podium probability and mean finishing position decay as the
   starting grid position worsens?

A predictive component then trains the following models:

- Logistic regression and random-forest classifiers, predicting whether a
  driver finishes on the podium given the starting grid position, circuit
  type, and a constructor-strength index.
- An ordinary least-squares model with an interaction term, used to assess
  whether the effect of grid position is significantly stronger on street
  circuits.
- A linear regression and an LSTM regression for finishing position.

## Execution

From the repository root (`applied_stats_project/`):

```bash
python -m src.starting_grid_advantage.main
```

The script locks the random-number generators of NumPy, Python's `random`
module, and TensorFlow, and enables TensorFlow operator-level determinism,
so that results are exactly reproducible.

## Inputs

`data/raw/results.csv`, `races.csv`, `circuits.csv`.

## Outputs

| Path | Contents |
|---|---|
| `outputs/starting_grid_advantage/plots/01_podium_decay.png` | Podium probability as a function of starting grid position. |
| `outputs/starting_grid_advantage/plots/02_mean_finish_decay.png` | Mean finishing position as a function of starting grid position. |
| `outputs/starting_grid_advantage/metrics/circuit_grid_correlations.csv` | Per-circuit Pearson correlation between starting and finishing position. |
| `outputs/starting_grid_advantage/models/*.pkl`, `*.keras` | Trained classifiers, linear regression, and LSTM model. |
