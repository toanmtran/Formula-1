# Lap-Time Distributions and Goodness-of-Fit Analysis

**Topic:** Probability distributions and goodness-of-fit testing.

## Research question

Which probability distribution best describes lap times within a single
Formula 1 race, and does the distributional form depend on the circuit or
the regulation era?

## Methodology

1. Select representative races spanning different circuit types and
   regulation eras.
2. Clean the lap-time observations by removing the formation lap, laps
   affected by safety-car or virtual safety-car deployments, pit in-laps
   and out-laps, and outliers identified by the interquartile-range rule.
3. Fit three candidate parametric families by maximum likelihood: Normal,
   Log-Normal, and Gamma.
4. Rank the candidate distributions using the Akaike Information
   Criterion (AIC) and Bayesian Information Criterion (BIC).
5. Produce diagnostic visualisations: histograms with overlaid probability
   density functions, quantile–quantile plots, and empirical versus
   theoretical cumulative distribution functions.
6. Compare the selected best-fit distributions across circuits (Monza,
   Monaco, Silverstone) and across regulation eras (V10, V8, turbo-hybrid,
   ground-effect).

## Execution

From the repository root (`applied_stats_project/`):

```bash
python -m src.lap_time_distributions.lap_distributions
```

## Inputs

`data/raw/lap_times.csv`, `races.csv`, `circuits.csv`, `results.csv`.

## Outputs

Six figures are written to `outputs/lap_time_distributions/`:

| File | Content |
|---|---|
| `fig1_distribution_overview.png` | Histogram with fitted PDFs, AIC and BIC comparison, Q-Q plot for the best-fitting distribution, and CDF comparison. |
| `fig2_qq_all_distributions.png` | Q-Q plots for all three candidate distributions. |
| `fig3_gof_summary_table.png` | Tabulated goodness-of-fit results. |
| `fig4_cross_circuit.png` | Distributional comparison across circuit types. |
| `fig5_cross_era.png` | Distributional comparison across regulation eras. |
| `fig6_skewness_synthesis.png` | Skewness summary and frequency of selected best-fit distributions. |

Refer to [`docs/f1_data_explainer.md`](../../docs/f1_data_explainer.md) for
the shared primer on the Formula 1 dataset.
