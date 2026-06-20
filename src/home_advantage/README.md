# Home-Race Advantage in Formula 1

**Topics:** Paired t-test, linear mixed-effects modelling, time-series
cross-validation, XGBoost regression, and SHAP-based interpretation.

## Research question

Do Formula 1 drivers perform better, relative to their teammate, at their
home Grand Prix than at away races, and does any such effect interact with
the driver's level of experience?

## Methodology

1. **Era-adjusted teammate delta.** Each driver's race points are
   normalised by the race winner's points, which compensates for the four
   different points systems used between 1950 and 2026. The target variable
   Δ is the deviation of the driver's normalised points from the
   constructor (teammate) average for that race.
2. **Paired t-test.** For every driver who has competed at their home
   Grand Prix, the mean of Δ at home races is compared with the mean of Δ
   at away races. The procedure reports the t-statistic, p-value,
   Cohen's d effect size, and the 95% confidence interval of the paired
   difference.
3. **Linear mixed-effects model.** The model
   `Δ ~ home_race × experience_years` is fitted with a random intercept
   for each driver, to test whether the home effect strengthens or
   attenuates as experience accumulates.
4. **XGBoost regression with SHAP interpretation.** The teammate delta is
   regressed on experience, the home indicator, grid position, and the
   circuit crash rate. SHAP beeswarm and dependence plots are produced for
   interpretability.
5. **Time-series cross-validation.** A chronological five-fold split is
   used to estimate out-of-sample RMSE and MAE, reported in both
   normalised units and era-adjusted points.

## Execution

From the repository root (`applied_stats_project/`):

```bash
python -m src.home_advantage.run_analysis
```

A modular entry point is also provided for inspecting individual stages
without writing any figures or tables to disk:

```bash
python -m src.home_advantage.main
```

The canonical, report-generating pipeline is `run_analysis.py`.

## Inputs

`data/raw/results.csv`, `races.csv`, `drivers.csv`, `circuits.csv`,
`countries.csv` (a nationality-to-country mapping).

## Outputs

| Path | Contents |
|---|---|
| `outputs/home_advantage/figures/01_..._13_*.png` | The 13 figures referenced in the report. |
| `outputs/home_advantage/tables/results.json` | All numerical results: t-test, mixed-effects model, backtest folds, and dataset statistics. |
| `outputs/home_advantage/tables/mixedlm_summary.txt` | Complete mixed-effects model summary. |
| `outputs/home_advantage/tables/backtest_folds.csv` | Per-fold RMSE and MAE. |

The full written report is available at
[`docs/home_advantage_report.pdf`](../../docs/home_advantage_report.pdf).
