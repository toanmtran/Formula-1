import os
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from .config import FIG_DIR


def train_and_explain_xgboost(df: pd.DataFrame):
    print("\n--- Training XGBoost & Generating SHAP ---")

    df['grid'] = pd.to_numeric(df['grid'], errors='coerce')

    features = ['experience_years', 'home_race', 'grid', 'circuit_crash_rate']
    target = 'teammate_delta'

    model_df = df.dropna(subset=features + [target])
    X = model_df[features]
    y = model_df[target]

    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    xgb_model.fit(X, y)

    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)

    shap.dependence_plot(
        "experience_years",
        shap_values,
        X,
        interaction_index="home_race",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "shap_dependence_exp_home.png"), dpi=160, bbox_inches="tight")
    plt.close("all")


def backtest_predictive_model(df: pd.DataFrame):
    print("\n--- Running Time-Series Backtesting ---")

    df['grid'] = pd.to_numeric(df['grid'], errors='coerce')
    features = ['experience_years', 'home_race', 'grid', 'circuit_crash_rate']
    target = 'teammate_delta'

    model_df = df.dropna(subset=features + [target, 'year']).copy()
    # Chronological order is critical for time-series CV
    model_df = model_df.sort_values(by=['year', 'raceId']).reset_index(drop=True)

    X = model_df[features]
    y = model_df[target]

    tscv = TimeSeriesSplit(n_splits=5)

    fold = 1
    rmse_scores = []
    mae_scores  = []

    xgb_params = {
        'n_estimators': 100,
        'max_depth': 4,
        'learning_rate': 0.1,
        'random_state': 42,
    }

    print(f"Testing across {tscv.get_n_splits()} chronological folds...")

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        train_years = model_df.iloc[train_index]['year'].unique()
        test_years  = model_df.iloc[test_index]['year'].unique()

        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train, y_train)

        predictions_norm = model.predict(X_test)

        rmse_norm = np.sqrt(mean_squared_error(y_test, predictions_norm))
        mae_norm  = mean_absolute_error(y_test, predictions_norm)

        rmse_scores.append(rmse_norm)
        mae_scores.append(mae_norm)

        # Convert normalised predictions back to real era-adjusted points
        test_max_points = model_df.iloc[test_index]['max_race_points'].values
        predictions_real = predictions_norm * test_max_points
        y_test_real      = y_test * test_max_points

        mae_real  = mean_absolute_error(y_test_real, predictions_real)
        rmse_real = root_mean_squared_error(y_test_real, predictions_real)

        print(f"Fold {fold}:")
        print(f"  Train: {train_years.min()}-{train_years.max()} | Test: {test_years.min()}-{test_years.max()}")
        print(f"  Normalized MAE: {mae_norm:.4f} | Normalized RMSE: {rmse_norm:.4f} (Model scale)")
        print(f"  Real Points MAE: {mae_real:.2f} | Real Points RMSE: {rmse_real:.4f} (Era-adjusted points scale)")

        fold += 1

    print("\n--- Final Predictive Performance ---")
    print(f"Average RMSE: {np.mean(rmse_scores):.4f} (+/- {np.std(rmse_scores):.4f})")
    print(f"Average MAE:  {np.mean(mae_scores):.4f} (+/- {np.std(mae_scores):.4f})")

    return np.mean(rmse_scores)
