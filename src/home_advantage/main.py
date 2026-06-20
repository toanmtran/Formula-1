"""Lightweight entry point. For the full pipeline see run_analysis.py."""

from .data_loader import load_and_merge_data
from .features import build_all_features
from .statistics import run_paired_ttest, run_mixed_effects_model
from .models import train_and_explain_xgboost, backtest_predictive_model


def main():
    df = load_and_merge_data()
    df = build_all_features(df)

    run_paired_ttest(df)
    run_mixed_effects_model(df)

    backtest_predictive_model(df)
    train_and_explain_xgboost(df)


if __name__ == "__main__":
    main()
