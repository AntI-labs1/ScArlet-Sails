"""
Rolling retrain utility for XGBoost v3.
"""

import argparse
from pathlib import Path

import pandas as pd

from core.feature_loader import FeatureLoader
from core.feature_engine_v2 import FeatureSpecV3
from scripts.train_xgboost_v3 import (
    build_dmatrices,
    compute_targets,
    evaluate_model,
    load_config,
    save_model,
    train_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Rolling retrain for XGBoost v3")
    parser.add_argument("--config-path", type=str, default="configs/model2_training.yaml")
    parser.add_argument("--experiment-name", type=str, default="rolling-retrain")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config_path)

    model_cfg = cfg["model2"]
    auto_cfg = cfg.get("auto_retrain", {})
    costs_cfg = cfg.get("costs", {})

    loader = FeatureLoader(data_dir=model_cfg.get("data_dir", "data/features"))
    coin = model_cfg.get("coin", "BTC")
    timeframe = model_cfg.get("timeframe", "15m")

    df = loader.load_features(coin=coin, timeframe=timeframe, validate=True)
    end_date = df.index.max()
    rolling_months = auto_cfg.get("rolling_months", 18)
    start_date = end_date - pd.DateOffset(months=rolling_months)

    df = df[df.index >= start_date]

    df_with_targets = compute_targets(
        df,
        horizon=model_cfg["horizon_bars"],
        commission=costs_cfg.get("commission", 0.001),
        slippage=costs_cfg.get("slippage", 0.0005),
        target_type=model_cfg.get("target_type", "fee_ret"),
    )

    # Sliding split: last 25% -> test, preceding 25% -> val, rest -> train
    n = len(df_with_targets)
    test_cut = int(n * 0.75)
    val_cut = int(n * 0.5)
    train_df = df_with_targets.iloc[:val_cut]
    val_df = df_with_targets.iloc[val_cut:test_cut]
    test_df = df_with_targets.iloc[test_cut:]

    feature_spec = FeatureSpecV3.from_dataframe(train_df)
    X_train, X_val, X_test, y_train, y_val, y_test = build_dmatrices(train_df, val_df, test_df, feature_spec)

    model = train_model(X_train, y_train, X_val, y_val, model_cfg.get("xgboost_params", {}))

    metrics_train = evaluate_model(model, X_train, y_train)
    metrics_val = evaluate_model(model, X_val, y_val)
    metrics_test = evaluate_model(model, X_test, y_test)

    output_path = Path(model_cfg.get("output_path", f"models/xgboost_v3_{coin.lower()}_{timeframe}.json"))
    timestamped = output_path.with_name(output_path.stem + f"_{end_date.date()}" + output_path.suffix)

    metadata = {
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "coin": coin,
        "timeframe": timeframe,
        "experiment": args.experiment_name,
        "source_data": str(loader.get_file_path(coin, timeframe)),
        "window_start": str(start_date),
        "window_end": str(end_date),
        "metrics": {"train": metrics_train, "val": metrics_val, "test": metrics_test},
        "horizon_bars": model_cfg.get("horizon_bars"),
        "target_type": model_cfg.get("target_type", "fee_ret"),
    }

    save_model(model, timestamped, feature_spec, metadata)
    print(f"Saved rolling retrained model to {timestamped}")


if __name__ == "__main__":
    main()
