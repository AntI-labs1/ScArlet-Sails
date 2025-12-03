"""
"""
Train XGBoost v3 with temporal split and fee-adjusted targets.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from backtesting.honest_backtest_v2 import HonestBacktestV2
from core.feature_engine_v2 import FeatureSpecV3
from core.feature_loader import FeatureLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost v3 model")
    parser.add_argument("--config-path", type=str, default="configs/model2_training.yaml")
    parser.add_argument("--experiment-name", type=str, default="default")
    parser.add_argument("--no-backtest", action="store_true", help="Skip validation backtest and threshold search")
    parser.add_argument("--verbose", action="store_true", help="Verbose diagnostics (probability stats, etc.)")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text())


def compute_targets(df: pd.DataFrame, horizon: int, commission: float, slippage: float, target_type: str) -> pd.DataFrame:
    df = df.copy()
    entry_price = df["open"].shift(-1)
    exit_price = df["close"].shift(-horizon)

    raw_ret = (exit_price - entry_price) / entry_price
    round_trip_cost = (commission + slippage) * 2
    fee_ret = raw_ret - round_trip_cost

    df["raw_ret"] = raw_ret
    df["fee_ret"] = fee_ret
    df["rapnl"] = fee_ret  # Placeholder until RAPnL is formalised

    if target_type == "fee_ret":
        target_series = (df["fee_ret"] > 0).astype(int)
    elif target_type == "raw_ret":
        target_series = (df["raw_ret"] > 0).astype(int)
    else:
        target_series = (df["rapnl"] > 0).astype(int)

    df["target"] = target_series
    df = df.iloc[:-horizon]  # drop tail with NaNs/unknown targets

    return df.dropna(subset=["target"])


def temporal_split(df: pd.DataFrame, train_start: str, train_end: str, val_end: str, test_end: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_index()

    train_mask = (df_sorted.index >= pd.Timestamp(train_start)) & (df_sorted.index < pd.Timestamp(train_end))
    val_mask = (df_sorted.index >= pd.Timestamp(train_end)) & (df_sorted.index < pd.Timestamp(val_end))
    if test_end:
        test_mask = (df_sorted.index >= pd.Timestamp(val_end)) & (df_sorted.index < pd.Timestamp(test_end))
    else:
        test_mask = df_sorted.index >= pd.Timestamp(val_end)

    return df_sorted[train_mask], df_sorted[val_mask], df_sorted[test_mask]


def build_dmatrices(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, spec: FeatureSpecV3):
    X_train = spec.enforce(train_df, raise_on_missing=True).drop(columns=["target"], errors="ignore")
    X_val = spec.enforce(val_df, raise_on_missing=True).drop(columns=["target"], errors="ignore")
    X_test = spec.enforce(test_df, raise_on_missing=True).drop(columns=["target"], errors="ignore")

    y_train = train_df["target"]
    y_val = val_df["target"]
    y_test = test_df["target"]

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, params: Dict) -> xgb.XGBClassifier:
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    params = params.copy()
    params.setdefault("objective", "binary:logistic")
    params.setdefault("eval_metric", "auc")
    params.setdefault("scale_pos_weight", scale_pos_weight)
    params.setdefault("random_state", 42)

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    return model


def evaluate_model(model: xgb.XGBClassifier, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> Dict:
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    return {
        "auc": float(roc_auc_score(y, proba)),
        "f1": float(f1_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "accuracy": float(accuracy_score(y, preds)),
        "threshold": threshold,
        "samples": int(len(y)),
        "class_balance": float(y.mean()),
    }


def backtest_thresholds(probabilities: np.ndarray, val_df: pd.DataFrame, thresholds: List[float], backtest_cfg: Dict) -> Dict:
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    ohlcv = val_df[ohlcv_cols]
    results = {}

    for th in thresholds:
        signals = (probabilities >= th).astype(int)
        engine = HonestBacktestV2(
            commission=backtest_cfg.get("commission", 0.001),
            slippage=backtest_cfg.get("slippage", 0.0005),
            max_hold_bars=backtest_cfg.get("max_hold_bars", 100),
            take_profit=backtest_cfg.get("take_profit", 0.02),
            stop_loss=backtest_cfg.get("stop_loss", 0.01),
            cooldown_bars=backtest_cfg.get("cooldown_bars", 10),
        )
        metrics = engine.run(ohlcv, signals)
        results[th] = metrics

    return results


def select_optimal_threshold(backtest_results: Dict[float, Dict], max_dd_limit: float) -> Dict:
    best_th = None
    best_sharpe = -np.inf
    for th, metrics in backtest_results.items():
        if metrics.get("max_drawdown_pct", 0) <= max_dd_limit and metrics.get("sharpe_ratio", -np.inf) > best_sharpe:
            best_sharpe = metrics["sharpe_ratio"]
            best_th = th
    all_sharpes = [m.get("sharpe_ratio", -np.inf) for m in backtest_results.values()]
    no_profitable = all(s <= 0 for s in all_sharpes)
    return {
        "threshold": best_th,
        "sharpe": best_sharpe,
        "backtest_metrics": backtest_results.get(best_th, {}),
        "no_profitable_threshold": no_profitable,
    }


def save_model(model: xgb.XGBClassifier, output_path: Path, feature_spec: FeatureSpecV3, metadata: Dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))

    meta_path = output_path.with_name(output_path.stem + "_metadata.json")
    metadata = metadata.copy()
    metadata.update({
        "feature_names": feature_spec.feature_names,
        "n_features": feature_spec.n_features,
    })
    meta_path.write_text(json.dumps(metadata, indent=2))


def main():
    args = parse_args()
    cfg = load_config(args.config_path)

    model_cfg = cfg["model2"]
    costs_cfg = cfg.get("costs", {})
    backtest_cfg = cfg.get("backtest", {})

    loader = FeatureLoader(data_dir=model_cfg.get("data_dir", "data/features"))
    coin = model_cfg.get("coin", "BTC")
    timeframe = model_cfg.get("timeframe", "15m")
    file_path = loader.get_file_path(coin, timeframe)
    df = loader.load_features(coin=coin, timeframe=timeframe, start_date=model_cfg["train_start"], end_date=model_cfg.get("test_end"), validate=True)

    df_with_targets = compute_targets(
        df,
        horizon=model_cfg["horizon_bars"],
        commission=costs_cfg.get("commission", 0.001),
        slippage=costs_cfg.get("slippage", 0.0005),
        target_type=model_cfg.get("target_type", "fee_ret"),
    )

    train_df, val_df, test_df = temporal_split(
        df_with_targets,
        train_start=model_cfg["train_start"],
        train_end=model_cfg["train_end"],
        val_end=model_cfg["val_end"],
        test_end=model_cfg.get("test_end"),
    )

    min_train = model_cfg.get("min_train_samples", 1000)
    min_val = model_cfg.get("min_val_samples", 200)

    if len(train_df) < min_train or len(val_df) < min_val:
        print("SKIPPING TRAINING: insufficient data for robust training/validation")
        print(f"  Train samples: {len(train_df)} (need >= {min_train})")
        print(f"  Val samples:   {len(val_df)} (need >= {min_val})")
        if isinstance(df.index, pd.DatetimeIndex):
            print(f"  Full data range: {df.index.min()} to {df.index.max()}")

        output_path = Path(model_cfg.get("output_path", f"models/xgboost_v3_{coin.lower()}_{timeframe}.json"))
        meta = {
            "created_at": pd.Timestamp.utcnow().isoformat(),
            "coin": coin,
            "timeframe": timeframe,
            "experiment": args.experiment_name,
            "source_data": str(file_path),
            "status": "SKIPPED",
            "reason": "insufficient_history",
            "train_samples": int(len(train_df)),
            "val_samples": int(len(val_df)),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path = output_path.with_name(output_path.stem + "_metadata.json")
        meta_path.write_text(json.dumps(meta, indent=2))
        return

    feature_spec = FeatureSpecV3.from_dataframe(train_df)
    if feature_spec.n_features != 74:
        raise ValueError(f"Expected 74 features, got {feature_spec.n_features}")

    X_train, X_val, X_test, y_train, y_val, y_test = build_dmatrices(train_df, val_df, test_df, feature_spec)

    model_params = model_cfg.get("xgboost_params", {})
    model = train_model(X_train, y_train, X_val, y_val, model_params)

    metrics_train = evaluate_model(model, X_train, y_train, threshold=0.5)
    metrics_val = evaluate_model(model, X_val, y_val, threshold=0.5)
    metrics_test = evaluate_model(model, X_test, y_test, threshold=0.5)

    optimal = {"threshold": None, "sharpe": None, "backtest_metrics": {}}
    if not args.no_backtest:
        print("Optimizing threshold on validation set...")
        val_prob = model.predict_proba(X_val)[:, 1]

        val_df_clean = val_df.loc[X_val.index].copy()
        val_df_clean["P_ml"] = val_prob

        if args.verbose:
            import numpy as np

            print("\nProbability distribution on Val:")
            print(f"  Min:    {val_prob.min():.4f}")
            print(f"  Max:    {val_prob.max():.4f}")
            print(f"  Mean:   {val_prob.mean():.4f}")
            print(f"  Median: {np.median(val_prob):.4f}")
            for thr in [0.3, 0.5, 0.7]:
                count = (val_prob > thr).sum()
                frac = count / len(val_prob) * 100.0
                print(f"  > {thr:.1f}: {count} samples ({frac:.2f}%)")
            print()

        thresholds = backtest_cfg.get("threshold_grid", [round(x, 2) for x in np.linspace(0.5, 0.9, 5)])
        bt_results = backtest_thresholds(val_prob, val_df, thresholds, backtest_cfg)
        optimal = select_optimal_threshold(bt_results, max_dd_limit=backtest_cfg.get("max_dd_pct", 20))

        best_thr = optimal["threshold"]
        bt_metrics = optimal.get("backtest_metrics", {})
        n_trades = bt_metrics.get("n_trades", 0)

        if best_thr is not None and n_trades > 0:
            trades_mask = val_df_clean["P_ml"] >= best_thr
            trades_df = val_df_clean.loc[trades_mask].copy()

            cols_to_keep = [
                "open",
                "high",
                "low",
                "close",
                "raw_ret",
                "fee_ret",
                "target",
                "P_ml",
            ]
            cols_existing = [c for c in cols_to_keep if c in trades_df.columns]
            trades_export = trades_df[cols_existing]

            diagnostics_dir = Path(backtest_cfg.get("diagnostics_dir", "diagnostics"))
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
            trades_path = diagnostics_dir / f"val_trades_{coin.lower()}_{timeframe}_{best_thr:.2f}.csv"
            trades_export.to_csv(trades_path)
            print(f"Exported {len(trades_export)} validation trades to: {trades_path}")

    output_path = Path(model_cfg.get("output_path", f"models/xgboost_v3_{coin.lower()}_{timeframe}.json"))
    metadata = {
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "coin": coin,
        "timeframe": timeframe,
        "experiment": args.experiment_name,
        "source_data": str(file_path),
        "metrics": {
            "train": metrics_train,
            "val": metrics_val,
            "test": metrics_test,
        },
        "optimal_threshold_trading": optimal,
        "target_type": model_cfg.get("target_type", "fee_ret"),
        "horizon_bars": model_cfg.get("horizon_bars"),
    }

    save_model(model, output_path, feature_spec, metadata)

    summary = {
        "train": metrics_train,
        "val": metrics_val,
        "test": metrics_test,
        "best_threshold": optimal,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
