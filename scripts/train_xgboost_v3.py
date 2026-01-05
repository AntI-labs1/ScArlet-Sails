"""
SCRIPT: TRAIN XGBOOST V3 (Canonical Pipeline Edition)
Training script adapted for the new Canonical Data Pipeline architecture.
"""
import sys
import os
import json
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

# Setup Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from scripts/ since canonical_pipeline is there
from scripts.canonical_pipeline import CanonicalPipeline

# --- CONFIG ---
COIN = "BTC"
TIMEFRAME = "4h"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_NAME = f"xgboost_v3_{COIN.lower()}_{TIMEFRAME}"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TRAINER")


def create_target(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """
    Simple target: will price go up in next N candles?
    Returns 1 if Close(t+horizon) > Close(t), else 0
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must have 'close' column")
    
    future_close = df['close'].shift(-horizon)
    target = (future_close > df['close']).astype(int)
    return target


def main():
    print(f"\n🚀 STARTING TRAINING: {MODEL_NAME}")
    print("=" * 60)

    # 1. Initialize Pipeline
    print("[1/6] Initializing Canonical Pipeline...")
    pipeline = CanonicalPipeline()
    print(f"    ✅ Pipeline initialized")

    # 2. Load and validate data
    print("[2/6] Loading market data...")
    
    # For now, we'll load raw data from data/ directory
    # In production, this would come from the data fetcher
    data_dir = PROJECT_ROOT / "data"
    data_file = data_dir / f"{COIN.lower()}_{TIMEFRAME}.csv"
    
    if not data_file.exists():
        print(f"    ❌ Error: Data file not found: {data_file}")
        print(f"    Please ensure you have market data in: {data_file}")
        return
    
    # Load raw data
    try:
        raw_df = pd.read_csv(data_file)
        print(f"    ✅ Loaded {len(raw_df)} candles from {data_file}")
    except Exception as e:
        print(f"    ❌ Error loading data: {e}")
        return

    # 3. Validate data through pipeline
    print("[3/6] Validating data through Canonical Pipeline...")
    
    # Convert DataFrame rows to dict for validation
    validated_data = []
    for idx, row in raw_df.iterrows():
        data_point = {
            "timestamp": row.get('timestamp', row.get('time', datetime.now().isoformat())),
            "symbol": f"{COIN}/USDT",
            "price": float(row['close']),
            "volume": float(row.get('volume', 0)),
            "source": "historical"
        }
        
        try:
            canonical = pipeline.validate(data_point)
            validated_data.append(canonical)
        except Exception:
            continue
    
    print(f"    ✅ Validated {len(validated_data)}/{len(raw_df)} data points")
    stats = pipeline.get_stats()
    print(f"    Success rate: {stats['success_rate']:.2f}%")

    # 4. Prepare features and target
    print("[4/6] Preparing features and target...")
    
    # For this simplified version, we'll use the original DataFrame
    # In production, features would come from the validated data
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in raw_df.columns]
    if missing_cols:
        print(f"    ❌ Missing required columns: {missing_cols}")
        return
    
    # Create target
    y = create_target(raw_df)
    
    # Use OHLCV as basic features (you can add more technical indicators here)
    feature_cols = ['open', 'high', 'low', 'close', 'volume']
    X = raw_df[feature_cols].copy()
    
    # Remove rows with NaN in target
    valid_mask = ~y.isna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]
    
    # Fill any remaining NaN with 0
    X = X.fillna(0)
    
    print(f"    Features: {list(X.columns)}")
    print(f"    Samples: {len(X)}")

    # 5. Train/Test split (temporal)
    print("[5/6] Training XGBoost model...")
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"    Train samples: {len(X_train)}")
    print(f"    Test samples: {len(X_test)}")
    
    # Train XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X.columns))
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=list(X.columns))
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'eta': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=50,
        verbose_eval=50
    )

    # 6. Evaluate and save
    print("[6/6] Evaluating and saving model...")
    y_prob = model.predict(dtest)
    y_pred = (y_prob > 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n    ✅ Test Accuracy:  {acc:.4f}")
    print(f"    ✅ Test Precision: {prec:.4f}")
    print(f"    ✅ Test ROC AUC:   {auc:.4f}")
    
    if auc < 0.52:
        print(f"    ⚠️ Warning: Model AUC is low ({auc:.4f}). Consider:")
        print(f"       - Adding more technical indicators as features")
        print(f"       - Tuning hyperparameters")
        print(f"       - Using more training data")
    
    # Save model
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / f"{MODEL_NAME}.json"
    model.save_model(str(model_path))
    
    # Save metadata
    meta_path = MODEL_DIR / f"{MODEL_NAME}_meta.json"
    metadata = {
        "feature_names": list(X.columns),
        "n_features": len(X.columns),
        "trained_at": datetime.now().isoformat(),
        "metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "auc": float(auc)
        },
        "coin": COIN,
        "timeframe": TIMEFRAME,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "validation_stats": stats
    }
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✨ SUCCESS!")
    print(f"    Model saved to: {model_path}")
    print(f"    Metadata saved to: {meta_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
