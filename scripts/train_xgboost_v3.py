"""
SCRIPT: TRAIN XGBOOST V3 (Canonical Pipeline Edition)
Training on 75 features using the new Single Source of Truth.
"""
import sys
import os
import json
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

# Setup Paths to include project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.canonical_pipeline import CanonicalPipeline

# --- CONFIG ---
COIN = "BTC"
TIMEFRAME = "4h"  # Используем 4h, так как в логах ты тестировал его успешно
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_NAME = f"xgboost_v3_{COIN.lower()}_{TIMEFRAME}"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TRAINER")

def create_target(ohlcv: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """
    Простой таргет: цена закрытия выросла через horizon бар?
    """
    future_close = ohlcv['close'].shift(-horizon)
    target = (future_close > ohlcv['close']).astype(int)
    return target

def main():
    print(f"\n🚀 STARTING TRAINING: {MODEL_NAME}")
    print("=" * 60)

    # 1. Load Data via Canonical Pipeline
    print("[1/5] Loading Canonical State...")
    # strict_mode=False позволяет загрузить данные даже если реестр (feature_registry) еще пуст
    pipeline = CanonicalPipeline(strict_mode=False) 
    
    try:
        # МАГИЯ V3: Пайплайн сам находит файл, чистит NaN и валидирует
        state = pipeline.load_state(COIN, TIMEFRAME)
        print(f"    ✅ Loaded State: {state.shape} features")
        print(f"    ℹ️  Version: {state.version}")
    except Exception as e:
        print(f"❌ Critical Error loading state: {e}")
        print(f"    Check if 'data/features/{COIN}_USDT_{TIMEFRAME}_features.parquet' exists.")
        return

    # 2. Prepare X and y
    print("[2/5] Preparing Dataset...")
    X = state.features
    y = create_target(state.raw_ohlcv)
    
    # Убираем NaN в таргете (последний бар, т.к. мы смотрим в будущее)
    valid_mask = ~y.isna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]
    
    # Дополнительная страховка от NaN (хотя Pipeline должен был убрать)
    if X.isna().any().any():
        print("    ⚠️ Warning: NaN found in features. Filling with 0.")
        X = X.fillna(0)

    # Time-series split (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"    Train: {len(X_train)} samples")
    print(f"    Test:  {len(X_test)} samples")
    print(f"    Features used: {len(X.columns)}")

    # 3. Train XGBoost
    print("[3/5] Training XGBoost...")
    # DMatrix нужен для сохранения имен фич внутри модели
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
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=50,
        verbose_eval=50
    )

    # 4. Evaluate
    print("[4/5] Evaluating...")
    y_prob = model.predict(dtest)
    y_pred = (y_prob > 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"    ✅ Accuracy:  {acc:.4f}")
    print(f"    ✅ Precision: {prec:.4f}")
    print(f"    ✅ ROC AUC:   {auc:.4f}")

    if auc < 0.51:
        print("    ⚠️ WARNING: Model is barely learning. Check targets.")

    # 5. Save Artifacts
    print("[5/5] Saving Model & Metadata...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save Model (JSON format)
    model_json_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.json")
    model.save_model(model_json_path)
    
    # Save Feature List (CRITICAL for Canonical State validation)
    meta_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_meta.json")
    metadata = {
        "feature_names": list(X.columns),
        "n_features": len(X.columns),
        "trained_at": datetime.now().isoformat(),
        "metrics": {"auc": auc, "precision": prec},
        "coin": COIN,
        "timeframe": TIMEFRAME,
        "version": state.version
    }
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\n✨ SUCCESS! Model saved to: {model_json_path}")
    print(f"📝 Metadata saved to: {meta_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
