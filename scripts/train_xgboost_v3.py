"""
SCRIPT: TRAIN XGBOOST V3 (Clean)
Training on 70 features. STRICTLY REMOVES 'TARGET' FROM INPUTS.
"""
import sys
import os
import json
import logging
import pandas as pd
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.canonical_pipeline import CanonicalPipeline

COIN = "BTC"
TIMEFRAME = "4h"
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_NAME = f"xgboost_v3_{COIN.lower()}_{TIMEFRAME}"

def create_target(ohlcv: pd.DataFrame, horizon: int = 1) -> pd.Series:
    future_close = ohlcv['close'].shift(-horizon)
    target = (future_close > ohlcv['close']).astype(int)
    return target

def main():
    print(f"\n🚀 TRAINING: {MODEL_NAME}")
    
    # 1. Load Data
    pipeline = CanonicalPipeline(strict_mode=False)
    state = pipeline.load_state(COIN, TIMEFRAME)
    print(f"✅ Data: {state.shape}")

    # 2. Prepare X (Features) and y (Target)
    # CRITICAL: Copy features and ensure 'target' is NOT in X
    X = state.features.copy()
    if 'target' in X.columns:
        X = X.drop(columns=['target'])
    
    y = create_target(state.raw_ohlcv)
    
    valid_mask = ~y.isna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]
    
    # Train/Test Split
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    print(f"Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(X.columns)}")

    # 3. Train
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X.columns))
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=list(X.columns))
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'eta': 0.05,
        'seed': 42
    }
    
    model = xgb.train(params, dtrain, num_boost_round=300, evals=[(dtest, 'test')], early_stopping_rounds=20, verbose_eval=50)

    # 4. Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(os.path.join(MODEL_DIR, f"{MODEL_NAME}.json"))
    
    # Save Metadata
    meta = {
        "feature_names": list(X.columns),
        "coin": COIN,
        "timeframe": TIMEFRAME,
        "metrics": {"auc": float(model.best_score)}
    }
    with open(os.path.join(MODEL_DIR, f"{MODEL_NAME}_meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)
        
    print(f"✨ SAVED MODEL & META. Features count: {len(X.columns)}")

if __name__ == "__main__":
    main()
