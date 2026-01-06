import os

print("🛠️ FINAL SYSTEM REPAIR STARTED...")

# 1. ПРАВИЛЬНЫЙ СКРИПТ ОБУЧЕНИЯ (УДАЛЯЕТ TARGET ЧТОБЫ НЕ БЫЛО ОШИБОК)
train_code = r'''"""
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
'''

# 2. ПРАВИЛЬНЫЙ СКРИПТ ЗАПУСКА (ИНФЕРЕНС)
inference_code = r'''"""
SCARLET SAILS: INFERENCE ENGINE
Fixed: Passes raw prices to Rule-Based Strategy.
"""
import sys
import os
import pandas as pd
import numpy as np
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.canonical_pipeline import CanonicalPipeline
from strategies.xgboost_ml_v3 import XGBoostMLStrategyV3
from strategies.rule_based_v2 import RuleBasedStrategy
from strategies.hybrid_q_learner import HybridQLearner

logging.basicConfig(level=logging.WARNING)

def main():
    print("\n🚀 INFERENCE ENGINE STARTING...")
    COIN = "BTC"
    TIMEFRAME = "4h"
    
    pipeline = CanonicalPipeline(strict_mode=False)
    state = pipeline.load_state(COIN, TIMEFRAME)

    try:
        model_path = f"models/xgboost_v3_{COIN.lower()}_{TIMEFRAME}.json"
        ml_strat = XGBoostMLStrategyV3(model_path=model_path)
        rb_strat = RuleBasedStrategy()
        rl_agent = HybridQLearner(model_dir="models")
    except Exception as e:
        print(f"❌ Init Error: {e}")
        return

    print("="*40)
    print(f"🧠 ANALYZING: {COIN} {TIMEFRAME}")
    print("="*40)
    
    # 1. ML SIGNAL
    try:
        raw_ml = ml_strat.generate_signal(state)
        p_ml = float(raw_ml.get('probability', 0.0)) if isinstance(raw_ml, dict) else float(raw_ml)
    except Exception as e:
        print(f"⚠️ ML Error: {e}")
        p_ml = 0.0
    
    # 2. RB SIGNAL (FIX: Add 'close' price)
    try:
        rb_input = state.features.copy()
        rb_input['close'] = state.raw_ohlcv['close'] # ВАЖНО: Добавляем цену
        
        rb_out = rb_strat.generate_signals(rb_input)
        
        if isinstance(rb_out, pd.DataFrame):
            col = 'signal' if 'signal' in rb_out.columns else rb_out.columns[0]
            p_rb_raw = float(rb_out[col].iloc[-1])
        elif isinstance(rb_out, pd.Series):
            p_rb_raw = float(rb_out.iloc[-1])
        else:
            p_rb_raw = float(rb_out)
    except Exception as e:
        print(f"⚠️ RB Error: {e}")
        p_rb_raw = 0.0

    p_rb_prob = (p_rb_raw + 1) / 2 # Normalize -1..1 to 0..1
    
    # 3. RL DECISION
    try:
        state_key = rl_agent.get_state_key(state)
        weights = rl_agent.select_action(state_key, is_training=False)
        w_rb, w_ml = weights
        vol = rl_agent._get_volatility_state(state)
    except Exception as e:
        print(f"⚠️ RL Error: {e}")
        w_rb, w_ml = 0.5, 0.5
        state_key = "UNKNOWN"
        vol = "UNKNOWN"
    
    # 4. FINAL
    p_final = (w_rb * p_rb_prob) + (w_ml * p_ml)
    last_price = state.raw_ohlcv['close'].iloc[-1]
    
    print(f"📉 Market State: [{vol}] | Key: {state_key}")
    print(f"\n🤖 Signals:")
    print(f"   ML (XGBoost): {p_ml:.4f}")
    print(f"   RB (Logic):   {p_rb_raw:.0f} (Prob: {p_rb_prob:.2f})")
    print(f"\n⚖️  Manager Weights:")
    print(f"   Trust RB: {w_rb*100:.0f}%")
    print(f"   Trust ML: {w_ml*100:.0f}%")
    
    print(f"\n🎯 FINAL SCORE: {p_final:.4f}")
    
    if p_final > 0.6: print(f"👉 ACTION: LONG 🟢 (${last_price:,.0f})")
    elif p_final < 0.4: print(f"👉 ACTION: SHORT 🔴 (${last_price:,.0f})")
    else: print(f"👉 ACTION: WAIT ⚪️")
    print("="*40)

if __name__ == "__main__":
    main()
'''

with open('scripts/train_xgboost_v3.py', 'w') as f:
    f.write(train_code)
print("✅ Fixed scripts/train_xgboost_v3.py (Removes target leakage)")

with open('scripts/run_inference.py', 'w') as f:
    f.write(inference_code)
print("✅ Fixed scripts/run_inference.py (Adds close price for RB)")