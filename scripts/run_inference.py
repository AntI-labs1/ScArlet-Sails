"""
SCARLET SAILS: INFERENCE ENGINE
===============================
The "Brain" that connects all components.
Final Architecture: Passes FULL Context (Features + Raw OHLCV) to strategies.
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
    
    # 1. ML SIGNAL (Stateless, Pure Math)
    try:
        raw_ml = ml_strat.generate_signal(state)
        p_ml = float(raw_ml.get('probability', 0.0)) if isinstance(raw_ml, dict) else float(raw_ml)
    except Exception as e:
        print(f"⚠️ ML Error: {e}")
        p_ml = 0.0
    
    # 2. RB SIGNAL (Context Aware)
    try:
        # ARCHITECTURAL FIX: Merge Features with ALL Raw Data (Open, High, Low, Close, Volume)
        # This gives the strategy full vision of the market.
        rb_input = state.features.copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in state.raw_ohlcv.columns:
                rb_input[col] = state.raw_ohlcv[col]
        
        rb_out = rb_strat.generate_signals(rb_input)
        
        if isinstance(rb_out, pd.DataFrame):
            # Try to find 'signal' column, otherwise take first column
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
    
    # 3. RL DECISION (Manager)
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
    
    # 4. FINAL CALCULATION
    p_final = (w_rb * p_rb_prob) + (w_ml * p_ml)
    last_price = state.raw_ohlcv['close'].iloc[-1]
    
    print(f"📉 Market State: [{vol}] | Key: {state_key}")
    print(f"\n🤖 Signals:")
    print(f"   ML (XGBoost): {p_ml:.4f} (Honest Alpha)")
    print(f"   RB (Logic):   {p_rb_raw:.0f} (Prob: {p_rb_prob:.2f})")
    print(f"\n⚖️  Manager Weights:")
    print(f"   Trust RB: {w_rb*100:.0f}%")
    print(f"   Trust ML: {w_ml*100:.0f}%")
    
    print(f"\n🎯 FINAL SCORE: {p_final:.4f}")
    
    direction = "WAIT ⚪️"
    if p_final > 0.6: direction = "LONG 🟢"
    elif p_final < 0.4: direction = "SHORT 🔴"
    
    print(f"👉 ACTION: {direction}")
    print(f"💰 PRICE:  ${last_price:,.2f}")
    print("="*40)

if __name__ == "__main__":
    main()
