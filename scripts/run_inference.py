"""
SCARLET SAILS: INFERENCE ENGINE
===============================
The "Brain" that connects all components:
Pipeline -> Features -> (ML + RB) -> RL Weighting -> Decision.
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

# Setup Logging
logging.basicConfig(level=logging.WARNING)
print("🚀 Initializing Inference Engine...")

def main():
    COIN = "BTC"
    TIMEFRAME = "4h"
    
    # 1. Pipeline (Eyes)
    pipeline = CanonicalPipeline(strict_mode=False)
    try:
        state = pipeline.load_state(COIN, TIMEFRAME)
    except Exception as e:
        print(f"❌ Data Error: {e}")
        return

    # 2. Load Strategies
    ml_strat = XGBoostMLStrategyV3(model_path=f"models/xgboost_v3_{COIN.lower()}_{TIMEFRAME}.json")
    rb_strat = RuleBasedStrategy()
    rl_agent = HybridQLearner(model_dir="models")

    print("\n" + "="*50)
    print(f"🧠 MARKET SNAPSHOT: {COIN} {TIMEFRAME}")
    print("="*50)
    
    # --- GET SIGNALS ---
    # ML Signal
    ml_result = ml_strat.generate_signal(state.features)
    p_ml = ml_result['probability']
    
    # RB Signal (need to implement generate_signal that takes state)
    # For now, mock it based on RSI
    rsi_high = state.features['regime_rsi_high'].iloc[-1]
    rsi_low = state.features['regime_rsi_low'].iloc[-1]
    
    if rsi_high > 0.5:
        p_rb_prob = 0.2  # Overbought -> Short bias
    elif rsi_low > 0.5:
        p_rb_prob = 0.8  # Oversold -> Long bias
    else:
        p_rb_prob = 0.5  # Neutral
    
    # --- RL DECISION ---
    vol_state = rl_agent._get_volatility_state(state)
    trend_state = rl_agent._get_trend_state(state)
    ood_state = rl_agent._get_ood_state(state)
    
    state_key = rl_agent.get_state_key(state)
    weights = rl_agent.select_action(state_key, is_training=False)
    w_rb, w_ml = weights
    
    # --- HYBRID CALCULATION ---
    p_final = (w_rb * p_rb_prob) + (w_ml * p_ml)
    
    # --- OUTPUT ---
    last_price = state.raw_ohlcv['close'].iloc[-1]
    
    print(f"📉 Context:")
    print(f"   • Volatility: {vol_state}")
    print(f"   • Trend:      {trend_state}")
    print(f"   • OOD Status: {ood_state}")
    print(f"   • State Key:  [{state_key}]")
    
    print(f"\n🤖 Strategies:")
    print(f"   • Rule-Based: {p_rb_prob:.2f}")
    print(f"   • ML Model:   {p_ml:.4f}")
    
    print(f"\n⚖️  RL Manager Decision:")
    print(f"   • Trust RB:   {w_rb*100:.0f}%")
    print(f"   • Trust ML:   {w_ml*100:.0f}%")
    
    print(f"\n🎯 FINAL VERDICT:")
    print(f"   • Probability: {p_final:.4f}")
    
    direction = "WAIT ⏸️"
    if p_final > 0.6: direction = "LONG 🟢"
    elif p_final < 0.4: direction = "SHORT 🔴"
    
    print(f"   • Signal:      {direction}")
    print(f"   • Price:       ${last_price:,.2f}")
    print("="*50)

if __name__ == "__main__":
    main()
