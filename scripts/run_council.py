"""
SCARLET SAILS: THE COUNCIL (Production Mode)
============================================
The Human-in-the-Loop Interface.
Connects directly to the Production Engine (Inference -> RL -> Decision).
NO FAKE DATA. REAL PIPELINE ONLY.
"""
import sys
import os
import pandas as pd
import numpy as np
import logging
import uuid
import json
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.canonical_pipeline import CanonicalPipeline
from strategies.xgboost_ml_v3 import XGBoostMLStrategyV3
from strategies.rule_based_v2 import RuleBasedStrategy
from strategies.hybrid_q_learner import HybridQLearner

# Настройка логов для файла (чтобы сохранять решения)
DECISIONS_FILE = os.path.join(PROJECT_ROOT, "rag", "trades", "trade_log.json")
os.makedirs(os.path.dirname(DECISIONS_FILE), exist_ok=True)

def load_history():
    if os.path.exists(DECISIONS_FILE):
        with open(DECISIONS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_decision(decision_data):
    history = load_history()
    history.append(decision_data)
    with open(DECISIONS_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def main():
    # 1. SETUP
    print("\n" + "="*60)
    print("  🏛️  THE COUNCIL SESSION (v3.0 Production)")
    print("="*60)
    
    COIN = "BTC"
    TIMEFRAME = "4h"

    # 2. LOAD DATA (REAL PIPELINE)
    print("⏳ Loading Market Context...")
    pipeline = CanonicalPipeline(strict_mode=False)
    try:
        state = pipeline.load_state(COIN, TIMEFRAME)
    except Exception as e:
        print(f"❌ Data Pipeline Error: {e}")
        return

    # 3. INIT STRATEGIES (REAL MODELS)
    try:
        model_path = f"models/xgboost_v3_{COIN.lower()}_{TIMEFRAME}.json"
        ml_strat = XGBoostMLStrategyV3(model_path=model_path)
        rb_strat = RuleBasedStrategy()
        rl_agent = HybridQLearner(model_dir="models")
    except Exception as e:
        print(f"❌ Strategy Init Error: {e}")
        return

    # 4. GET SIGNALS (Using the FIXED Logic)
    # --- ML ---
    try:
        raw_ml = ml_strat.generate_signal(state)
        p_ml = float(raw_ml.get('probability', 0.0)) if isinstance(raw_ml, dict) else float(raw_ml)
    except: p_ml = 0.0
    
    # --- RB (FIXED: With Close Price) ---
    try:
        rb_input = state.features.copy()
        rb_input['close'] = state.raw_ohlcv['close'] # CRITICAL FIX
        rb_out = rb_strat.generate_signals(rb_input)
        
        if isinstance(rb_out, pd.DataFrame):
            col = 'signal' if 'signal' in rb_out.columns else rb_out.columns[0]
            p_rb_raw = float(rb_out[col].iloc[-1])
        elif isinstance(rb_out, pd.Series):
            p_rb_raw = float(rb_out.iloc[-1])
        else:
            p_rb_raw = float(rb_out)
    except: p_rb_raw = 0.0
    
    p_rb_prob = (p_rb_raw + 1) / 2

    # --- RL MANAGER ---
    state_key = rl_agent.get_state_key(state)
    weights = rl_agent.select_action(state_key, is_training=False)
    w_rb, w_ml = weights
    vol_state = rl_agent._get_volatility_state(state)

    # --- FINAL VERDICT ---
    p_final = (w_rb * p_rb_prob) + (w_ml * p_ml)
    last_price = state.raw_ohlcv['close'].iloc[-1]
    
    # 5. PRESENTATION TO HUMAN
    print(f"\n📉 MARKET STATE: [{vol_state}] (Key: {state_key})")
    print("-" * 60)
    print(f"🤖 AGENTS:")
    print(f"  • ML (XGBoost):  {p_ml:.2f}  (Weight: {w_ml*100:.0f}%)")
    print(f"  • RB (Logic):    {p_rb_raw:.0f}     (Weight: {w_rb*100:.0f}%)")
    print("-" * 60)
    
    signal_str = "WAIT ⚪️"
    if p_final > 0.6: signal_str = "LONG 🟢"
    elif p_final < 0.4: signal_str = "SHORT 🔴"
    
    print(f"🎯 COUNCIL RECOMMENDATION:")
    print(f"   ACTION:     {signal_str}")
    print(f"   SCORE:      {p_final:.4f}")
    print(f"   PRICE:      ${last_price:,.2f}")
    print("=" * 60)

    # 6. HUMAN INTERVENTION
    choice = input("\n👨‍⚖️  YOUR DECISION (A=Accept / R=Reject / S=Skip): ").lower().strip()
    
    if choice == 'a':
        trade_id = uuid.uuid4().hex[:8]
        decision = {
            "id": trade_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "coin": COIN,
            "action": signal_str,
            "score": p_final,
            "price": last_price,
            "strategies": {"ml": p_ml, "rb": p_rb_raw},
            "status": "OPEN"
        }
        save_decision(decision)
        print(f"✅ Trade {trade_id} ACCEPTED and logged to database.")
        print(f"   Run: python scripts/update_outcome.py {trade_id} --close <price> later.")
        
    elif choice == 'r':
        print("❌ Trade REJECTED.")
    else:
        print("🤷‍♂️ Skipped.")

if __name__ == "__main__":
    main()
