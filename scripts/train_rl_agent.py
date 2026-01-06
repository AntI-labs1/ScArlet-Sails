"""
SCRIPT: TRAIN RL AGENT v2 (Fixed Feature Names)
================================================
"""
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.canonical_pipeline import CanonicalPipeline
from strategies.hybrid_q_learner import HybridQLearner

COIN = "BTC"
TIMEFRAME = "4h"
EPISODES = 5  # Увеличили для лучшего обучения

def get_simulated_signals(state):
    """Mock signals for training"""
    close = state.raw_ohlcv['close']
    returns = close.pct_change()
    
    # ML Signal (60% accuracy)
    future_dir = (close.shift(-1) > close).astype(int)
    mask = np.random.rand(len(close)) < 0.60 
    p_ml = np.where(mask, future_dir, 1 - future_dir).astype(float)
    
    # RB Signal (RSI logic)
    p_rb = np.zeros(len(close))
    rsi_proxy = returns.rolling(14).mean()
    p_rb[rsi_proxy < -0.01] = 0.9
    p_rb[rsi_proxy > 0.01] = 0.1
    p_rb[(p_rb == 0)] = 0.5
    
    return p_ml, p_rb

def main():
    print("="*60)
    print("🚀 TRAINING RL AGENT v2 (Fixed Features)")
    print("="*60)
    
    # Load Data
    print("[1/4] Loading History...")
    pipeline = CanonicalPipeline(strict_mode=False)
    try:
        state = pipeline.load_state(COIN, TIMEFRAME)
        print(f"   ✅ Data: {len(state.features)} bars")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return

    # Init Agent
    print("[2/4] Initializing Agent...")
    agent = HybridQLearner(learning_rate=0.1, epsilon=0.5)
    print(f"   Path: {agent.model_path}")

    # Prepare Signals
    print("[3/4] Preparing Signals...")
    p_ml_arr, p_rb_arr = get_simulated_signals(state)

    # Training Loop
    print(f"[4/4] Training {EPISODES} Episodes...")
    
    for episode in range(EPISODES):
        total_reward = 0
        agent.epsilon = max(0.05, 0.5 * (0.8 ** episode))
        
        for t in tqdm(range(100, len(state.features)-1), desc=f"Ep {episode+1}"):
            # === VOLATILITY STATE (FIXED) ===
            # Используем реальные колонки: regime_atr_low/mid/high
            atr_low = state.features['regime_atr_low'].iloc[t]
            atr_mid = state.features['regime_atr_mid'].iloc[t]
            atr_high = state.features['regime_atr_high'].iloc[t]
            
            if atr_high > 0.5:
                vol_state = "HIGH"
            elif atr_low > 0.5:
                vol_state = "LOW"
            else:
                vol_state = "NORM"
            
            # === TREND STATE (FIXED) ===
            # Используем regime_rsi_low/mid/high для определения тренда
            rsi_high = state.features['regime_rsi_high'].iloc[t]
            rsi_low = state.features['regime_rsi_low'].iloc[t]
            
            if rsi_high > 0.5:
                trend_state = "BULL"  # Перекупленность -> восходящий тренд
            elif rsi_low > 0.5:
                trend_state = "BEAR"  # Перепроданность -> нисходящий тренд
            else:
                trend_state = "RANGE"
            
            # === STATE KEY (теперь будет варьироваться!) ===
            state_key = f"{vol_state}|{trend_state}|SAFE|NORM|FLAT|FLAT"
            
            # Agent action
            weights = agent.select_action(state_key, is_training=True)
            w_rb, w_ml = weights
            
            # Calculate result
            p_ml = p_ml_arr[t]
            p_rb = p_rb_arr[t]
            p_final = w_rb * p_rb + w_ml * p_ml
            
            actual_move = state.raw_ohlcv['close'].iloc[t+1] - state.raw_ohlcv['close'].iloc[t]
            position = 1.0 if p_final > 0.5 else -1.0
            
            # Sizing factor
            size = 1.0
            if vol_state == "HIGH": size = 0.7
            if vol_state == "LOW": size = 1.2
            
            reward = position * actual_move * size
            
            # Update
            next_state_key = state_key
            agent.update(state_key, weights, reward, next_state_key, sizing_factor=size)
            
            total_reward += reward
            
        print(f"   Ep {episode+1}: Reward={total_reward:.2f} | ε={agent.epsilon:.2f} | States={len(agent.q_table)}")
        agent.save()

    print("\n✅ Training Complete!")
    print(f"   Brain: {agent.model_path}")
    print(f"   Learned States: {len(agent.q_table)}")

if __name__ == "__main__":
    main()
