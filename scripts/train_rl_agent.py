"""
SCRIPT: TRAIN RL AGENT (Historical Simulation)
==============================================
Прогоняет HybridQLearner по истории, чтобы заполнить Q-Table.
Использует CanonicalPipeline для получения данных.
"""
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Setup Project Root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.canonical_pipeline import CanonicalPipeline
from strategies.hybrid_q_learner import HybridQLearner

# CONFIG
COIN = "BTC"
TIMEFRAME = "4h"
EPISODES = 3  # Сколько раз пройти по истории

def get_simulated_signals(state):
    """
    Для обучения RL нам нужны сигналы от P_rb и P_ml.
    Чтобы не запускать тяжелые стратегии на каждом шаге, 
    мы сгенерируем их приближенно на основе цены (Mocking).
    """
    close = state.raw_ohlcv['close']
    returns = close.pct_change()
    
    # 1. Mock ML Signal (Slightly predictive)
    # Добавляем шум к реальному движению
    future_dir = (close.shift(-1) > close).astype(int)
    # С вероятностью 60% ML угадывает
    mask = np.random.rand(len(close)) < 0.60 
    p_ml = np.where(mask, future_dir, 1 - future_dir).astype(float)
    
    # 2. Mock RB Signal (RSI logic)
    p_rb = np.zeros(len(close))
    rsi_proxy = returns.rolling(14).mean()
    p_rb[rsi_proxy < -0.01] = 0.9  # Oversold
    p_rb[rsi_proxy > 0.01] = 0.1   # Overbought
    p_rb[(p_rb == 0)] = 0.5        # Neutral
    
    return p_ml, p_rb

def main():
    print("="*60)
    print("🚀 TRAINING RL AGENT (Historical Bootstrapping)")
    print("="*60)
    
    # 1. Load Data
    print("[1/4] Loading History via Pipeline...")
    pipeline = CanonicalPipeline(strict_mode=False)
    try:
        state = pipeline.load_state(COIN, TIMEFRAME)
        print(f"   ✅ Data loaded: {len(state.features)} bars")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return

    # 2. Init Agent
    print("[2/4] Initializing HybridQLearner v3...")
    agent = HybridQLearner(learning_rate=0.1, epsilon=0.5)
    print(f"   Agent memory path: {agent.model_path}")

    # 3. Prepare Signals
    print("[3/4] Preparing Simulation Signals...")
    p_ml_arr, p_rb_arr = get_simulated_signals(state)
    
    # Volatility
    if 'norm_atr_pct' in state.features:
        volatility = state.features['norm_atr_pct']
    else:
        volatility = pd.Series(0, index=state.features.index)
    
    # 4. Training Loop
    print(f"[4/4] Running {EPISODES} Episodes...")
    
    for episode in range(EPISODES):
        total_reward = 0
        updates = 0
        
        # Уменьшаем epsilon
        agent.epsilon = max(0.05, 0.5 * (0.8 ** episode))
        
        # Проходим по истории
        for t in tqdm(range(100, len(state.features)-1), desc=f"Episode {episode+1}"):
            # 1. Define State
            vol_val = volatility.iloc[t]
            vol_state = "NORM"
            if vol_val > 2.0: vol_state = "CRISIS"
            elif vol_val > 1.0: vol_state = "HIGH"
            elif vol_val < -0.5: vol_state = "LOW"
            
            # Trend
            trend_val = state.features['regime_trend'].iloc[t] if 'regime_trend' in state.features else 0
            trend_state = "BULL" if trend_val > 0.5 else ("BEAR" if trend_val < -0.5 else "RANGE")
            
            state_key = f"{vol_state}|{trend_state}|SAFE|NORM|FLAT|FLAT"
            
            # 2. Agent chooses action
            weights = agent.select_action(state_key, is_training=True)
            w_rb, w_ml = weights
            
            # 3. Calculate Result
            p_ml = p_ml_arr[t]
            p_rb = p_rb_arr[t]
            
            # Combined Probability
            p_final = w_rb * p_rb + w_ml * p_ml
            
            # Actual move
            actual_move = state.raw_ohlcv['close'].iloc[t+1] - state.raw_ohlcv['close'].iloc[t]
            
            # Position
            position = 1.0 if p_final > 0.5 else -1.0
            
            # Sizing Factor
            size = 1.0
            if vol_state == "CRISIS": size = 0.3
            if vol_state == "HIGH": size = 0.7
            if vol_state == "LOW": size = 1.2
            
            # Reward
            reward = position * actual_move * size
            
            # 4. Update Agent
            next_state_key = state_key 
            
            agent.update(state_key, weights, reward, next_state_key, sizing_factor=size)
            
            total_reward += reward
            updates += 1
            
        print(f"   Episode {episode+1} Result: Reward={total_reward:.2f} | Epsilon={agent.epsilon:.2f} | States={len(agent.q_table)}")
        
        # Save after each episode
        agent.save()

    print("\n✅ Training Complete.")
    print(f"   Brain saved to: {agent.model_path}")
    print("   Now the agent is not blind!")

if __name__ == "__main__":
    main()
