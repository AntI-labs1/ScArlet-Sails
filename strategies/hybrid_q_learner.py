"""
Hybrid Q-Learner v3 (Enterprise Edition)
========================================
Advanced Meta-Strategy using Contextual Q-Learning.

Replaces legacy 'Linear Function Approximation' with High-Dimensional Tabular Q-Learning.
Why Tabular? 
1. Deterministic: Easy to audit why the agent chose 'Strong ML'.
2. Stable: No gradient explosion risks.
3. Persistable: Saves as simple JSON.

State Space (6 Dimensions):
1. Volatility (Low/Norm/High/Crisis)
2. Trend (Bear/Range/Bull)
3. OOD Status (Safe/Danger)
4. Dispersion (Agreed/Chaos)
5. RB Performance (Win/Loss)
6. ML Performance (Win/Loss)

Action Space:
- Weights combinations for P_rb and P_ml
"""

import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

from core.canonical_state import CanonicalState

logger = logging.getLogger(__name__)

class HybridQLearner:
    """
    Risk-Aware Q-Learning Agent.
    Learns to allocate capital between Rule-Based (Logic) and ML (Stats) strategies.
    """
    
    def __init__(self, model_dir: str = "models", learning_rate=0.05, discount_factor=0.9, epsilon=0.15):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # Path management
        self.model_path = Path(model_dir) / "q_table_v3_advanced.json"
        
        # Action Space: (Weight_RB, Weight_ML)
        self.actions = [
            (1.0, 0.0), # 0: Trust Logic Only (High OOD?)
            (0.0, 1.0), # 1: Trust ML Only (Clean patterns?)
            (0.5, 0.5), # 2: Balanced
            (0.8, 0.2), # 3: Conservative
            (0.2, 0.8)  # 4: Aggressive
        ]
        
        # Memory
        self.q_table = {}
        self._rb_history = []
        self._ml_history = []
        
        self.load()

    def _get_volatility_state(self, state: CanonicalState) -> str:
        """Analyze Volatility Regime from features."""
        # Using feature_loader's normalized ATR if available
        if 'norm_atr_pct' in state.features.columns:
            val = state.features['norm_atr_pct'].iloc[-1]
            if val > 2.0: return "CRISIS"
            if val > 1.0: return "HIGH"
            if val < -0.5: return "LOW"
        return "NORM"

    def _get_trend_state(self, state: CanonicalState) -> str:
        """Analyze Trend Direction."""
        if 'regime_trend' in state.features.columns:
            val = state.features['regime_trend'].iloc[-1]
            if val > 0.5: return "BULL"
            if val < -0.5: return "BEAR"
        return "RANGE"

    def _get_ood_state(self, state: CanonicalState) -> str:
        """Check if we are in uncharted territory."""
        # This usually comes from OODDetector, passed via metadata or external check
        # For now, we infer from z-scores in features
        z_cols = [c for c in state.features.columns if 'zscore' in c]
        if z_cols:
            # If any feature is > 3 sigma, it's OOD-ish
            max_z = state.features[z_cols].iloc[-1].abs().max()
            if max_z > 4.0: return "DANGER"
            if max_z > 2.5: return "WARN"
        return "SAFE"

    def get_state_key(self, state: CanonicalState, dispersion_level: str = "NORM", 
                      rb_perf: str = "FLAT", ml_perf: str = "FLAT") -> str:
        """
        Constructs the 6-Dimensional State Key.
        Format: VOL|TREND|OOD|DISP|RB|ML
        """
        vol = self._get_volatility_state(state)
        trend = self._get_trend_state(state)
        ood = self._get_ood_state(state)
        
        # Combine all dimensions
        return f"{vol}|{trend}|{ood}|{dispersion_level}|{rb_perf}|{ml_perf}"

    def select_action(self, state_key: str, is_training: bool = False) -> Tuple[float, float]:
        """Epsilon-Greedy Selection"""
        # Init Q-values if new state
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.actions))
        
        # Explore
        if is_training and np.random.random() < self.epsilon:
            idx = np.random.randint(len(self.actions))
        # Exploit
        else:
            idx = np.argmax(self.q_table[state_key])
            
        return self.actions[idx]

    def update(self, state_key: str, action_weights: Tuple[float, float], 
               raw_reward: float, next_state_key: str, 
               sizing_factor: float = 1.0):
        """
        Risk-Aware Q-Learning Update.
        The 'sizing_factor' is crucial: it penalizes strategies that perform well 
        only when the Risk Manager (Sizer) would forbid trading.
        """
        # Map weights back to index
        try:
            action_idx = self.actions.index(action_weights)
        except ValueError:
            return # Should not happen
            
        # Ensure state exists
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(len(self.actions))
            
        # SCALED REWARD LOGIC
        # Если Risk Manager говорит "снизить позу" (sizing_factor < 1.0),
        # мы уменьшаем награду. Агент учится избегать ситуаций, 
        # где прибыль есть, но риск слишком велик.
        real_reward = raw_reward * sizing_factor
        
        # Bellman Equation
        current_q = self.q_table[state_key][action_idx]
        max_next_q = np.max(self.q_table[next_state_key])
        
        new_q = current_q + self.lr * (real_reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action_idx] = new_q

    def save(self):
        """Save memory to JSON"""
        serializable = {k: v.tolist() for k, v in self.q_table.items()}
        try:
            with open(self.model_path, 'w') as f:
                json.dump(serializable, f, indent=2)
            logger.info(f"💾 Q-Table saved: {len(self.q_table)} states")
        except Exception as e:
            logger.error(f"Save failed: {e}")

    def load(self):
        """Load memory"""
        if not self.model_path.exists():
            logger.info("New Q-Learner initialized.")
            return
        try:
            with open(self.model_path, 'r') as f:
                data = json.load(f)
            self.q_table = {k: np.array(v) for k, v in data.items()}
            logger.info(f"✅ Q-Table loaded: {len(self.q_table)} states")
        except Exception as e:
            logger.error(f"Load failed: {e}")
