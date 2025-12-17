"""
Hybrid Strategy with Q-Learning.

Learns optimal weights α(t), β(t) for combining P_rb and P_ml:
    P_hyb = α(t) * P_rb + β(t) * P_ml
    
Where weights adapt based on recent performance of each strategy.

Uses tabular Q-learning (no DQN — simpler, more interpretable).
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from enum import Enum
import json
from pathlib import Path


class WeightAction(Enum):
    """Possible weight adjustments."""
    FAVOR_RB = "favor_rb"      # α ↑, β ↓
    FAVOR_ML = "favor_ml"      # α ↓, β ↑
    BALANCED = "balanced"      # α = β = 0.5
    STRONG_RB = "strong_rb"    # α = 0.8, β = 0.2
    STRONG_ML = "strong_ml"    # α = 0.2, β = 0.8


# Action to weight mapping
ACTION_WEIGHTS = {
    WeightAction.FAVOR_RB: (0.6, 0.4),
    WeightAction.FAVOR_ML: (0.4, 0.6),
    WeightAction.BALANCED: (0.5, 0.5),
    WeightAction.STRONG_RB: (0.8, 0.2),
    WeightAction.STRONG_ML: (0.2, 0.8),
}


@dataclass
class MarketState:
    """Discretized market state for Q-table."""
    trend_bin: int        # -2 to +2 (strong down to strong up)
    volatility_bin: int   # 0 to 2 (low, medium, high)
    rb_recent_win: int    # 0 or 1 (recent P_rb performance)
    ml_recent_win: int    # 0 or 1 (recent P_ml performance)
    
    def to_tuple(self) -> tuple:
        return (self.trend_bin, self.volatility_bin, 
                self.rb_recent_win, self.ml_recent_win)


class HybridQLearner:
    """
    Q-Learning for hybrid strategy weights.
    
    State space: (trend, volatility, rb_performance, ml_performance)
    Action space: 5 weight configurations
    Reward: Trade PnL
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
    ):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Q-table: state_tuple -> {action: Q-value}
        self.q_table: Dict[tuple, Dict[WeightAction, float]] = {}
        
        # Performance tracking
        self._rb_returns: list = []
        self._ml_returns: list = []
        self._lookback = 10
        
        # Current weights
        self.alpha = 0.5  # P_rb weight
        self.beta = 0.5   # P_ml weight
        
        # Statistics
        self.total_updates = 0
        self.exploration_count = 0
    
    def _get_q_values(self, state: tuple) -> Dict[WeightAction, float]:
        """Get Q-values for state, initialize if new."""
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in WeightAction}
        return self.q_table[state]
    
    def _discretize_trend(self, returns: np.ndarray, periods: int = 20) -> int:
        """Discretize trend into bins."""
        if len(returns) < periods:
            return 0
        
        trend = np.mean(returns[-periods:])
        
        if trend < -0.02:
            return -2  # Strong down
        elif trend < -0.005:
            return -1  # Down
        elif trend < 0.005:
            return 0   # Flat
        elif trend < 0.02:
            return 1   # Up
        else:
            return 2   # Strong up
    
    def _discretize_volatility(self, returns: np.ndarray, periods: int = 20) -> int:
        """Discretize volatility into bins."""
        if len(returns) < periods:
            return 1
        
        vol = np.std(returns[-periods:])
        
        if vol < 0.01:
            return 0  # Low
        elif vol < 0.03:
            return 1  # Medium
        else:
            return 2  # High
    
    def _recent_performance(self, returns: list) -> int:
        """Check if recent performance is positive."""
        if len(returns) < 3:
            return 0
        return 1 if np.mean(returns[-3:]) > 0 else 0
    
    def get_state(self, market_returns: np.ndarray) -> MarketState:
        """Convert market data to discrete state."""
        trend = self._discretize_trend(market_returns)
        vol = self._discretize_volatility(market_returns)
        rb_win = self._recent_performance(self._rb_returns)
        ml_win = self._recent_performance(self._ml_returns)
        
        return MarketState(trend, vol, rb_win, ml_win)
    
    def select_action(self, state: MarketState, training: bool = True) -> WeightAction:
        """Select action using epsilon-greedy policy."""
        state_tuple = state.to_tuple()
        
        # Exploration
        if training and np.random.random() < self.epsilon:
            self.exploration_count += 1
            return np.random.choice(list(WeightAction))
        
        # Exploitation
        q_values = self._get_q_values(state_tuple)
        best_action = max(q_values, key=q_values.get)
        return best_action
    
    def update(
        self,
        state: MarketState,
        action: WeightAction,
        reward: float,
        next_state: MarketState,
    ):
        """Q-learning update."""
        state_tuple = state.to_tuple()
        next_tuple = next_state.to_tuple()
        
        # Current Q-value
        q_current = self._get_q_values(state_tuple)[action]
        
        # Best next Q-value
        next_q_values = self._get_q_values(next_tuple)
        q_next_max = max(next_q_values.values())
        
        # TD update
        td_target = reward + self.gamma * q_next_max
        td_error = td_target - q_current
        
        self.q_table[state_tuple][action] = q_current + self.lr * td_error
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        self.total_updates += 1
    
    def get_weights(self, action: WeightAction) -> Tuple[float, float]:
        """Get alpha, beta for action."""
        self.alpha, self.beta = ACTION_WEIGHTS[action]
        return self.alpha, self.beta
    
    def compute_hybrid(self, p_rb: float, p_ml: float) -> float:
        """Compute P_hyb with current weights."""
        return self.alpha * p_rb + self.beta * p_ml
    
    def record_performance(self, p_rb: float, p_ml: float, actual_return: float):
        """Record strategy performance for state computation."""
        # Simplified: assume strategy is "long if p > 0.5"
        rb_return = actual_return if p_rb > 0.5 else -actual_return
        ml_return = actual_return if p_ml > 0.5 else -actual_return
        
        self._rb_returns.append(rb_return)
        self._ml_returns.append(ml_return)
        
        # Keep only lookback
        if len(self._rb_returns) > self._lookback:
            self._rb_returns = self._rb_returns[-self._lookback:]
            self._ml_returns = self._ml_returns[-self._lookback:]
    
    def get_statistics(self) -> dict:
        """Get learning statistics."""
        return {
            'total_updates': self.total_updates,
            'exploration_count': self.exploration_count,
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
            'current_alpha': self.alpha,
            'current_beta': self.beta,
        }
    
    def save(self, path: str):
        """Save Q-table to file."""
        data = {
            'q_table': {
                str(k): {a.value: v for a, v in actions.items()}
                for k, actions in self.q_table.items()
            },
            'epsilon': self.epsilon,
            'total_updates': self.total_updates,
            'lr': self.lr,
            'gamma': self.gamma,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load Q-table from file."""
        with open(path) as f:
            data = json.load(f)
        
        self.q_table = {}
        for k, actions in data['q_table'].items():
            state_tuple = eval(k)  # Convert string back to tuple
            self.q_table[state_tuple] = {
                WeightAction(a): v for a, v in actions.items()
            }
        
        self.epsilon = data.get('epsilon', 0.1)
        self.total_updates = data.get('total_updates', 0)


def train_hybrid_learner(
    predictions_path: str,
    features_path: str,
    output_path: str,
    episodes: int = 3,
) -> HybridQLearner:
    """
    Train Q-learner on historical data.
    
    Args:
        predictions_path: Path to predictions parquet
        features_path: Path to features parquet
        output_path: Path to save trained Q-table
        episodes: Number of passes through data
        
    Returns:
        Trained HybridQLearner
    """
    import pandas as pd
    
    print("Loading data...")
    pred = pd.read_parquet(predictions_path)
    
    # Simulate P_rb (in real system, would come from RuleBasedStrategy)
    np.random.seed(42)
    p_ml = pred['y_pred'].values
    p_rb = np.clip(p_ml + np.random.randn(len(p_ml)) * 0.15, 0, 1)
    returns = pred['returns'].values
    
    learner = HybridQLearner()
    
    print(f"Training for {episodes} episodes...")
    
    for episode in range(episodes):
        learner._rb_returns = []
        learner._ml_returns = []
        
        total_reward = 0
        
        for i in range(50, len(pred)):  # Skip initial warmup
            # Get state
            state = learner.get_state(returns[:i])
            
            # Select action
            action = learner.select_action(state, training=True)
            alpha, beta = learner.get_weights(action)
            
            # Compute hybrid and reward
            p_hyb = learner.compute_hybrid(p_rb[i], p_ml[i])
            
            # Reward = return if correct direction
            if p_hyb > 0.5:
                reward = returns[i]
            else:
                reward = -returns[i]
            
            total_reward += reward
            
            # Record for next state
            learner.record_performance(p_rb[i], p_ml[i], returns[i])
            
            # Get next state and update
            if i + 1 < len(pred):
                next_state = learner.get_state(returns[:i+1])
                learner.update(state, action, reward, next_state)
        
        print(f"  Episode {episode+1}: Total reward = {total_reward:.4f}, "
              f"Epsilon = {learner.epsilon:.4f}, "
              f"Q-table size = {len(learner.q_table)}")
    
    # Save
    learner.save(output_path)
    print(f"\nSaved to: {output_path}")
    print(f"Final statistics: {learner.get_statistics()}")
    
    return learner



