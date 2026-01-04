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
    """Continuous market state for linear function approximation."""
    trend: float          # Continuous trend value
    vol: float            # Continuous volatility value
    rb_perf: float       # Recent P_rb performance (0-1)
    ml_perf: float       # Recent P_ml performance (0-1)
    dispersion: float    # Dispersion between strategies (0-1)
    ood_score: float     # Out-of-distribution score (0-1)
    
    def to_vector(self) -> np.ndarray:
        """Convert to state vector for linear approximation."""
        return np.array([
            self.trend,
            self.vol,
            self.rb_perf,
            self.ml_perf,
            self.dispersion,
            self.ood_score,
        ])


class HybridQLearner:
    """
    Q-Learning with Linear Function Approximation for hybrid strategy weights.
    
    Uses weight matrix instead of Q-table:
    - State: [trend, vol, rb_perf, ml_perf, dispersion, ood_score] (6 features)
    - Actions: 5 weight configurations
    - Q(s,a) = weights[a] · state_vector
    
    Updates: Gradient descent on TD error
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,  # Lower LR for gradient descent
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        n_state_features: int = 6,
        n_actions: int = 5,
    ):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Linear function approximation: weights matrix
        # Shape: (n_actions, n_state_features)
        # Each row is weights for one action
        np.random.seed(42)  # For reproducibility
        self.weights = np.random.normal(0, 0.1, size=(n_actions, n_state_features))
        
        # Performance tracking
        self._rb_returns: list = []
        self._ml_returns: list = []
        self._lookback = 10
        
        # Current weights for hybrid calculation
        self.alpha = 0.5  # P_rb weight
        self.beta = 0.5   # P_ml weight
        
        # Statistics
        self.total_updates = 0
        self.exploration_count = 0
    
    def _get_q_values(self, state_vector: np.ndarray) -> Dict[WeightAction, float]:
        """
        Get Q-values for state using linear function approximation.
        
        Args:
            state_vector: State vector [trend, vol, rb_perf, ml_perf, dispersion, ood_score]
        
        Returns:
            Dict mapping actions to Q-values
        """
        # Q(s,a) = weights[a] · state_vector
        q_values = {}
        actions = list(WeightAction)
        
        for i, action in enumerate(actions):
            q_values[action] = float(np.dot(self.weights[i], state_vector))
        
        return q_values
    
    def _normalize_trend(self, returns: np.ndarray, periods: int = 20) -> float:
        """Calculate normalized trend value (continuous)."""
        if len(returns) < periods:
            return 0.0
        
        trend = np.mean(returns[-periods:])
        # Normalize to [-1, 1] range (clip extreme values)
        return float(np.clip(trend * 10, -1.0, 1.0))
    
    def _normalize_volatility(self, returns: np.ndarray, periods: int = 20) -> float:
        """Calculate normalized volatility value (continuous)."""
        if len(returns) < periods:
            return 0.5  # Medium volatility
        
        vol = np.std(returns[-periods:])
        # Normalize to [0, 1] range
        return float(np.clip(vol * 10, 0.0, 1.0))
    
    def _recent_performance(self, returns: list) -> float:
        """Calculate recent performance as continuous value [0, 1]."""
        if len(returns) < 3:
            return 0.5  # Neutral
        
        mean_return = np.mean(returns[-3:])
        # Normalize: positive returns → 1.0, negative → 0.0
        return float(np.clip((mean_return + 0.05) * 10, 0.0, 1.0))
    
    def get_state(
        self,
        market_returns: np.ndarray,
        dispersion: float = 0.5,
        ood_score: float = 0.5,
    ) -> MarketState:
        """
        Convert market data to continuous state vector.
        
        Args:
            market_returns: Market return series
            dispersion: Dispersion between strategies (0-1)
            ood_score: Out-of-distribution score (0-1)
        
        Returns:
            MarketState with continuous features
        """
        trend = self._normalize_trend(market_returns)
        vol = self._normalize_volatility(market_returns)
        rb_perf = self._recent_performance(self._rb_returns)
        ml_perf = self._recent_performance(self._ml_returns)
        
        return MarketState(
            trend=trend,
            vol=vol,
            rb_perf=rb_perf,
            ml_perf=ml_perf,
            dispersion=float(np.clip(dispersion, 0.0, 1.0)),
            ood_score=float(np.clip(ood_score, 0.0, 1.0)),
        )
    
    def select_action(self, state: MarketState, training: bool = True) -> WeightAction:
        """Select action using epsilon-greedy policy with linear approximation."""
        state_vector = state.to_vector()
        
        # Exploration
        if training and np.random.random() < self.epsilon:
            self.exploration_count += 1
            return np.random.choice(list(WeightAction))
        
        # Exploitation: Q(s,a) = weights[a] · state_vector
        q_values = self._get_q_values(state_vector)
        best_action = max(q_values, key=q_values.get)
        return best_action
    
    def update(
        self,
        state: MarketState,
        action: WeightAction,
        reward: float,
        next_state: MarketState,
    ):
        """
        Q-learning update with gradient descent (Linear Function Approximation).
        
        TD error: δ = r + γ * max_a' Q(s', a') - Q(s, a)
        Gradient: ∇_w Q(s, a) = state_vector
        Update: w[a] ← w[a] + α * δ * state_vector
        """
        state_vector = state.to_vector()
        next_vector = next_state.to_vector()
        
        # Get action index
        actions = list(WeightAction)
        action_idx = actions.index(action)
        
        # Current Q-value: Q(s, a) = weights[a] · state_vector
        q_current = float(np.dot(self.weights[action_idx], state_vector))
        
        # Best next Q-value: max_a' Q(s', a')
        next_q_values = self._get_q_values(next_vector)
        q_next_max = max(next_q_values.values())
        
        # TD target and error
        td_target = reward + self.gamma * q_next_max
        td_error = td_target - q_current
        
        # Gradient descent update: w[a] ← w[a] + α * δ * state_vector
        self.weights[action_idx] += self.lr * td_error * state_vector
        
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
            'weights_shape': self.weights.shape,
            'weights_mean': float(np.mean(self.weights)),
            'weights_std': float(np.std(self.weights)),
            'current_alpha': self.alpha,
            'current_beta': self.beta,
        }
    
    def save(self, path: str):
        """Save weights matrix to file."""
        data = {
            'weights': self.weights.tolist(),
            'epsilon': self.epsilon,
            'total_updates': self.total_updates,
            'lr': self.lr,
            'gamma': self.gamma,
            'weights_shape': list(self.weights.shape),
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load weights matrix from file."""
        with open(path) as f:
            data = json.load(f)
        
        weights_list = data['weights']
        self.weights = np.array(weights_list)
        
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
            # Calculate dispersion (simplified: absolute difference)
            dispersion = abs(p_rb[i] - p_ml[i])
            ood_score = 0.5  # Placeholder - would come from OOD detector
            
            # Get state with dispersion and OOD score
            state = learner.get_state(
                returns[:i],
                dispersion=dispersion,
                ood_score=ood_score,
            )
            
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
                next_dispersion = abs(p_rb[i+1] - p_ml[i+1]) if i+1 < len(pred) else 0.5
                next_state = learner.get_state(
                    returns[:i+1],
                    dispersion=next_dispersion,
                    ood_score=ood_score,
                )
                learner.update(state, action, reward, next_state)
        
        print(f"  Episode {episode+1}: Total reward = {total_reward:.4f}, "
              f"Epsilon = {learner.epsilon:.4f}, "
              f"Weights shape = {learner.weights.shape}")
    
    # Save
    learner.save(output_path)
    print(f"\nSaved to: {output_path}")
    print(f"Final statistics: {learner.get_statistics()}")
    
    return learner



