"""
Tests for Hybrid Q-Learner.
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.hybrid_q_learner import (
    HybridQLearner,
    WeightAction,
    MarketState,
    ACTION_WEIGHTS,
)


class TestHybridQLearner:
    """Tests for Q-learning hybrid strategy."""
    
    @pytest.fixture
    def learner(self):
        return HybridQLearner()
    
    def test_initialization(self, learner):
        """Learner initializes correctly."""
        assert learner.alpha == 0.5
        assert learner.beta == 0.5
        assert learner.epsilon == 0.1
        assert len(learner.q_table) == 0
    
    def test_get_weights(self, learner):
        """Get weights for each action."""
        for action in WeightAction:
            alpha, beta = learner.get_weights(action)
            assert 0 <= alpha <= 1
            assert 0 <= beta <= 1
            assert abs(alpha + beta - 1.0) < 0.01  # Sum to 1
    
    def test_compute_hybrid(self, learner):
        """Compute hybrid probability."""
        learner.alpha = 0.6
        learner.beta = 0.4
        
        p_hyb = learner.compute_hybrid(0.8, 0.6)
        expected = 0.6 * 0.8 + 0.4 * 0.6
        
        assert abs(p_hyb - expected) < 0.001
    
    def test_state_discretization(self, learner):
        """State discretization works."""
        returns = np.random.randn(100) * 0.01
        state = learner.get_state(returns)
        
        assert isinstance(state, MarketState)
        assert -2 <= state.trend_bin <= 2
        assert 0 <= state.volatility_bin <= 2
        assert state.rb_recent_win in [0, 1]
        assert state.ml_recent_win in [0, 1]
    
    def test_action_selection_exploration(self, learner):
        """Exploration selects random actions."""
        learner.epsilon = 1.0  # Always explore
        
        state = MarketState(0, 1, 0, 0)
        actions = [learner.select_action(state, training=True) 
                   for _ in range(100)]
        
        # Should see variety
        unique_actions = set(actions)
        assert len(unique_actions) > 1
    
    def test_action_selection_exploitation(self, learner):
        """Exploitation selects best action."""
        learner.epsilon = 0.0  # Never explore
        
        state = MarketState(0, 1, 0, 0)
        state_tuple = state.to_tuple()
        
        # Set one action as clearly best
        learner.q_table[state_tuple] = {
            WeightAction.BALANCED: 0.0,
            WeightAction.FAVOR_ML: 10.0,  # Best
            WeightAction.FAVOR_RB: 0.0,
            WeightAction.STRONG_ML: 0.0,
            WeightAction.STRONG_RB: 0.0,
        }
        
        action = learner.select_action(state, training=True)
        assert action == WeightAction.FAVOR_ML
    
    def test_q_update(self, learner):
        """Q-value updates correctly."""
        state = MarketState(0, 1, 0, 0)
        next_state = MarketState(0, 1, 1, 0)
        action = WeightAction.BALANCED
        reward = 0.05
        
        # Initial Q-value is 0
        state_tuple = state.to_tuple()
        q_before = learner._get_q_values(state_tuple)[action]
        assert q_before == 0.0
        
        learner.update(state, action, reward, next_state)
        
        q_after = learner._get_q_values(state_tuple)[action]
        assert q_after > 0  # Should increase with positive reward
    
    def test_epsilon_decay(self, learner):
        """Epsilon decays over updates."""
        initial_epsilon = learner.epsilon
        
        state = MarketState(0, 1, 0, 0)
        next_state = MarketState(0, 1, 1, 0)
        
        for _ in range(10):
            learner.update(state, WeightAction.BALANCED, 0.01, next_state)
        
        assert learner.epsilon < initial_epsilon
        assert learner.epsilon >= learner.min_epsilon
    
    def test_save_and_load(self, learner):
        """Q-table can be saved and loaded."""
        # Train a bit
        state = MarketState(0, 1, 0, 0)
        next_state = MarketState(0, 1, 1, 0)
        learner.update(state, WeightAction.FAVOR_ML, 0.05, next_state)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        
        learner.save(path)
        
        loaded = HybridQLearner()
        loaded.load(path)
        
        assert len(loaded.q_table) == len(learner.q_table)
        
        Path(path).unlink()
    
    def test_record_performance(self, learner):
        """Performance recording works."""
        for i in range(5):
            learner.record_performance(0.6, 0.7, 0.01)
        
        assert len(learner._rb_returns) == 5
        assert len(learner._ml_returns) == 5
    
    def test_statistics(self, learner):
        """Statistics are tracked."""
        state = MarketState(0, 1, 0, 0)
        next_state = MarketState(0, 1, 1, 0)
        learner.update(state, WeightAction.BALANCED, 0.01, next_state)
        
        stats = learner.get_statistics()
        
        assert 'total_updates' in stats
        assert stats['total_updates'] == 1
        assert 'q_table_size' in stats


class TestMarketState:
    """Tests for MarketState."""
    
    def test_to_tuple(self):
        """State converts to tuple."""
        state = MarketState(1, 2, 0, 1)
        t = state.to_tuple()
        
        assert t == (1, 2, 0, 1)
        assert isinstance(t, tuple)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


