"""
RL TRADING ENVIRONMENT
Gym-compatible environment for training DQN on trading decisions

State: Market features + Portfolio state
Actions: {NO_ACTION, ENTER_LONG, EXIT_LONG}
Reward: PnL - costs - risk_penalty

Author: STAR_ANT + Claude Sonnet 4.5
Date: November 17, 2025
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def normalize_price_features(close_window: pd.Series) -> Dict[str, float]:
    """Convert recent prices to stable, leak-free features."""
    if close_window is None or len(close_window) < 2:
        return {'latest_ret': 0.0, 'mean_ret': 0.0, 'vol': 0.0, 'trend': 0.0, 'rel_price': 0.0}

    log_returns = np.log(close_window).diff().dropna()
    if log_returns.empty:
        return {'latest_ret': 0.0, 'mean_ret': 0.0, 'vol': 0.0, 'trend': 0.0, 'rel_price': 0.0}

    latest_ret = float(log_returns.iloc[-1])
    mean_ret = float(log_returns.mean())
    vol = float(log_returns.std())
    trend = float((close_window.iloc[-1] - close_window.iloc[0]) / close_window.iloc[0])
    rel_price = float(close_window.iloc[-1] / close_window.mean() - 1)
    return {
        'latest_ret': latest_ret,
        'mean_ret': mean_ret,
        'vol': vol,
        'trend': trend,
        'rel_price': rel_price,
    }


def normalize_volume_features(volume_window: pd.Series) -> Dict[str, float]:
    """Return z-scored volume features to stabilize RL inputs."""
    if volume_window is None or len(volume_window) < 2:
        return {'z_vol': 0.0, 'vol_mean': 0.0, 'vol_std': 1.0}
    mean = float(volume_window.mean())
    std = float(volume_window.std())
    std = std if std > 0 else 1.0
    z_vol = float((volume_window.iloc[-1] - mean) / std)
    return {'z_vol': z_vol, 'vol_mean': mean, 'vol_std': std}


class TradingEnvironment:
    """
    Trading Environment for Reinforcement Learning
    
    Implements gym-like interface for training DQN
    """
    
    # Action space
    NO_ACTION = 0
    ENTER_LONG = 1
    EXIT_LONG = 2
    
    def __init__(self, df: pd.DataFrame, config: Dict = None):
        """
        Initialize trading environment
        
        Parameters:
        -----------
        df : DataFrame
            Historical OHLCV data
        config : dict
            Configuration parameters
        """
        self.df = df.reset_index(drop=True)
        self.config = config or {}
        
        # Environment parameters
        self.max_position_size = self.config.get('max_position_size', 1.0)
        self.transaction_cost = self.config.get('transaction_cost', 0.001)
        self.slippage = self.config.get('slippage', 0.0005)
        self.total_cost = self.transaction_cost + self.slippage
        
        # Risk parameters
        self.max_drawdown_limit = self.config.get('max_drawdown_limit', 0.20)
        self.volatility_penalty = self.config.get('volatility_penalty', 0.1)
        
        # State space dimensions
        self.state_dim = 12  # price features (5) + portfolio (4) + regime (3)
        self.action_dim = 3  # NO_ACTION, ENTER_LONG, EXIT_LONG
        
        # Episode tracking
        self.current_step = 0
        self.max_steps = len(self.df) - 1
        
        # Portfolio state
        self.position = 0.0  # Current position size
        self.entry_price = 0.0
        self.equity = 100000.0  # Starting capital
        self.peak_equity = self.equity
        self.total_pnl = 0.0
        
        # Episode history
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_states = []
        
        logger.info(f"TradingEnvironment initialized: {len(self.df)} bars")
    
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state
        
        Returns:
        --------
        np.ndarray : Initial state
        """
        self.current_step = 0
        self.position = 0.0
        self.entry_price = 0.0
        self.equity = 100000.0
        self.peak_equity = self.equity
        self.total_pnl = 0.0
        
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_states = []
        
        state = self._get_state()
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return next state
        
        Parameters:
        -----------
        action : int
            Action to take {NO_ACTION, ENTER_LONG, EXIT_LONG}
        
        Returns:
        --------
        tuple : (next_state, reward, done, info)
        """
        # Get current price
        current_price = self.df.loc[self.current_step, 'close']
        
        # Execute action
        reward = self._execute_action(action, current_price)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Get next state
        next_state = self._get_state() if not done else np.zeros(self.state_dim, dtype=np.float32)
        
        # Track episode
        self.episode_rewards.append(reward)
        self.episode_actions.append(action)
        self.episode_states.append(next_state)
        
        # Info dict
        info = {
            'position': self.position,
            'equity': self.equity,
            'pnl': self.total_pnl,
            'drawdown': (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        }
        
        return next_state, reward, done, info
    
    def _execute_action(self, action: int, current_price: float) -> float:
        """
        Execute trading action and compute reward
        
        ИСПРАВЛЕНО: Более реалистичные rewards
        
        Parameters:
        -----------
        action : int
            Action to execute
        current_price : float
            Current market price
        
        Returns:
        --------
        float : Reward for this action
        """
        reward = 0.0
        
        if action == self.ENTER_LONG:
            # Enter long position
            if self.position == 0:
                self.position = self.max_position_size
                self.entry_price = current_price * (1 + self.total_cost)
                reward = -self.total_cost * 10  # Penalty for costs (scaled up)
            else:
                reward = -0.1  # Penalty for invalid action
        
        elif action == self.EXIT_LONG:
            # Exit long position
            if self.position > 0:
                exit_price = current_price * (1 - self.total_cost)
                pnl = (exit_price - self.entry_price) / self.entry_price
                
                # Update equity
                self.equity *= (1 + pnl * self.position)
                self.total_pnl += pnl * self.position
                
                # Update peak
                if self.equity > self.peak_equity:
                    self.peak_equity = self.equity
                
                # Reward = PnL scaled up for learning
                reward = pnl * self.position * 100  # Scale for better gradients
                
                # Subtract risk penalty
                reward -= self._compute_risk_penalty() * 10
                
                # Reset position
                self.position = 0.0
                self.entry_price = 0.0
            else:
                reward = -0.1  # Penalty for invalid action
        
        elif action == self.NO_ACTION:
            # No action
            if self.position > 0:
                # Holding position: compute unrealized P&L
                unrealized_pnl = (current_price - self.entry_price) / self.entry_price
                # Small reward/penalty for holding
                reward = unrealized_pnl * 5  # Scaled down (less than exit)
                
                # Time penalty for long holds (encourage action)
                reward -= 0.001
            else:
                # Not in position: small penalty for inaction
                reward = -0.001
        
        # Clip reward to prevent explosions
        reward = np.clip(reward, -10.0, 10.0)
        
        return float(reward)
    
    def _compute_risk_penalty(self) -> float:
        """
        Compute risk penalty based on recent volatility and drawdown
        
        Returns:
        --------
        float : Risk penalty
        """
        # Volatility penalty
        if self.current_step < 20:
            vol_penalty = 0.0
        else:
            recent_returns = self.df['close'].pct_change().iloc[max(0, self.current_step-20):self.current_step]
            volatility = recent_returns.std()
            vol_penalty = self.volatility_penalty * volatility
        
        # Drawdown penalty
        current_dd = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        dd_penalty = 0.0
        if current_dd > self.max_drawdown_limit:
            dd_penalty = (current_dd - self.max_drawdown_limit) * 10  # Heavy penalty
        
        return vol_penalty + dd_penalty
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state representation
        
        ИСПРАВЛЕНО: Правильное объединение массивов
        
        State components:
        1. Price features (5): returns, volatility, trend, etc.
        2. Portfolio state (4): position, unrealized_pnl, equity, drawdown
        3. Regime indicators (3): low_vol, normal, high_vol
        
        Returns:
        --------
        np.ndarray : State vector (12 dim)
        """
        if self.current_step >= len(self.df):
            return np.zeros(self.state_dim, dtype=np.float32)
        
        # Initialize state array
        state = np.zeros(self.state_dim, dtype=np.float32)

        try:
            # Price and volume windows
            close = self.df['close'].iloc[max(0, self.current_step - 20):self.current_step + 1]
            volume = self.df['volume'].iloc[max(0, self.current_step - 20):self.current_step + 1]

            price_feats = normalize_price_features(close)
            vol_feats = normalize_volume_features(volume)

            state[0] = price_feats['latest_ret']
            state[1] = price_feats['mean_ret']
            state[2] = price_feats['vol']
            state[3] = price_feats['trend']
            state[4] = price_feats['rel_price']

            # Portfolio features (indices 5-8)
            current_price = float(self.df.loc[self.current_step, 'close'])

            state[5] = float(self.position)

            if self.position > 0 and self.entry_price > 0:
                state[6] = float((current_price - self.entry_price) / self.entry_price)  # Unrealized PnL
            else:
                state[6] = 0.0

            state[7] = float((self.equity - 100000) / 100000)  # Equity change

            if self.peak_equity > 0:
                state[8] = float((self.peak_equity - self.equity) / self.peak_equity)  # Drawdown
            else:
                state[8] = 0.0

            # Regime features (indices 9-11) - one-hot encoded
            volatility = abs(price_feats['vol'])
            if volatility < 0.015:
                state[9:12] = [1.0, 0.0, 0.0]  # Low vol
            elif volatility < 0.03:
                state[9:12] = [0.0, 1.0, 0.0]  # Normal vol
            else:
                state[9:12] = [0.0, 0.0, 1.0]  # High vol

            # Blend in normalized volume (z-score) into volatility slot for richer context
            state[2] += vol_feats['z_vol']

            # Clip to reasonable bounds
            state = np.clip(state, -10.0, 10.0)

            # Replace any NaN or Inf
            state = np.nan_to_num(state, nan=0.0, posinf=10.0, neginf=-10.0)

        except Exception as e:
            logger.warning(f"Error computing state at step {self.current_step}: {e}")
            state = np.zeros(self.state_dim, dtype=np.float32)

        return state.astype(np.float32)
    
    def get_episode_stats(self) -> Dict:
        """
        Get statistics for completed episode
        
        Returns:
        --------
        dict : Episode statistics
        """
        if len(self.episode_rewards) == 0:
            return {}
        
        total_reward = sum(self.episode_rewards)
        avg_reward = np.mean(self.episode_rewards)
        
        # Count actions
        action_counts = {
            'no_action': sum(1 for a in self.episode_actions if a == self.NO_ACTION),
            'enter': sum(1 for a in self.episode_actions if a == self.ENTER_LONG),
            'exit': sum(1 for a in self.episode_actions if a == self.EXIT_LONG)
        }
        
        return {
            'total_reward': total_reward,
            'avg_reward': avg_reward,
            'final_equity': self.equity,
            'total_pnl': self.total_pnl,
            'max_drawdown': (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0.0,
            'actions': action_counts,
            'steps': len(self.episode_rewards)
        }


# Test environment
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("RL TRADING ENVIRONMENT TEST")
    print("="*80)
    
    # Generate test data
    np.random.seed(42)
    n_bars = 500
    dates = pd.date_range('2024-01-01', periods=n_bars, freq='h')
    close_prices = 50000 * np.exp(np.cumsum(np.random.normal(0.0001, 0.02, n_bars)))
    
    df = pd.DataFrame({
        'open': close_prices * (1 + np.random.normal(0, 0.001, n_bars)),
        'high': close_prices * (1 + np.abs(np.random.normal(0.002, 0.005, n_bars))),
        'low': close_prices * (1 - np.abs(np.random.normal(0.002, 0.005, n_bars))),
        'close': close_prices,
        'volume': np.random.lognormal(5, 0.5, n_bars)
    })
    
    # Create environment
    env = TradingEnvironment(df)
    
    print(f"\nEnvironment created:")
    print(f"  State dim: {env.state_dim}")
    print(f"  Action dim: {env.action_dim}")
    print(f"  Max steps: {env.max_steps}")
    
    # Test random episode
    print("\nTesting random episode...")
    state = env.reset()
    print(f"  Initial state shape: {state.shape}")
    print(f"  Initial state: {state}")
    
    total_reward = 0
    for step in range(100):
        action = np.random.randint(0, 3)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        if step < 5:  # Print first few steps
            print(f"  Step {step}: action={action}, reward={reward:.4f}, state_shape={next_state.shape}")
        
        if done:
            break
    
    stats = env.get_episode_stats()
    
    print(f"\n{'='*80}")
    print("EPISODE COMPLETE!")
    print(f"{'='*80}")
    print(f"  Total reward: {stats['total_reward']:.4f}")
    print(f"  Avg reward: {stats['avg_reward']:.4f}")
    print(f"  Final equity: ${stats['final_equity']:.2f}")
    print(f"  Total PnL: {stats['total_pnl']:.2%}")
    print(f"  Max drawdown: {stats['max_drawdown']:.2%}")
    print(f"  Actions: {stats['actions']}")
    print(f"  Steps: {stats['steps']}")
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE - NO ERRORS!")
    print("="*80)
