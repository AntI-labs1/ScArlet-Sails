"""
HYBRID STRATEGY - FULL P_j(S) IMPLEMENTATION
Mathematical formula from MATHEMATICAL_FRAMEWORK.md

P_hyb(S) = α(t)·P_rb(S) + β(t)·P_ml(S) + γ·𝔼[V_future(S)]

ADAPTED: Works with 74 advanced features for ML component

Author: STAR_ANT + Claude Sonnet 4.5
Date: November 24, 2025 (PRODUCTION READY)
"""

import logging
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.rule_based_v2 import RuleBasedStrategy
try:  # Prefer the v3 ML strategy when available
    from strategies.xgboost_ml_v3 import XGBoostMLStrategyV3 as DefaultMLStrategy
except Exception:  # pragma: no cover - fallback for environments without v3 model
    from strategies.xgboost_ml_v2 import XGBoostMLStrategy as DefaultMLStrategy

logger = logging.getLogger(__name__)


class AdaptiveWeightCalculator:
    """
    Calculate adaptive weights α(t), β(t) based on rolling performance
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}

        self.window = self.config.get('window', 50)
        self.gamma_weight = self.config.get('gamma_weight', 0.10)
        
        # Performance tracking
        self.rb_returns = []
        self.ml_returns = []
        
        # Default weights
        self.default_alpha = 0.45
        self.default_beta = 0.45
        
        logger.info(f"AdaptiveWeightCalculator initialized: window={self.window}, γ_weight={self.gamma_weight}")
    
    def update_performance(self, rb_return: Optional[float], ml_return: Optional[float]):
        """Update performance tracking with realized PnL returns."""
        if rb_return is not None:
            self.rb_returns.append(rb_return)
        if ml_return is not None:
            self.ml_returns.append(ml_return)

        if len(self.rb_returns) > self.window:
            self.rb_returns = self.rb_returns[-self.window:]
        if len(self.ml_returns) > self.window:
            self.ml_returns = self.ml_returns[-self.window:]
    
    def calculate_sharpe(self, returns: list) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return 0.0
        
        return mean_return / std_return
    
    def calculate_weights(self) -> Tuple[float, float]:
        """Calculate adaptive weights α(t), β(t)"""
        if len(self.rb_returns) < 5 or len(self.ml_returns) < 5:
            return self.default_alpha, self.default_beta
        
        # Calculate Sharpe ratios
        sharpe_rb = self.calculate_sharpe(self.rb_returns)
        sharpe_ml = self.calculate_sharpe(self.ml_returns)
        
        # Softmax
        exp_rb = np.exp(sharpe_rb + 1e-6)
        exp_ml = np.exp(sharpe_ml + 1e-6)
        
        total_exp = exp_rb + exp_ml
        
        available_weight = 1.0 - self.gamma_weight
        
        alpha = (exp_rb / total_exp) * available_weight
        beta = (exp_ml / total_exp) * available_weight
        
        # Clip
        alpha = np.clip(alpha, 0.1, 0.8)
        beta = np.clip(beta, 0.1, 0.8)
        
        # Renormalize
        total = alpha + beta
        if total > available_weight:
            scale = available_weight / total
            alpha *= scale
            beta *= scale
        
        return float(alpha), float(beta)


class HybridStrategy:
    """
    Hybrid Trading Strategy
    
    Decision function:
    P_hyb(S) = α(t)·P_rb(S) + β(t)·P_ml(S) + γ·𝔼[V_future(S)]
    """
    
    def __init__(self, config: Dict = None,
                 rb_strategy: RuleBasedStrategy = None,
                 ml_strategy: Optional[object] = None,
                 rl_advisor: Optional[object] = None):
        self.config = config or {}

        # Initialize strategies
        self.rb_strategy = rb_strategy or RuleBasedStrategy(self.config.get('rule_based', {}))
        self.ml_strategy = ml_strategy or DefaultMLStrategy(self.config.get('xgboost_ml', {}))

        # Weighting configuration
        self.weights_config = self.config.get('weights', {})
        self.mode = self.weights_config.get('mode', 'static')
        self.weight_calculator = AdaptiveWeightCalculator(self.weights_config)

        static_alpha = self.weights_config.get('alpha', self.weight_calculator.default_alpha)
        static_beta = self.weights_config.get('beta', 1 - static_alpha)
        self.alpha = float(static_alpha)
        self.beta = float(static_beta)
        self.gamma_weight = self.weight_calculator.gamma_weight

        # RL controls
        self.use_rl = bool(self.config.get('use_rl', False))
        self.threshold = float(self.config.get('threshold', 0.5))
        self.rl_component = rl_advisor if self.use_rl else None

        logger.info(
            f"HybridStrategy initialized: mode={self.mode}, α={self.alpha:.2f}, β={self.beta:.2f}, "
            f"γ_weight={self.gamma_weight:.2f}, use_rl={self.use_rl}"
        )
    
    def calculate_pjs(self, rb_result: Tuple[float, Dict],
                           ml_result: Tuple[float, Dict],
                           market_state: Dict) -> Tuple[float, Dict]:
        """Calculate P_hyb(S)"""
        P_rb, rb_components = rb_result
        P_ml, ml_components = ml_result
        
        # Calculate weights
        if self.mode == 'adaptive':
            self.alpha, self.beta = self.weight_calculator.calculate_weights()
        else:
            # Static mode keeps configured weights
            self.alpha = float(self.weights_config.get('alpha', self.alpha))
            self.beta = float(self.weights_config.get('beta', 1 - self.alpha))

        # Calculate RL component
        rl_value = 0.0
        if self.use_rl and self.rl_component is not None:
            rl_value = float(getattr(self.rl_component, 'calculate_rl_component', lambda s: 0.0)(market_state))

        # Calculate P_hyb
        P_hyb = self.alpha * P_rb + self.beta * P_ml + rl_value
        
        components = {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma_weight': self.gamma_weight,
            'P_rb': P_rb,
            'P_ml': P_ml,
            'rl_value': rl_value,
            'P_hyb': P_hyb,
            'rb_components': rb_components,
            'ml_components': ml_components
        }
        
        return P_hyb, components
    
    def generate_signals(self, df: pd.DataFrame, market_states: list = None) -> pd.DataFrame:
        """
        Generate hybrid trading signals
        
        Parameters:
        -----------
        df : DataFrame
            DataFrame with 75 columns (74 features + target) from REAL DATA
        market_states : list, optional
            Pre-computed market states
        
        Returns:
        --------
        DataFrame with hybrid signals
        """
        logger.info(f"Generating hybrid signals for {len(df)} bars...")
        
        # Generate Rule-Based signals
        rb_signals = self.rb_strategy.generate_signals(df, market_states)

        # Generate ML signals
        ml_signals = self.ml_strategy.generate_signals(df, market_states)
        
        # Ensure same index
        common_index = rb_signals.index.intersection(ml_signals.index)
        rb_signals = rb_signals.loc[common_index]
        ml_signals = ml_signals.loc[common_index]
        
        results = []
        
        for idx in common_index:
            # Get component signals
            P_rb = rb_signals.loc[idx, 'P_rb']
            P_ml = ml_signals.loc[idx, 'P_ml']
            
            rb_components = {
                'W_opportunity': rb_signals.loc[idx, 'W_opportunity'],
                'filters_product': rb_signals.loc[idx, 'filters_product'],
                'costs': rb_signals.loc[idx, 'costs'],
                'risk_penalty': rb_signals.loc[idx, 'risk_penalty']
            }
            
            ml_components = {
                'ml_score': ml_signals.loc[idx, 'ml_score'],
                'filters_product': ml_signals.loc[idx, 'filters_product'],
                'costs': ml_signals.loc[idx, 'costs'],
                'R_ood': ml_signals.loc[idx, 'R_ood']
            }
            
            # Market state
            if market_states and len(market_states) > len(results):
                market_state = market_states[len(results)]
            else:
                market_state = {
                    'regime': 'normal',
                    'crisis_level': 0.0
                }
            
            # Calculate hybrid
            P_hyb, hyb_components = self.calculate_pjs(
                (P_rb, rb_components),
                (P_ml, ml_components),
                market_state
            )
            
            # Update performance with realized returns if available
            rb_perf = df.loc[idx, 'rb_return'] if 'rb_return' in df.columns else None
            ml_perf = df.loc[idx, 'ml_return'] if 'ml_return' in df.columns else None
            self.weight_calculator.update_performance(rb_perf, ml_perf)

            # Generate signal
            signal = 1 if P_hyb > self.threshold else 0
            
            results.append({
                'timestamp': idx,
                'P_hyb': P_hyb,
                'signal': signal,
                'alpha': hyb_components['alpha'],
                'beta': hyb_components['beta'],
                'P_rb': P_rb,
                'P_ml': P_ml,
                'rl_value': hyb_components['rl_value']
            })
        
        results_df = pd.DataFrame(results)
        results_df.set_index('timestamp', inplace=True)

        logger.info(f"Generated {results_df['signal'].sum()} hybrid signals (out of {len(results_df)} bars)")

        return results_df

    def generate_signal(self, df: pd.DataFrame, market_states: list = None) -> int:
        """Generate a single hybrid signal with graceful fallbacks."""
        if df is None or df.empty:
            raise ValueError("Input dataframe is empty; cannot generate hybrid signal")

        has_features = getattr(self.ml_strategy, 'has_required_features', lambda _: True)(df)
        if has_features:
            signals_df = self.generate_signals(df, market_states)
            if signals_df.empty:
                raise ValueError("No hybrid signals generated from input dataframe")
            return int(signals_df['signal'].iloc[-1])

        logger.warning("HybridStrategy falling back to simplified aggregation (missing ML features)")
        rb_signal = self.rb_strategy.generate_signal(df, market_states)
        ml_signal = self.ml_strategy.generate_signal(df, market_states)

        alpha, beta = self.weight_calculator.calculate_weights()
        weighted_score = alpha * rb_signal + beta * ml_signal
        threshold = 0.5 * (alpha + beta)

        return 1 if weighted_score > threshold else 0


# Testing
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("HYBRID STRATEGY TEST")
    print("=" * 80)
    
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
    }, index=dates)
    
    # Initialize strategy
    strategy = HybridStrategy()
    
    # Generate signals
    signals_df = strategy.generate_signals(df)
    
    print(f"\nSignals generated: {len(signals_df)}")
    print(f"Total hybrid signals: {signals_df['signal'].sum()}")
    print(f"Signal rate: {signals_df['signal'].mean():.2%}")
    
    print("\nP_hyb Statistics:")
    print(signals_df[['P_hyb', 'P_rb', 'P_ml', 'alpha', 'beta', 'signal']].describe())
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)