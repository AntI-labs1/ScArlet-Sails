"""
RULE-BASED STRATEGY - MINIMAL FIX
Only 1 line changed: added 'returns' to market_state

Original beautiful code preserved!

Author: STAR_ANT + Claude Sonnet 4.5
Date: November 24, 2025 (MINIMAL FIX)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging
import sys
import os
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.opportunity_scorer import OpportunityScorer
from core.feature_engine_v2 import CanonicalMarketStateBuilder
from models.pjs_components import CostCalculator, compute_risk_penalty_from_market_state

logger = logging.getLogger(__name__)


class TechnicalFilters:
    """Technical indicator-based filters for Rule-Based strategy"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # RSI parameters
        self.rsi_period = self.config.get('rsi_period', 14)
        self.rsi_lower = self.config.get('rsi_lower', 20)
        self.rsi_upper = self.config.get('rsi_upper', 80)
        
        # EMA parameters
        self.ema_fast = self.config.get('ema_fast', 9)
        
        # BB parameters
        self.bb_period = self.config.get('bb_period', 20)
        self.bb_std = self.config.get('bb_std', 2.0)
        
        logger.info("TechnicalFilters initialized")
    
    def calculate_rsi(self, prices: pd.Series, period: int = None) -> pd.Series:
        """Calculate RSI indicator"""
        if period is None:
            period = self.rsi_period
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def rsi_filter(self, df: pd.DataFrame) -> pd.Series:
        """I₁: RSI filter"""
        rsi = self.calculate_rsi(df['close'])
        filter_signal = ((rsi > self.rsi_lower) & (rsi < self.rsi_upper)).astype(int)
        return filter_signal
    
    def ema_filter(self, df: pd.DataFrame) -> pd.Series:
        """I₂: EMA trend filter"""
        ema = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        filter_signal = (df['close'] > ema).astype(int)
        return filter_signal
    
    def volume_filter(self, df: pd.DataFrame) -> pd.Series:
        """I₃: Volume confirmation filter"""
        volume_ma = df['volume'].rolling(window=14).mean()
        filter_signal = (df['volume'] > volume_ma).astype(int)
        return filter_signal
    
    def bollinger_filter(self, df: pd.DataFrame) -> pd.Series:
        """I₄: Bollinger Band position filter"""
        sma = df['close'].rolling(window=self.bb_period).mean()
        std = df['close'].rolling(window=self.bb_period).std()
        
        bb_upper = sma + (self.bb_std * std)
        bb_lower = sma - (self.bb_std * std)
        
        bb_position = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        filter_signal = ((bb_position > 0.25) & (bb_position < 0.75)).astype(int)
        
        return filter_signal
    
    def calculate_all_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all filters and return as DataFrame"""
        filters = pd.DataFrame(index=df.index)
        
        filters['I1_rsi'] = self.rsi_filter(df)
        filters['I2_ema'] = self.ema_filter(df)
        filters['I3_volume'] = self.volume_filter(df)
        filters['I4_bb'] = self.bollinger_filter(df)
        
        filters['product'] = (filters['I1_rsi'] * filters['I2_ema'] * 
                             filters['I3_volume'] * filters['I4_bb'])
        
        return filters


class RuleBasedStrategy:
    """
    Rule-Based Trading Strategy with full P_j(S) formula
    
    Decision function:
    P_rb(S) = W_opportunity(S) · ∏ᵢ Iᵢ(S) - C_fixed - R_penalty(S)
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}

        # Initialize components
        self.opportunity_scorer = OpportunityScorer(self.config.get('opportunity', {}))
        self.technical_filters = TechnicalFilters(self.config.get('filters', {}))
        self.cost_calculator = CostCalculator.from_config(self.config)

        # Costs
        self.C_fixed = self.cost_calculator.get_round_trip_cost(
            self.config.get('use_maker', True)
        )

        # Risk penalty
        self.risk_calculator = None
        self.signal_threshold = self.config.get('threshold', 0.05)

        logger.info(f"RuleBasedStrategy initialized: C_fixed={self.C_fixed:.4f}, threshold={self.signal_threshold:.4f}")
    
    def set_risk_calculator(self, risk_calc):
        """Set the AdvancedRiskPenalty calculator instance"""
        self.risk_calculator = risk_calc
        logger.info("Risk calculator set")
    
    def calculate_pjs(self, market_state: Dict, df: pd.DataFrame, idx: int) -> Tuple[float, Dict]:
        """Calculate P_rb(S) for a given market state"""
        # Component 1: Opportunity score
        W_opportunity = self.opportunity_scorer.calculate_opportunity(market_state)
        
        # Component 2: Technical filters
        df_history = df.iloc[:idx+1]
        
        if len(df_history) < 50:
            filters_product = 0
        else:
            filters = self.technical_filters.calculate_all_filters(df_history)
            filters_product = filters['product'].iloc[-1]
        
        # Component 3: Costs
        costs = self.C_fixed
        
        # Component 4: Risk penalty
        if self.risk_calculator is not None:
            try:
                current_return_shock = market_state.get('return_shock', 0.0)
                historical_returns = market_state.get('returns', np.array([0]))
                current_state_features = market_state.get('state_features', np.zeros(3))
                current_value = market_state.get('portfolio_value', 1.0)
                spread = market_state.get('spread', 0.001)
                atr = market_state.get('atr', 0.02)
                regime = market_state.get('regime', 'normal')

                risk_results = self.risk_calculator.calculate_total_risk_penalty(
                    current_return_shock=current_return_shock,
                    historical_returns=historical_returns,
                    current_state=current_state_features,
                    current_value=current_value,
                    spread=spread,
                    atr=atr,
                    regime=regime.upper()
                )

                risk_penalty = risk_results['total_penalty_adjusted']
            except Exception as e:
                logger.warning(f"Risk calculation failed: {e}. Using default penalty.")
                risk_penalty = compute_risk_penalty_from_market_state(market_state, self.config)
        else:
            risk_penalty = compute_risk_penalty_from_market_state(market_state, self.config)
        
        # Calculate P_rb(S)
        P_rb = W_opportunity * filters_product - costs - risk_penalty
        
        components = {
            'W_opportunity': W_opportunity,
            'filters_product': filters_product,
            'costs': costs,
            'risk_penalty': risk_penalty,
            'P_rb': P_rb
        }

        return P_rb, components

    def generate_signal(self, df: pd.DataFrame, market_states: list = None) -> int:
        """Generate a single signal for the latest bar."""
        if df is None or df.empty:
            raise ValueError("Input dataframe is empty; cannot generate signal")

        signals_df = self.generate_signals(df, market_states)
        if signals_df.empty:
            raise ValueError("No signals generated from input dataframe")

        return int(signals_df['signal'].iloc[-1])

    def generate_signals(self, df: pd.DataFrame, market_states: list = None) -> pd.DataFrame:
        """Generate trading signals for an entire dataframe using canonical S_t."""
        results = []

        logger.info(f"Generating signals for {len(df)} bars...")

        builder = CanonicalMarketStateBuilder(df)
        iterator = tqdm(range(len(df)), desc="Generating signals") if len(df) > 10000 else range(len(df))

        for idx in iterator:
            if market_states and idx < len(market_states):
                market_state = market_states[idx]
            else:
                market_state = builder.build_for_index(idx)

            P_rb, components = self.calculate_pjs(market_state, df, idx)
            signal = 1 if P_rb > self.signal_threshold else 0

            results.append({
                'timestamp': df.index[idx],
                'P_rb': P_rb,
                'signal': signal,
                **components
            })

        results_df = pd.DataFrame(results)
        results_df.set_index('timestamp', inplace=True)

        logger.info(f"Generated {results_df['signal'].sum()} signals (out of {len(df)} bars)")

        return results_df


# Testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("RULE-BASED STRATEGY TEST (MINIMAL FIX)")
    print("=" * 80)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1H')
    
    close_prices = 50000 * np.exp(np.cumsum(np.random.normal(0.0001, 0.02, 1000)))
    
    df = pd.DataFrame({
        'open': close_prices * (1 + np.random.normal(0, 0.001, 1000)),
        'high': close_prices * (1 + np.abs(np.random.normal(0.002, 0.005, 1000))),
        'low': close_prices * (1 - np.abs(np.random.normal(0.002, 0.005, 1000))),
        'close': close_prices,
        'volume': np.random.lognormal(5, 0.5, 1000)
    }, index=dates)
    
    print(f"\nTest data: {len(df)} bars")
    
    # Initialize strategy
    strategy = RuleBasedStrategy()
    
    # Generate signals
    signals_df = strategy.generate_signals(df)
    
    print(f"\n✅ Signals generated: {len(signals_df)}")
    print(f"Buy signals: {signals_df['signal'].sum()}")
    print(f"Signal rate: {signals_df['signal'].mean():.2%}")
    
    print("\nP_rb Statistics:")
    print(signals_df[['P_rb', 'W_opportunity', 'filters_product', 'signal']].describe())
    
    print("\n" + "=" * 80)
    print("MINIMAL FIX COMPLETE - ORIGINAL CODE PRESERVED!")
    print("=" * 80)