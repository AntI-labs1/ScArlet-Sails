"""
OPPORTUNITY SCORER - IMPLEMENTATION
Based on mathematical formulation from AI Market Microstructure Specialist

W_opportunity(S) = w_vol·W_vol(S) + w_liq·W_liq(S) + w_micro·W_micro(S)

Author: STAR_ANT + Claude Sonnet 4.5
Date: November 16, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class OpportunityScorer:
    """
    Calculates opportunity score for trading based on market microstructure
    
    Components:
    - Volatility: Regime-adaptive volatility opportunities
    - Liquidity: Spread, depth, imbalance analysis
    - Microstructure: Quality assessment (simplified for crypto)
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize OpportunityScorer
        
        Parameters:
        -----------
        config : dict
            Configuration parameters (optional)
        """
        self.config = config or {}
        
        # Component weights (default from AI specification)
        self.w_vol = float(self.config.get('w_vol', 0.40))
        self.w_liq = float(self.config.get('w_liq', 0.35))
        self.w_micro = float(self.config.get('w_micro', 0.25))
        
        # Volatility parameters
        self.lambda_params = {
            'low_vol': self.config.get('lambda_low', 15),
            'normal': self.config.get('lambda_normal', 10),
            'high_vol': self.config.get('lambda_high', 6),
            'crisis': self.config.get('lambda_crisis', 3)
        }
        
        # Liquidity parameters
        self.kappa_s = self.config.get('kappa_s', 100)  # Spread sensitivity
        self.kappa_d = self.config.get('kappa_d', 2)    # Depth sensitivity
        
        # Validate/normalize weights so we never crash on bad config
        weight_sum = self.w_vol + self.w_liq + self.w_micro
        if weight_sum <= 0:
            # Fall back to uniform weights if config is broken
            self.w_vol = self.w_liq = self.w_micro = 1 / 3
        elif abs(weight_sum - 1.0) > 1e-6:
            self.w_vol /= weight_sum
            self.w_liq /= weight_sum
            self.w_micro /= weight_sum
        
        logger.info(f"OpportunityScorer initialized: w_vol={self.w_vol}, w_liq={self.w_liq}, w_micro={self.w_micro}")
    
    def calculate_opportunity(self, market_state: Dict) -> float:
        """
        Main function: Calculate W_opportunity(S)
        
        Parameters:
        -----------
        market_state : dict
            Dictionary containing:
            - 'returns': recent returns (array)
            - 'regime': market regime ('low_vol', 'normal', 'high_vol', 'crisis')
            - 'crisis_level': crisis indicator [0,1]
            - 'orderbook': {'bids': [(price, vol)], 'asks': [(price, vol)]}
            - 'volume': current volume
            - 'volume_ma': moving average volume
            
        Returns:
        --------
        float : W_opportunity ∈ [0,1]
        """
        # Calculate components
        W_vol = self._calculate_volatility_score(
            market_state['returns'],
            market_state.get('regime', 'normal'),
            market_state.get('crisis_level', 0.0)
        )
        
        W_liq = self._calculate_liquidity_score(
            market_state.get('orderbook', None),
            market_state.get('volume', 0),
            market_state.get('volume_ma', 1)
        )
        
        W_micro = self._calculate_microstructure_score(
            market_state
        )
        
        # Weighted combination
        W_opportunity = self.w_vol * W_vol + self.w_liq * W_liq + self.w_micro * W_micro
        
        # Ensure bounds
        W_opportunity = np.clip(W_opportunity, 0.0, 1.0)
        
        logger.debug(f"Opportunity: vol={W_vol:.3f}, liq={W_liq:.3f}, micro={W_micro:.3f} → total={W_opportunity:.3f}")
        
        return W_opportunity
    
    def _calculate_volatility_score(self, returns: np.ndarray, regime: str, crisis_level: float) -> float:
        """
        Calculate volatility component W_vol(S)
        
        Formula:
        W_vol(S) = Φ(σ_realized, ψ) · (1 - R_t)
        where Φ(σ, ψ) = 1 - exp(-λ_ψ · σ)
        
        Parameters:
        -----------
        returns : array
            Recent returns for volatility calculation
        regime : str
            Market regime
        crisis_level : float
            Crisis indicator [0,1]
        
        Returns:
        --------
        float : W_vol ∈ [0,1]
        """
        if len(returns) == 0:
            return 0.0
        
        # Calculate realized volatility (annualized for hourly data)
        # Assuming 24 hours/day, 365 days/year
        sigma_realized = np.sqrt(np.mean(returns**2)) * np.sqrt(365 * 24)
        
        # Get regime-specific lambda
        lambda_regime = self.lambda_params.get(regime, self.lambda_params['normal'])
        
        # Transformation Φ
        Phi = 1 - np.exp(-lambda_regime * sigma_realized)
        
        # Crisis adjustment
        W_vol = Phi * (1 - crisis_level)
        
        return float(np.clip(W_vol, 0.0, 1.0))
    
    def _calculate_liquidity_score(self, orderbook: Optional[Dict], volume: float, volume_ma: float) -> float:
        """
        Calculate liquidity component W_liq(S)
        
        Formula:
        W_liq = w_spread·L_spread + w_depth·L_depth + w_imbalance·L_imbalance
        
        Parameters:
        -----------
        orderbook : dict or None
            Order book with 'bids' and 'asks'
        volume : float
            Current bar volume
        volume_ma : float
            Moving average volume
        
        Returns:
        --------
        float : W_liq ∈ [0,1]
        """
        # Handle missing orderbook gracefully using volume context
        if orderbook is None or not orderbook:
            if volume_ma <= 0:
                return 0.0
            volume_ratio = volume / volume_ma if volume_ma else 0
            # Map volume ratio into [0,1] with a soft cap
            return float(np.clip(volume_ratio / 2.0, 0.0, 1.0))
        
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if len(bids) == 0 or len(asks) == 0:
            return 0.0
        
        # Extract best bid/ask
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid = (best_bid + best_ask) / 2
        
        # Component 1: Spread score
        s_relative = (best_ask - best_bid) / mid if mid > 0 else 1.0
        L_spread = 1 - np.tanh(self.kappa_s * s_relative)
        
        # Component 2: Depth score
        # Sum top 10 levels (or all if less than 10)
        bid_depth = sum(vol for _, vol in bids[:min(10, len(bids))])
        ask_depth = sum(vol for _, vol in asks[:min(10, len(asks))])
        total_depth = bid_depth + ask_depth
        
        # Normalize by typical depth (calibration constant)
        typical_depth = 10.0  # Can be adjusted per asset
        L_depth = min(1.0, total_depth / typical_depth)
        
        # Component 3: Imbalance score
        if bid_depth + ask_depth > 0:
            OBI = (bid_depth - ask_depth) / (bid_depth + ask_depth)
        else:
            OBI = 0.0
        L_imbalance = 1 - abs(OBI)
        
        # Aggregate with sub-weights
        w_spread = self.config.get('w_spread', 0.4)
        w_depth = self.config.get('w_depth', 0.4)
        w_imbalance = self.config.get('w_imbalance', 0.2)
        weight_sum = w_spread + w_depth + w_imbalance
        if weight_sum <= 0:
            w_spread = w_depth = w_imbalance = 1 / 3
        elif abs(weight_sum - 1.0) > 1e-6:
            w_spread /= weight_sum
            w_depth /= weight_sum
            w_imbalance /= weight_sum
        
        W_liq = w_spread * L_spread + w_depth * L_depth + w_imbalance * L_imbalance
        
        return float(np.clip(W_liq, 0.0, 1.0))
    
    def _calculate_microstructure_score(self, market_state: Dict) -> float:
        """
        Calculate microstructure quality component W_micro(S)
        
        Simplified for crypto (no HFT data):
        - Price efficiency (autocorrelation)
        - Volume consistency
        - Regime quality
        
        Parameters:
        -----------
        market_state : dict
            Market state data
        
        Returns:
        --------
        float : W_micro ∈ [0,1]
        """
        # For crypto without tick data, use simplified version
        
        # 1. Price efficiency via return autocorrelation
        returns = market_state.get('returns', np.array([]))
        if len(returns) > 2:
            # Calculate lag-1 autocorrelation
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            epsilon_pricing = abs(autocorr)
            Lambda_efficiency = np.exp(-epsilon_pricing)
        else:
            Lambda_efficiency = 0.5  # Neutral if insufficient data
        
        # 2. Volume consistency (not too erratic)
        volume = market_state.get('volume', 0)
        volume_ma = market_state.get('volume_ma', 1)
        if volume_ma > 0:
            volume_ratio = volume / volume_ma
            # Penalize extreme ratios (too high or too low)
            volume_score = 1.0 - abs(np.log(volume_ratio)) / 2.0  # Normalized
            volume_score = np.clip(volume_score, 0.0, 1.0)
        else:
            volume_score = 0.5
        
        # 3. Simple manipulation detection (very basic)
        # In crypto, hard to detect without order flow data
        # Placeholder: assume no manipulation for now
        manipulation_penalty = 0.0
        
        # Combine
        W_micro = (1 - manipulation_penalty) * Lambda_efficiency * volume_score
        
        return float(np.clip(W_micro, 0.0, 1.0))
    
    def get_component_scores(self, market_state: Dict) -> Dict[str, float]:
        """
        Get individual component scores for analysis
        
        Returns:
        --------
        dict : {'W_vol': float, 'W_liq': float, 'W_micro': float, 'W_total': float}
        """
        W_vol = self._calculate_volatility_score(
            market_state['returns'],
            market_state.get('regime', 'normal'),
            market_state.get('crisis_level', 0.0)
        )
        
        W_liq = self._calculate_liquidity_score(
            market_state.get('orderbook', None),
            market_state.get('volume', 0),
            market_state.get('volume_ma', 1)
        )
        
        W_micro = self._calculate_microstructure_score(market_state)
        
        W_total = self.w_vol * W_vol + self.w_liq * W_liq + self.w_micro * W_micro
        
        return {
            'W_vol': W_vol,
            'W_liq': W_liq,
            'W_micro': W_micro,
            'W_total': W_total
        }


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    
    print("=" * 80)
    print("OPPORTUNITY SCORER TEST")
    print("=" * 80)
    
    # Initialize scorer
    scorer = OpportunityScorer()
    
    # Sample market state
    market_state = {
        'returns': np.random.normal(0.001, 0.02, 100),  # 100 recent returns
        'regime': 'normal',
        'crisis_level': 0.0,
        'orderbook': {
            'bids': [(50000, 1.5), (49990, 2.0), (49980, 1.8)],
            'asks': [(50010, 1.2), (50020, 1.9), (50030, 1.5)]
        },
        'volume': 100,
        'volume_ma': 120
    }
    
    # Calculate opportunity
    W_opp = scorer.calculate_opportunity(market_state)
    
    print(f"\nTotal Opportunity Score: {W_opp:.4f}")
    
    # Get component breakdown
    components = scorer.get_component_scores(market_state)
    print("\nComponent Breakdown:")
    for key, value in components.items():
        print(f"  {key}: {value:.4f}")
    
    # Test different regimes
    print("\n" + "-" * 40)
    print("REGIME SENSITIVITY TEST")
    print("-" * 40)
    
    for regime in ['low_vol', 'normal', 'high_vol', 'crisis']:
        market_state['regime'] = regime
        market_state['crisis_level'] = 0.0 if regime != 'crisis' else 0.8
        
        W_opp = scorer.calculate_opportunity(market_state)
        print(f"  {regime:12s}: W_opportunity = {W_opp:.4f}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)