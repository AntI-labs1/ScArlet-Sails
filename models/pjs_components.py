"""
P_j(S) Formula Components
=========================

Implementation of all components for the P_j(S) decision formula:

P_j(S) = ML(...) · ∏_k I_k + opportunity(S) - costs(S) - risk_penalty(S) + γ·E[V_future]

Components:
1. ML(...) - Already exists (XGBoost model)
2. ∏_k I_k - Filters (crisis, regime, correlation, etc.)
3. opportunity(S) - Opportunity scorer (simplified version)
4. costs(S) - Trading costs calculator
5. risk_penalty(S) - Risk penalty calculator
6. γ·E[V_future] - Out of scope for 1MVP (RL component)

Author: Scarlet Sails Team
Date: 2025-11-13
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


# ============================================================
# 1. OPPORTUNITY SCORER (Simplified)
# ============================================================

@dataclass
class OpportunityScore:
    """Opportunity scoring result."""
    timestamp: datetime
    score: float  # 0-1
    factors: Dict[str, float]  # Contributing factors

    def get_recommendation(self) -> str:
        """Get recommendation based on score."""
        if self.score >= 0.7:
            return "EXCELLENT"
        elif self.score >= 0.5:
            return "GOOD"
        elif self.score >= 0.3:
            return "MODERATE"
        else:
            return "LOW"


class SimpleOpportunityScorer:
    """
    Simplified opportunity scorer using available features.

    Unlike full OpportunityScorer (38 features), this uses only:
    - RSI (oversold indicator)
    - Volume ratio (spike detection)
    - EMA trend (momentum)
    - ATR (volatility)

    Score interpretation:
    - 0.7-1.0: EXCELLENT opportunity
    - 0.5-0.7: GOOD opportunity
    - 0.3-0.5: MODERATE opportunity
    - 0.0-0.3: LOW opportunity
    """

    def __init__(self):
        """Initialize simple opportunity scorer."""
        pass

    def score(
        self,
        rsi: float,
        volume_ratio: float,
        ema_9: float,
        ema_21: float,
        close: float,
        atr_pct: float,
    ) -> OpportunityScore:
        """
        Calculate opportunity score from basic features.

        Args:
            rsi: RSI value (0-100)
            volume_ratio: Volume / avg_volume
            ema_9: 9-period EMA
            ema_21: 21-period EMA
            close: Current price
            atr_pct: ATR as percentage of price

        Returns:
            OpportunityScore with score and factors
        """
        score = 0.5  # Base score
        factors = {}

        # Factor 1: RSI Oversold (+)
        # More oversold = more opportunity (mean reversion)
        if rsi < 30:
            rsi_factor = (30 - rsi) / 30  # 0.0-1.0
            score += rsi_factor * 0.25
            factors['rsi_oversold'] = rsi_factor
        else:
            factors['rsi_oversold'] = 0.0

        # Factor 2: Volume Spike - Moderate (+)
        # Moderate volume (1.5-3x) = interest, not panic
        # High volume (>5x) = panic = bad
        if 1.5 <= volume_ratio <= 3.0:
            volume_factor = 1.0 - abs(volume_ratio - 2.0) / 1.5
            score += volume_factor * 0.15
            factors['volume_moderate'] = volume_factor
        elif volume_ratio > 5.0:
            # Panic volume - bad
            panic_penalty = min(0.2, (volume_ratio - 5.0) / 10.0)
            score -= panic_penalty
            factors['volume_moderate'] = -panic_penalty
        else:
            factors['volume_moderate'] = 0.0

        # Factor 3: EMA Trend (+/-)
        # EMA_9 > EMA_21 = uptrend = good
        # EMA_9 < EMA_21 = downtrend = bad (but oversold can be opportunity)
        ema_ratio = ema_9 / ema_21 if ema_21 > 0 else 1.0
        if ema_ratio > 1.0:
            # Uptrend
            trend_factor = min(0.1, (ema_ratio - 1.0) * 2.0)
            score += trend_factor
            factors['ema_uptrend'] = trend_factor
        else:
            # Downtrend (not necessarily bad if oversold)
            trend_factor = min(0.05, (1.0 - ema_ratio) * 1.0)
            score -= trend_factor
            factors['ema_uptrend'] = -trend_factor

        # Factor 4: Volatility (ATR)
        # Low-moderate volatility = good
        # High volatility = risky = reduce opportunity
        if atr_pct < 0.03:
            # Low volatility - good
            score += 0.1
            factors['volatility_low'] = 0.1
        elif atr_pct > 0.08:
            # High volatility - risky
            vol_penalty = min(0.15, (atr_pct - 0.08) / 0.1)
            score -= vol_penalty
            factors['volatility_low'] = -vol_penalty
        else:
            factors['volatility_low'] = 0.0

        # Clip to 0-1 range
        score = np.clip(score, 0.0, 1.0)

        return OpportunityScore(
            timestamp=datetime.now(),
            score=score,
            factors=factors
        )


# ============================================================
# 2. COST CALCULATOR
# ============================================================

class CostCalculator:
    """
    Calculate trading costs.

    Costs include:
    - Trading fees (maker/taker)
    - Slippage estimation
    - Opportunity cost (optional)
    """

    def __init__(
        self,
        maker_fee: float = 0.001,  # 0.1% Binance maker fee
        taker_fee: float = 0.001,  # 0.1% Binance taker fee
        slippage: float = 0.0005,  # 0.05% estimated slippage
    ):
        """
        Initialize cost calculator.

        Args:
            maker_fee: Maker fee as decimal (0.001 = 0.1%)
            taker_fee: Taker fee as decimal
            slippage: Estimated slippage as decimal
        """
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage = slippage

    def calculate_entry_cost(self, use_maker: bool = True) -> float:
        """
        Calculate entry cost.

        Args:
            use_maker: Whether using maker order (limit) or taker (market)

        Returns:
            Cost as decimal (e.g., 0.0015 = 0.15%)
        """
        fee = self.maker_fee if use_maker else self.taker_fee
        return fee + self.slippage

    def calculate_exit_cost(self, use_maker: bool = True) -> float:
        """
        Calculate exit cost.

        Args:
            use_maker: Whether using maker order (limit) or taker (market)

        Returns:
            Cost as decimal
        """
        fee = self.maker_fee if use_maker else self.taker_fee
        return fee + self.slippage

    def calculate_round_trip_cost(self, use_maker: bool = True) -> float:
        """
        Calculate round-trip (entry + exit) cost.

        Args:
            use_maker: Whether using maker orders

        Returns:
            Total cost as decimal (e.g., 0.003 = 0.3%)
        """
        return self.calculate_entry_cost(use_maker) + self.calculate_exit_cost(use_maker)

    @classmethod
    def from_config(cls, config: Dict) -> "CostCalculator":
        """Instantiate calculator from nested strategy config."""
        fees = config.get("fees", {}) if config else {}
        return cls(
            maker_fee=fees.get("maker", config.get("maker_fee", 0.001)),
            taker_fee=fees.get("taker", config.get("taker_fee", 0.001)),
            slippage=config.get("slippage", 0.0005),
        )

    def get_round_trip_cost(self, use_maker: bool = True) -> float:
        """Convenience wrapper mirroring terminology in architecture docs."""
        return self.calculate_round_trip_cost(use_maker)

    def get_min_profit_threshold(self, use_maker: bool = True) -> float:
        """
        Get minimum profit threshold to break even.

        Args:
            use_maker: Whether using maker orders

        Returns:
            Minimum profit as decimal
        """
        return self.calculate_round_trip_cost(use_maker)


# ============================================================
# 3. RISK PENALTY CALCULATOR
# ============================================================

class RiskPenaltyCalculator:
    """
    Calculate risk penalties.

    Penalties for:
    - High volatility
    - Low liquidity (volume)
    - Crisis mode
    - OOD (out-of-distribution) uncertainty
    """

    def __init__(
        self,
        volatility_threshold: float = 0.05,  # 5% ATR
        volume_threshold: float = 0.5,  # 50% of average
        crisis_threshold: int = 3,  # Crisis level
        ood_threshold: float = 0.5,  # 50% OOD ratio
    ):
        """
        Initialize risk penalty calculator.

        Args:
            volatility_threshold: ATR % above which to penalize
            volume_threshold: Volume ratio below which to penalize
            crisis_threshold: Crisis level above which to penalize
            ood_threshold: OOD ratio above which to penalize
        """
        self.volatility_threshold = volatility_threshold
        self.volume_threshold = volume_threshold
        self.crisis_threshold = crisis_threshold
        self.ood_threshold = ood_threshold

    def calculate_penalty(
        self,
        atr_pct: float,
        volume_ratio: float,
        crisis_level: int = 0,
        ood_ratio: float = 0.0,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total risk penalty.

        Args:
            atr_pct: ATR as percentage of price
            volume_ratio: Volume / average volume
            crisis_level: Crisis severity (0-5)
            ood_ratio: Out-of-distribution ratio (0-1)

        Returns:
            (total_penalty, penalty_breakdown)
            Penalty as decimal (e.g., 0.02 = 2% penalty)
        """
        penalties = {}
        total = 0.0

        # 1. Volatility Penalty
        if atr_pct > self.volatility_threshold:
            vol_penalty = min(0.01, (atr_pct - self.volatility_threshold) / 0.05)
            penalties['volatility'] = vol_penalty
            total += vol_penalty
        else:
            penalties['volatility'] = 0.0

        # 2. Liquidity Penalty (low volume)
        if volume_ratio < self.volume_threshold:
            liq_penalty = min(0.015, (self.volume_threshold - volume_ratio) / 0.5)
            penalties['liquidity'] = liq_penalty
            total += liq_penalty
        else:
            penalties['liquidity'] = 0.0

        # 3. Crisis Penalty
        if crisis_level > self.crisis_threshold:
            crisis_penalty = min(0.03, (crisis_level - self.crisis_threshold) * 0.01)
            penalties['crisis'] = crisis_penalty
            total += crisis_penalty
        else:
            penalties['crisis'] = 0.0

        # 4. OOD Uncertainty Penalty
        if ood_ratio > self.ood_threshold:
            ood_penalty = min(0.005, (ood_ratio - self.ood_threshold) / 1.0)
            penalties['ood_uncertainty'] = ood_penalty
            total += ood_penalty
        else:
            penalties['ood_uncertainty'] = 0.0

        return total, penalties


def compute_risk_penalty_from_market_state(
    market_state: Dict,
    config: Optional[Dict] = None,
) -> float:
    """Thin wrapper to create RiskPenaltyCalculator and compute penalty.

    The function expects the canonical market_state fields produced by
    CanonicalMarketStateBuilder/RuleBasedStrategy and keeps defaults safe
    when some features are missing.
    """

    rpc_cfg = (config or {}).get("risk", {}) if config else {}
    calculator = RiskPenaltyCalculator(
        volatility_threshold=rpc_cfg.get("volatility_threshold", 0.05),
        volume_threshold=rpc_cfg.get("volume_threshold", 0.5),
        crisis_threshold=rpc_cfg.get("crisis_threshold", 3),
        ood_threshold=rpc_cfg.get("ood_threshold", 0.5),
    )

    atr_pct = market_state.get("atr", 0.0) / market_state.get("spread", 1) if market_state.get("spread", 1) != 0 else market_state.get("atr", 0.0)
    volume_ma = market_state.get("volume_ma", 1)
    volume_ratio = (market_state.get("volume", 0) / volume_ma) if volume_ma else 0

    total_penalty, _ = calculator.calculate_penalty(
        atr_pct=atr_pct,
        volume_ratio=volume_ratio,
        crisis_level=market_state.get("crisis_level", 0),
        ood_ratio=market_state.get("ood_ratio", 0.0),
    )
    return total_penalty


# ============================================================
# 4. IMPROVED RULE-BASED STRATEGY
# ============================================================

class ImprovedRuleBasedStrategy:
    """
    Improved Rule-Based strategy with multiple filters.

    Original: RSI < 30

    Improved:
    1. RSI < 30 (oversold)
    2. EMA_9 > EMA_21 (uptrend) - optional
    3. Volume > 1.5x average (interest)
    4. ATR < 5% (not too volatile)
    5. Multi-TF confirmation (1h trend check)
    """

    def __init__(
        self,
        rsi_threshold: float = 30.0,
        use_ema_filter: bool = True,
        use_volume_filter: bool = True,
        use_atr_filter: bool = True,
        volume_min: float = 1.5,
        atr_max: float = 0.05,
        ema_min_ratio: float = 1.0,
    ):
        """
        Initialize improved rule-based strategy.

        Args:
            rsi_threshold: RSI threshold for oversold
            use_ema_filter: Whether to require EMA_9 > EMA_21
            use_volume_filter: Whether to require volume spike
            use_atr_filter: Whether to limit ATR
            volume_min: Minimum volume ratio
            atr_max: Maximum ATR percentage
            ema_min_ratio: Minimum EMA_9/EMA_21 ratio (default 1.0 = strict uptrend)
                          Use 0.99 for less strict (allows nearly equal EMAs)
        """
        self.rsi_threshold = rsi_threshold
        self.use_ema_filter = use_ema_filter
        self.use_volume_filter = use_volume_filter
        self.use_atr_filter = use_atr_filter
        self.volume_min = volume_min
        self.atr_max = atr_max
        self.ema_min_ratio = ema_min_ratio

    def should_enter(
        self,
        rsi: float,
        ema_9: float,
        ema_21: float,
        volume_ratio: float,
        atr_pct: float,
    ) -> bool:
        """
        Determine if should enter trade.

        Args:
            rsi: RSI value (0-100)
            ema_9: 9-period EMA
            ema_21: 21-period EMA
            volume_ratio: Volume / average volume
            atr_pct: ATR as percentage of price

        Returns:
            True if all filters pass
        """
        # Filter 1: RSI oversold (REQUIRED)
        if rsi >= self.rsi_threshold:
            return False

        # Filter 2: EMA trend (OPTIONAL)
        if self.use_ema_filter:
            # Check if EMA_9 >= EMA_21 * ema_min_ratio
            # ema_min_ratio=1.0 → strict uptrend (EMA_9 > EMA_21)
            # ema_min_ratio=0.99 → less strict (EMA_9 >= EMA_21 * 0.99)
            if ema_9 < ema_21 * self.ema_min_ratio:
                return False

        # Filter 3: Volume spike (OPTIONAL)
        if self.use_volume_filter:
            if volume_ratio < self.volume_min:
                return False

        # Filter 4: ATR limit (OPTIONAL)
        if self.use_atr_filter:
            if atr_pct > self.atr_max:
                return False

        # All filters passed
        return True


# ============================================================
# 5. P_j(S) CALCULATOR
# ============================================================

class PjS_Calculator:
    """
    Main P_j(S) calculator combining all components.

    Formula:
    P_j(S) = ML(...) · ∏_k I_k + opportunity(S) - costs(S) - risk_penalty(S)

    Components:
    - ML(...): XGBoost prediction probability
    - ∏_k I_k: Product of filters (crisis, regime, etc.)
    - opportunity(S): Opportunity score
    - costs(S): Trading costs
    - risk_penalty(S): Risk penalties
    """

    def __init__(
        self,
        opportunity_scorer: Optional[SimpleOpportunityScorer] = None,
        cost_calculator: Optional[CostCalculator] = None,
        risk_calculator: Optional[RiskPenaltyCalculator] = None,
    ):
        """
        Initialize P_j(S) calculator.

        Args:
            opportunity_scorer: Opportunity scorer instance
            cost_calculator: Cost calculator instance
            risk_calculator: Risk penalty calculator instance
        """
        self.opportunity_scorer = opportunity_scorer or SimpleOpportunityScorer()
        self.cost_calculator = cost_calculator or CostCalculator()
        self.risk_calculator = risk_calculator or RiskPenaltyCalculator()

    def calculate(
        self,
        ml_score: float,
        filters: Dict[str, float],
        rsi: float,
        volume_ratio: float,
        ema_9: float,
        ema_21: float,
        close: float,
        atr_pct: float,
        crisis_level: int = 0,
        ood_ratio: float = 0.0,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate P_j(S) score.

        Args:
            ml_score: ML model prediction (0-1)
            filters: Dict of filter values (0-1, where 0=blocked, 1=pass)
            rsi: RSI value
            volume_ratio: Volume ratio
            ema_9: 9-period EMA
            ema_21: 21-period EMA
            close: Current price
            atr_pct: ATR percentage
            crisis_level: Crisis level (0-5)
            ood_ratio: OOD ratio (0-1)

        Returns:
            (pjs_score, components) where components show breakdown
        """
        # Component 1: ML score weighted by filters
        filter_product = 1.0
        for filter_name, filter_value in filters.items():
            filter_product *= filter_value
        ml_component = ml_score * filter_product

        # Component 2: Opportunity score
        opp_score_obj = self.opportunity_scorer.score(
            rsi=rsi,
            volume_ratio=volume_ratio,
            ema_9=ema_9,
            ema_21=ema_21,
            close=close,
            atr_pct=atr_pct,
        )
        opportunity = opp_score_obj.score

        # Component 3: Costs
        costs = self.cost_calculator.calculate_round_trip_cost(use_maker=True)

        # Component 4: Risk penalties
        risk_penalty, penalty_breakdown = self.risk_calculator.calculate_penalty(
            atr_pct=atr_pct,
            volume_ratio=volume_ratio,
            crisis_level=crisis_level,
            ood_ratio=ood_ratio,
        )

        # Final P_j(S) calculation
        pjs = ml_component + opportunity - costs - risk_penalty

        # Component breakdown for analysis
        components = {
            'ml_score': ml_score,
            'filter_product': filter_product,
            'ml_component': ml_component,
            'opportunity': opportunity,
            'costs': costs,
            'risk_penalty': risk_penalty,
            'pjs_total': pjs,
        }

        return pjs, components


# ============================================================
# DEMO & TESTS
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("P_j(S) Components Demo")
    print("=" * 60)

    # 1. Opportunity Scorer Test
    print("\n1. Simple Opportunity Scorer")
    print("-" * 60)
    scorer = SimpleOpportunityScorer()

    # High opportunity scenario
    opp1 = scorer.score(
        rsi=25,  # Oversold
        volume_ratio=2.0,  # Moderate spike
        ema_9=100,
        ema_21=98,  # Uptrend
        close=100,
        atr_pct=0.02,  # Low volatility
    )
    print(f"High opportunity: {opp1.score:.2f} ({opp1.get_recommendation()})")
    print(f"  Factors: {opp1.factors}")

    # Low opportunity scenario
    opp2 = scorer.score(
        rsi=15,  # Very oversold (crash)
        volume_ratio=8.0,  # Panic volume
        ema_9=95,
        ema_21=100,  # Downtrend
        close=100,
        atr_pct=0.12,  # High volatility
    )
    print(f"\nLow opportunity: {opp2.score:.2f} ({opp2.get_recommendation()})")
    print(f"  Factors: {opp2.factors}")

    # 2. Cost Calculator Test
    print("\n2. Cost Calculator")
    print("-" * 60)
    cost_calc = CostCalculator()
    print(f"Entry cost (maker): {cost_calc.calculate_entry_cost():.4f} (0.15%)")
    print(f"Exit cost (maker): {cost_calc.calculate_exit_cost():.4f} (0.15%)")
    print(f"Round-trip cost: {cost_calc.calculate_round_trip_cost():.4f} (0.30%)")
    print(f"Min profit to break even: {cost_calc.get_min_profit_threshold():.4f}")

    # 3. Risk Penalty Calculator Test
    print("\n3. Risk Penalty Calculator")
    print("-" * 60)
    risk_calc = RiskPenaltyCalculator()

    # Low risk
    penalty1, breakdown1 = risk_calc.calculate_penalty(
        atr_pct=0.02,
        volume_ratio=1.2,
        crisis_level=0,
        ood_ratio=0.1,
    )
    print(f"Low risk penalty: {penalty1:.4f}")
    print(f"  Breakdown: {breakdown1}")

    # High risk
    penalty2, breakdown2 = risk_calc.calculate_penalty(
        atr_pct=0.10,  # High volatility
        volume_ratio=0.3,  # Low liquidity
        crisis_level=4,  # Crisis
        ood_ratio=0.8,  # High OOD
    )
    print(f"\nHigh risk penalty: {penalty2:.4f}")
    print(f"  Breakdown: {breakdown2}")

    # 4. P_j(S) Calculator Test
    print("\n4. P_j(S) Calculator")
    print("-" * 60)
    pjs_calc = PjS_Calculator()

    # Good scenario
    pjs1, comp1 = pjs_calc.calculate(
        ml_score=0.65,
        filters={'crisis': 1.0, 'regime': 1.0},
        rsi=28,
        volume_ratio=2.0,
        ema_9=100,
        ema_21=98,
        close=100,
        atr_pct=0.03,
        crisis_level=0,
        ood_ratio=0.05,
    )
    print(f"Good scenario P_j(S): {pjs1:.4f}")
    print(f"  Components: {comp1}")

    # Bad scenario
    pjs2, comp2 = pjs_calc.calculate(
        ml_score=0.45,
        filters={'crisis': 0.5, 'regime': 0.8},  # Some filters blocking
        rsi=20,
        volume_ratio=7.0,  # Panic
        ema_9=95,
        ema_21=100,  # Downtrend
        close=100,
        atr_pct=0.15,  # High vol
        crisis_level=4,
        ood_ratio=0.7,
    )
    print(f"\nBad scenario P_j(S): {pjs2:.4f}")
    print(f"  Components: {comp2}")

    print("\n" + "=" * 60)
    print("✅ All P_j(S) components ready for integration!")
    print("=" * 60)
