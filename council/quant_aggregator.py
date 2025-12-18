"""
ScArlet-Sails Quant Aggregator

Collects signals from quantitative strategies (P_rb, P_ml, P_hyb)
and packages them into QuantSignals for Council consumption.

This module is the bridge between existing strategies and Council contracts.
It does NOT modify strategy logic — only calls and aggregates.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd

# Rolling dispersion for position sizing
from core.rolling_dispersion import RollingDispersionCalculator, integrate_dispersion_with_position_sizing
from core.regime_detector import RegimeDetector
from core.dynamic_position_sizer import DynamicPositionSizer, PositionSizingInput

# Council contracts
from council.contracts import (
    QuantSignals,
    AgentOpinion,
    AgentRole,
    ActionType,
    Regime,
)

logger = logging.getLogger(__name__)


@dataclass
class StrategyResult:
    """Result from a single strategy call."""
    name: str
    signal: float          # Raw signal value (probability or score)
    action: ActionType     # Interpreted action
    confidence: float      # Confidence in the signal
    metadata: Dict[str, Any]
    success: bool          # Whether strategy executed without error
    error: Optional[str] = None


class QuantAggregator:
    """
    Aggregates signals from multiple quantitative strategies.
    
    Responsibilities:
    - Call each strategy with current market state
    - Normalize outputs to common format
    - Calculate agreement between strategies
    - Package into QuantSignals for Council
    
    Does NOT:
    - Modify strategy logic
    - Make trading decisions
    - Execute trades
    
    Usage:
        aggregator = QuantAggregator()
        aggregator.register_strategy('rule_based', rb_strategy)
        aggregator.register_strategy('xgboost_ml', ml_strategy)
        
        signals = aggregator.aggregate(features_df, market_state)
    """
    
    def __init__(self):
        """Initialize aggregator with empty strategy registry."""
        self._strategies: Dict[str, Any] = {}
        self._last_results: Dict[str, StrategyResult] = {}
        self._signal_threshold: float = 0.5  # Above = bullish, below = bearish
        # Rolling dispersion calculator for position sizing
        self._dispersion_calc = RollingDispersionCalculator(window=100)
        self._regime_detector = RegimeDetector()
        self._position_sizer = DynamicPositionSizer()
        self._current_drawdown = 0.0
        self._last_ohlcv = None
    
    def register_strategy(self, name: str, strategy: Any) -> None:
        """
        Register a strategy for aggregation.
        
        Args:
            name: Unique identifier (e.g., 'rule_based', 'xgboost_ml')
            strategy: Strategy instance with generate_signals() or predict() method
        """
        self._strategies[name] = strategy
        logger.info(f"Registered strategy: {name}")
    
    def aggregate(
        self,
        features: pd.DataFrame,
        market_state: Optional[Dict[str, Any]] = None,
    ) -> QuantSignals:
        """
        Run all strategies and aggregate results.
        
        Args:
            features: DataFrame with technical features (from FeatureEngine)
            market_state: Optional additional market context
            
        Returns:
            QuantSignals with P_rb, P_ml, P_hyb, agreement
        """
        results = {}
        
        # Call each registered strategy
        for name, strategy in self._strategies.items():
            result = self._call_strategy(name, strategy, features, market_state)
            results[name] = result
            self._last_results[name] = result
        
        # Extract signals
        p_rb = self._extract_signal(results.get('rule_based'))
        p_ml = self._extract_signal(results.get('xgboost_ml'))
        p_hyb = self._extract_signal(results.get('hybrid'))
        
        # Build QuantSignals
        signals = QuantSignals(
            p_rb=p_rb,
            p_ml=p_ml,
            p_hyb=p_hyb,
        )
        
        # Calculate agreement
        signals.compute_agreement()
        
        return signals
    
    def _call_strategy(
        self,
        name: str,
        strategy: Any,
        features: pd.DataFrame,
        market_state: Optional[Dict[str, Any]],
    ) -> StrategyResult:
        """
        Call a single strategy safely.
        
        Handles different strategy interfaces:
        - generate_signals(df) -> DataFrame with 'signal' column
        - predict(df) -> array of probabilities
        - calculate_signal(row) -> single float
        """
        try:
            signal_value = None
            metadata = {}
            
            # Try different interfaces
            if hasattr(strategy, 'generate_signals'):
                # Rule-based style
                result_df = strategy.generate_signals(features)
                if 'signal' in result_df.columns:
                    signal_value = float(result_df['signal'].iloc[-1])
                if 'opportunity_score' in result_df.columns:
                    metadata['opportunity_score'] = float(result_df['opportunity_score'].iloc[-1])
                if 'risk_penalty' in result_df.columns:
                    metadata['risk_penalty'] = float(result_df['risk_penalty'].iloc[-1])
                    
            elif hasattr(strategy, 'predict'):
                # ML style (XGBoost)
                predictions = strategy.predict(features)
                if hasattr(predictions, '__len__') and len(predictions) > 0:
                    signal_value = float(predictions[-1])
                else:
                    signal_value = float(predictions)
                    
            elif hasattr(strategy, 'predict_proba'):
                # Probability output
                proba = strategy.predict_proba(features)
                if hasattr(proba, 'shape') and len(proba.shape) > 1:
                    signal_value = float(proba[-1, 1])  # Probability of class 1
                else:
                    signal_value = float(proba[-1])
                    
            elif hasattr(strategy, 'calculate_signal'):
                # Single row calculation
                last_row = features.iloc[-1] if isinstance(features, pd.DataFrame) else features
                signal_value = float(strategy.calculate_signal(last_row))
            
            else:
                raise AttributeError(f"Strategy {name} has no known interface")
            
            # Normalize signal to [0, 1]
            if signal_value is not None:
                signal_value = np.clip(signal_value, 0.0, 1.0)
            
            # Interpret action
            action = self._interpret_action(signal_value)
            
            # Estimate confidence
            confidence = self._estimate_confidence(signal_value)
            
            return StrategyResult(
                name=name,
                signal=signal_value,
                action=action,
                confidence=confidence,
                metadata=metadata,
                success=True,
            )
            
        except Exception as e:
            logger.error(f"Strategy {name} failed: {e}")
            return StrategyResult(
                name=name,
                signal=0.5,
                action=ActionType.HOLD,
                confidence=0.0,
                metadata={},
                success=False,
                error=str(e),
            )
    
    def _extract_signal(self, result: Optional[StrategyResult]) -> Optional[float]:
        """Extract signal value from result, None if failed or missing."""
        if result is None:
            return None
        if not result.success:
            return None
        return result.signal
    
    def _interpret_action(self, signal: Optional[float]) -> ActionType:
        """
        Convert signal value to action.
        
        Signal interpretation:
        - > 0.6: LONG (bullish)
        - < 0.4: SHORT (bearish) 
        - 0.4-0.6: HOLD (neutral)
        """
        if signal is None:
            return ActionType.HOLD
        
        if signal > 0.6:
            return ActionType.LONG
        elif signal < 0.4:
            return ActionType.SHORT
        else:
            return ActionType.HOLD
    
    def _estimate_confidence(self, signal: Optional[float]) -> float:
        """
        Estimate confidence from signal strength.
        
        Confidence is higher when signal is far from 0.5 (neutral).
        """
        if signal is None:
            return 0.0
        
        # Distance from neutral (0.5)
        distance = abs(signal - 0.5)
        
        # Scale to [0, 1] where 0.5 distance = 1.0 confidence
        confidence = min(distance * 2, 1.0)
        
        return confidence
    
    def get_detailed_results(self) -> Dict[str, StrategyResult]:
        """Get detailed results from last aggregation."""
        return self._last_results.copy()
    
    def get_dispersion_statistics(self) -> dict:
        """Get rolling dispersion statistics."""
        return self._dispersion_calc.get_statistics()
    
    def reset_dispersion(self) -> None:
        """Reset dispersion calculator (e.g., for new backtest run)."""
        self._dispersion_calc.reset()
    
    def to_agent_opinion(
        self,
        signals: QuantSignals,
        agent_id: str = "quant_aggregator",
    ) -> AgentOpinion:
        """
        Convert aggregated signals to AgentOpinion for Council.
        
        Args:
            signals: QuantSignals from aggregate()
            agent_id: Identifier for this opinion
            
        Returns:
            AgentOpinion ready for Council Stage 1
        """
        # Determine overall action from signals
        action = self._determine_consensus_action(signals)
        
        # Calculate aggregate confidence
        confidence = self._calculate_aggregate_confidence(signals)
        
        # Build justification
        justification = self._build_justification(signals)
        
        # Suggested position size based on rolling dispersion
        position_size = self._suggest_position_size(signals)
        
        return AgentOpinion(
            agent_id=agent_id,
            role=AgentRole.QUANT,
            proposed_action=action,
            position_size_pct=position_size,
            confidence=confidence,
            justification=justification,
            raw_metrics={
                "p_rb": signals.p_rb,
                "p_ml": signals.p_ml,
                "p_hyb": signals.p_hyb,
                "agreement": signals.agreement,
            },
        )
    
    def _determine_consensus_action(self, signals: QuantSignals) -> ActionType:
        """Determine action from consensus of available signals."""
        actions = []
        
        for p in [signals.p_rb, signals.p_ml, signals.p_hyb]:
            if p is not None:
                actions.append(self._interpret_action(p))
        
        if not actions:
            return ActionType.HOLD
        
        # Count votes
        long_votes = sum(1 for a in actions if a == ActionType.LONG)
        short_votes = sum(1 for a in actions if a == ActionType.SHORT)
        
        # Majority wins, tie = HOLD
        if long_votes > short_votes and long_votes > len(actions) / 2:
            return ActionType.LONG
        elif short_votes > long_votes and short_votes > len(actions) / 2:
            return ActionType.SHORT
        else:
            return ActionType.HOLD
    
    def _calculate_aggregate_confidence(self, signals: QuantSignals) -> float:
        """
        Calculate aggregate confidence.
        
        Higher when:
        - Agreement is high
        - Individual signals are strong (far from 0.5)
        """
        confidences = []
        
        for p in [signals.p_rb, signals.p_ml, signals.p_hyb]:
            if p is not None:
                confidences.append(self._estimate_confidence(p))
        
        if not confidences:
            return 0.0
        
        # Base confidence = average of individual confidences
        base_confidence = np.mean(confidences)
        
        # Adjust by agreement
        agreement_factor = signals.agreement if signals.agreement else 0.5
        
        # Final confidence
        final_confidence = base_confidence * (0.5 + 0.5 * agreement_factor)
        
        return float(np.clip(final_confidence, 0.0, 1.0))
    
    def _build_justification(self, signals: QuantSignals) -> str:
        """Build human-readable justification."""
        parts = []
        
        if signals.p_rb is not None:
            rb_action = self._interpret_action(signals.p_rb)
            parts.append(f"Rule-Based: {signals.p_rb:.2f} ({rb_action.value})")
        
        if signals.p_ml is not None:
            ml_action = self._interpret_action(signals.p_ml)
            parts.append(f"XGBoost ML: {signals.p_ml:.2f} ({ml_action.value})")
        
        if signals.p_hyb is not None:
            hyb_action = self._interpret_action(signals.p_hyb)
            parts.append(f"Hybrid: {signals.p_hyb:.2f} ({hyb_action.value})")
        
        if signals.agreement is not None:
            parts.append(f"Agreement: {signals.agreement:.0%}")
        
        # Add dispersion info
        disp_state = self._dispersion_calc.get_state()
        if disp_state and disp_state.n_samples > 0:
            parts.append(f"Dispersion: {disp_state.current_std:.3f} (x{disp_state.confidence_multiplier:.2f})")
        
        return " | ".join(parts) if parts else "No signals available"
    
    def _suggest_position_size(self, signals: QuantSignals) -> float:
        """Dynamic position sizing based on multiple factors."""
        
        # Get dispersion state
        disp_state = None
        if signals.p_rb is not None and signals.p_ml is not None:
        disp_state = self._dispersion_calc.update(
            p_rb=signals.p_rb,
            p_ml=signals.p_ml,
                p_hyb=signals.p_hyb or 0.5
        )
        
        # Get regime state
        regime_state = None
        if self._last_ohlcv is not None and len(self._last_ohlcv) > 20:
            regime_state = self._regime_detector.detect(self._last_ohlcv)
        
        # Build inputs
        inputs = PositionSizingInput(
            p_hyb=signals.p_hyb or 0.5,
            agreement=signals.agreement or 0.5,
            dispersion_state=disp_state,
            regime_state=regime_state,
            current_drawdown=self._current_drawdown,
        )
        
        # Calculate
        output = self._position_sizer.calculate(inputs)
        return output.position_size
    
    def set_ohlcv(self, df):
        """Set current OHLCV for regime detection."""
        self._last_ohlcv = df
    
    def set_drawdown(self, dd: float):
        """Update current drawdown."""
        self._current_drawdown = dd


def detect_regime(features: pd.DataFrame) -> Regime:
    """
    Detect market regime from features.
    
    Simple volatility-based classification.
    """
    if features is None or features.empty:
        return Regime.NORMAL
    
    # Try to get ATR percentage
    atr_col = None
    for col in ['atr_pct', 'ATR_pct', 'atr_pct_15m', 'ATR_pct_15m']:
        if col in features.columns:
            atr_col = col
            break
    
    if atr_col is None:
        return Regime.NORMAL
    
    atr_pct = features[atr_col].iloc[-1]
    
    if atr_pct < 0.015:
        return Regime.LOW_VOL
    elif atr_pct < 0.03:
        return Regime.NORMAL
    elif atr_pct < 0.05:
        return Regime.HIGH_VOL
    else:
        return Regime.CRISIS


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_quant_aggregator_with_strategies(
    model_path: Optional[str] = None,
    include_hybrid: bool = False,
) -> QuantAggregator:
    """
    Factory function to create aggregator with registered strategies.
    
    Args:
        model_path: Path to trained XGBoost model (optional)
        include_hybrid: Whether to include hybrid strategy
        
    Returns:
        Configured QuantAggregator
    """
    aggregator = QuantAggregator()
    
    # Try to import and register Rule-Based
    try:
        from strategies.rule_based_v2 import RuleBasedStrategy
        rb_strategy = RuleBasedStrategy()
        aggregator.register_strategy('rule_based', rb_strategy)
        logger.info("Registered Rule-Based strategy")
    except ImportError as e:
        logger.warning(f"Could not import Rule-Based strategy: {e}")
    except Exception as e:
        logger.error(f"Error initializing Rule-Based strategy: {e}")
    
    # Try to import and register XGBoost ML
    if model_path:
        try:
            from strategies.xgboost_ml_v3 import XGBoostMLStrategy
            ml_strategy = XGBoostMLStrategy()
            ml_strategy.load_model(model_path)
            aggregator.register_strategy('xgboost_ml', ml_strategy)
            logger.info(f"Registered XGBoost ML strategy from {model_path}")
        except ImportError as e:
            logger.warning(f"Could not import XGBoost ML strategy: {e}")
        except Exception as e:
            logger.error(f"Error initializing XGBoost ML strategy: {e}")
    
    # Try to import and register Hybrid
    if include_hybrid:
        try:
            from strategies.hybrid_v2 import HybridStrategy
            hyb_strategy = HybridStrategy()
            aggregator.register_strategy('hybrid', hyb_strategy)
            logger.info("Registered Hybrid strategy")
        except ImportError as e:
            logger.warning(f"Could not import Hybrid strategy: {e}")
        except Exception as e:
            logger.error(f"Error initializing Hybrid strategy: {e}")
    
    return aggregator