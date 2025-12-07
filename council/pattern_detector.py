"""
ScArlet-Sails Pattern Detector Agent

Council agent responsible for identifying market patterns.

Two implementations:
1. RuleBasedPatternDetector - Uses predefined rules (no LLM)
2. LLMPatternDetector - Uses LLM with RAG context (requires API/local model)

This agent receives CouncilContext and returns AgentOpinion.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod

from council.contracts import (
    CouncilContext,
    AgentOpinion,
    AgentRole,
    ActionType,
    SeverityLevel,
    Regime,
    QuantSignals,
    RAGContext,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PATTERN DEFINITIONS
# =============================================================================

@dataclass
class PatternDefinition:
    """Definition of a trading pattern."""
    pattern_id: str
    name: str
    description: str
    direction: ActionType  # Expected direction if pattern triggers
    
    # Conditions (all must be true)
    rsi_range: Tuple[float, float]  # (min, max)
    price_to_ma_range: Tuple[float, float]  # Relative to SMA50
    volume_ratio_min: float
    regime_allowed: List[Regime]
    
    # Risk parameters
    default_sl_pct: float
    default_tp_pct: float
    min_confidence: float
    
    # Historical stats (updated from RAG)
    historical_win_rate: Optional[float] = None
    historical_avg_pnl: Optional[float] = None
    sample_count: int = 0


# Pre-defined patterns library
PATTERN_LIBRARY: Dict[str, PatternDefinition] = {
    "ma50_bounce": PatternDefinition(
        pattern_id="ma50_bounce",
        name="MA50 Bounce",
        description="Price touches MA50 from above with oversold RSI, expecting bounce",
        direction=ActionType.LONG,
        rsi_range=(20, 40),
        price_to_ma_range=(-0.03, 0.02),  # -3% to +2% from MA50
        volume_ratio_min=1.0,
        regime_allowed=[Regime.NORMAL, Regime.LOW_VOL],
        default_sl_pct=4.0,
        default_tp_pct=8.0,
        min_confidence=0.5,
    ),
    
    "oversold_reversal": PatternDefinition(
        pattern_id="oversold_reversal",
        name="Oversold Reversal",
        description="Extreme oversold RSI with price stabilization",
        direction=ActionType.LONG,
        rsi_range=(10, 30),
        price_to_ma_range=(-0.15, 0.0),  # Below MA50
        volume_ratio_min=1.2,
        regime_allowed=[Regime.NORMAL, Regime.HIGH_VOL],
        default_sl_pct=5.0,
        default_tp_pct=10.0,
        min_confidence=0.6,
    ),
    
    "overbought_reversal": PatternDefinition(
        pattern_id="overbought_reversal",
        name="Overbought Reversal",
        description="Extreme overbought RSI with price exhaustion",
        direction=ActionType.SHORT,
        rsi_range=(70, 90),
        price_to_ma_range=(0.0, 0.15),  # Above MA50
        volume_ratio_min=1.2,
        regime_allowed=[Regime.NORMAL, Regime.HIGH_VOL],
        default_sl_pct=5.0,
        default_tp_pct=10.0,
        min_confidence=0.6,
    ),
    
    "breakout_long": PatternDefinition(
        pattern_id="breakout_long",
        name="Bullish Breakout",
        description="Price breaks above MA50 with high volume",
        direction=ActionType.LONG,
        rsi_range=(45, 70),
        price_to_ma_range=(0.01, 0.05),  # Just above MA50
        volume_ratio_min=1.5,
        regime_allowed=[Regime.NORMAL, Regime.LOW_VOL],
        default_sl_pct=3.0,
        default_tp_pct=6.0,
        min_confidence=0.55,
    ),
    
    "breakout_short": PatternDefinition(
        pattern_id="breakout_short",
        name="Bearish Breakout",
        description="Price breaks below MA50 with high volume",
        direction=ActionType.SHORT,
        rsi_range=(30, 55),
        price_to_ma_range=(-0.05, -0.01),  # Just below MA50
        volume_ratio_min=1.5,
        regime_allowed=[Regime.NORMAL, Regime.LOW_VOL],
        default_sl_pct=3.0,
        default_tp_pct=6.0,
        min_confidence=0.55,
    ),
}


# =============================================================================
# BASE PATTERN DETECTOR
# =============================================================================

class BasePatternDetector(ABC):
    """Abstract base class for pattern detectors."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.role = AgentRole.RULE  # or AgentRole.LLM for LLM detector
    
    @abstractmethod
    def detect(self, context: CouncilContext) -> AgentOpinion:
        """
        Detect patterns and return opinion.
        
        Args:
            context: Full Council context with market state, signals, RAG
            
        Returns:
            AgentOpinion with detected pattern and recommendation
        """
        pass
    
    def _create_opinion(
        self,
        action: ActionType,
        confidence: float,
        justification: str,
        pattern_id: Optional[str] = None,
        sl_pct: Optional[float] = None,
        tp_pct: Optional[float] = None,
        position_size_pct: float = 0.5,
        raw_metrics: Optional[Dict[str, Any]] = None,
    ) -> AgentOpinion:
        """Helper to create AgentOpinion."""
        return AgentOpinion(
            agent_id=self.agent_id,
            role=self.role,
            proposed_action=action,
            position_size_pct=position_size_pct,
            confidence=confidence,
            justification=justification,
            pattern_detected=pattern_id,
            suggested_sl_pct=sl_pct,
            suggested_tp_pct=tp_pct,
            raw_metrics=raw_metrics or {},
        )


# =============================================================================
# RULE-BASED PATTERN DETECTOR
# =============================================================================

class RuleBasedPatternDetector(BasePatternDetector):
    """
    Pattern detector using predefined rules.
    
    No LLM required. Fast and deterministic.
    Good baseline for comparison with LLM detector.
    """
    
    def __init__(
        self,
        agent_id: str = "pattern_detector_rule_v1",
        patterns: Optional[Dict[str, PatternDefinition]] = None,
    ):
        super().__init__(agent_id)
        self.role = AgentRole.RULE
        self.patterns = patterns or PATTERN_LIBRARY
    
    def detect(self, context: CouncilContext) -> AgentOpinion:
        """
        Detect patterns using rules.
        
        Process:
        1. Check each pattern definition against market state
        2. Score matches by how well conditions are met
        3. Boost/penalize based on quant signals agreement
        4. Boost/penalize based on RAG similar cases
        5. Return best match or HOLD if no pattern
        """
        market = context.market
        quant = context.quant_signals
        rag = context.rag
        
        # Check all patterns
        matches: List[Tuple[PatternDefinition, float, Dict[str, Any]]] = []
        
        for pattern_id, pattern in self.patterns.items():
            match_score, match_details = self._check_pattern(pattern, market, quant)
            
            if match_score > 0:
                # Adjust score based on RAG
                rag_adjustment = self._get_rag_adjustment(pattern_id, rag)
                final_score = match_score * rag_adjustment
                
                match_details['rag_adjustment'] = rag_adjustment
                match_details['final_score'] = final_score
                
                if final_score >= pattern.min_confidence:
                    matches.append((pattern, final_score, match_details))
        
        # No patterns matched
        if not matches:
            return self._create_opinion(
                action=ActionType.HOLD,
                confidence=0.3,
                justification="No clear pattern detected",
                raw_metrics={
                    "patterns_checked": len(self.patterns),
                    "rsi": market.rsi,
                    "price_to_sma50": market.price_to_sma50_pct,
                    "regime": market.regime.value,
                },
            )
        
        # Sort by score, get best match
        matches.sort(key=lambda x: x[1], reverse=True)
        best_pattern, best_score, details = matches[0]
        
        # Calculate position size based on confidence and agreement
        position_size = self._calculate_position_size(
            best_score,
            quant.agreement,
            context.constraints,
        )
        
        # Build justification
        justification = self._build_justification(best_pattern, details, quant)
        
        return self._create_opinion(
            action=best_pattern.direction,
            confidence=best_score,
            justification=justification,
            pattern_id=best_pattern.pattern_id,
            sl_pct=best_pattern.default_sl_pct,
            tp_pct=best_pattern.default_tp_pct,
            position_size_pct=position_size,
            raw_metrics=details,
        )
    
    def _check_pattern(
        self,
        pattern: PatternDefinition,
        market: Any,  # MarketSnapshot
        quant: QuantSignals,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Check if pattern conditions are met.
        
        Returns:
            (score, details) where score is 0 if not matched, 0-1 if matched
        """
        details = {
            "pattern_id": pattern.pattern_id,
            "checks": {},
        }
        
        score = 1.0
        
        # Check regime
        if market.regime not in pattern.regime_allowed:
            details["checks"]["regime"] = f"FAIL: {market.regime.value} not in {[r.value for r in pattern.regime_allowed]}"
            return 0, details
        details["checks"]["regime"] = "PASS"
        
        # Check RSI range
        rsi_min, rsi_max = pattern.rsi_range
        if not (rsi_min <= market.rsi <= rsi_max):
            details["checks"]["rsi"] = f"FAIL: {market.rsi:.1f} not in [{rsi_min}, {rsi_max}]"
            return 0, details
        
        # Score RSI (closer to center of range = higher score)
        rsi_center = (rsi_min + rsi_max) / 2
        rsi_range_half = (rsi_max - rsi_min) / 2
        rsi_distance = abs(market.rsi - rsi_center) / rsi_range_half
        rsi_score = 1.0 - (rsi_distance * 0.3)  # Max 30% penalty
        score *= rsi_score
        details["checks"]["rsi"] = f"PASS: {market.rsi:.1f} (score: {rsi_score:.2f})"
        
        # Check price to MA50
        price_min, price_max = pattern.price_to_ma_range
        price_to_ma = market.price_to_sma50_pct / 100  # Convert from % to ratio
        
        if not (price_min <= price_to_ma <= price_max):
            details["checks"]["price_to_ma"] = f"FAIL: {price_to_ma:.3f} not in [{price_min}, {price_max}]"
            return 0, details
        details["checks"]["price_to_ma"] = f"PASS: {price_to_ma:.3f}"
        
        # Check volume
        if market.volume_ratio < pattern.volume_ratio_min:
            details["checks"]["volume"] = f"FAIL: {market.volume_ratio:.2f} < {pattern.volume_ratio_min}"
            return 0, details
        
        # Score volume (higher = better, up to 2x)
        volume_score = min(market.volume_ratio / pattern.volume_ratio_min, 2.0) / 2.0
        volume_score = 0.7 + (volume_score * 0.3)  # Scale to 0.7-1.0
        score *= volume_score
        details["checks"]["volume"] = f"PASS: {market.volume_ratio:.2f} (score: {volume_score:.2f})"
        
        # Adjust by quant signals agreement
        if quant.agreement is not None:
            agreement_factor = 0.7 + (quant.agreement * 0.3)  # 0.7-1.0
            score *= agreement_factor
            details["quant_agreement"] = quant.agreement
        
        # Check if quant signals support the direction
        direction_support = self._check_quant_direction_support(pattern.direction, quant)
        score *= direction_support
        details["direction_support"] = direction_support
        
        details["base_score"] = score
        return score, details
    
    def _check_quant_direction_support(
        self,
        direction: ActionType,
        quant: QuantSignals,
    ) -> float:
        """
        Check if quant signals support the pattern direction.
        
        Returns multiplier: 0.5 (against) to 1.2 (strongly support)
        """
        signals = []
        
        for p in [quant.p_rb, quant.p_ml, quant.p_hyb]:
            if p is not None:
                signals.append(p)
        
        if not signals:
            return 1.0  # Neutral
        
        avg_signal = sum(signals) / len(signals)
        
        if direction == ActionType.LONG:
            # Higher signal = more support
            if avg_signal > 0.6:
                return 1.2  # Strong support
            elif avg_signal > 0.5:
                return 1.0  # Neutral
            elif avg_signal > 0.4:
                return 0.8  # Weak against
            else:
                return 0.5  # Against
                
        elif direction == ActionType.SHORT:
            # Lower signal = more support for short
            if avg_signal < 0.4:
                return 1.2  # Strong support
            elif avg_signal < 0.5:
                return 1.0  # Neutral
            elif avg_signal < 0.6:
                return 0.8  # Weak against
            else:
                return 0.5  # Against
        
        return 1.0
    
    def _get_rag_adjustment(
        self,
        pattern_id: str,
        rag: RAGContext,
    ) -> float:
        """
        Adjust score based on similar cases from RAG.
        
        Returns multiplier: 0.5 (bad history) to 1.3 (good history)
        """
        if not rag.similar_patterns:
            return 1.0  # No data, neutral
        
        # Find cases with same pattern
        relevant_cases = [
            p for p in rag.similar_patterns
            if p.get('pattern_id') == pattern_id
        ]
        
        if not relevant_cases:
            return 1.0  # No history for this pattern
        
        # Calculate win rate from history
        wins = sum(1 for c in relevant_cases if c.get('outcome') == 'win')
        total = len(relevant_cases)
        
        if total < 3:
            return 1.0  # Not enough data
        
        win_rate = wins / total
        
        # Convert to multiplier
        if win_rate > 0.6:
            return 1.2 + (win_rate - 0.6) * 0.5  # Up to 1.4
        elif win_rate > 0.5:
            return 1.0 + (win_rate - 0.5) * 2  # 1.0-1.2
        elif win_rate > 0.4:
            return 0.8 + (win_rate - 0.4) * 2  # 0.8-1.0
        else:
            return 0.5 + win_rate  # 0.5-0.9
    
    def _calculate_position_size(
        self,
        confidence: float,
        agreement: Optional[float],
        constraints: Any,
    ) -> float:
        """Calculate suggested position size."""
        # Base size from confidence
        base_size = 0.3 + (confidence * 0.5)  # 0.3% to 0.8%
        
        # Adjust by agreement
        if agreement is not None:
            agreement_factor = 0.7 + (agreement * 0.3)
            base_size *= agreement_factor
        
        # Cap at constraints
        max_size = constraints.max_risk_per_trade_pct
        
        return min(base_size, max_size)
    
    def _build_justification(
        self,
        pattern: PatternDefinition,
        details: Dict[str, Any],
        quant: QuantSignals,
    ) -> str:
        """Build human-readable justification."""
        parts = [
            f"Pattern: {pattern.name}",
            f"Confidence: {details.get('final_score', 0):.0%}",
        ]
        
        if quant.agreement is not None:
            parts.append(f"Quant Agreement: {quant.agreement:.0%}")
        
        rag_adj = details.get('rag_adjustment', 1.0)
        if rag_adj != 1.0:
            parts.append(f"RAG Adjustment: {rag_adj:.2f}x")
        
        return " | ".join(parts)


# =============================================================================
# LLM PATTERN DETECTOR (PLACEHOLDER)
# =============================================================================

class LLMPatternDetector(BasePatternDetector):
    """
    Pattern detector using LLM with RAG context.
    
    Requires:
    - LLM API (OpenAI, Anthropic) or local model
    - Properly formatted prompts
    - JSON parsing of responses
    
    This is a placeholder implementation.
    Full implementation requires LLM integration.
    """
    
    def __init__(
        self,
        agent_id: str = "pattern_detector_llm_v1",
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
    ):
        super().__init__(agent_id)
        self.role = AgentRole.LLM
        self.model_name = model_name
        self.api_key = api_key
        self._llm_client = None  # Initialize when needed
    
    def detect(self, context: CouncilContext) -> AgentOpinion:
        """
        Detect patterns using LLM.
        
        Falls back to rule-based if LLM unavailable.
        """
        # Try LLM detection
        try:
            if self._is_llm_available():
                return self._detect_with_llm(context)
        except Exception as e:
            logger.warning(f"LLM detection failed: {e}, falling back to rules")
        
        # Fallback to rule-based
        fallback = RuleBasedPatternDetector(agent_id=f"{self.agent_id}_fallback")
        opinion = fallback.detect(context)
        
        # Mark as fallback
        opinion.raw_metrics['llm_fallback'] = True
        return opinion
    
    def _is_llm_available(self) -> bool:
        """Check if LLM is configured and available."""
        # TODO: Implement actual check
        return False
    
    def _detect_with_llm(self, context: CouncilContext) -> AgentOpinion:
        """
        Call LLM for pattern detection.
        
        Prompt structure:
        1. System: Define role and output format
        2. User: Market state + RAG context + question
        3. Parse JSON response into AgentOpinion
        """
        # Build prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(context)
        
        # TODO: Call LLM API
        # response = self._call_llm(system_prompt, user_prompt)
        # return self._parse_response(response)
        
        # Placeholder
        raise NotImplementedError("LLM integration not implemented")
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for LLM."""
        return """You are a trading pattern detector for the ScArlet-Sails system.

Your task:
1. Analyze the market state and indicators
2. Compare with similar historical patterns from RAG
3. Identify if any known pattern is present
4. Provide a trading recommendation

You MUST respond with ONLY a JSON object in this exact format:
{
    "pattern_detected": "pattern_id or null",
    "proposed_action": "long|short|hold",
    "confidence": 0.0-1.0,
    "position_size_pct": 0.0-1.0,
    "suggested_sl_pct": number or null,
    "suggested_tp_pct": number or null,
    "justification": "1-2 sentence explanation"
}

Known patterns:
- ma50_bounce: Price touches MA50 with oversold RSI
- oversold_reversal: Extreme oversold with stabilization
- overbought_reversal: Extreme overbought with exhaustion
- breakout_long: Bullish break above MA50 with volume
- breakout_short: Bearish break below MA50 with volume

Be conservative. Only identify a pattern if you are reasonably confident.
If unsure, return "hold" with explanation."""
    
    def _build_user_prompt(self, context: CouncilContext) -> str:
        """Build user prompt with context."""
        market = context.market.to_dict()
        quant = context.quant_signals.to_dict()
        rag_text = context.rag.to_prompt_text()
        
        return f"""## Current Market State
Symbol: {market['symbol']}
Timeframe: {market['timeframe']}
Regime: {market['regime']}

## Indicators
RSI: {market['rsi']:.1f}
Price vs EMA9: {market['price_to_ema9_pct']:.2f}%
Price vs EMA21: {market['price_to_ema21_pct']:.2f}%
Price vs SMA50: {market['price_to_sma50_pct']:.2f}%
ATR: {market['atr_pct']:.2f}%
Volume Ratio: {market['volume_ratio']:.2f}

## Quant Strategy Signals
P_rb (Rule-Based): {quant['p_rb']}
P_ml (XGBoost): {quant['p_ml']}
P_hyb (Hybrid): {quant['p_hyb']}
Agreement: {quant['agreement']}

## Historical Context (RAG)
{rag_text}

## Screenshot Description
{context.rag.screenshot_description or 'Not available'}

Based on this information, identify any trading pattern and provide your recommendation."""


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_pattern_detector(
    use_llm: bool = False,
    llm_model: str = "gpt-4",
    api_key: Optional[str] = None,
) -> BasePatternDetector:
    """
    Factory function to create appropriate pattern detector.
    
    Args:
        use_llm: Whether to use LLM-based detection
        llm_model: LLM model name (if use_llm=True)
        api_key: API key for LLM (if use_llm=True)
        
    Returns:
        Pattern detector instance
    """
    if use_llm:
        return LLMPatternDetector(
            model_name=llm_model,
            api_key=api_key,
        )
    else:
        return RuleBasedPatternDetector()
```
