"""
ScArlet-Sails Council Package

Phase 1 exports for Council decision-making framework.
"""

# =============================================================================
# CONTRACTS - Core data structures (Stage 0/1/2/3)
# =============================================================================
from council.contracts import (
    # Enums
    AgentRole,
    ActionType,
    SeverityLevel,
    Regime,
    HumanDecision,
    
    # Stage 0: Input Context
    MarketSnapshot,
    PositionState,
    RiskConstraints,
    QuantSignals,
    RAGContext,
    CouncilContext,
    
    # Stage 1: Agent Opinions
    AgentOpinion,
    
    # Stage 2: Peer Review
    AgentReview,
    
    # Stage 3: Aggregated Recommendation
    CouncilRecommendation,
    
    # Human Response
    HumanResponse,
    
    # Trade Log
    TradeLogEntry,
    
    # Validation helpers
    validate_opinion,
    validate_recommendation,
)

# =============================================================================
# QUANT AGGREGATOR
# =============================================================================
from council.quant_aggregator import QuantAggregator

# =============================================================================
# PATTERN DETECTOR
# =============================================================================
from council.pattern_detector import (
    RuleBasedPatternDetector,
    # LLMPatternDetector would be imported here when implemented
)

# =============================================================================
# BASE AGENT
# =============================================================================
from council.base_agent import BaseAgent

__all__ = [
    # Enums
    "AgentRole",
    "ActionType",
    "SeverityLevel",
    "Regime",
    "HumanDecision",
    
    # Stage 0
    "MarketSnapshot",
    "PositionState",
    "RiskConstraints",
    "QuantSignals",
    "RAGContext",
    "CouncilContext",
    
    # Stage 1
    "AgentOpinion",
    
    # Stage 2
    "AgentReview",
    
    # Stage 3
    "CouncilRecommendation",
    
    # Human
    "HumanResponse",
    
    # Trade Log
    "TradeLogEntry",
    
    # Validation
    "validate_opinion",
    "validate_recommendation",
    
    # Agents
    "BaseAgent",
    "QuantAggregator",
    "RuleBasedPatternDetector",
]
