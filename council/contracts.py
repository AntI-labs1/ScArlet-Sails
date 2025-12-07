"""
ScArlet-Sails Council Contracts

Strict type definitions and procedures for Council decision-making.
Separates "what" (contracts) from "how" (implementations).

Architecture based on llm-council pattern:
- Stage 0: Input context
- Stage 1: Agent opinions
- Stage 2: Peer review
- Stage 3: Aggregation (Chairman)

Any agent (Quant/LLM/Rule/Risk/Human) must return data
in these formats. No exceptions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional
import json


# =============================================================================
# ENUMS
# =============================================================================

class AgentRole(Enum):
    """Roles in the Council."""
    QUANT = "quant"           # Quantitative strategies (P_rb, P_ml, P_hyb)
    LLM = "llm"               # LLM-based pattern detection
    RULE = "rule"             # Rule-based pattern detection
    RISK = "risk"             # Risk assessment
    HUMAN = "human"           # Human-in-the-loop
    CONTRARIAN = "contrarian" # Devil's advocate


class ActionType(Enum):
    """Possible trading actions."""
    LONG = "long"
    SHORT = "short"
    HOLD = "hold"
    CLOSE = "close"
    REDUCE = "reduce"


class SeverityLevel(Enum):
    """Risk severity levels."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class Regime(Enum):
    """Market regime classification."""
    LOW_VOL = "low_vol"
    NORMAL = "normal"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"


# =============================================================================
# STAGE 0: INPUT CONTEXT
# =============================================================================

@dataclass
class MarketSnapshot:
    """
    Current market state snapshot.
    Subset of CanonicalState for Council consumption.
    """
    symbol: str
    timeframe: str
    timestamp: datetime
    
    # Price data
    current_price: float
    spread_pct: float
    volume_24h: float
    
    # Technical features (aggregated)
    rsi: float
    price_to_ema9_pct: float
    price_to_ema21_pct: float
    price_to_sma50_pct: float
    atr_pct: float
    bb_width_pct: float
    volume_ratio: float
    
    # Derived
    regime: Regime
    opportunity_score: float
    risk_penalty: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat(),
            "current_price": self.current_price,
            "spread_pct": self.spread_pct,
            "volume_24h": self.volume_24h,
            "rsi": self.rsi,
            "price_to_ema9_pct": self.price_to_ema9_pct,
            "price_to_ema21_pct": self.price_to_ema21_pct,
            "price_to_sma50_pct": self.price_to_sma50_pct,
            "atr_pct": self.atr_pct,
            "bb_width_pct": self.bb_width_pct,
            "volume_ratio": self.volume_ratio,
            "regime": self.regime.value,
            "opportunity_score": self.opportunity_score,
            "risk_penalty": self.risk_penalty,
        }


@dataclass
class PositionState:
    """Current position state."""
    current_action: ActionType  # LONG, SHORT, or HOLD (flat)
    size: float                 # Position size (0 if flat)
    entry_price: Optional[float]
    unrealized_pnl_pct: float
    leverage: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_action": self.current_action.value,
            "size": self.size,
            "entry_price": self.entry_price,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "leverage": self.leverage,
        }


@dataclass
class RiskConstraints:
    """Risk limits for this decision."""
    max_position_size_pct: float = 10.0
    max_risk_per_trade_pct: float = 0.5
    max_leverage: float = 1.0
    daily_loss_remaining_pct: float = 3.0
    weekly_loss_remaining_pct: float = 7.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_position_size_pct": self.max_position_size_pct,
            "max_risk_per_trade_pct": self.max_risk_per_trade_pct,
            "max_leverage": self.max_leverage,
            "daily_loss_remaining_pct": self.daily_loss_remaining_pct,
            "weekly_loss_remaining_pct": self.weekly_loss_remaining_pct,
        }


@dataclass
class QuantSignals:
    """Signals from quantitative strategies."""
    p_rb: Optional[float] = None   # Rule-based probability
    p_ml: Optional[float] = None   # ML probability
    p_hyb: Optional[float] = None  # Hybrid probability
    agreement: Optional[float] = None  # 1 - max_spread
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "p_rb": self.p_rb,
            "p_ml": self.p_ml,
            "p_hyb": self.p_hyb,
            "agreement": self.agreement,
        }
    
    def compute_agreement(self) -> float:
        """Calculate agreement between available signals."""
        values = [v for v in [self.p_rb, self.p_ml, self.p_hyb] if v is not None]
        if len(values) < 2:
            return 1.0
        spread = max(values) - min(values)
        self.agreement = 1.0 - min(spread, 1.0)
        return self.agreement


@dataclass
class RAGContext:
    """Context retrieved from RAG."""
    similar_patterns: List[Dict[str, Any]] = field(default_factory=list)
    recent_trades: List[Dict[str, Any]] = field(default_factory=list)
    relevant_lessons: List[Dict[str, Any]] = field(default_factory=list)
    screenshot_description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "similar_patterns": self.similar_patterns,
            "recent_trades": self.recent_trades,
            "relevant_lessons": self.relevant_lessons,
            "screenshot_description": self.screenshot_description,
        }
    
    def to_prompt_text(self) -> str:
        """Format as text for LLM prompt."""
        lines = []
        
        if self.similar_patterns:
            lines.append("## Similar Patterns")
            for p in self.similar_patterns[:3]:
                lines.append(f"- {p.get('pattern_id', 'unknown')}: "
                           f"outcome={p.get('outcome', 'N/A')}, "
                           f"pnl={p.get('pnl_pct', 'N/A')}%")
        
        if self.recent_trades:
            lines.append("\n## Recent Trades")
            for t in self.recent_trades[:3]:
                lines.append(f"- {t.get('action', 'N/A')}: "
                           f"outcome={t.get('outcome', 'N/A')}, "
                           f"pnl={t.get('pnl_pct', 'N/A')}%")
        
        if self.relevant_lessons:
            lines.append("\n## Lessons")
            for l in self.relevant_lessons[:3]:
                lines.append(f"- {l.get('lesson', 'N/A')}")
        
        if self.screenshot_description:
            lines.append(f"\n## Chart Description\n{self.screenshot_description}")
        
        return "\n".join(lines)


@dataclass
class CouncilContext:
    """
    STAGE 0: Complete input context for Council.
    
    This is the single input structure that all agents receive.
    Agents must not request additional data outside this contract.
    """
    # Market state
    market: MarketSnapshot
    position: PositionState
    
    # Constraints
    constraints: RiskConstraints
    
    # Signals from quant strategies
    quant_signals: QuantSignals
    
    # RAG context
    rag: RAGContext
    
    # Metadata
    request_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "market": self.market.to_dict(),
            "position": self.position.to_dict(),
            "constraints": self.constraints.to_dict(),
            "quant_signals": self.quant_signals.to_dict(),
            "rag": self.rag.to_dict(),
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


# =============================================================================
# STAGE 1: AGENT OPINIONS
# =============================================================================

@dataclass
class AgentOpinion:
    """
    STAGE 1: Opinion from a single Council agent.
    
    Every agent (Quant, LLM, Rule, Risk, Human) MUST return
    this exact structure. No variations allowed.
    """
    # Identity
    agent_id: str              # e.g., "quant_xgboost_v3", "llm_pattern_detector"
    role: AgentRole
    
    # Decision
    proposed_action: ActionType
    position_size_pct: float   # Suggested position size (% of equity)
    confidence: float          # 0.0 to 1.0
    
    # Explanation (required for audit trail)
    justification: str         # Human-readable, 1-3 sentences
    
    # Optional details
    suggested_sl_pct: Optional[float] = None  # Stop loss distance
    suggested_tp_pct: Optional[float] = None  # Take profit distance
    pattern_detected: Optional[str] = None    # Pattern ID if applicable
    raw_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "proposed_action": self.proposed_action.value,
            "position_size_pct": self.position_size_pct,
            "confidence": self.confidence,
            "justification": self.justification,
            "suggested_sl_pct": self.suggested_sl_pct,
            "suggested_tp_pct": self.suggested_tp_pct,
            "pattern_detected": self.pattern_detected,
            "raw_metrics": self.raw_metrics,
            "timestamp": self.timestamp.isoformat(),
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentOpinion':
        """Parse from dict (e.g., LLM JSON response)."""
        return cls(
            agent_id=data.get("agent_id", "unknown"),
            role=AgentRole(data.get("role", "rule")),
            proposed_action=ActionType(data.get("proposed_action", "hold")),
            position_size_pct=float(data.get("position_size_pct", 0)),
            confidence=float(data.get("confidence", 0)),
            justification=data.get("justification", ""),
            suggested_sl_pct=data.get("suggested_sl_pct"),
            suggested_tp_pct=data.get("suggested_tp_pct"),
            pattern_detected=data.get("pattern_detected"),
            raw_metrics=data.get("raw_metrics", {}),
        )


# =============================================================================
# STAGE 2: PEER REVIEW
# =============================================================================

@dataclass
class AgentReview:
    """
    STAGE 2: Review of one agent's opinion by another.
    
    Used for:
    - Risk agent reviewing Quant/LLM opinions
    - Contrarian agent challenging assumptions
    - LLM agent evaluating other opinions
    """
    # Who is reviewing whom
    reviewer_id: str
    reviewer_role: AgentRole
    target_agent_id: str
    
    # Assessment
    score: float               # 0.0 to 1.0 (quality/agreement score)
    risk_flag: SeverityLevel   # Risk concern level
    
    # Feedback
    comments: str              # Specific concerns or endorsement
    suggested_adjustments: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reviewer_id": self.reviewer_id,
            "reviewer_role": self.reviewer_role.value,
            "target_agent_id": self.target_agent_id,
            "score": self.score,
            "risk_flag": self.risk_flag.value,
            "comments": self.comments,
            "suggested_adjustments": self.suggested_adjustments,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# STAGE 3: AGGREGATED RECOMMENDATION
# =============================================================================

@dataclass
class CouncilRecommendation:
    """
    STAGE 3: Final aggregated recommendation from Council.
    
    This is what the Human sees and decides on.
    """
    # Final decision
    final_action: ActionType
    final_position_size_pct: float
    aggregate_confidence: float
    
    # Risk assessment
    risk_level: SeverityLevel
    sl_pct: Optional[float] = None
    tp_pct: Optional[float] = None
    
    # Explanation
    rationale: str             # Summary of why this decision
    dissent_summary: Optional[str] = None  # Any disagreements
    
    # Constraint violations (if any forced adjustments)
    violated_constraints: List[str] = field(default_factory=list)
    
    # Audit trail
    opinions_used: List[AgentOpinion] = field(default_factory=list)
    reviews: List[AgentReview] = field(default_factory=list)
    
    # Metadata
    request_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "final_action": self.final_action.value,
            "final_position_size_pct": self.final_position_size_pct,
            "aggregate_confidence": self.aggregate_confidence,
            "risk_level": self.risk_level.value,
            "sl_pct": self.sl_pct,
            "tp_pct": self.tp_pct,
            "rationale": self.rationale,
            "dissent_summary": self.dissent_summary,
            "violated_constraints": self.violated_constraints,
            "opinions_count": len(self.opinions_used),
            "reviews_count": len(self.reviews),
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def to_human_display(self) -> str:
        """Format for CLI display to human."""
        lines = [
            "=" * 50,
            "COUNCIL RECOMMENDATION",
            "=" * 50,
            f"Action: {self.final_action.value.upper()}",
            f"Position Size: {self.final_position_size_pct:.1f}%",
            f"Confidence: {self.aggregate_confidence:.0%}",
            f"Risk Level: {self.risk_level.name}",
            "",
        ]
        
        if self.sl_pct or self.tp_pct:
            lines.append(f"SL: {self.sl_pct}% | TP: {self.tp_pct}%")
            lines.append("")
        
        lines.append(f"Rationale: {self.rationale}")
        
        if self.dissent_summary:
            lines.append(f"\nDissent: {self.dissent_summary}")
        
        if self.violated_constraints:
            lines.append(f"\n⚠️  Constraints: {', '.join(self.violated_constraints)}")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)


# =============================================================================
# HUMAN DECISION
# =============================================================================

class HumanDecision(Enum):
    """Human response to Council recommendation."""
    ACCEPT = "accept"
    MODIFY = "modify"
    REJECT = "reject"
    SKIP = "skip"


@dataclass
class HumanResponse:
    """
    Human operator's response to Council recommendation.
    Logged to RAG for future learning.
    """
    decision: HumanDecision
    
    # If modified
    modified_action: Optional[ActionType] = None
    modified_size_pct: Optional[float] = None
    modified_sl_pct: Optional[float] = None
    modified_tp_pct: Optional[float] = None
    
    # Reasoning (important for learning)
    reasoning: str = ""
    
    # Reference
    request_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "decision": self.decision.value,
            "modified_action": self.modified_action.value if self.modified_action else None,
            "modified_size_pct": self.modified_size_pct,
            "modified_sl_pct": self.modified_sl_pct,
            "modified_tp_pct": self.modified_tp_pct,
            "reasoning": self.reasoning,
        }


# =============================================================================
# TRADE LOG ENTRY
# =============================================================================

@dataclass
class TradeLogEntry:
    """
    Complete record of a trading decision.
    Stored in rag/trades/trade_log.json
    """
    # Identifiers
    trade_id: str
    request_id: str
    timestamp: datetime
    
    # Context snapshot
    symbol: str
    timeframe: str
    regime: str
    
    # Quant signals at decision time
    p_rb: Optional[float]
    p_ml: Optional[float]
    p_hyb: Optional[float]
    agreement: Optional[float]
    
    # Council recommendation
    council_action: str
    council_confidence: float
    council_rationale: str
    
    # Human decision
    human_decision: str
    human_reasoning: str
    
    # Execution (filled after trade)
    executed_action: Optional[str] = None
    executed_size_pct: Optional[float] = None
    entry_price: Optional[float] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    
    # Outcome (filled after close)
    outcome: Optional[str] = None  # "win", "loss", "breakeven"
    pnl_pct: Optional[float] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # "tp", "sl", "manual", "timeout"
    duration_hours: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "regime": self.regime,
            "p_rb": self.p_rb,
            "p_ml": self.p_ml,
            "p_hyb": self.p_hyb,
            "agreement": self.agreement,
            "council_action": self.council_action,
            "council_confidence": self.council_confidence,
            "council_rationale": self.council_rationale,
            "human_decision": self.human_decision,
            "human_reasoning": self.human_reasoning,
            "executed_action": self.executed_action,
            "executed_size_pct": self.executed_size_pct,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "outcome": self.outcome,
            "pnl_pct": self.pnl_pct,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "duration_hours": self.duration_hours,
        }


# =============================================================================
# PROTOCOL INTERFACES
# =============================================================================

def validate_opinion(opinion: AgentOpinion) -> List[str]:
    """
    Validate that opinion follows contract.
    Returns list of violations (empty if valid).
    """
    violations = []
    
    if not opinion.agent_id:
        violations.append("agent_id is required")
    
    if not 0 <= opinion.confidence <= 1:
        violations.append(f"confidence must be 0-1, got {opinion.confidence}")
    
    if opinion.position_size_pct < 0:
        violations.append(f"position_size_pct must be >= 0, got {opinion.position_size_pct}")
    
    if not opinion.justification:
        violations.append("justification is required")
    
    return violations


def validate_recommendation(rec: CouncilRecommendation, constraints: RiskConstraints) -> List[str]:
    """
    Validate recommendation against constraints.
    Returns list of violations.
    """
    violations = []
    
    if rec.final_position_size_pct > constraints.max_position_size_pct:
        violations.append(f"position_size {rec.final_position_size_pct}% > max {constraints.max_position_size_pct}%")
    
    if constraints.daily_loss_remaining_pct <= 0:
        violations.append("daily loss limit reached")
    
    if constraints.weekly_loss_remaining_pct <= 0:
        violations.append("weekly loss limit reached")
    
    return violations
```

---

## ИНСТРУКЦИЯ: КАК ВСТАВИТЬ В GITHUB ВРУЧНУЮ

### Способ 1: Через веб-интерфейс GitHub

1. Открой https://github.com/AntI-labs1/ScArlet-Sails

2. Перейди в папку `council/`

3. Нажми **Add file** → **Create new file**

4. В поле имени файла напиши: `contracts.py`

5. Скопируй весь код выше и вставь в редактор

6. Внизу страницы:
   - Commit message: `feat: Add Council contracts (Stage 0/1/2/3)`
   - Description: 
```
   Strict type definitions for Council decision-making:
   - Stage 0: CouncilContext (market, position, constraints, quant signals, RAG)
   - Stage 1: AgentOpinion (action, confidence, justification)
   - Stage 2: AgentReview (peer review, risk flags)
   - Stage 3: CouncilRecommendation (final aggregated decision)
   - HumanResponse and TradeLogEntry for audit trail
   
   All agents must return data in these formats. No exceptions.
