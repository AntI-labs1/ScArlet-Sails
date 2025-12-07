"""
ScArlet-Sails Human Interface (CLI)

Command-line interface for human-in-the-loop decision making.

Displays Council recommendation and allows human to:
- Accept: Execute as recommended
- Modify: Change parameters (size, SL/TP)
- Reject: Skip with reason logged
- Skip: Skip without logging

All decisions are logged to RAG for future learning.
"""

import json
import logging
from typing import Optional, Tuple
from datetime import datetime
from pathlib import Path
import uuid

from council.contracts import (
    CouncilRecommendation,
    HumanResponse,
    HumanDecision,
    ActionType,
    SeverityLevel,
    TradeLogEntry,
    CouncilContext,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    # Colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    
    # Background
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


def colorize(text: str, color: str) -> str:
    """Add color to text."""
    return f"{color}{text}{Colors.RESET}"


def bold(text: str) -> str:
    """Make text bold."""
    return f"{Colors.BOLD}{text}{Colors.RESET}"


def print_separator(char: str = "=", length: int = 60) -> None:
    """Print a separator line."""
    print(char * length)


def print_header(title: str) -> None:
    """Print a header with separators."""
    print_separator("=")
    print(colorize(bold(f"  {title}"), Colors.CYAN))
    print_separator("=")


def print_section(title: str) -> None:
    """Print a section header."""
    print()
    print(colorize(bold(f"▶ {title}"), Colors.YELLOW))
    print_separator("-", 40)


# =============================================================================
# RECOMMENDATION DISPLAY
# =============================================================================

def display_recommendation(
    recommendation: CouncilRecommendation,
    context: Optional[CouncilContext] = None,
) -> None:
    """
    Display Council recommendation in formatted CLI output.
    
    Args:
        recommendation: CouncilRecommendation from Council
        context: Optional CouncilContext for additional info
    """
    print()
    print_header("COUNCIL RECOMMENDATION")
    
    # Main action
    action_color = _get_action_color(recommendation.final_action)
    action_text = recommendation.final_action.value.upper()
    
    print()
    print(f"  Action:      {colorize(bold(action_text), action_color)}")
    print(f"  Position:    {recommendation.final_position_size_pct:.2f}% of equity")
    print(f"  Confidence:  {_format_confidence(recommendation.aggregate_confidence)}")
    print(f"  Risk Level:  {_format_risk(recommendation.risk_level)}")
    
    # SL/TP if available
    if recommendation.sl_pct or recommendation.tp_pct:
        print()
        if recommendation.sl_pct:
            print(f"  Stop Loss:   {colorize(f'-{recommendation.sl_pct}%', Colors.RED)}")
        if recommendation.tp_pct:
            print(f"  Take Profit: {colorize(f'+{recommendation.tp_pct}%', Colors.GREEN)}")
    
    # Rationale
    print_section("Rationale")
    print(f"  {recommendation.rationale}")
    
    # Dissent if any
    if recommendation.dissent_summary:
        print_section("Dissent")
        print(f"  {colorize(recommendation.dissent_summary, Colors.YELLOW)}")
    
    # Constraint violations if any
    if recommendation.violated_constraints:
        print_section("⚠️  Constraint Warnings")
        for violation in recommendation.violated_constraints:
            print(f"  • {colorize(violation, Colors.RED)}")
    
    # Quant signals from context
    if context and context.quant_signals:
        print_section("Quant Signals")
        qs = context.quant_signals
        
        if qs.p_rb is not None:
            print(f"  P_rb (Rule-Based): {_format_signal(qs.p_rb)}")
        if qs.p_ml is not None:
            print(f"  P_ml (XGBoost):    {_format_signal(qs.p_ml)}")
        if qs.p_hyb is not None:
            print(f"  P_hyb (Hybrid):    {_format_signal(qs.p_hyb)}")
        if qs.agreement is not None:
            print(f"  Agreement:         {_format_confidence(qs.agreement)}")
    
    # Market info from context
    if context and context.market:
        print_section("Market State")
        m = context.market
        print(f"  Symbol:    {m.symbol} | {m.timeframe}")
        print(f"  Regime:    {m.regime.value}")
        print(f"  RSI:       {m.rsi:.1f}")
        print(f"  ATR:       {m.atr_pct:.2f}%")
    
    # Similar cases from RAG
    if context and context.rag and context.rag.similar_patterns:
        print_section("Similar Historical Cases")
        for i, case in enumerate(context.rag.similar_patterns[:3], 1):
            outcome = case.get('outcome', 'N/A')
            outcome_color = Colors.GREEN if outcome == 'win' else Colors.RED if outcome == 'loss' else Colors.WHITE
            pnl = case.get('pnl_pct', 'N/A')
            pattern = case.get('pattern_id', 'unknown')
            print(f"  {i}. {pattern}: {colorize(outcome, outcome_color)} ({pnl}%)")
    
    print()
    print_separator("=")


def _get_action_color(action: ActionType) -> str:
    """Get color for action."""
    if action == ActionType.LONG:
        return Colors.GREEN
    elif action == ActionType.SHORT:
        return Colors.RED
    elif action in [ActionType.CLOSE, ActionType.REDUCE]:
        return Colors.YELLOW
    else:
        return Colors.WHITE


def _format_confidence(value: float) -> str:
    """Format confidence with color."""
    pct = f"{value:.0%}"
    if value >= 0.7:
        return colorize(pct, Colors.GREEN)
    elif value >= 0.5:
        return colorize(pct, Colors.YELLOW)
    else:
        return colorize(pct, Colors.RED)


def _format_risk(level: SeverityLevel) -> str:
    """Format risk level with color."""
    name = level.name
    if level == SeverityLevel.NONE:
        return colorize(name, Colors.GREEN)
    elif level == SeverityLevel.LOW:
        return colorize(name, Colors.CYAN)
    elif level == SeverityLevel.MEDIUM:
        return colorize(name, Colors.YELLOW)
    elif level == SeverityLevel.HIGH:
        return colorize(name, Colors.RED)
    else:  # CRITICAL
        return colorize(bold(name), Colors.BG_RED + Colors.WHITE)


def _format_signal(value: float) -> str:
    """Format signal value with color."""
    text = f"{value:.2f}"
    if value > 0.6:
        return colorize(text, Colors.GREEN) + " (bullish)"
    elif value < 0.4:
        return colorize(text, Colors.RED) + " (bearish)"
    else:
        return colorize(text, Colors.WHITE) + " (neutral)"


# =============================================================================
# HUMAN INPUT
# =============================================================================

def get_human_decision(
    recommendation: CouncilRecommendation,
) -> HumanResponse:
    """
    Get human decision via CLI input.
    
    Args:
        recommendation: CouncilRecommendation to decide on
        
    Returns:
        HumanResponse with decision and optional modifications
    """
    print()
    print(colorize(bold("YOUR DECISION"), Colors.MAGENTA))
    print()
    print("  [A] Accept   - Execute as recommended")
    print("  [M] Modify   - Change parameters")
    print("  [R] Reject   - Skip and log reason")
    print("  [S] Skip     - Skip without logging")
    print()
    
    while True:
        choice = input(colorize("  Enter choice (A/M/R/S): ", Colors.CYAN)).strip().upper()
        
        if choice == 'A':
            return _handle_accept(recommendation)
        elif choice == 'M':
            return _handle_modify(recommendation)
        elif choice == 'R':
            return _handle_reject(recommendation)
        elif choice == 'S':
            return _handle_skip(recommendation)
        else:
            print(colorize("  Invalid choice. Please enter A, M, R, or S.", Colors.RED))


def _handle_accept(recommendation: CouncilRecommendation) -> HumanResponse:
    """Handle Accept decision."""
    reasoning = input("  Reason (optional, press Enter to skip): ").strip()
    
    return HumanResponse(
        decision=HumanDecision.ACCEPT,
        reasoning=reasoning or "Accepted Council recommendation",
        request_id=recommendation.request_id,
    )


def _handle_modify(recommendation: CouncilRecommendation) -> HumanResponse:
    """Handle Modify decision."""
    print()
    print("  Current values:")
    print(f"    Action: {recommendation.final_action.value}")
    print(f"    Size:   {recommendation.final_position_size_pct:.2f}%")
    print(f"    SL:     {recommendation.sl_pct}%")
    print(f"    TP:     {recommendation.tp_pct}%")
    print()
    
    # Get modifications
    modified_action = None
    modified_size = None
    modified_sl = None
    modified_tp = None
    
    # Action
    action_input = input("  New action (long/short/hold, Enter to keep): ").strip().lower()
    if action_input:
        try:
            modified_action = ActionType(action_input)
        except ValueError:
            print(colorize(f"  Invalid action '{action_input}', keeping original", Colors.YELLOW))
    
    # Size
    size_input = input("  New size % (Enter to keep): ").strip()
    if size_input:
        try:
            modified_size = float(size_input)
        except ValueError:
            print(colorize(f"  Invalid size '{size_input}', keeping original", Colors.YELLOW))
    
    # SL
    sl_input = input("  New SL % (Enter to keep): ").strip()
    if sl_input:
        try:
            modified_sl = float(sl_input)
        except ValueError:
            print(colorize(f"  Invalid SL '{sl_input}', keeping original", Colors.YELLOW))
    
    # TP
    tp_input = input("  New TP % (Enter to keep): ").strip()
    if tp_input:
        try:
            modified_tp = float(tp_input)
        except ValueError:
            print(colorize(f"  Invalid TP '{tp_input}', keeping original", Colors.YELLOW))
    
    # Reasoning
    reasoning = input("  Reason for modification: ").strip()
    
    return HumanResponse(
        decision=HumanDecision.MODIFY,
        modified_action=modified_action,
        modified_size_pct=modified_size,
        modified_sl_pct=modified_sl,
        modified_tp_pct=modified_tp,
        reasoning=reasoning or "Modified parameters",
        request_id=recommendation.request_id,
    )


def _handle_reject(recommendation: CouncilRecommendation) -> HumanResponse:
    """Handle Reject decision."""
    reasoning = input("  Reason for rejection (required): ").strip()
    
    while not reasoning:
        print(colorize("  Reason is required for rejection.", Colors.RED))
        reasoning = input("  Reason for rejection: ").strip()
    
    return HumanResponse(
        decision=HumanDecision.REJECT,
        reasoning=reasoning,
        request_id=recommendation.request_id,
    )


def _handle_skip(recommendation: CouncilRecommendation) -> HumanResponse:
    """Handle Skip decision."""
    return HumanResponse(
        decision=HumanDecision.SKIP,
        reasoning="Skipped without logging",
        request_id=recommendation.request_id,
    )


# =============================================================================
# RAG LOGGING
# =============================================================================

class DecisionLogger:
    """
    Logs human decisions to RAG for future learning.
    """
    
    def __init__(self, rag_path: str = "rag/trades/trade_log.json"):
        self.rag_path = Path(rag_path)
        self._ensure_file_exists()
    
    def _ensure_file_exists(self) -> None:
        """Ensure trade log file exists."""
        if not self.rag_path.exists():
            self.rag_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_log({"version": "1.0", "trades": []})
    
    def _load_log(self) -> dict:
        """Load trade log from disk."""
        try:
            with open(self.rag_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load trade log: {e}")
            return {"version": "1.0", "trades": []}
    
    def _save_log(self, data: dict) -> None:
        """Save trade log to disk."""
        try:
            with open(self.rag_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save trade log: {e}")
    
    def log_decision(
        self,
        recommendation: CouncilRecommendation,
        response: HumanResponse,
        context: Optional[CouncilContext] = None,
    ) -> str:
        """
        Log a decision to RAG.
        
        Args:
            recommendation: Council's recommendation
            response: Human's response
            context: Optional context for additional info
            
        Returns:
            trade_id of logged entry
        """
        # Skip logging for SKIP decisions
        if response.decision == HumanDecision.SKIP:
            return ""
        
        trade_id = str(uuid.uuid4())[:8]
        
        # Build log entry
        entry = {
            "trade_id": trade_id,
            "request_id": recommendation.request_id,
            "timestamp": datetime.utcnow().isoformat(),
            
            # Context
            "symbol": context.market.symbol if context else "unknown",
            "timeframe": context.market.timeframe if context else "unknown",
            "regime": context.market.regime.value if context else "unknown",
            
            # Quant signals
            "p_rb": context.quant_signals.p_rb if context else None,
            "p_ml": context.quant_signals.p_ml if context else None,
            "p_hyb": context.quant_signals.p_hyb if context else None,
            "agreement": context.quant_signals.agreement if context else None,
            
            # Council recommendation
            "council_action": recommendation.final_action.value,
            "council_size_pct": recommendation.final_position_size_pct,
            "council_confidence": recommendation.aggregate_confidence,
            "council_sl_pct": recommendation.sl_pct,
            "council_tp_pct": recommendation.tp_pct,
            "council_rationale": recommendation.rationale,
            
            # Human decision
            "human_decision": response.decision.value,
            "human_reasoning": response.reasoning,
            
            # Modifications (if any)
            "modified_action": response.modified_action.value if response.modified_action else None,
            "modified_size_pct": response.modified_size_pct,
            "modified_sl_pct": response.modified_sl_pct,
            "modified_tp_pct": response.modified_tp_pct,
            
            # Outcome (to be filled later)
            "executed": response.decision in [HumanDecision.ACCEPT, HumanDecision.MODIFY],
            "outcome": None,
            "pnl_pct": None,
            "exit_reason": None,
        }
        
        # Load, append, save
        log_data = self._load_log()
        log_data["trades"].append(entry)
        log_data["last_updated"] = datetime.utcnow().isoformat()
        self._save_log(log_data)
        
        logger.info(f"Logged decision {trade_id}: {response.decision.value}")
        
        return trade_id
    
    def update_outcome(
        self,
        trade_id: str,
        outcome: str,
        pnl_pct: float,
        exit_reason: str,
    ) -> bool:
        """
        Update trade outcome after position is closed.
        
        Args:
            trade_id: Trade ID to update
            outcome: "win", "loss", or "breakeven"
            pnl_pct: Actual P&L percentage
            exit_reason: "tp", "sl", "manual", "timeout"
            
        Returns:
            True if updated, False if trade not found
        """
        log_data = self._load_log()
        
        for trade in log_data["trades"]:
            if trade["trade_id"] == trade_id:
                trade["outcome"] = outcome
                trade["pnl_pct"] = pnl_pct
                trade["exit_reason"] = exit_reason
                trade["closed_at"] = datetime.utcnow().isoformat()
                
                self._save_log(log_data)
                logger.info(f"Updated outcome for {trade_id}: {outcome} ({pnl_pct}%)")
                return True
        
        logger.warning(f"Trade {trade_id} not found for outcome update")
        return False


# =============================================================================
# MAIN CLI FUNCTION
# =============================================================================

def run_decision_cli(
    recommendation: CouncilRecommendation,
    context: Optional[CouncilContext] = None,
    log_to_rag: bool = True,
) -> Tuple[HumanResponse, Optional[str]]:
    """
    Run the full CLI decision flow.
    
    Args:
        recommendation: CouncilRecommendation to decide on
        context: Optional context for display
        log_to_rag: Whether to log decision to RAG
        
    Returns:
        (HumanResponse, trade_id or None)
    """
    # Display recommendation
    display_recommendation(recommendation, context)
    
    # Get human decision
    response = get_human_decision(recommendation)
    
    # Log if requested
    trade_id = None
    if log_to_rag:
        logger_instance = DecisionLogger()
        trade_id = logger_instance.log_decision(recommendation, response, context)
    
    # Confirmation
    print()
    if response.decision == HumanDecision.ACCEPT:
        print(colorize("✓ Decision: ACCEPT", Colors.GREEN))
    elif response.decision == HumanDecision.MODIFY:
        print(colorize("✓ Decision: MODIFY", Colors.YELLOW))
    elif response.decision == HumanDecision.REJECT:
        print(colorize("✗ Decision: REJECT", Colors.RED))
    else:
        print(colorize("○ Decision: SKIP", Colors.WHITE))
    
    if trade_id:
        print(f"  Trade ID: {trade_id}")
    
    print()
    
    return response, trade_id


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    """Test CLI with mock data."""
    from council.contracts import (
        MarketSnapshot,
        PositionState,
        RiskConstraints,
        QuantSignals,
        RAGContext,
        Regime,
    )
    
    # Create mock recommendation
    recommendation = CouncilRecommendation(
        final_action=ActionType.LONG,
        final_position_size_pct=0.5,
        aggregate_confidence=0.72,
        risk_level=SeverityLevel.MEDIUM,
        sl_pct=4.0,
        tp_pct=8.0,
        rationale="MA50 Bounce pattern detected with 72% confidence. RSI oversold (28), price near MA50 (-1.2%). Quant signals agree (85%).",
        dissent_summary="Volume slightly below average",
        request_id="test_001",
    )
    
    # Create mock context
    context = CouncilContext(
        market=MarketSnapshot(
            symbol="BTC_USDT",
            timeframe="4h",
            timestamp=datetime.utcnow(),
            current_price=43500.0,
            spread_pct=0.01,
            volume_24h=1500000000,
            rsi=28.5,
            price_to_ema9_pct=-0.8,
            price_to_ema21_pct=-1.5,
            price_to_sma50_pct=-1.2,
            atr_pct=2.3,
            bb_width_pct=4.5,
            volume_ratio=0.95,
            regime=Regime.NORMAL,
            opportunity_score=0.68,
            risk_penalty=0.25,
        ),
        position=PositionState(
            current_action=ActionType.HOLD,
            size=0,
            entry_price=None,
            unrealized_pnl_pct=0,
        ),
        constraints=RiskConstraints(),
        quant_signals=QuantSignals(
            p_rb=0.65,
            p_ml=0.58,
            p_hyb=0.61,
            agreement=0.85,
        ),
        rag=RAGContext(
            similar_patterns=[
                {"pattern_id": "ma50_bounce", "outcome": "win", "pnl_pct": 3.2},
                {"pattern_id": "ma50_bounce", "outcome": "win", "pnl_pct": 2.8},
                {"pattern_id": "ma50_bounce", "outcome": "loss", "pnl_pct": -1.5},
            ],
        ),
        request_id="test_001",
    )
    
    # Run CLI
    response, trade_id = run_decision_cli(recommendation, context, log_to_rag=False)
    
    print(f"Response: {response.to_dict()}")
