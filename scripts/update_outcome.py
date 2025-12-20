#!/usr/bin/env python3
"""
ScArlet-Sails Outcome Updater

Updates trade outcomes after position is closed.
Links Council recommendations to actual results for learning loop.

Usage:
    python scripts/update_outcome.py TRADE_ID --outcome win --pnl 2.5
    python scripts/update_outcome.py TRADE_ID --outcome loss --pnl -1.2 --reason stopped_out
    python scripts/update_outcome.py --list                    # List pending trades
    python scripts/update_outcome.py --stats                   # Show statistics

Philosophy:
    "Логировать всё для learning loop. Council was right? Human was right?"
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class TradeOutcome(Enum):
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    CANCELLED = "cancelled"


class ExitReason(Enum):
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    MANUAL = "manual"
    TIME_EXIT = "time_exit"
    SIGNAL_REVERSAL = "signal_reversal"
    RISK_LIMIT = "risk_limit"
    OTHER = "other"


@dataclass
class OutcomeUpdate:
    """Outcome update for a trade."""
    trade_id: str
    outcome: TradeOutcome
    pnl_pct: float
    exit_reason: ExitReason
    exit_price: Optional[float]
    exit_timestamp: datetime
    notes: Optional[str]
    
    # Learning metrics (computed)
    council_was_right: Optional[bool] = None
    human_was_right: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trade_id': self.trade_id,
            'outcome': self.outcome.value,
            'pnl_pct': self.pnl_pct,
            'exit_reason': self.exit_reason.value,
            'exit_price': self.exit_price,
            'exit_timestamp': self.exit_timestamp.isoformat(),
            'notes': self.notes,
            'council_was_right': self.council_was_right,
            'human_was_right': self.human_was_right,
        }


# =============================================================================
# OUTCOME MANAGER
# =============================================================================

class OutcomeManager:
    """
    Manages trade outcomes and learning loop.
    
    Storage structure:
        rag/decisions/
        ├── pending/           # Trades awaiting outcome
        │   └── TRADE_ID.json
        ├── completed/         # Trades with outcome
        │   └── TRADE_ID.json
        └── stats.json         # Aggregate statistics
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or PROJECT_ROOT / "rag" / "decisions"
        self.pending_dir = self.base_dir / "pending"
        self.completed_dir = self.base_dir / "completed"
        self.stats_file = self.base_dir / "stats.json"
        
        # Ensure directories exist
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.completed_dir.mkdir(parents=True, exist_ok=True)
    
    def get_pending_trades(self) -> List[Dict[str, Any]]:
        """Get all trades pending outcome."""
        trades = []
        for f in self.pending_dir.glob("*.json"):
            try:
                with open(f, 'r') as fp:
                    trade = json.load(fp)
                    trade['_file'] = f.name
                    trades.append(trade)
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
        
        # Sort by timestamp
        trades.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return trades
    
    def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific trade by ID."""
        # Check pending
        pending_file = self.pending_dir / f"{trade_id}.json"
        if pending_file.exists():
            with open(pending_file, 'r') as f:
                return json.load(f)
        
        # Check completed
        completed_file = self.completed_dir / f"{trade_id}.json"
        if completed_file.exists():
            with open(completed_file, 'r') as f:
                return json.load(f)
        
        return None
    
    def update_outcome(self, update: OutcomeUpdate) -> bool:
        """Update trade with outcome."""
        # Load original trade
        pending_file = self.pending_dir / f"{update.trade_id}.json"
        
        if not pending_file.exists():
            logger.error(f"Trade not found in pending: {update.trade_id}")
            return False
        
        try:
            with open(pending_file, 'r') as f:
                trade = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load trade: {e}")
            return False
        
        # Compute learning metrics
        update.council_was_right = self._compute_council_correctness(trade, update)
        update.human_was_right = self._compute_human_correctness(trade, update)
        
        # Merge outcome into trade
        trade['outcome'] = update.to_dict()
        trade['status'] = 'completed'
        
        # Save to completed
        completed_file = self.completed_dir / f"{update.trade_id}.json"
        try:
            with open(completed_file, 'w') as f:
                json.dump(trade, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save completed trade: {e}")
            return False
        
        # Remove from pending
        try:
            pending_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to remove pending file: {e}")
        
        # Update stats
        self._update_stats(trade, update)
        
        logger.info(f"Trade {update.trade_id} outcome updated: {update.outcome.value}")
        return True
    
    def _compute_council_correctness(
        self,
        trade: Dict[str, Any],
        update: OutcomeUpdate,
    ) -> Optional[bool]:
        """
        Was Council's recommendation correct?
        
        Logic:
        - If Council recommended LONG and outcome is WIN → True
        - If Council recommended LONG and outcome is LOSS → False
        - If Council recommended HOLD and human took trade → compare to breakeven
        """
        recommendation = trade.get('recommendation', {})
        council_action = recommendation.get('final_action', '').lower()
        
        if not council_action:
            return None
        
        if update.outcome == TradeOutcome.CANCELLED:
            return None
        
        # Did council recommend taking a position?
        council_took_position = council_action in ['long', 'short']
        
        if council_took_position:
            # Council recommended trade
            if update.outcome == TradeOutcome.WIN:
                return True
            elif update.outcome == TradeOutcome.LOSS:
                return False
            else:  # breakeven
                return None
        else:
            # Council recommended HOLD
            # If human took trade and lost → Council was right to say HOLD
            # If human took trade and won → Council was wrong
            if update.outcome == TradeOutcome.WIN:
                return False  # Should have recommended trade
            elif update.outcome == TradeOutcome.LOSS:
                return True   # Correctly said HOLD
            else:
                return None
    
    def _compute_human_correctness(
        self,
        trade: Dict[str, Any],
        update: OutcomeUpdate,
    ) -> Optional[bool]:
        """
        Was Human's decision correct?
        
        Logic:
        - If Human accepted and outcome is WIN → True
        - If Human rejected and would have been WIN → False
        - etc.
        """
        human_decision = trade.get('human_response', {}).get('decision', '').lower()
        
        if not human_decision:
            return None
        
        if update.outcome == TradeOutcome.CANCELLED:
            return None
        
        if human_decision == 'accept':
            # Human accepted recommendation
            if update.outcome == TradeOutcome.WIN:
                return True
            elif update.outcome == TradeOutcome.LOSS:
                return False
            else:
                return None
        
        elif human_decision == 'reject':
            # Human rejected recommendation
            # This is complex - we don't know what would have happened
            # For now, assume rejection of losing trade = correct
            if update.outcome == TradeOutcome.LOSS:
                return True  # Good rejection
            elif update.outcome == TradeOutcome.WIN:
                return False  # Missed opportunity
            else:
                return None
        
        elif human_decision == 'modify':
            # Human modified - same as accept for correctness
            if update.outcome == TradeOutcome.WIN:
                return True
            elif update.outcome == TradeOutcome.LOSS:
                return False
            else:
                return None
        
        return None
    
    def _update_stats(self, trade: Dict[str, Any], update: OutcomeUpdate) -> None:
        """Update aggregate statistics."""
        # Load existing stats
        stats = self._load_stats()
        
        # Update counters
        stats['total_trades'] = stats.get('total_trades', 0) + 1
        
        if update.outcome == TradeOutcome.WIN:
            stats['wins'] = stats.get('wins', 0) + 1
            stats['total_pnl'] = stats.get('total_pnl', 0.0) + update.pnl_pct
        elif update.outcome == TradeOutcome.LOSS:
            stats['losses'] = stats.get('losses', 0) + 1
            stats['total_pnl'] = stats.get('total_pnl', 0.0) + update.pnl_pct
        elif update.outcome == TradeOutcome.BREAKEVEN:
            stats['breakeven'] = stats.get('breakeven', 0) + 1
        
        # Learning metrics
        if update.council_was_right is True:
            stats['council_correct'] = stats.get('council_correct', 0) + 1
        elif update.council_was_right is False:
            stats['council_incorrect'] = stats.get('council_incorrect', 0) + 1
        
        if update.human_was_right is True:
            stats['human_correct'] = stats.get('human_correct', 0) + 1
        elif update.human_was_right is False:
            stats['human_incorrect'] = stats.get('human_incorrect', 0) + 1
        
        # Win rate
        total = stats.get('wins', 0) + stats.get('losses', 0)
        if total > 0:
            stats['win_rate'] = stats.get('wins', 0) / total
        
        # Council accuracy
        council_total = stats.get('council_correct', 0) + stats.get('council_incorrect', 0)
        if council_total > 0:
            stats['council_accuracy'] = stats.get('council_correct', 0) / council_total
        
        # Human accuracy
        human_total = stats.get('human_correct', 0) + stats.get('human_incorrect', 0)
        if human_total > 0:
            stats['human_accuracy'] = stats.get('human_correct', 0) / human_total
        
        stats['last_updated'] = datetime.now().isoformat()
        
        # Save stats
        self._save_stats(stats)
    
    def _load_stats(self) -> Dict[str, Any]:
        """Load statistics."""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def _save_stats(self, stats: Dict[str, Any]) -> None:
        """Save statistics."""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics."""
        return self._load_stats()
    
    def print_stats(self) -> None:
        """Print formatted statistics."""
        stats = self.get_stats()
        
        if not stats:
            print("No statistics available yet.")
            return
        
        print()
        print("=" * 50)
        print("  TRADING STATISTICS")
        print("=" * 50)
        
        print(f"\n  Total Trades:    {stats.get('total_trades', 0)}")
        print(f"  Wins:            {stats.get('wins', 0)}")
        print(f"  Losses:          {stats.get('losses', 0)}")
        print(f"  Breakeven:       {stats.get('breakeven', 0)}")
        
        win_rate = stats.get('win_rate', 0)
        print(f"\n  Win Rate:        {win_rate:.1%}")
        print(f"  Total PnL:       {stats.get('total_pnl', 0):+.2f}%")
        
        print("\n  LEARNING METRICS")
        print("-" * 50)
        
        council_acc = stats.get('council_accuracy')
        if council_acc is not None:
            print(f"  Council Accuracy: {council_acc:.1%}")
            print(f"    Correct:   {stats.get('council_correct', 0)}")
            print(f"    Incorrect: {stats.get('council_incorrect', 0)}")
        
        human_acc = stats.get('human_accuracy')
        if human_acc is not None:
            print(f"\n  Human Accuracy:   {human_acc:.1%}")
            print(f"    Correct:   {stats.get('human_correct', 0)}")
            print(f"    Incorrect: {stats.get('human_incorrect', 0)}")
        
        print(f"\n  Last Updated: {stats.get('last_updated', 'N/A')}")
        print()
    
    def print_pending(self) -> None:
        """Print pending trades."""
        trades = self.get_pending_trades()
        
        if not trades:
            print("\nNo pending trades.")
            return
        
        print()
        print("=" * 70)
        print("  PENDING TRADES")
        print("=" * 70)
        
        for t in trades:
            trade_id = t.get('trade_id', t.get('request_id', 'unknown'))
            timestamp = t.get('timestamp', 'N/A')
            rec = t.get('recommendation', {})
            action = rec.get('final_action', 'N/A')
            size = rec.get('final_position_size_pct', 0)
            
            human = t.get('human_response', {})
            decision = human.get('decision', 'N/A')
            
            print(f"\n  ID:        {trade_id}")
            print(f"  Timestamp: {timestamp}")
            print(f"  Action:    {action} ({size:.1f}%)")
            print(f"  Decision:  {decision}")
            print("-" * 70)
        
        print(f"\n  Total pending: {len(trades)}")
        print()


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="ScArlet-Sails Outcome Updater",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/update_outcome.py BTC_20231220_143022 --outcome win --pnl 2.5
    python scripts/update_outcome.py BTC_20231220_143022 --outcome loss --pnl -1.2 --reason stop_loss
    python scripts/update_outcome.py --list
    python scripts/update_outcome.py --stats
        """
    )
    
    parser.add_argument(
        'trade_id',
        type=str,
        nargs='?',
        help='Trade ID to update'
    )
    
    parser.add_argument(
        '--outcome',
        type=str,
        choices=['win', 'loss', 'breakeven', 'cancelled'],
        help='Trade outcome'
    )
    
    parser.add_argument(
        '--pnl',
        type=float,
        default=0.0,
        help='PnL percentage (e.g., 2.5 for +2.5%%, -1.2 for -1.2%%)'
    )
    
    parser.add_argument(
        '--reason',
        type=str,
        choices=['take_profit', 'stop_loss', 'trailing_stop', 'manual', 
                 'time_exit', 'signal_reversal', 'risk_limit', 'other'],
        default='manual',
        help='Exit reason'
    )
    
    parser.add_argument(
        '--price',
        type=float,
        help='Exit price'
    )
    
    parser.add_argument(
        '--notes',
        type=str,
        help='Additional notes'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List pending trades'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show statistics'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    manager = OutcomeManager()
    
    # List pending trades
    if args.list:
        manager.print_pending()
        return 0
    
    # Show statistics
    if args.stats:
        manager.print_stats()
        return 0
    
    # Update outcome
    if not args.trade_id:
        print("Error: trade_id required (or use --list / --stats)")
        return 1
    
    if not args.outcome:
        print("Error: --outcome required")
        return 1
    
    # Build update
    update = OutcomeUpdate(
        trade_id=args.trade_id,
        outcome=TradeOutcome(args.outcome),
        pnl_pct=args.pnl,
        exit_reason=ExitReason(args.reason),
        exit_price=args.price,
        exit_timestamp=datetime.now(),
        notes=args.notes,
    )
    
    # Apply update
    success = manager.update_outcome(update)
    
    if success:
        print(f"\n[OK] Trade {args.trade_id} updated:")
        print(f"     Outcome: {args.outcome}")
        print(f"     PnL: {args.pnl:+.2f}%")
        print(f"     Reason: {args.reason}")
        
        if update.council_was_right is not None:
            status = "CORRECT" if update.council_was_right else "INCORRECT"
            print(f"     Council: {status}")
        
        if update.human_was_right is not None:
            status = "CORRECT" if update.human_was_right else "INCORRECT"
            print(f"     Human: {status}")
        
        print(f"\n     Run --stats to see updated statistics")
        return 0
    else:
        print(f"\n[X] Failed to update trade {args.trade_id}")
        print("     Use --list to see pending trades")
        return 1


if __name__ == "__main__":
    sys.exit(main())