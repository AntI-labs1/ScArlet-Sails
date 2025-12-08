"""
Pattern Outcome Updater

Links trade results to patterns for continuous learning.

Workflow:
1. Trade closes
2. Find the pattern that triggered it
3. Record outcome (PnL, duration, exit_reason)
4. Update pattern statistics
5. Rebuild index if needed
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import numpy as np


class PatternUpdater:
    """
    Updates patterns with real trading outcomes.
    
    This is CRITICAL for learning:
    - Pattern A has 70% win rate → increase confidence
    - Pattern B has 30% win rate → decrease confidence or exclude
    """
    
    def __init__(self, patterns_dir: str = "rag/patterns"):
        """
        Initialize updater.
        
        Args:
            patterns_dir: Directory with JSON patterns
        """
        self.patterns_dir = Path(patterns_dir)
        self.outcomes_file = self.patterns_dir / "outcomes.json"
        
        # Load existing outcomes summary
        if self.outcomes_file.exists():
            with open(self.outcomes_file, 'r', encoding='utf-8') as f:
                self.outcomes_summary = json.load(f)
        else:
            self.outcomes_summary = {}
    
    def link_trade(
        self,
        pattern_id: str,
        trade_result: dict,
        verbose: bool = True
    ) -> bool:
        """
        Link a closed trade to its pattern.
        
        Args:
            pattern_id: Pattern ID (e.g., "BTC_1h_20241208_1500")
            trade_result: {
                'entry_price': float,
                'exit_price': float,
                'pnl_pct': float,
                'duration_bars': int,
                'exit_reason': 'tp' | 'sl' | 'manual' | 'timeout',
                'max_drawdown_pct': float,
                'max_profit_pct': float,
            }
            verbose: Print result
            
        Returns:
            True if successful
        """
        # Find pattern file
        pattern_file = self.patterns_dir / f"{pattern_id}.json"
        
        if not pattern_file.exists():
            if verbose:
                print(f"⚠️ Pattern not found: {pattern_id}")
            return False
        
        try:
            # Load pattern
            with open(pattern_file, 'r', encoding='utf-8') as f:
                pattern = json.load(f)
            
            # Initialize outcomes list if needed
            if 'outcomes' not in pattern:
                pattern['outcomes'] = []
            
            # Add this outcome
            outcome = {
                **trade_result,
                'recorded_at': datetime.now().isoformat(),
            }
            pattern['outcomes'].append(outcome)
            
            # Calculate statistics
            stats = self._calculate_statistics(pattern['outcomes'])
            pattern['statistics'] = stats
            
            # Save updated pattern
            with open(pattern_file, 'w', encoding='utf-8') as f:
                json.dump(pattern, f, indent=2, ensure_ascii=False)
            
            # Update summary
            self.outcomes_summary[pattern_id] = stats
            self._save_summary()
            
            if verbose:
                print(f"✅ Outcome recorded for {pattern_id}")
                print(f"   Total trades: {stats['total_trades']}")
                print(f"   Win rate: {stats['win_rate']:.1%}")
                print(f"   Avg PnL: {stats['avg_pnl']:.2f}%")
                print(f"   Profit factor: {stats['profit_factor']:.2f}")
            
            return True
            
        except Exception as e:
            if verbose:
                print(f"❌ Error: {e}")
            return False
    
    def _calculate_statistics(self, outcomes: List[Dict]) -> Dict:
        """Calculate aggregate statistics from outcomes."""
        if not outcomes:
            return {'total_trades': 0}
        
        pnls = [o['pnl_pct'] for o in outcomes]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        return {
            'total_trades': len(outcomes),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(outcomes),
            'avg_pnl': np.mean(pnls),
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'max_pnl': max(pnls),
            'min_pnl': min(pnls),
            'std_pnl': np.std(pnls),
            'profit_factor': abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf'),
            'expectancy': np.mean(pnls),
            'last_updated': datetime.now().isoformat(),
        }
    
    def _save_summary(self):
        """Save outcomes summary."""
        with open(self.outcomes_file, 'w', encoding='utf-8') as f:
            json.dump(self.outcomes_summary, f, indent=2, ensure_ascii=False)
    
    def get_pattern_stats(self, pattern_id: str) -> Optional[Dict]:
        """Get statistics for a pattern."""
        return self.outcomes_summary.get(pattern_id)
    
    def get_all_stats(self) -> Dict:
        """Get all patterns' statistics."""
        return self.outcomes_summary
    
    def get_aggregate_stats(self) -> Dict:
        """
        Get aggregate statistics across ALL patterns.
        
        Useful for overall system performance.
        """
        if not self.outcomes_summary:
            return {'total_patterns': 0}
        
        all_win_rates = []
        all_pnls = []
        total_trades = 0
        
        for stats in self.outcomes_summary.values():
            if stats.get('total_trades', 0) > 0:
                all_win_rates.append(stats['win_rate'])
                all_pnls.append(stats['avg_pnl'])
                total_trades += stats['total_trades']
        
        return {
            'total_patterns': len(self.outcomes_summary),
            'patterns_with_trades': len(all_win_rates),
            'total_trades': total_trades,
            'avg_win_rate': np.mean(all_win_rates) if all_win_rates else 0,
            'avg_pnl': np.mean(all_pnls) if all_pnls else 0,
        }
