"""
Trade Logger for Scarlet Sails Backtesting Framework

Logs trades and provides trade analysis.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict
from pathlib import Path
import json


@dataclass
class Trade:
    """Single trade record."""
    entry_time: datetime
    entry_price: float
    size: float
    direction: int  # 1 = long, -1 = short
    
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    
    strategy: str = 'unknown'
    coin: str = 'unknown'
    timeframe: str = 'unknown'
    
    signal_strength: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    def close(self, exit_time: datetime, exit_price: float, commission: float = 0.0):
        """Close the trade and calculate PnL."""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.commission += commission
        
        # Calculate PnL
        if self.direction == 1:  # Long
            self.pnl = (self.exit_price - self.entry_price) * self.size - self.commission - self.slippage
            self.pnl_pct = (self.exit_price / self.entry_price - 1) * 100
        else:  # Short
            self.pnl = (self.entry_price - self.exit_price) * self.size - self.commission - self.slippage
            self.pnl_pct = (self.entry_price / self.exit_price - 1) * 100
    
    @property
    def is_open(self) -> bool:
        return self.exit_time is None
    
    @property
    def is_winner(self) -> bool:
        return self.pnl > 0
    
    @property
    def duration(self) -> Optional[pd.Timedelta]:
        if self.exit_time and self.entry_time:
            return pd.Timedelta(self.exit_time - self.entry_time)
        return None
    
    @property
    def duration_hours(self) -> Optional[float]:
        d = self.duration
        return d.total_seconds() / 3600 if d else None
    
    def to_dict(self) -> Dict:
        return {
            'entry_time': str(self.entry_time),
            'entry_price': self.entry_price,
            'size': self.size,
            'direction': 'LONG' if self.direction == 1 else 'SHORT',
            'exit_time': str(self.exit_time) if self.exit_time else None,
            'exit_price': self.exit_price,
            'pnl': round(self.pnl, 2),
            'pnl_pct': round(self.pnl_pct, 4),
            'commission': round(self.commission, 4),
            'slippage': round(self.slippage, 4),
            'duration_hours': round(self.duration_hours, 2) if self.duration_hours else None,
            'strategy': self.strategy,
            'coin': self.coin,
            'timeframe': self.timeframe,
            'is_winner': self.is_winner,
        }


class TradeLogger:
    """Logs and analyzes trades."""
    
    def __init__(self):
        self.trades: List[Trade] = []
        self._current_trade: Optional[Trade] = None
    
    def open_trade(
        self,
        entry_time: datetime,
        entry_price: float,
        size: float,
        direction: int,
        commission: float = 0.0,
        slippage: float = 0.0,
        strategy: str = 'unknown',
        coin: str = 'unknown',
        timeframe: str = 'unknown',
        signal_strength: float = 0.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Trade:
        """Open a new trade."""
        trade = Trade(
            entry_time=entry_time,
            entry_price=entry_price,
            size=size,
            direction=direction,
            commission=commission,
            slippage=slippage,
            strategy=strategy,
            coin=coin,
            timeframe=timeframe,
            signal_strength=signal_strength,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        self._current_trade = trade
        return trade
    
    def close_trade(
        self,
        exit_time: datetime,
        exit_price: float,
        commission: float = 0.0
    ) -> Optional[Trade]:
        """Close current trade."""
        if self._current_trade is None:
            return None
        
        self._current_trade.close(exit_time, exit_price, commission)
        self.trades.append(self._current_trade)
        
        closed = self._current_trade
        self._current_trade = None
        return closed
    
    def add_trade(self, trade: Trade):
        """Add completed trade directly."""
        if not trade.is_open:
            self.trades.append(trade)
    
    @property
    def has_open_position(self) -> bool:
        return self._current_trade is not None
    
    @property
    def current_trade(self) -> Optional[Trade]:
        return self._current_trade
    
    def get_closed_trades(self) -> List[Trade]:
        """Get all closed trades."""
        return [t for t in self.trades if not t.is_open]
    
    def get_statistics(self) -> Dict:
        """Calculate trade statistics."""
        closed = self.get_closed_trades()
        
        if not closed:
            return {
                'total_trades': 0,
                'winners': 0,
                'losers': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'avg_pnl_pct': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'avg_duration_hours': 0.0,
                'total_commission': 0.0,
            }
        
        winners = [t for t in closed if t.is_winner]
        losers = [t for t in closed if not t.is_winner]
        
        win_pnls = [t.pnl for t in winners] if winners else [0]
        loss_pnls = [t.pnl for t in losers] if losers else [0]
        all_pnls = [t.pnl for t in closed]
        all_pnl_pcts = [t.pnl_pct for t in closed]
        durations = [t.duration_hours for t in closed if t.duration_hours]
        
        return {
            'total_trades': len(closed),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(closed) * 100,
            'avg_pnl': np.mean(all_pnls),
            'avg_pnl_pct': np.mean(all_pnl_pcts),
            'avg_win': np.mean(win_pnls) if winners else 0.0,
            'avg_loss': np.mean(loss_pnls) if losers else 0.0,
            'largest_win': max(win_pnls) if winners else 0.0,
            'largest_loss': min(loss_pnls) if losers else 0.0,
            'avg_duration_hours': np.mean(durations) if durations else 0.0,
            'total_commission': sum(t.commission for t in closed),
            'profit_factor': abs(sum(win_pnls) / sum(loss_pnls)) if sum(loss_pnls) != 0 else float('inf'),
            'expectancy': np.mean(all_pnls),
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        records = [t.to_dict() for t in self.trades if not t.is_open]
        df = pd.DataFrame(records)
        
        if 'entry_time' in df.columns:
            df['entry_time'] = pd.to_datetime(df['entry_time'])
        if 'exit_time' in df.columns:
            df['exit_time'] = pd.to_datetime(df['exit_time'])
        
        return df
    
    def save_to_csv(self, filepath: str):
        """Save trades to CSV."""
        df = self.to_dataframe()
        if not df.empty:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(filepath, index=False)
            print(f"Saved {len(df)} trades to {filepath}")
    
    def save_to_json(self, filepath: str):
        """Save trades to JSON."""
        data = {
            'trades': [t.to_dict() for t in self.trades if not t.is_open],
            'statistics': self.get_statistics()
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Saved trades to {filepath}")
    
    def analyze_by_hour(self) -> pd.DataFrame:
        """Analyze performance by hour of day."""
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()
        
        df['hour'] = pd.to_datetime(df['entry_time']).dt.hour
        
        return df.groupby('hour').agg({
            'pnl': ['count', 'sum', 'mean'],
            'pnl_pct': 'mean',
            'is_winner': 'mean'
        }).round(4)
    
    def analyze_by_day(self) -> pd.DataFrame:
        """Analyze performance by day of week."""
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()
        
        df['day'] = pd.to_datetime(df['entry_time']).dt.day_name()
        
        return df.groupby('day').agg({
            'pnl': ['count', 'sum', 'mean'],
            'pnl_pct': 'mean',
            'is_winner': 'mean'
        }).round(4)
    
    def get_consecutive_stats(self) -> Dict:
        """Get consecutive wins/losses statistics."""
        closed = self.get_closed_trades()
        if not closed:
            return {'max_consecutive_wins': 0, 'max_consecutive_losses': 0}
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in closed:
            if trade.is_winner:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return {
            'max_consecutive_wins': max_wins,
            'max_consecutive_losses': max_losses,
        }