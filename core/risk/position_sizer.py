"""
Position Sizer and Risk Manager for Scarlet Sails Backtesting Framework

Handles position sizing and risk management.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class PositionConfig:
    """Position sizing configuration."""
    method: str = 'fixed_pct'  # fixed_pct, risk_based, kelly, atr_based
    fixed_pct: float = 0.1  # 10% of capital per trade
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_position_pct: float = 0.25  # Max 25% of capital in single position
    kelly_fraction: float = 0.25  # Fraction of Kelly criterion to use


class PositionSizer:
    """
    Calculate position sizes using various methods.
    
    Methods
    -------
    - fixed_pct: Fixed percentage of capital
    - risk_based: Based on stop-loss distance and risk per trade
    - kelly: Kelly criterion (capped)
    - atr_based: Based on ATR for volatility-adjusted sizing
    """
    
    def __init__(self, config: Optional[PositionConfig] = None):
        self.config = config or PositionConfig()
    
    def calculate_size(
        self,
        capital: float,
        price: float,
        stop_loss: Optional[float] = None,
        atr: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win_loss_ratio: Optional[float] = None,
        signal_strength: float = 1.0,
    ) -> float:
        """
        Calculate position size.
        
        Parameters
        ----------
        capital : float
            Available capital
        price : float
            Entry price
        stop_loss : float, optional
            Stop loss price (for risk_based)
        atr : float, optional
            Average True Range (for atr_based)
        win_rate : float, optional
            Historical win rate (for kelly)
        avg_win_loss_ratio : float, optional
            Average win / average loss (for kelly)
        signal_strength : float
            Signal strength [0, 1] to scale position
        
        Returns
        -------
        float
            Position size (number of units)
        """
        method = self.config.method
        
        if method == 'fixed_pct':
            size = self._fixed_pct_size(capital, price)
        elif method == 'risk_based':
            size = self._risk_based_size(capital, price, stop_loss)
        elif method == 'kelly':
            size = self._kelly_size(capital, price, win_rate, avg_win_loss_ratio)
        elif method == 'atr_based':
            size = self._atr_based_size(capital, price, atr)
        else:
            size = self._fixed_pct_size(capital, price)
        
        # Scale by signal strength
        size *= signal_strength
        
        # Apply max position limit
        max_size = (capital * self.config.max_position_pct) / price
        size = min(size, max_size)
        
        # Ensure non-negative
        return max(0.0, size)
    
    def _fixed_pct_size(self, capital: float, price: float) -> float:
        """Fixed percentage of capital."""
        position_value = capital * self.config.fixed_pct
        return position_value / price
    
    def _risk_based_size(
        self,
        capital: float,
        price: float,
        stop_loss: Optional[float]
    ) -> float:
        """Risk-based position sizing."""
        if stop_loss is None or stop_loss >= price:
            # Fallback to fixed
            return self._fixed_pct_size(capital, price)
        
        risk_amount = capital * self.config.risk_per_trade
        risk_per_unit = abs(price - stop_loss)
        
        if risk_per_unit == 0:
            return self._fixed_pct_size(capital, price)
        
        return risk_amount / risk_per_unit
    
    def _kelly_size(
        self,
        capital: float,
        price: float,
        win_rate: Optional[float],
        avg_win_loss_ratio: Optional[float]
    ) -> float:
        """Kelly criterion sizing."""
        if win_rate is None or avg_win_loss_ratio is None:
            return self._fixed_pct_size(capital, price)
        
        if win_rate <= 0 or win_rate >= 1 or avg_win_loss_ratio <= 0:
            return self._fixed_pct_size(capital, price)
        
        # Kelly formula: f = (p * b - q) / b
        # p = win probability, q = loss probability, b = win/loss ratio
        p = win_rate
        q = 1 - win_rate
        b = avg_win_loss_ratio
        
        kelly_pct = (p * b - q) / b
        
        # Cap at fraction of Kelly (safer)
        kelly_pct = max(0, kelly_pct * self.config.kelly_fraction)
        
        # Cap at max position
        kelly_pct = min(kelly_pct, self.config.max_position_pct)
        
        position_value = capital * kelly_pct
        return position_value / price
    
    def _atr_based_size(
        self,
        capital: float,
        price: float,
        atr: Optional[float]
    ) -> float:
        """ATR-based volatility-adjusted sizing."""
        if atr is None or atr <= 0:
            return self._fixed_pct_size(capital, price)
        
        # Target risk per trade as multiple of ATR
        # Higher ATR = smaller position
        risk_amount = capital * self.config.risk_per_trade
        atr_multiplier = 2.0  # Stop at 2x ATR
        risk_per_unit = atr * atr_multiplier
        
        return risk_amount / risk_per_unit


@dataclass
class RiskLimits:
    """Risk management limits."""
    max_position_pct: float = 0.25  # Max 25% in single position
    max_drawdown_pct: float = 0.20  # Max 20% drawdown
    daily_loss_limit_pct: float = 0.05  # Max 5% daily loss
    max_open_trades: int = 5
    stop_loss_pct: float = 0.02  # Default 2% stop loss


class RiskManager:
    """
    Enforces risk management rules.
    
    Checks
    ------
    - Max position size
    - Max drawdown (circuit breaker)
    - Daily loss limit
    - Max open trades
    """
    
    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self._daily_pnl: float = 0.0
        self._current_date: Optional[pd.Timestamp] = None
        self._peak_value: float = 0.0
        self._current_value: float = 0.0
        self._open_trades: int = 0
        self._is_halted: bool = False
        self._halt_reason: str = ''
    
    def reset(self, initial_capital: float):
        """Reset risk manager state."""
        self._daily_pnl = 0.0
        self._current_date = None
        self._peak_value = initial_capital
        self._current_value = initial_capital
        self._open_trades = 0
        self._is_halted = False
        self._halt_reason = ''
    
    def update(
        self,
        timestamp: pd.Timestamp,
        portfolio_value: float,
        trade_pnl: float = 0.0
    ):
        """Update risk manager state."""
        # New day check
        if self._current_date is None or timestamp.date() != self._current_date.date():
            self._daily_pnl = 0.0
            self._current_date = timestamp
        
        # Update values
        self._current_value = portfolio_value
        self._daily_pnl += trade_pnl
        
        # Update peak
        if portfolio_value > self._peak_value:
            self._peak_value = portfolio_value
        
        # Check limits
        self._check_limits()
    
    def _check_limits(self):
        """Check if any limits are breached."""
        # Drawdown check
        if self._peak_value > 0:
            drawdown = (self._peak_value - self._current_value) / self._peak_value
            if drawdown >= self.limits.max_drawdown_pct:
                self._is_halted = True
                self._halt_reason = f"Max drawdown breached: {drawdown:.1%}"
                return
        
        # Daily loss check
        if self._current_value > 0:
            daily_loss_pct = -self._daily_pnl / self._current_value
            if daily_loss_pct >= self.limits.daily_loss_limit_pct:
                self._is_halted = True
                self._halt_reason = f"Daily loss limit breached: {daily_loss_pct:.1%}"
                return
    
    def can_open_trade(
        self,
        position_value: float,
        portfolio_value: float
    ) -> tuple[bool, str]:
        """
        Check if new trade is allowed.
        
        Returns
        -------
        tuple[bool, str]
            (allowed, reason)
        """
        if self._is_halted:
            return False, f"Trading halted: {self._halt_reason}"
        
        # Position size check
        if portfolio_value > 0:
            position_pct = position_value / portfolio_value
            if position_pct > self.limits.max_position_pct:
                return False, f"Position too large: {position_pct:.1%} > {self.limits.max_position_pct:.1%}"
        
        # Open trades check
        if self._open_trades >= self.limits.max_open_trades:
            return False, f"Max open trades: {self._open_trades} >= {self.limits.max_open_trades}"
        
        return True, "OK"
    
    def on_trade_open(self):
        """Called when trade opened."""
        self._open_trades += 1
    
    def on_trade_close(self, pnl: float):
        """Called when trade closed."""
        self._open_trades = max(0, self._open_trades - 1)
        self._daily_pnl += pnl
    
    def calculate_stop_loss(self, entry_price: float, direction: int) -> float:
        """Calculate default stop loss price."""
        if direction == 1:  # Long
            return entry_price * (1 - self.limits.stop_loss_pct)
        else:  # Short
            return entry_price * (1 + self.limits.stop_loss_pct)
    
    def get_current_drawdown(self) -> float:
        """Get current drawdown percentage."""
        if self._peak_value <= 0:
            return 0.0
        return (self._peak_value - self._current_value) / self._peak_value
    
    def get_status(self) -> Dict:
        """Get current risk status."""
        return {
            'is_halted': self._is_halted,
            'halt_reason': self._halt_reason,
            'current_drawdown': self.get_current_drawdown(),
            'daily_pnl': self._daily_pnl,
            'open_trades': self._open_trades,
            'peak_value': self._peak_value,
            'current_value': self._current_value,
        }