"""
Metrics Calculator for Scarlet Sails Backtesting Framework

Calculates comprehensive performance metrics.
Target metrics:
- Sharpe Ratio > 1.0
- Profit Factor > 2.0
- Max Drawdown < 15%
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

from .trade_logger import Trade


@dataclass
class BacktestMetrics:
    """Container for backtest metrics."""
    # Identifiers
    strategy: str = 'unknown'
    coin: str = 'unknown'
    timeframe: str = 'unknown'
    start_date: str = ''
    end_date: str = ''
    
    # Capital
    initial_capital: float = 0.0
    final_value: float = 0.0
    
    # Return metrics
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    cagr: float = 0.0
    
    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_days: float = 0.0
    volatility: float = 0.0
    
    # Trade metrics
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    expectancy: float = 0.0
    
    # Advanced
    var_95: float = 0.0
    cvar_95: float = 0.0
    tail_ratio: float = 0.0
    avg_trade_duration_hours: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    # Status
    meets_sharpe_target: bool = False
    meets_pf_target: bool = False
    meets_drawdown_target: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'identifiers': {
                'strategy': self.strategy,
                'coin': self.coin,
                'timeframe': self.timeframe,
                'start_date': self.start_date,
                'end_date': self.end_date,
            },
            'capital': {
                'initial': round(self.initial_capital, 2),
                'final': round(self.final_value, 2),
            },
            'returns': {
                'total_return': round(self.total_return, 2),
                'total_return_pct': round(self.total_return_pct, 2),
                'annualized_return': round(self.annualized_return, 2),
                'cagr': round(self.cagr, 2),
            },
            'risk': {
                'sharpe_ratio': round(self.sharpe_ratio, 3),
                'sortino_ratio': round(self.sortino_ratio, 3),
                'calmar_ratio': round(self.calmar_ratio, 3),
                'max_drawdown': round(self.max_drawdown, 4),
                'max_drawdown_duration_days': round(self.max_drawdown_duration_days, 1),
                'volatility': round(self.volatility, 4),
            },
            'trades': {
                'total': self.total_trades,
                'winners': self.winners,
                'losers': self.losers,
                'win_rate': round(self.win_rate, 2),
                'profit_factor': round(self.profit_factor, 3),
                'avg_win': round(self.avg_win, 2),
                'avg_loss': round(self.avg_loss, 2),
                'largest_win': round(self.largest_win, 2),
                'largest_loss': round(self.largest_loss, 2),
                'expectancy': round(self.expectancy, 4),
            },
            'advanced': {
                'var_95': round(self.var_95, 4),
                'cvar_95': round(self.cvar_95, 4),
                'tail_ratio': round(self.tail_ratio, 3),
                'avg_duration_hours': round(self.avg_trade_duration_hours, 2),
                'max_consecutive_wins': self.max_consecutive_wins,
                'max_consecutive_losses': self.max_consecutive_losses,
            },
            'targets': {
                'sharpe_target_1.0': self.meets_sharpe_target,
                'profit_factor_target_2.0': self.meets_pf_target,
                'max_drawdown_target_15pct': self.meets_drawdown_target,
            }
        }
    
    def save_json(self, filepath: str):
        """Save metrics to JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        def convert_numpy(obj):
            """Convert numpy types to Python native types."""
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(v) for v in obj]
            elif pd.isna(obj):
                return None
            return obj
        
        data = convert_numpy(self.to_dict())
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def print_summary(self):
        """Print formatted summary."""
        status_sharpe = '✅' if self.meets_sharpe_target else '❌'
        status_pf = '✅' if self.meets_pf_target else '❌'
        status_dd = '✅' if self.meets_drawdown_target else '❌'
        
        print(f"""
{'='*70}
BACKTEST RESULTS: {self.strategy} on {self.coin}_{self.timeframe}
{'='*70}

Period: {self.start_date} to {self.end_date}
Initial Capital: ${self.initial_capital:,.2f}
Final Value: ${self.final_value:,.2f}

RETURN METRICS:
{'─'*70}
Total Return: {self.total_return_pct:+.2f}%
Annualized Return: {self.annualized_return:+.2f}%
CAGR: {self.cagr:+.2f}%

RISK METRICS:
{'─'*70}
Sharpe Ratio: {self.sharpe_ratio:.3f} {status_sharpe} (target > 1.0)
Sortino Ratio: {self.sortino_ratio:.3f}
Max Drawdown: {self.max_drawdown*100:.2f}% {status_dd} (target < 15%)
Calmar Ratio: {self.calmar_ratio:.3f}
Volatility: {self.volatility*100:.2f}%

TRADE METRICS:
{'─'*70}
Total Trades: {self.total_trades}
Win Rate: {self.win_rate:.2f}%
Profit Factor: {self.profit_factor:.3f} {status_pf} (target > 2.0)
Avg Win: ${self.avg_win:.2f} ({self.avg_win_pct:.2f}%)
Avg Loss: ${self.avg_loss:.2f} ({self.avg_loss_pct:.2f}%)
Expectancy: ${self.expectancy:.4f}

ADVANCED:
{'─'*70}
VaR (95%): {self.var_95*100:.2f}%
CVaR (95%): {self.cvar_95*100:.2f}%
Avg Trade Duration: {self.avg_trade_duration_hours:.1f} hours
Max Consecutive Wins: {self.max_consecutive_wins}
Max Consecutive Losses: {self.max_consecutive_losses}

{'='*70}
""")


class MetricsCalculator:
    """
    Calculate comprehensive backtest metrics.
    
    Target metrics:
    - Sharpe Ratio > 1.0
    - Profit Factor > 2.0
    - Max Drawdown < 15%
    """
    
    RISK_FREE_RATE = 0.05  # 5% annual risk-free rate
    TARGET_SHARPE = 1.0
    TARGET_PROFIT_FACTOR = 2.0
    TARGET_MAX_DRAWDOWN = 0.15
    
    @classmethod
    def calculate_all(
        cls,
        equity_curve: pd.DataFrame,
        trades: List[Trade],
        initial_capital: float,
        bars_per_year: int = 35040,  # 15m default
        strategy: str = 'unknown',
        coin: str = 'unknown',
        timeframe: str = 'unknown',
    ) -> BacktestMetrics:
        """
        Calculate all metrics from equity curve and trades.
        
        Parameters
        ----------
        equity_curve : pd.DataFrame
            DataFrame with 'value' column and DatetimeIndex
        trades : List[Trade]
            List of completed trades
        initial_capital : float
            Starting capital
        bars_per_year : int
            Number of bars per year for annualization
        strategy, coin, timeframe : str
            Identifiers for the backtest
        
        Returns
        -------
        BacktestMetrics
            Comprehensive metrics object
        """
        metrics = BacktestMetrics()
        
        # Identifiers
        metrics.strategy = strategy
        metrics.coin = coin
        metrics.timeframe = timeframe
        
        if len(equity_curve) == 0:
            return metrics
        
        metrics.start_date = str(equity_curve.index[0].date())
        metrics.end_date = str(equity_curve.index[-1].date())
        
        # Capital
        metrics.initial_capital = initial_capital
        metrics.final_value = equity_curve['value'].iloc[-1]
        
        # Returns
        returns = cls._calculate_returns(equity_curve)
        metrics.total_return = metrics.final_value - initial_capital
        metrics.total_return_pct = (metrics.final_value / initial_capital - 1) * 100
        metrics.annualized_return = cls._annualized_return(returns, bars_per_year)
        metrics.cagr = cls._cagr(initial_capital, metrics.final_value, len(equity_curve), bars_per_year)
        
        # Risk
        metrics.volatility = cls._volatility(returns, bars_per_year)
        metrics.sharpe_ratio = cls._sharpe_ratio(returns, bars_per_year)
        metrics.sortino_ratio = cls._sortino_ratio(returns, bars_per_year)
        
        dd_info = cls._drawdown_analysis(equity_curve)
        metrics.max_drawdown = dd_info['max_drawdown']
        metrics.max_drawdown_duration_days = dd_info['max_duration_days']
        
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.annualized_return / (metrics.max_drawdown * 100)
        
        # VaR / CVaR
        metrics.var_95 = cls._var(returns, 0.95)
        metrics.cvar_95 = cls._cvar(returns, 0.95)
        metrics.tail_ratio = cls._tail_ratio(returns)
        
        # Trades
        trade_stats = cls._trade_statistics(trades)
        metrics.total_trades = trade_stats['total']
        metrics.winners = trade_stats['winners']
        metrics.losers = trade_stats['losers']
        metrics.win_rate = trade_stats['win_rate']
        metrics.profit_factor = trade_stats['profit_factor']
        metrics.avg_win = trade_stats['avg_win']
        metrics.avg_loss = trade_stats['avg_loss']
        metrics.avg_win_pct = trade_stats['avg_win_pct']
        metrics.avg_loss_pct = trade_stats['avg_loss_pct']
        metrics.largest_win = trade_stats['largest_win']
        metrics.largest_loss = trade_stats['largest_loss']
        metrics.expectancy = trade_stats['expectancy']
        metrics.avg_trade_duration_hours = trade_stats['avg_duration_hours']
        metrics.max_consecutive_wins = trade_stats['max_consecutive_wins']
        metrics.max_consecutive_losses = trade_stats['max_consecutive_losses']
        
        # Targets
        metrics.meets_sharpe_target = metrics.sharpe_ratio >= cls.TARGET_SHARPE
        metrics.meets_pf_target = metrics.profit_factor >= cls.TARGET_PROFIT_FACTOR
        metrics.meets_drawdown_target = metrics.max_drawdown <= cls.TARGET_MAX_DRAWDOWN
        
        return metrics
    
    @staticmethod
    def _calculate_returns(equity_curve: pd.DataFrame) -> pd.Series:
        """Calculate returns from equity curve."""
        return equity_curve['value'].pct_change().dropna()
    
    @classmethod
    def _annualized_return(cls, returns: pd.Series, bars_per_year: int) -> float:
        """Calculate annualized return."""
        if len(returns) == 0:
            return 0.0
        total_return = (1 + returns).prod() - 1
        n_periods = len(returns)
        years = n_periods / bars_per_year
        if years <= 0:
            return 0.0
        return ((1 + total_return) ** (1 / years) - 1) * 100
    
    @staticmethod
    def _cagr(initial: float, final: float, n_bars: int, bars_per_year: int) -> float:
        """Calculate CAGR."""
        if initial <= 0 or n_bars <= 0:
            return 0.0
        years = n_bars / bars_per_year
        if years <= 0:
            return 0.0
        return ((final / initial) ** (1 / years) - 1) * 100
    
    @staticmethod
    def _volatility(returns: pd.Series, bars_per_year: int) -> float:
        """Calculate annualized volatility."""
        if len(returns) == 0:
            return 0.0
        return returns.std() * np.sqrt(bars_per_year)
    
    @classmethod
    def _sharpe_ratio(cls, returns: pd.Series, bars_per_year: int) -> float:
        """
        Calculate Sharpe Ratio.
        
        Target: > 1.0
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - cls.RISK_FREE_RATE / bars_per_year
        
        if excess_returns.std() == 0:
            return 0.0
        
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(bars_per_year)
    
    @classmethod
    def _sortino_ratio(cls, returns: pd.Series, bars_per_year: int) -> float:
        """Calculate Sortino Ratio (downside volatility only)."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - cls.RISK_FREE_RATE / bars_per_year
        downside = returns[returns < 0]
        
        if len(downside) == 0 or downside.std() == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        
        downside_std = downside.std() * np.sqrt(bars_per_year)
        return (excess_returns.mean() * bars_per_year) / downside_std
    
    @staticmethod
    def _drawdown_analysis(equity_curve: pd.DataFrame) -> Dict:
        """
        Analyze drawdowns.
        
        Target: Max DD < 15%
        """
        values = equity_curve['value']
        peak = values.expanding().max()
        drawdown = (values - peak) / peak
        
        max_dd = abs(drawdown.min())
        
        # Duration analysis
        is_underwater = drawdown < 0
        underwater_periods = []
        current_start = None
        
        for i, underwater in enumerate(is_underwater):
            if underwater and current_start is None:
                current_start = i
            elif not underwater and current_start is not None:
                underwater_periods.append(i - current_start)
                current_start = None
        
        if current_start is not None:
            underwater_periods.append(len(is_underwater) - current_start)
        
        max_duration = max(underwater_periods) if underwater_periods else 0
        
        # Convert to days (approximate)
        idx = equity_curve.index
        if len(idx) >= 2:
            avg_bar_duration = (idx[-1] - idx[0]).total_seconds() / len(idx) / 86400
        else:
            avg_bar_duration = 0.01  # ~15 minutes
        
        max_duration_days = max_duration * avg_bar_duration
        
        return {
            'max_drawdown': max_dd,
            'max_duration_bars': max_duration,
            'max_duration_days': max_duration_days,
            'drawdown_series': drawdown,
        }
    
    @staticmethod
    def _var(returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        if len(returns) == 0:
            return 0.0
        return abs(np.percentile(returns, (1 - confidence) * 100))
    
    @staticmethod
    def _cvar(returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        if len(returns) == 0:
            return 0.0
        var = np.percentile(returns, (1 - confidence) * 100)
        return abs(returns[returns <= var].mean())
    
    @staticmethod
    def _tail_ratio(returns: pd.Series) -> float:
        """Calculate tail ratio (right tail / left tail)."""
        if len(returns) == 0:
            return 1.0
        
        right_tail = np.percentile(returns, 95)
        left_tail = abs(np.percentile(returns, 5))
        
        if left_tail == 0:
            return float('inf') if right_tail > 0 else 1.0
        
        return right_tail / left_tail
    
    @staticmethod
    def _trade_statistics(trades: List[Trade]) -> Dict:
        """
        Calculate trade statistics.
        
        Target: Profit Factor > 2.0
        """
        closed = [t for t in trades if not t.is_open]
        
        if not closed:
            return {
                'total': 0, 'winners': 0, 'losers': 0,
                'win_rate': 0.0, 'profit_factor': 0.0,
                'avg_win': 0.0, 'avg_loss': 0.0,
                'avg_win_pct': 0.0, 'avg_loss_pct': 0.0,
                'largest_win': 0.0, 'largest_loss': 0.0,
                'expectancy': 0.0, 'avg_duration_hours': 0.0,
                'max_consecutive_wins': 0, 'max_consecutive_losses': 0,
            }
        
        winners = [t for t in closed if t.pnl > 0]
        losers = [t for t in closed if t.pnl <= 0]
        
        win_pnls = [t.pnl for t in winners]
        loss_pnls = [t.pnl for t in losers]
        win_pcts = [t.pnl_pct for t in winners]
        loss_pcts = [t.pnl_pct for t in losers]
        
        gross_profit = sum(win_pnls) if win_pnls else 0
        gross_loss = abs(sum(loss_pnls)) if loss_pnls else 0
        
        # Profit factor
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float('inf') if gross_profit > 0 else 0.0
        
        # Durations
        durations = [t.duration_hours for t in closed if t.duration_hours is not None]
        
        # Consecutive wins/losses
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for t in closed:
            if t.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return {
            'total': len(closed),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(closed) * 100 if closed else 0,
            'profit_factor': profit_factor,
            'avg_win': np.mean(win_pnls) if win_pnls else 0.0,
            'avg_loss': np.mean(loss_pnls) if loss_pnls else 0.0,
            'avg_win_pct': np.mean(win_pcts) if win_pcts else 0.0,
            'avg_loss_pct': np.mean(loss_pcts) if loss_pcts else 0.0,
            'largest_win': max(win_pnls) if win_pnls else 0.0,
            'largest_loss': min(loss_pnls) if loss_pnls else 0.0,
            'expectancy': np.mean([t.pnl for t in closed]),
            'avg_duration_hours': np.mean(durations) if durations else 0.0,
            'max_consecutive_wins': max_wins,
            'max_consecutive_losses': max_losses,
        }
    
    @staticmethod
    def compare_strategies(metrics_list: List[BacktestMetrics]) -> pd.DataFrame:
        """Create comparison table for multiple strategies."""
        data = []
        for m in metrics_list:
            data.append({
                'Strategy': m.strategy,
                'Coin': m.coin,
                'TF': m.timeframe,
                'Return %': round(m.total_return_pct, 1),
                'Ann. Return %': round(m.annualized_return, 1),
                'Sharpe': round(m.sharpe_ratio, 2),
                'PF': round(m.profit_factor, 2),
                'Max DD %': round(m.max_drawdown * 100, 1),
                'Win Rate %': round(m.win_rate, 1),
                'Trades': m.total_trades,
            })
        
        return pd.DataFrame(data)