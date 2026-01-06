"""
Backtest Engine for Scarlet Sails Trading Framework

Production-ready backtester with:
- Multi-asset support (14 coins × 4 timeframes)
- Realistic transaction costs (commission + slippage)
- Position sizing and risk management
- Comprehensive metrics calculation

Target metrics:
- Sharpe Ratio > 1.0
- Profit Factor > 2.0
- Max Drawdown < 15%
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Protocol
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .data_loader import load_market_data, AVAILABLE_COINS, AVAILABLE_TIMEFRAMES, get_bars_per_year
from .trade_logger import TradeLogger, Trade
from .position_sizer import PositionSizer, RiskManager, PositionConfig, RiskLimits
from .metrics_calculator import MetricsCalculator, BacktestMetrics


class Strategy(Protocol):
    """Protocol for trading strategies."""
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals.
        
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with any additional features
        
        Returns
        -------
        pd.Series
            Signal series with values:
            - 1: Buy/Long
            - 0: Hold/No signal
            - -1: Sell/Short (if supported)
        """
        ...


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    initial_capital: float = 10000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005   # 0.05%
    
    # Position sizing
    position_size_pct: float = 0.1  # 10% of capital per trade
    max_position_pct: float = 0.25  # Max 25% in single position
    
    # Risk management
    use_stop_loss: bool = True
    stop_loss_pct: float = 0.02  # 2% stop loss
    use_take_profit: bool = False
    take_profit_pct: float = 0.04  # 4% take profit
    
    # Execution
    entry_on_close: bool = True  # Enter at close of signal bar
    
    # Output
    verbose: bool = True
    save_results: bool = True
    output_dir: str = 'backtest_results'


@dataclass
class BacktestResult:
    """Container for backtest results."""
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: List[Trade] = field(default_factory=list)
    metrics: Optional[BacktestMetrics] = None
    
    coin: str = ''
    timeframe: str = ''
    strategy_name: str = ''
    config: Optional[BacktestConfig] = None
    
    signals: pd.Series = field(default_factory=pd.Series)
    drawdown_series: pd.Series = field(default_factory=pd.Series)
    
    def save(self, output_dir: str = 'backtest_results'):
        """Save results to files."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        prefix = f"{self.strategy_name}_{self.coin}_{self.timeframe}"
        
        # Save equity curve
        self.equity_curve.to_csv(path / f"{prefix}_equity.csv")
        
        # Save trades
        if self.trades:
            trades_df = pd.DataFrame([t.to_dict() for t in self.trades])
            trades_df.to_csv(path / f"{prefix}_trades.csv", index=False)
        
        # Save metrics
        if self.metrics:
            self.metrics.save_json(str(path / f"{prefix}_metrics.json"))
        
        print(f"Results saved to {path}")


class BacktestEngine:
    """
    Production-ready backtesting engine.
    
    Features
    --------
    - Multi-asset support (14 coins, 4 timeframes)
    - Realistic transaction costs
    - Position sizing and risk management
    - Comprehensive metrics
    
    Example
    -------
    >>> engine = BacktestEngine()
    >>> result = engine.run(strategy, coin='BTC', timeframe='15m')
    >>> result.metrics.print_summary()
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        
        # Initialize components
        self.position_sizer = PositionSizer(PositionConfig(
            method='fixed_pct',
            fixed_pct=self.config.position_size_pct,
            max_position_pct=self.config.max_position_pct,
        ))
        
        self.risk_manager = RiskManager(RiskLimits(
            max_position_pct=self.config.max_position_pct,
            stop_loss_pct=self.config.stop_loss_pct,
        ))
        
        self.trade_logger = TradeLogger()
    
    def run(
        self,
        strategy: Strategy,
        coin: str = 'BTC',
        timeframe: str = '15m',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        data_dir: str = 'data/raw',
    ) -> BacktestResult:
        """
        Run backtest on single coin/timeframe.
        
        Parameters
        ----------
        strategy : Strategy
            Trading strategy with generate_signals() method
        coin : str
            Coin symbol (e.g., 'BTC', 'ETH', 'ENA')
        timeframe : str
            Timeframe ('15m', '1h', '4h', '1d')
        start_date, end_date : str, optional
            Date range filter (YYYY-MM-DD)
        data_dir : str
            Path to data directory
        
        Returns
        -------
        BacktestResult
            Complete backtest results
        """
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Running backtest: {strategy.__class__.__name__} on {coin}_{timeframe}")
            print(f"{'='*60}")
        
        # Load data
        df = load_market_data(coin, timeframe, start_date, end_date, data_dir)
        
        if self.config.verbose:
            print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
        
        # Generate signals
        signals = strategy.generate_signals(df)
        
        if self.config.verbose:
            signal_counts = signals.value_counts()
            print(f"Signals generated: {dict(signal_counts)}")
        
        # Run simulation
        equity_curve, trades = self._simulate(df, signals, coin, timeframe)
        
        # Calculate metrics
        bars_per_year = get_bars_per_year(timeframe)
        metrics = MetricsCalculator.calculate_all(
            equity_curve=equity_curve,
            trades=trades,
            initial_capital=self.config.initial_capital,
            bars_per_year=bars_per_year,
            strategy=strategy.__class__.__name__,
            coin=coin,
            timeframe=timeframe,
        )
        
        # Create result
        result = BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            metrics=metrics,
            coin=coin,
            timeframe=timeframe,
            strategy_name=strategy.__class__.__name__,
            config=self.config,
            signals=signals,
        )
        
        if self.config.verbose:
            metrics.print_summary()
        
        if self.config.save_results:
            result.save(self.config.output_dir)
        
        return result
    
    def _simulate(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        coin: str,
        timeframe: str,
    ) -> tuple[pd.DataFrame, List[Trade]]:
        """Run trading simulation."""
        # Reset state
        self.trade_logger = TradeLogger()
        self.risk_manager.reset(self.config.initial_capital)
        
        capital = self.config.initial_capital
        position = 0.0  # Current position size
        entry_price = 0.0
        
        equity_values = []
        timestamps = []
        
        for i in range(len(df)):
            timestamp = df.index[i]
            row = df.iloc[i]
            price = row['close']
            
            # Get signal (aligned by index)
            if timestamp in signals.index:
                signal = signals.loc[timestamp]
            else:
                signal = 0
            
            # Current portfolio value
            portfolio_value = capital + position * price
            
            # Update risk manager
            self.risk_manager.update(timestamp, portfolio_value)
            
            # Check stop loss / take profit if in position
            if position > 0:
                pnl_pct = (price - entry_price) / entry_price
                
                # Stop loss
                if self.config.use_stop_loss and pnl_pct <= -self.config.stop_loss_pct:
                    capital, position = self._close_position(
                        timestamp, price, capital, position, coin, timeframe, 'stop_loss'
                    )
                
                # Take profit
                elif self.config.use_take_profit and pnl_pct >= self.config.take_profit_pct:
                    capital, position = self._close_position(
                        timestamp, price, capital, position, coin, timeframe, 'take_profit'
                    )
            
            # Process signal
            if signal == 1 and position == 0:
                # Open long position
                capital, position, entry_price = self._open_position(
                    timestamp, price, capital, coin, timeframe
                )
            
            elif signal == -1 and position > 0:
                # Close position on sell signal
                capital, position = self._close_position(
                    timestamp, price, capital, position, coin, timeframe, 'signal'
                )
            
            # Record equity
            portfolio_value = capital + position * price
            equity_values.append(portfolio_value)
            timestamps.append(timestamp)
        
        # Close any remaining position at end
        if position > 0:
            final_price = df['close'].iloc[-1]
            final_timestamp = df.index[-1]
            capital, position = self._close_position(
                final_timestamp, final_price, capital, position, coin, timeframe, 'end'
            )
            equity_values[-1] = capital
        
        # Create equity curve
        equity_curve = pd.DataFrame({
            'value': equity_values
        }, index=timestamps)
        
        return equity_curve, self.trade_logger.trades
    
    def _open_position(
        self,
        timestamp: datetime,
        price: float,
        capital: float,
        coin: str,
        timeframe: str,
    ) -> tuple[float, float, float]:
        """Open a new position."""
        # Calculate position size
        size = self.position_sizer.calculate_size(capital, price)
        position_value = size * price
        
        # Check risk limits
        can_trade, reason = self.risk_manager.can_open_trade(position_value, capital)
        if not can_trade:
            return capital, 0.0, 0.0
        
        # Calculate costs
        commission = position_value * self.config.commission
        slippage = position_value * self.config.slippage
        total_cost = commission + slippage
        
        # Adjust for slippage (worse entry)
        adjusted_price = price * (1 + self.config.slippage)
        
        # Execute
        capital -= position_value + total_cost
        
        # Log trade
        self.trade_logger.open_trade(
            entry_time=timestamp,
            entry_price=adjusted_price,
            size=size,
            direction=1,
            commission=commission,
            slippage=slippage,
            strategy='backtest',
            coin=coin,
            timeframe=timeframe,
        )
        
        self.risk_manager.on_trade_open()
        
        return capital, size, adjusted_price
    
    def _close_position(
        self,
        timestamp: datetime,
        price: float,
        capital: float,
        position: float,
        coin: str,
        timeframe: str,
        reason: str = 'signal',
    ) -> tuple[float, float]:
        """Close existing position."""
        # Calculate costs
        position_value = position * price
        commission = position_value * self.config.commission
        
        # Adjust for slippage (worse exit)
        adjusted_price = price * (1 - self.config.slippage)
        actual_value = position * adjusted_price
        
        # Execute
        capital += actual_value - commission
        
        # Close trade in logger
        trade = self.trade_logger.close_trade(
            exit_time=timestamp,
            exit_price=adjusted_price,
            commission=commission,
        )
        
        if trade:
            self.risk_manager.on_trade_close(trade.pnl)
        
        return capital, 0.0
    
    def run_multi_asset(
        self,
        strategy: Strategy,
        coins: List[str],
        timeframe: str = '15m',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        data_dir: str = 'data/raw',
    ) -> Dict[str, BacktestResult]:
        """
        Run backtest on multiple coins.
        
        Parameters
        ----------
        strategy : Strategy
            Trading strategy
        coins : List[str]
            List of coin symbols
        timeframe : str
            Single timeframe
        start_date, end_date : str, optional
            Date range
        data_dir : str
            Data directory
        
        Returns
        -------
        Dict[str, BacktestResult]
            Results for each coin
        """
        results = {}
        
        for coin in coins:
            try:
                result = self.run(
                    strategy=strategy,
                    coin=coin,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    data_dir=data_dir,
                )
                results[coin] = result
            except Exception as e:
                print(f"Error backtesting {coin}: {e}")
        
        # Print comparison
        if results and self.config.verbose:
            metrics_list = [r.metrics for r in results.values() if r.metrics]
            if metrics_list:
                comparison = MetricsCalculator.compare_strategies(metrics_list)
                print("\n" + "="*70)
                print("MULTI-ASSET COMPARISON")
                print("="*70)
                print(comparison.to_string(index=False))
        
        return results


def run_quick_backtest(
    strategy: Strategy,
    coin: str = 'BTC',
    timeframe: str = '15m',
    initial_capital: float = 10000,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> BacktestResult:
    """
    Quick backtest with default settings.
    
    Example
    -------
    >>> from strategies import HybridStrategy
    >>> result = run_quick_backtest(HybridStrategy(), coin='ENA', timeframe='15m')
    """
    config = BacktestConfig(
        initial_capital=initial_capital,
        verbose=True,
        save_results=True,
    )
    
    engine = BacktestEngine(config)
    return engine.run(strategy, coin, timeframe, start_date, end_date)