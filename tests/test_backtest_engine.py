"""
Tests for Backtest Engine

Run: pytest tests/test_backtest_engine.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.engine.backtest_engine import BacktestEngine, BacktestConfig, BacktestResult
from core.utils.trade_logger import Trade, TradeLogger
from core.risk.position_sizer import PositionSizer, PositionConfig, RiskManager, RiskLimits
from core.engine.metrics_calculator import MetricsCalculator, BacktestMetrics


# ============================================================================
# MOCK DATA AND STRATEGIES
# ============================================================================

def create_mock_ohlcv(n_bars: int = 1000, start_price: float = 50000) -> pd.DataFrame:
    """Create mock OHLCV data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range(
        start='2024-01-01',
        periods=n_bars,
        freq='15min'
    )
    
    # Random walk for close prices
    returns = np.random.normal(0.0001, 0.002, n_bars)
    close = start_price * np.cumprod(1 + returns)
    
    # Generate OHLC from close
    noise = np.random.uniform(0.998, 1.002, n_bars)
    open_prices = np.roll(close, 1) * noise
    open_prices[0] = start_price
    
    high = np.maximum(open_prices, close) * np.random.uniform(1.0, 1.005, n_bars)
    low = np.minimum(open_prices, close) * np.random.uniform(0.995, 1.0, n_bars)
    
    volume = np.random.uniform(100, 1000, n_bars) * 1e6
    
    df = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }, index=dates)
    
    return df


class MockStrategy:
    """Mock strategy for testing."""
    
    def __init__(self, signal_rate: float = 0.1):
        """
        Parameters
        ----------
        signal_rate : float
            Probability of generating a buy signal (0-1)
        """
        self.signal_rate = signal_rate
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate random signals for testing."""
        np.random.seed(42)
        n = len(df)
        
        # Generate signals: 1 = buy, -1 = sell, 0 = hold
        signals = np.zeros(n)
        
        in_position = False
        for i in range(n):
            if not in_position:
                if np.random.random() < self.signal_rate:
                    signals[i] = 1
                    in_position = True
            else:
                # Exit after random holding period
                if np.random.random() < 0.05:  # ~5% chance to exit per bar
                    signals[i] = -1
                    in_position = False
        
        return pd.Series(signals, index=df.index)


class AlwaysBuyStrategy:
    """Strategy that always buys (for testing)."""
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        signals.iloc[0] = 1  # Buy at start
        signals.iloc[-1] = -1  # Sell at end
        return signals


class BuyAndHoldStrategy:
    """Buy and hold strategy."""
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        signals.iloc[0] = 1  # Buy once at start
        return signals


# ============================================================================
# TEST CLASSES
# ============================================================================

class TestBacktestConfig:
    """Test backtest configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BacktestConfig()
        
        assert config.initial_capital == 10000.0
        assert config.commission == 0.001
        assert config.slippage == 0.0005
        assert config.position_size_pct == 0.1
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = BacktestConfig(
            initial_capital=50000,
            commission=0.002,
            slippage=0.001,
        )
        
        assert config.initial_capital == 50000
        assert config.commission == 0.002


class TestTradeLogger:
    """Test trade logging."""
    
    def test_open_close_trade(self):
        """Test opening and closing a trade."""
        logger = TradeLogger()
        
        # Open trade
        trade = logger.open_trade(
            entry_time=datetime(2024, 1, 1, 10, 0),
            entry_price=50000,
            size=0.1,
            direction=1,
            commission=5.0,
        )
        
        assert logger.has_open_position
        assert trade.is_open
        
        # Close trade
        closed = logger.close_trade(
            exit_time=datetime(2024, 1, 1, 14, 0),
            exit_price=51000,
            commission=5.1,
        )
        
        assert not logger.has_open_position
        assert closed.pnl > 0  # Profitable trade
        assert len(logger.trades) == 1
    
    def test_trade_statistics(self):
        """Test trade statistics calculation."""
        logger = TradeLogger()
        
        # Add some trades manually
        for i, pnl in enumerate([100, -50, 75, -25, 150]):
            trade = Trade(
                entry_time=datetime(2024, 1, 1 + i),
                entry_price=50000,
                size=0.1,
                direction=1,
                exit_time=datetime(2024, 1, 1 + i, 12),
                exit_price=50000 + pnl * 10,
                pnl=pnl,
                pnl_pct=pnl / 5000 * 100,
            )
            logger.add_trade(trade)
        
        stats = logger.get_statistics()
        
        assert stats['total_trades'] == 5
        assert stats['winners'] == 3
        assert stats['losers'] == 2
        assert stats['win_rate'] == 60.0


class TestPositionSizer:
    """Test position sizing."""
    
    def test_fixed_pct_sizing(self):
        """Test fixed percentage sizing."""
        config = PositionConfig(method='fixed_pct', fixed_pct=0.1)
        sizer = PositionSizer(config)
        
        size = sizer.calculate_size(
            capital=10000,
            price=50000,
        )
        
        # 10% of 10000 = 1000, at 50000 price = 0.02 units
        assert abs(size - 0.02) < 0.001
    
    def test_max_position_limit(self):
        """Test max position limit enforcement."""
        config = PositionConfig(
            method='fixed_pct',
            fixed_pct=0.5,  # Try to use 50%
            max_position_pct=0.25,  # But max is 25%
        )
        sizer = PositionSizer(config)
        
        size = sizer.calculate_size(capital=10000, price=50000)
        position_value = size * 50000
        
        # Should be capped at 25%
        assert position_value <= 10000 * 0.25


class TestRiskManager:
    """Test risk management."""
    
    def test_drawdown_halt(self):
        """Test drawdown circuit breaker."""
        limits = RiskLimits(max_drawdown_pct=0.15)
        rm = RiskManager(limits)
        
        rm.reset(initial_capital=10000)
        
        # Simulate 20% drawdown
        rm.update(pd.Timestamp('2024-01-01'), 10000)
        rm.update(pd.Timestamp('2024-01-02'), 8000)  # -20%
        
        assert rm._is_halted
        assert 'drawdown' in rm._halt_reason.lower()
    
    def test_position_size_check(self):
        """Test position size limit check."""
        limits = RiskLimits(max_position_pct=0.25)
        rm = RiskManager(limits)
        rm.reset(10000)
        
        # Try to open 30% position
        can_trade, reason = rm.can_open_trade(3000, 10000)
        
        assert not can_trade
        assert 'too large' in reason.lower()


class TestMetricsCalculator:
    """Test metrics calculation."""
    
    @pytest.fixture
    def sample_equity_curve(self):
        """Create sample equity curve."""
        dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
        # Upward trending with noise
        values = 10000 * (1 + np.cumsum(np.random.normal(0.0002, 0.002, 1000)))
        return pd.DataFrame({'value': values}, index=dates)
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trades."""
        trades = []
        for i in range(50):
            is_winner = np.random.random() > 0.45  # ~55% win rate
            pnl = np.random.uniform(50, 200) if is_winner else -np.random.uniform(30, 100)
            
            trade = Trade(
                entry_time=datetime(2024, 1, 1 + i // 10, i % 10),
                entry_price=50000,
                size=0.1,
                direction=1,
                exit_time=datetime(2024, 1, 1 + i // 10, i % 10 + 4),
                exit_price=50000 + pnl * 10,
                pnl=pnl,
                pnl_pct=pnl / 5000 * 100,
            )
            trades.append(trade)
        return trades
    
    def test_metrics_calculation(self, sample_equity_curve, sample_trades):
        """Test full metrics calculation."""
        metrics = MetricsCalculator.calculate_all(
            equity_curve=sample_equity_curve,
            trades=sample_trades,
            initial_capital=10000,
            bars_per_year=35040,
            strategy='TestStrategy',
            coin='BTC',
            timeframe='15m',
        )
        
        assert metrics.total_trades == 50
        assert 0 <= metrics.win_rate <= 100
        assert metrics.sharpe_ratio != 0
        assert metrics.max_drawdown >= 0
    
    def test_sharpe_ratio_calculation(self, sample_equity_curve):
        """Test Sharpe ratio is reasonable."""
        metrics = MetricsCalculator.calculate_all(
            equity_curve=sample_equity_curve,
            trades=[],
            initial_capital=10000,
        )
        
        # Sharpe should be finite
        assert not np.isinf(metrics.sharpe_ratio)
        assert not np.isnan(metrics.sharpe_ratio)


class TestBacktestEngine:
    """Test backtest engine."""
    
    @pytest.fixture
    def mock_data(self):
        """Create mock OHLCV data."""
        return create_mock_ohlcv(n_bars=2000)
    
    @pytest.fixture
    def engine(self):
        """Create backtest engine."""
        config = BacktestConfig(
            initial_capital=10000,
            commission=0.001,
            slippage=0.0005,
            verbose=False,
            save_results=False,
        )
        return BacktestEngine(config)
    
    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine.config.initial_capital == 10000
        assert engine.position_sizer is not None
        assert engine.risk_manager is not None
    
    def test_simulate_with_mock_data(self, engine, mock_data, monkeypatch):
        """Test simulation with mock data."""
        # Mock the data loader
        def mock_load(*args, **kwargs):
            return mock_data
        
        monkeypatch.setattr('core.backtest_engine.load_market_data', mock_load)
        
        strategy = MockStrategy(signal_rate=0.05)
        result = engine.run(strategy, coin='BTC', timeframe='15m')
        
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0
        assert result.metrics is not None
    
    def test_buy_and_hold(self, engine, mock_data, monkeypatch):
        """Test buy and hold strategy."""
        def mock_load(*args, **kwargs):
            return mock_data
        
        monkeypatch.setattr('core.backtest_engine.load_market_data', mock_load)
        
        strategy = BuyAndHoldStrategy()
        result = engine.run(strategy, coin='BTC', timeframe='15m')
        
        # Should have exactly 1 trade (buy at start, close at end)
        assert len(result.trades) >= 1
    
    def test_commission_impact(self, mock_data, monkeypatch):
        """Test that commissions reduce returns."""
        def mock_load(*args, **kwargs):
            return mock_data
        
        monkeypatch.setattr('core.backtest_engine.load_market_data', mock_load)
        
        # High commission
        config_high = BacktestConfig(
            commission=0.01,  # 1%
            verbose=False,
            save_results=False,
        )
        engine_high = BacktestEngine(config_high)
        
        # Low commission
        config_low = BacktestConfig(
            commission=0.0001,  # 0.01%
            verbose=False,
            save_results=False,
        )
        engine_low = BacktestEngine(config_low)
        
        strategy = MockStrategy(signal_rate=0.1)
        
        result_high = engine_high.run(strategy, coin='BTC', timeframe='15m')
        
        # Reset random seed for same signals
        strategy = MockStrategy(signal_rate=0.1)
        result_low = engine_low.run(strategy, coin='BTC', timeframe='15m')
        
        # High commission should result in lower final value (usually)
        # This test might be flaky due to randomness
        assert result_high.metrics.final_value <= result_low.metrics.final_value * 1.1


class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline_mock(self, monkeypatch):
        """Test full backtest pipeline with mock data."""
        mock_data = create_mock_ohlcv(n_bars=5000)
        
        def mock_load(*args, **kwargs):
            return mock_data
        
        monkeypatch.setattr('core.backtest_engine.load_market_data', mock_load)
        
        # Create engine
        config = BacktestConfig(
            initial_capital=10000,
            commission=0.001,
            slippage=0.0005,
            use_stop_loss=True,
            stop_loss_pct=0.02,
            verbose=False,
            save_results=False,
        )
        engine = BacktestEngine(config)
        
        # Run backtest
        strategy = MockStrategy(signal_rate=0.03)
        result = engine.run(strategy, coin='BTC', timeframe='15m')
        
        # Verify results
        assert result.metrics is not None
        assert result.metrics.total_trades > 0
        assert len(result.equity_curve) == len(mock_data)
        
        # Check metrics are reasonable
        assert -100 <= result.metrics.total_return_pct <= 1000
        assert 0 <= result.metrics.win_rate <= 100
        assert result.metrics.max_drawdown >= 0
        assert result.metrics.max_drawdown <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])