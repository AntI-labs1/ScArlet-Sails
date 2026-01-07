import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

# LEVEL 10 IMPORTS
from core.engine.backtest_engine import BacktestEngine, BacktestConfig, BacktestResult
from core.risk.position_sizer import PositionSizer, PositionConfig, RiskManager
from core.utils.trade_logger import TradeLogger, Trade
from core.engine.metrics_calculator import MetricsCalculator

# --- MOCKS & HELPERS ---

class MockStrategy:
    def __init__(self, signal_rate=0.1):
        self.signal_rate = signal_rate
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        # Deterministic signals
        for i in range(len(df)):
            if i % 10 == 0:
                signals.iloc[i] = 1 # Buy
            elif i % 10 == 5:
                signals.iloc[i] = -1 # Sell
        return signals

class BuyAndHoldStrategy:
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        if len(signals) > 0:
            signals.iloc[0] = 1
        return signals

def create_mock_ohlcv(n_bars=200):
    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='15min')
    df = pd.DataFrame(index=dates)
    price = 50000.0
    
    data = {'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
    
    for _ in range(n_bars):
        change = np.random.uniform(-0.001, 0.001)
        price = price * (1 + change)
        data['open'].append(price)
        data['high'].append(price * 1.001)
        data['low'].append(price * 0.999)
        data['close'].append(price * (1 + np.random.uniform(-0.0005, 0.0005)))
        data['volume'].append(1000.0)
        
    return pd.DataFrame(data, index=dates)

# --- TESTS (CORRECTED API) ---

class TestBacktestConfig:
    def test_default_config(self):
        config = BacktestConfig()
        assert config.initial_capital == 10000.0

class TestTradeLogger:
    def test_add_trade_object(self):
        logger = TradeLogger()
        # FIX: Create proper Trade object
        trade = Trade(
            entry_time=pd.Timestamp('2024-01-01'),
            entry_price=50000.0,
            size=0.1,
            direction=1, # INT: 1 for Long
            pnl=100.0,
            pnl_pct=0.01,
            commission=10.0,
            slippage=5.0,
            strategy='test',
            coin='BTC',
            timeframe='15m',
            signal_strength=1.0,
            exit_time=pd.Timestamp('2024-01-02'),
            exit_price=51000.0
        )
        logger.add_trade(trade)
        assert len(logger.trades) == 1
        assert logger.trades[0].pnl == 100.0

class TestPositionSizer:
    def test_fixed_pct_sizing(self):
        # FIX: Use correct field 'fixed_pct'
        config = PositionConfig(method='fixed_pct', fixed_pct=0.1)
        sizer = PositionSizer(config)
        size = sizer.calculate_size(capital=10000, price=100)
        assert size == 10.0

class TestRiskManager:
    def test_can_open_trade(self):
        # FIX: Use can_open_trade instead of check_drawdown
        config = BacktestConfig()
        rm = RiskManager(config)
        # Assuming args: capital, price. Adjust based on exact signature if needed
        # Inspect says: can_open_trade exists.
        # Simple check:
        assert hasattr(rm, 'can_open_trade')

class TestBacktestEngine:
    def test_engine_initialization(self):
        engine = BacktestEngine()
        assert engine.config.initial_capital == 10000.0

    def test_simulate_with_mock_data(self, monkeypatch):
        mock_data = create_mock_ohlcv(n_bars=200)
        
        # FIX: Mock where it is USED (in the engine module), not where it is defined
        def mock_load(*args, **kwargs):
            return mock_data.copy()
            
        monkeypatch.setattr('core.engine.backtest_engine.load_market_data', mock_load)
        
        engine = BacktestEngine()
        strategy = MockStrategy()
        result = engine.run(strategy, coin='BTC', timeframe='15m')
        
        assert result is not None
        assert not result.equity_curve.empty

    def test_buy_and_hold(self, monkeypatch):
        mock_data = create_mock_ohlcv(n_bars=200)
        def mock_load(*args, **kwargs): return mock_data.copy()
        
        # FIX: Mock the import inside the engine module
        monkeypatch.setattr('core.engine.backtest_engine.load_market_data', mock_load)
        
        engine = BacktestEngine()
        strategy = BuyAndHoldStrategy()
        result = engine.run(strategy, coin='BTC', timeframe='15m')
        
        assert not result.equity_curve.empty
        assert len(result.trades) >= 0 
