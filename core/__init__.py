"""
Core components for Scarlet Sails Backtesting Framework
"""

from .data_loader import load_market_data, AVAILABLE_COINS, AVAILABLE_TIMEFRAMES
from .backtest_engine import BacktestEngine, BacktestConfig, BacktestResult
from .metrics_calculator import MetricsCalculator, BacktestMetrics
from .position_sizer import PositionSizer, RiskManager
from .trade_logger import TradeLogger, Trade

# Canonical Pipeline (v3)
from .canonical_state import CanonicalState
from .canonical_pipeline import CanonicalPipeline
from .feature_registry import FeatureRegistry

all = [
    'load_market_data',
    'AVAILABLE_COINS',
    'AVAILABLE_TIMEFRAMES',
    'BacktestEngine',
    'BacktestConfig',
    'BacktestResult',
    'MetricsCalculator',
    'BacktestMetrics',
    'PositionSizer',
    'RiskManager',
    'TradeLogger',
    'Trade'
        'CanonicalState',
    'CanonicalPipeline',
    'FeatureRegistry',
]
