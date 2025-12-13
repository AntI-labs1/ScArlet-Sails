"""
Tests for data loader module.
"""
import pytest
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_loader import (
    load_market_data,
    load_multiple_assets,
    validate_params,
    get_data_info,
    AVAILABLE_COINS,
    AVAILABLE_TIMEFRAMES,
)


class TestValidation:
    """Test input validation."""
    
    def test_valid_coin(self):
        """Valid coin in available list."""
        assert 'BTC' in AVAILABLE_COINS
    
    def test_invalid_coin_raises(self):
        """Invalid coin raises ValueError."""
        with pytest.raises(ValueError):
            validate_params('INVALID_COIN_XYZ', '15m')
    
    def test_valid_timeframe(self):
        """Valid timeframe in list."""
        assert '15m' in AVAILABLE_TIMEFRAMES
    
    def test_invalid_timeframe_raises(self):
        """Invalid timeframe raises ValueError."""
        with pytest.raises(ValueError):
            validate_params('BTC', 'invalid_tf')
    
    def test_available_coins_list(self):
        """Available coins is a non-empty list."""
        assert isinstance(AVAILABLE_COINS, (list, tuple))
        assert len(AVAILABLE_COINS) > 0
    
    def test_available_timeframes_list(self):
        """Available timeframes is a non-empty list."""
        assert isinstance(AVAILABLE_TIMEFRAMES, (list, tuple))
        assert len(AVAILABLE_TIMEFRAMES) > 0


class TestDataLoading:
    """Test data loading functionality."""
    
    def test_load_btc_15m(self):
        """Load BTC 15m data."""
        df = load_market_data('BTC', '15m')
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_load_with_date_filter(self):
        """Test loading data with date filter."""
        df = load_market_data('BTC', '15m', start_date='2024-01-01')
        
        if df is not None and len(df) > 0:
            first_idx = df.index[0]
            target_date = pd.Timestamp('2024-01-01')
            
            if hasattr(first_idx, 'tzinfo') and first_idx.tzinfo is not None:
                first_idx = first_idx.tz_localize(None)
            
            assert first_idx >= target_date, "Start date filter failed"
    
    def test_load_invalid_coin_raises(self):
        """Loading invalid coin raises ValueError."""
        with pytest.raises(ValueError):
            load_market_data('NONEXISTENT_COIN_12345', '15m')
    
    def test_ohlc_relationships(self):
        """OHLC relationships are valid."""
        df = load_market_data('BTC', '15m')
        
        assert (df['high'] >= df['low']).all(), "High >= Low"
        assert (df['high'] >= df['open']).all(), "High >= Open"
        assert (df['high'] >= df['close']).all(), "High >= Close"
        assert (df['low'] <= df['open']).all(), "Low <= Open"
        assert (df['low'] <= df['close']).all(), "Low <= Close"
    
    def test_no_negative_values(self):
        """Prices and volume are non-negative."""
        df = load_market_data('BTC', '15m')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert (df[col] >= 0).all(), f"{col} has negative values"


class TestMultiAssetLoading:
    """Test loading multiple assets."""
    
    def test_load_multiple_coins(self):
        """Can load multiple coins at once."""
        coins = AVAILABLE_COINS[:2]
        result = load_multiple_assets(coins, '15m')
        
        assert isinstance(result, dict)
        assert len(result) > 0


class TestDataInfo:
    """Test data info methods."""
    
    def test_get_data_info(self):
        """Get info about available data."""
        info = get_data_info()
        assert isinstance(info, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])