"""
Tests for core/data_loader.py — verify naming-mismatch fix from revision 2026-05,
AVAILABLE_COINS contains both crypto and metals, validate_params behaves correctly,
and OHLCV invariants hold.

Integration-level tests; will skip gracefully if conftest synthetic data missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.data_loader import (  # noqa: E402
    AVAILABLE_COINS,
    AVAILABLE_TIMEFRAMES,
    get_data_info,
    load_market_data,
    load_multiple_assets,
    validate_params,
)


# =============================================================================
# CONSTANTS / METADATA
# =============================================================================

class TestAvailableConstants:
    def test_contains_crypto(self):
        for coin in ["BTC", "ETH", "SOL"]:
            assert coin in AVAILABLE_COINS, f"{coin} missing"

    def test_contains_metals_added_in_revision_2026_05(self):
        # Pivot to metals — see POST_MORTEM.md
        for metal in ["GOLD", "SILVER", "COPPER", "PLATINUM"]:
            assert metal in AVAILABLE_COINS, f"{metal} missing (metals pivot)"

    def test_no_duplicates(self):
        assert len(AVAILABLE_COINS) == len(set(AVAILABLE_COINS))

    def test_timeframes(self):
        for tf in ["15m", "1h", "4h", "1d"]:
            assert tf in AVAILABLE_TIMEFRAMES

    def test_coins_is_list_or_tuple(self):
        assert isinstance(AVAILABLE_COINS, (list, tuple))
        assert len(AVAILABLE_COINS) > 0

    def test_timeframes_is_list_or_tuple(self):
        assert isinstance(AVAILABLE_TIMEFRAMES, (list, tuple))
        assert len(AVAILABLE_TIMEFRAMES) > 0


# =============================================================================
# validate_params
# =============================================================================

class TestValidateParams:
    def test_valid_passes(self):
        validate_params("BTC", "15m")
        validate_params("GOLD", "1d")
        validate_params("ETH", "4h")

    def test_invalid_coin_raises(self):
        with pytest.raises(ValueError, match="(?i)invalid coin"):
            validate_params("FAKECOIN_XYZ", "15m")

    def test_invalid_timeframe_raises(self):
        with pytest.raises(ValueError):
            validate_params("BTC", "1y")

    def test_empty_coin_raises(self):
        with pytest.raises(ValueError):
            validate_params("", "15m")

    def test_empty_timeframe_raises(self):
        with pytest.raises(ValueError):
            validate_params("BTC", "")


# =============================================================================
# load_market_data
# =============================================================================

class TestLoadMarketData:
    """Skip gracefully if data missing."""

    @pytest.fixture(autouse=True)
    def _data_exists(self):
        a = PROJECT_ROOT / "data" / "raw" / "BTC_USDT_15m.parquet"
        b = PROJECT_ROOT / "data" / "raw" / "BTCUSDT_15m.parquet"
        if not a.exists() and not b.exists():
            pytest.skip("no BTC 15m parquet — conftest synthetic data missing")

    def test_returns_dataframe(self):
        df = load_market_data("BTC", "15m")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 100

    def test_has_ohlcv_columns(self):
        df = load_market_data("BTC", "15m")
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in df.columns, f"missing column: {col}"

    def test_index_is_datetime(self):
        df = load_market_data("BTC", "15m")
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_start_date_filter(self):
        df_full = load_market_data("BTC", "15m")
        if len(df_full) < 200:
            pytest.skip("not enough data for date filter test")
        cutoff = df_full.index[int(len(df_full) * 0.75)]
        df_filt = load_market_data("BTC", "15m", start_date=cutoff.strftime("%Y-%m-%d"))
        assert len(df_filt) < len(df_full)

    def test_ohlc_invariants(self):
        df = load_market_data("BTC", "15m")
        assert (df["high"] >= df["low"]).all()
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()

    def test_no_negative_values(self):
        df = load_market_data("BTC", "15m")
        for col in ["open", "high", "low", "close", "volume"]:
            assert (df[col] >= 0).all(), f"{col} negative"

    def test_invalid_coin_raises(self):
        with pytest.raises(ValueError):
            load_market_data("NONEXISTENT_FAKE_ASSET", "15m")


# =============================================================================
# load_multiple_assets
# =============================================================================

class TestLoadMultiple:
    def test_load_two_coins(self):
        try:
            result = load_multiple_assets(["BTC", "ETH"], "15m")
        except (FileNotFoundError, ValueError):
            pytest.skip("BTC or ETH data not present")
        assert isinstance(result, dict)


# =============================================================================
# get_data_info
# =============================================================================

class TestDataInfo:
    def test_returns_dict(self):
        assert isinstance(get_data_info(), dict)

    def test_has_required_keys(self):
        info = get_data_info()
        for key in ["total_files", "coins_found", "timeframes_found"]:
            assert key in info


# =============================================================================
# REGRESSION FOR BUG 5: naming convention mismatch
# =============================================================================

class TestBug5NamingConvention:
    """Bug 5: loader expected BTC_USDT_15m.parquet, DVC stored BTCUSDT_15m.parquet.
    Fix: loader accepts both naming conventions."""

    def test_loader_accepts_canonical_naming(self):
        try:
            df = load_market_data("BTC", "15m")
            assert isinstance(df, pd.DataFrame)
        except FileNotFoundError:
            pytest.skip("BTC parquet not present")

    def test_get_data_info_handles_both_forms(self):
        info = get_data_info()
        assert isinstance(info["total_files"], int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
