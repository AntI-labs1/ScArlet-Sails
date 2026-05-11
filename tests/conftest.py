"""
Pytest bootstrap for ScArlet-Sails.

Generates synthetic OHLCV fixtures the first time tests are run so that the
`tests/` suite is self-sufficient on a clean machine (CI, Kaggle, fresh laptop)
without requiring DVC-tracked real data.

Real data, when present in `data/raw/`, is never overwritten — the fixtures
are only created if a coin/TF parquet is missing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Subset of coins/timeframes the test suite actually touches. Synthetic data is
# small but enough for load/validate/metric tests.
_FIXTURE_COINS = ("BTC", "ETH", "SOL", "ALGO", "AVAX")
_FIXTURE_TIMEFRAMES = ("15m", "1h")

# pd.date_range frequency strings keyed by timeframe (pandas 2.2+ aliases).
_FREQ_MAP = {"15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}


def _synthetic_ohlcv(n_bars: int, freq: str, seed: int) -> pd.DataFrame:
    """Generate a deterministic OHLCV series for a single asset."""
    rng = np.random.default_rng(seed)
    # GBM-ish close
    drift = 0.00005
    vol = 0.004
    log_returns = rng.normal(loc=drift, scale=vol, size=n_bars)
    close = 30_000.0 * np.exp(np.cumsum(log_returns))
    # Build OHLC such that high >= max(open,close), low <= min(open,close)
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    spread = np.abs(rng.normal(scale=vol, size=n_bars)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    low = np.maximum(low, 1e-3)  # never negative
    volume = np.abs(rng.normal(loc=10.0, scale=3.0, size=n_bars))
    index = pd.date_range(start="2024-01-01", periods=n_bars, freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def _ensure_fixture(coin: str, timeframe: str) -> None:
    """Write a synthetic parquet for (coin, tf) if no real file is present."""
    canonical = DATA_DIR / f"{coin}_USDT_{timeframe}.parquet"
    binance = DATA_DIR / f"{coin}USDT_{timeframe}.parquet"
    if canonical.exists() or binance.exists():
        return  # real data wins
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    n_bars = 4_000 if timeframe == "15m" else 1_000
    freq = _FREQ_MAP[timeframe]
    seed = abs(hash((coin, timeframe))) % (2**32)
    df = _synthetic_ohlcv(n_bars=n_bars, freq=freq, seed=seed)
    df.index.name = "timestamp"
    df.to_parquet(canonical)


def _bootstrap_fixtures(coins: Iterable[str], timeframes: Iterable[str]) -> None:
    for coin in coins:
        for tf in timeframes:
            _ensure_fixture(coin, tf)


# Run at module import (pytest discovers conftest.py before test collection).
_bootstrap_fixtures(_FIXTURE_COINS, _FIXTURE_TIMEFRAMES)
