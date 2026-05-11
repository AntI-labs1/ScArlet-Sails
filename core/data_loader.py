"""
Data Loader for Scarlet Sails Backtesting Framework

Loads OHLCV data from parquet files.
Data location: data/raw/{COIN}_USDT_{TIMEFRAME}.parquet

Available coins: ALGO, AVAX, BTC, DOT, ENA, ETH, HBAR, LDO, LINK, LTC, ONDO, SOL, SUI, UNI
Available timeframes: 15m, 1h, 4h, 1d
Total combinations: 56
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict

AVAILABLE_COINS = [
    # Crypto (legacy — оставлены как baseline; см. revision 2026-05)
    'ALGO', 'AVAX', 'BTC', 'DOT', 'ENA', 'ETH',
    'HBAR', 'LDO', 'LINK', 'LTC', 'ONDO', 'SOL', 'SUI', 'UNI',
    # Precious metals (pivot 2026-05) — fetched via scripts/fetch_metals.py
    'GOLD', 'SILVER', 'COPPER', 'PLATINUM',
]

AVAILABLE_TIMEFRAMES = ['15m', '1h', '4h', '1d']

# Timeframe to annual multiplier (for annualized returns)
TIMEFRAME_MULTIPLIERS = {
    '15m': 365 * 24 * 4,   # 35040 bars/year
    '1h': 365 * 24,        # 8760 bars/year
    '4h': 365 * 6,         # 2190 bars/year
    '1d': 365,             # 365 bars/year
}


def validate_params(coin: str, timeframe: str) -> None:
    """Validate coin and timeframe parameters."""
    if coin not in AVAILABLE_COINS:
        raise ValueError(
            f"Invalid coin: {coin}. Available: {AVAILABLE_COINS}"
        )
    if timeframe not in AVAILABLE_TIMEFRAMES:
        raise ValueError(
            f"Invalid timeframe: {timeframe}. Available: {AVAILABLE_TIMEFRAMES}"
        )


def load_market_data(
    coin: str,
    timeframe: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    data_dir: str = 'data/raw'
) -> pd.DataFrame:
    """
    Load OHLCV data for specific coin/timeframe.
    
    Parameters
    ----------
    coin : str
        Coin symbol (e.g., 'BTC', 'ETH', 'ENA')
    timeframe : str
        Timeframe (e.g., '15m', '1h', '4h', '1d')
    start_date : str, optional
        Start date filter (YYYY-MM-DD)
    end_date : str, optional
        End date filter (YYYY-MM-DD)
    data_dir : str
        Path to data directory
    
    Returns
    -------
    pd.DataFrame
        OHLCV data with DatetimeIndex
        Columns: ['open', 'high', 'low', 'close', 'volume']
    
    Raises
    ------
    ValueError
        If coin/timeframe invalid or data validation fails
    FileNotFoundError
        If data file not found
    """
    validate_params(coin, timeframe)

    # Support both naming conventions used in the repo:
    #   canonical: BTC_USDT_15m.parquet
    #   Binance-style / DVC-tracked: BTCUSDT_15m.parquet
    base = Path(data_dir)
    candidates = [
        base / f'{coin}_USDT_{timeframe}.parquet',
        base / f'{coin}USDT_{timeframe}.parquet',
    ]
    path = next((p for p in candidates if p.exists()), None)

    if path is None:
        searched = '\n  '.join(str(p.absolute()) for p in candidates)
        raise FileNotFoundError(
            f"Data file not found for {coin}/{timeframe}. Tried:\n  {searched}"
        )

    df = pd.read_parquet(path)
    
    # Validate columns
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        else:
            raise ValueError(
                f"Index must be DatetimeIndex, got {type(df.index).__name__}"
            )
    
    # Sort by time
    df = df.sort_index()
    
    # Apply date filters with timezone awareness
    if start_date:
        start_ts = pd.to_datetime(start_date)
        if df.index.tz is not None and start_ts.tz is None:
            start_ts = start_ts.tz_localize(df.index.tz)
        df = df[df.index >= start_ts]
    if end_date:
        end_ts = pd.to_datetime(end_date)
        if df.index.tz is not None and end_ts.tz is None:
            end_ts = end_ts.tz_localize(df.index.tz)
        df = df[df.index <= end_ts]
    
    if len(df) == 0:
        raise ValueError(
            f"No data for {coin} {timeframe} "
            f"between {start_date} and {end_date}"
        )
    
    return df[required]


def load_multiple_assets(
    coins: List[str],
    timeframe: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    data_dir: str = 'data/raw'
) -> Dict[str, pd.DataFrame]:
    """
    Load data for multiple coins.
    
    Parameters
    ----------
    coins : List[str]
        List of coin symbols
    timeframe : str
        Single timeframe
    start_date, end_date : str, optional
        Date filters
    data_dir : str
        Path to data directory
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping coin -> DataFrame
    """
    result = {}
    errors = []
    
    for coin in coins:
        try:
            result[coin] = load_market_data(
                coin, timeframe, start_date, end_date, data_dir
            )
        except Exception as e:
            errors.append(f"{coin}: {e}")
    
    if errors and not result:
        raise ValueError(f"Failed to load any data:\n" + "\n".join(errors))
    
    if errors:
        print(f"Warning: Failed to load {len(errors)} assets:")
        for err in errors:
            print(f"  {err}")
    
    return result


def get_data_info(data_dir: str = 'data/raw') -> Dict:
    """
    Get information about available data files.
    
    Returns
    -------
    Dict
        Information about available data
    """
    path = Path(data_dir)
    if path.exists():
        files = list(path.glob('*_USDT_*.parquet')) + list(path.glob('*USDT_*.parquet'))
        # Deduplicate (a single file matches both patterns if it uses underscore)
        files = sorted(set(files))
    else:
        files = []

    found_coins = set()
    found_timeframes = set()
    file_info = []

    for f in files:
        name = f.stem  # e.g., 'BTC_USDT_15m' or 'BTCUSDT_15m'
        parts = name.split('_')
        coin: Optional[str] = None
        tf: Optional[str] = None
        if len(parts) >= 3 and parts[1] == 'USDT':
            # canonical 'BTC_USDT_15m'
            coin, tf = parts[0], parts[2]
        elif len(parts) == 2 and parts[0].endswith('USDT'):
            # Binance-style 'BTCUSDT_15m'
            coin, tf = parts[0][:-4], parts[1]
        if coin is not None and tf is not None:
            found_coins.add(coin)
            found_timeframes.add(tf)
            file_info.append({
                'file': f.name,
                'coin': coin,
                'timeframe': tf,
                'size_mb': f.stat().st_size / (1024 * 1024)
            })
    
    return {
        'data_dir': str(path.absolute()),
        'total_files': len(files),
        'coins_found': sorted(found_coins),
        'timeframes_found': sorted(found_timeframes),
        'expected_coins': AVAILABLE_COINS,
        'expected_timeframes': AVAILABLE_TIMEFRAMES,
        'files': file_info
    }


def get_bars_per_year(timeframe: str) -> int:
    """Get number of bars per year for timeframe."""
    return TIMEFRAME_MULTIPLIERS.get(timeframe, 365)