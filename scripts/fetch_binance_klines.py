#!/usr/bin/env python3
"""
Download real OHLCV data from Binance Vision archive.

Источник: https://data.binance.vision/ — публичный архив klines, без ключей и
авторизации. Качаем месячные ZIP-CSV, склеиваем в один parquet на (монета, TF)
и кладём в data/raw/ в имени `{COIN}_USDT_{TF}.parquet` (совпадает с тем что
ждёт `core.data_loader.load_market_data`).

Использование:
    # Все 14 монет, 4 таймфрейма, последние 2 года (по умолчанию)
    python scripts/fetch_binance_klines.py

    # Только BTC/ETH, 15m, 2024 год
    python scripts/fetch_binance_klines.py --coins BTC ETH --timeframes 15m \
        --start 2024-01 --end 2024-12

    # Только swap, не перезаписывать существующее
    python scripts/fetch_binance_klines.py --skip-existing

Зависимости: только stdlib + pandas + pyarrow (уже в requirements.txt).
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import pandas as pd

logger = logging.getLogger("fetch_binance")

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
DEFAULT_COINS = [
    "BTC", "ETH", "SOL", "AVAX", "DOT", "LINK", "UNI", "LTC",
    "ALGO", "HBAR", "LDO", "SUI", "ENA", "ONDO",
]
DEFAULT_TIMEFRAMES = ["15m", "1h", "4h", "1d"]

# Binance Vision klines CSV columns (12-column legacy format).
_KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]


def _month_range(start: str, end: str) -> List[str]:
    """Inclusive list of YYYY-MM strings from start to end."""
    s = datetime.strptime(start, "%Y-%m")
    e = datetime.strptime(end, "%Y-%m")
    if e < s:
        raise ValueError(f"end {end} < start {start}")
    out: List[str] = []
    y, m = s.year, s.month
    while (y, m) <= (e.year, e.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            m, y = 1, y + 1
    return out


def _download_month(symbol: str, tf: str, ym: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
    """Download, parse, and normalize one month-ZIP. Returns canonical OHLCV
    (DatetimeIndex UTC), or None if the month isn't published yet (404)."""
    url = f"{BASE_URL}/{symbol}/{tf}/{symbol}-{tf}-{ym}.zip"
    for attempt in range(max_retries):
        try:
            req = Request(url, headers={"User-Agent": "scarlet-sails-fetch/0.2"})
            with urlopen(req, timeout=60) as resp:
                blob = resp.read()
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as fh:
                    raw = pd.read_csv(fh, header=None, names=_KLINE_COLUMNS)
            return _normalize(raw)
        except HTTPError as e:
            if e.code == 404:
                # Month not published yet — silently skip.
                return None
            logger.warning("HTTP %s on %s (attempt %d)", e.code, url, attempt + 1)
        except (URLError, TimeoutError) as e:
            logger.warning("Network error on %s: %s (attempt %d)", url, e, attempt + 1)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw Binance kline frame to canonical OHLCV.

    Binance Vision migrated klines timestamp from milliseconds to microseconds
    around 2025-01-01. Detection is per-frame (not per-batch) so concatenating
    pre- and post-migration months can't poison each other.
    """
    sample = int(df["open_time"].iloc[0])
    # ms for 2023+ are ~1.7e12; us for 2025+ are ~1.7e15. 1e14 cleanly separates.
    unit = "us" if sample > 10**14 else "ms"

    # CRITICAL: use .to_numpy() (drops the source RangeIndex) so the new
    # DatetimeIndex aligns positionally. Passing pd.Series with a different
    # index here makes pandas align-by-index → all values become NaN.
    out = pd.DataFrame(
        {
            "open": df["open"].astype(float).to_numpy(),
            "high": df["high"].astype(float).to_numpy(),
            "low": df["low"].astype(float).to_numpy(),
            "close": df["close"].astype(float).to_numpy(),
            "volume": df["volume"].astype(float).to_numpy(),
        },
        index=pd.to_datetime(df["open_time"].to_numpy(), unit=unit, utc=True),
    )
    out.index.name = "timestamp"
    # Defensive: if any close is NaN, the index-alignment bug is back.
    if out["close"].isna().any():
        raise ValueError(
            f"close has {out['close'].isna().sum()} NaN values after normalize; "
            "index alignment bug regressed."
        )
    # Sanity: every timestamp must fall in [2017, now+1] — guards against unit
    # detection bugs from changing Binance file formats.
    if (out.index.year < 2017).any() or (out.index.year > datetime.now(timezone.utc).year + 1).any():
        bad = out.index[(out.index.year < 2017) | (out.index.year > datetime.now(timezone.utc).year + 1)][:3]
        raise ValueError(
            f"timestamp out of plausible range — likely wrong unit detection. "
            f"sample={sample}, unit={unit!r}, examples={list(bad)}"
        )
    return out.sort_index()


def fetch_pair(
    coin: str,
    tf: str,
    months: List[str],
    out_dir: Path,
    skip_existing: bool = False,
) -> Path:
    """Download one (coin, tf) pair across the month range, write a single parquet."""
    symbol = f"{coin}USDT"
    out_path = out_dir / f"{coin}_USDT_{tf}.parquet"

    if skip_existing and out_path.exists():
        logger.info("[skip] %s already exists", out_path.name)
        return out_path

    frames: List[pd.DataFrame] = []
    for ym in months:
        df = _download_month(symbol, tf, ym)
        if df is None:
            continue
        frames.append(df)

    if not frames:
        logger.warning("[empty] no data for %s/%s in %s..%s", coin, tf, months[0], months[-1])
        return out_path

    # Each frame is already normalized (DatetimeIndex UTC) by _download_month.
    ohlcv = pd.concat(frames).sort_index()
    # de-dup any month-boundary overlap
    ohlcv = ohlcv[~ohlcv.index.duplicated(keep="first")]
    out_dir.mkdir(parents=True, exist_ok=True)
    ohlcv.to_parquet(out_path)
    logger.info(
        "[done] %s: %d bars, %s -> %s, %.1f MB",
        out_path.name,
        len(ohlcv),
        ohlcv.index[0].date(),
        ohlcv.index[-1].date(),
        out_path.stat().st_size / (1024 * 1024),
    )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--coins", nargs="+", default=DEFAULT_COINS,
                        help=f"Symbols to fetch. Default: {DEFAULT_COINS}")
    parser.add_argument("--timeframes", nargs="+", default=DEFAULT_TIMEFRAMES,
                        help=f"TFs to fetch. Default: {DEFAULT_TIMEFRAMES}")
    parser.add_argument("--start", default="2023-01", help="Earliest YYYY-MM (inclusive)")
    parser.add_argument("--end", default=None,
                        help="Latest YYYY-MM (inclusive). Default: previous month")
    parser.add_argument("--out-dir", default="data/raw", help="Output directory")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent (coin, tf) workers")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip pairs whose output parquet already exists")
    return parser.parse_args()


def _default_end_month() -> str:
    now = datetime.now(timezone.utc)
    # Binance Vision publishes last month around the 1st-3rd of the next month;
    # safest default is to ask for the prior month.
    y, m = now.year, now.month - 1
    if m == 0:
        m, y = 12, y - 1
    return f"{y:04d}-{m:02d}"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    end = args.end or _default_end_month()
    months = _month_range(args.start, end)
    out_dir = Path(args.out_dir)

    logger.info("Range: %s..%s (%d months)", months[0], months[-1], len(months))
    logger.info("Coins: %s", args.coins)
    logger.info("TFs:   %s", args.timeframes)
    logger.info("Out:   %s", out_dir.absolute())

    jobs = [(coin, tf) for coin in args.coins for tf in args.timeframes]
    failures: List[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_pair, c, tf, months, out_dir, args.skip_existing): (c, tf)
            for c, tf in jobs
        }
        for fut in as_completed(futures):
            c, tf = futures[fut]
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001 — surface to CLI summary
                logger.error("[FAIL] %s/%s: %s", c, tf, e)
                failures.append(f"{c}/{tf}")

    logger.info("Finished. %d ok, %d failed.", len(jobs) - len(failures), len(failures))
    if failures:
        logger.error("Failed pairs: %s", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
