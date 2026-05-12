#!/usr/bin/env python3
"""
Download real OHLCV data for precious metals via yfinance.

Источник: Yahoo Finance (бесплатно, без ключей).
Использует **futures continuous contracts** (GC=F, SI=F, HG=F, PL=F),
у которых самая длинная история и реальный объём, не ETF-обёртка.

Использование:
    # Все 4 металла, 1d, ~10 лет (по умолчанию)
    python scripts/fetch_metals.py

    # Точечно: gold + silver, 1h, 2 года
    python scripts/fetch_metals.py --metals gold silver \
        --timeframes 1h --period 2y

Выход: data/raw/{NAME}_USDT_{TF}.parquet — naming совпадает с тем что ждёт
core.data_loader.load_market_data. "USDT" в имени — рудимент crypto-схемы,
сохраняем для совместимости с существующим pipeline.

Зависимости: yfinance (добавь в requirements.txt), pandas, pyarrow.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

try:
    import yfinance as yf  # type: ignore
except ImportError as e:  # pragma: no cover — only triggered without yf installed
    raise ImportError(
        "yfinance is required to download metals data. "
        "Install via `pip install yfinance>=0.2.40`."
    ) from e

logger = logging.getLogger("fetch_metals")

# Маппинг "human name" → (yfinance symbol, internal name for filename).
# Используем futures (=F) — они дают глубокую историю и реальный volume.
METALS: Dict[str, str] = {
    "gold": "GC=F",
    "silver": "SI=F",
    "copper": "HG=F",
    "platinum": "PL=F",
}

# yfinance interval strings.
TF_TO_YF: Dict[str, str] = {
    "1h": "1h",
    "1d": "1d",
    "1wk": "1wk",
}

# yfinance ограничивает интрадей-историю: 1h максимум 730 дней, 1d — полная.
# Дефолт для каждого TF — максимально доступный период.
TF_TO_PERIOD: Dict[str, str] = {
    "1h": "730d",  # ~2 года
    "1d": "max",
    "1wk": "max",
}


def _fetch_one(symbol: str, name: str, tf: str, period: str, out_dir: Path) -> Path:
    """Download a single (metal, timeframe) pair, write to parquet."""
    yf_interval = TF_TO_YF[tf]
    out_path = out_dir / f"{name.upper()}_USDT_{tf}.parquet"

    logger.info("[%s/%s] downloading %s, period=%s, interval=%s", name, tf, symbol, period, yf_interval)
    df = yf.download(
        tickers=symbol,
        period=period,
        interval=yf_interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned empty frame for {symbol} {yf_interval}/{period}")

    # yfinance может возвращать MultiIndex columns когда tickers — список.
    # Для одного тикера разворачиваем.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normalize column names to lower (project convention: open/high/low/close/volume).
    df = df.rename(columns={c: c.lower() for c in df.columns})

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing columns {missing} in yfinance response for {symbol}")

    df = df[required].dropna()

    # yfinance index — DatetimeIndex; нормализуем в UTC.
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"

    if df["close"].isna().any():
        raise ValueError(f"close has NaN after dropna — yfinance corrupt response for {symbol}")

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    logger.info(
        "[done] %s: %d bars, %s -> %s, %.1f KB",
        out_path.name,
        len(df),
        df.index[0].date(),
        df.index[-1].date(),
        out_path.stat().st_size / 1024,
    )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metals", nargs="+", choices=list(METALS), default=list(METALS),
                        help=f"Metals to fetch. Default: all {list(METALS)}")
    parser.add_argument("--timeframes", nargs="+", choices=list(TF_TO_YF), default=["1d"],
                        help="Timeframes (yfinance intraday limited to 730d). Default: 1d only.")
    parser.add_argument("--period", default=None,
                        help="yfinance period like '2y', '5y', 'max'. Default: max for daily, 730d for hourly.")
    parser.add_argument("--out-dir", default="data/raw", help="Output directory")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    out_dir = Path(args.out_dir)

    failures: List[str] = []
    for name in args.metals:
        symbol = METALS[name]
        for tf in args.timeframes:
            period = args.period or TF_TO_PERIOD[tf]
            try:
                _fetch_one(symbol, name, tf, period, out_dir)
            except Exception as e:  # noqa: BLE001 — surface to CLI summary
                logger.error("[FAIL] %s/%s: %s", name, tf, e)
                failures.append(f"{name}/{tf}")

    logger.info("Finished. %d ok, %d failed.", len(args.metals) * len(args.timeframes) - len(failures), len(failures))
    if failures:
        logger.error("Failed: %s", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
