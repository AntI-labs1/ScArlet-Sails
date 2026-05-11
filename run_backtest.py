#!/usr/bin/env python3
"""
Scarlet Sails Backtesting CLI — единый запуск через vectorbt-движок.

Примеры:
    python run_backtest.py --strategy rsi --coin BTC --timeframe 15m
    python run_backtest.py --strategy combined --coin ENA --timeframe 15m --start 2024-01-01
    python run_backtest.py --strategy rsi --coins BTC ETH SOL --timeframe 15m
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure project imports work when running from any cwd.
sys.path.insert(0, str(Path(__file__).parent))

from backtesting.vbt_engine import (  # noqa: E402
    VBTBacktestConfig,
    VBTBacktestEngine,
    VBTBacktestResult,
)
from core.data_loader import AVAILABLE_COINS, AVAILABLE_TIMEFRAMES  # noqa: E402
from core.ood_detector import OODDetector  # noqa: E402
from strategies import (  # noqa: E402
    CombinedStrategy,
    HybridStrategy,
    RuleBasedStrategy,
    SimpleBollingerStrategy,
    SimpleMAStrategy,
    SimpleRSIStrategy,
)


def _make_combined() -> CombinedStrategy:
    return CombinedStrategy(ood_detector=OODDetector())


STRATEGY_FACTORIES = {
    "rsi": SimpleRSIStrategy,
    "ma": SimpleMAStrategy,
    "bollinger": SimpleBollingerStrategy,
    "combined": _make_combined,
    "rule_based": RuleBasedStrategy,
    "hybrid": HybridStrategy,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scarlet Sails Backtesting Framework (vectorbt-based).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_backtest.py --strategy rsi --coin BTC --timeframe 15m
  python run_backtest.py --strategy combined --coin ENA --timeframe 15m --start 2024-01-01
  python run_backtest.py --strategy rsi --coins BTC ETH SOL --timeframe 15m
""",
    )

    parser.add_argument(
        "--strategy", "-s",
        choices=list(STRATEGY_FACTORIES),
        default="rsi",
        help="Trading strategy to use",
    )
    parser.add_argument("--coin", "-c", default="BTC", help="Single coin (e.g. BTC)")
    parser.add_argument(
        "--coins",
        nargs="+",
        help="Multiple coins for comparison (overrides --coin)",
    )
    parser.add_argument(
        "--timeframe", "-t",
        choices=AVAILABLE_TIMEFRAMES,
        default="15m",
        help="Timeframe",
    )
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=10_000, help="Initial capital")
    parser.add_argument(
        "--commission",
        type=float,
        default=0.001,
        help="Commission per trade (default 0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.0005,
        help="Slippage per trade (default 0.0005 = 0.05%%)",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=0.02,
        help="Stop-loss fraction (default 0.02 = 2%%); pass 0 to disable",
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=0.0,
        help="Take-profit fraction (default 0 = disabled)",
    )
    parser.add_argument(
        "--output", "-o",
        default="backtest_results",
        help="Output directory for CSV summary",
    )
    parser.add_argument("--data-dir", default="data/raw", help="OHLCV data directory")
    parser.add_argument("--quiet", "-q", action="store_true", help="Less verbose output")

    return parser.parse_args()


def _build_engine(args: argparse.Namespace) -> VBTBacktestEngine:
    sl = args.stop_loss if args.stop_loss and args.stop_loss > 0 else None
    tp = args.take_profit if args.take_profit and args.take_profit > 0 else None
    return VBTBacktestEngine(
        VBTBacktestConfig(
            initial_capital=args.capital,
            commission=args.commission,
            slippage=args.slippage,
            stop_loss_pct=sl,
            take_profit_pct=tp,
        )
    )


def _run_one(
    engine: VBTBacktestEngine,
    strategy_name: str,
    coin: str,
    args: argparse.Namespace,
) -> VBTBacktestResult | None:
    factory = STRATEGY_FACTORIES[strategy_name]
    try:
        result = engine.run(
            factory(),
            coin=coin,
            timeframe=args.timeframe,
            start_date=args.start,
            end_date=args.end,
            data_dir=args.data_dir,
        )
    except FileNotFoundError as e:
        print(f"[{coin}] data missing: {e}", file=sys.stderr)
        return None
    except Exception as e:  # noqa: BLE001 — surface to CLI
        print(f"[{coin}] backtest failed: {e}", file=sys.stderr)
        return None

    if not args.quiet:
        print()
        print("=" * 60)
        print(result.summary())
        print("=" * 60)
    return result


def _save_summary(
    results: dict[str, VBTBacktestResult],
    output_dir: str,
    strategy_name: str,
    timeframe: str,
) -> Path:
    rows = []
    for coin, r in results.items():
        rows.append({"coin": coin, "timeframe": timeframe, **r.metrics})
    df = pd.DataFrame(rows)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"vbt_{strategy_name}_{timeframe}_summary.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    coins = args.coins if args.coins else [args.coin]
    invalid = [c for c in coins if c not in AVAILABLE_COINS]
    if invalid:
        print(f"Unknown coin(s): {invalid}. Available: {AVAILABLE_COINS}", file=sys.stderr)
        return 2

    engine = _build_engine(args)

    results: dict[str, VBTBacktestResult] = {}
    for coin in coins:
        r = _run_one(engine, args.strategy, coin, args)
        if r is not None:
            results[coin] = r

    if not results:
        print("No successful backtests.", file=sys.stderr)
        return 1

    csv_path = _save_summary(results, args.output, args.strategy, args.timeframe)
    if not args.quiet:
        print(f"\nSummary CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
