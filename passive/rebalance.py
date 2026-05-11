#!/usr/bin/env python3
"""
Passive portfolio quarterly rebalance calculator (Track D, ScArlet-Sails).

После закрытия активного-trading трека проекта (см. ../POST_MORTEM.md),
capital allocation перешёл на passive risk-parity portfolios. Этот скрипт —
единственный инструмент, который нужно запускать раз в квартал.

Использование:
    # Просто посмотреть target allocation
    python passive/rebalance.py --portfolio 60_40_us --total 100000

    # Реальная ребалансировка: указать текущие значения позиций
    python passive/rebalance.py --portfolio 60_40_us \
        --current "SPY:62000,TLT:38000"

    # Список доступных портфелей
    python passive/rebalance.py --list

    # Проверить нужна ли ребалансировка (drift-based)
    python passive/rebalance.py --portfolio 60_40_us \
        --current "SPY:65000,TLT:35000" --check-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import yaml

PORTFOLIOS_FILE = Path(__file__).parent / "portfolios.yaml"


def load_portfolios() -> dict:
    with open(PORTFOLIOS_FILE, "r") as f:
        return yaml.safe_load(f)["portfolios"]


def parse_current(spec: Optional[str]) -> Dict[str, float]:
    """Parse 'SPY:60000,TLT:42000' → {'SPY': 60000.0, 'TLT': 42000.0}."""
    if not spec:
        return {}
    out: Dict[str, float] = {}
    for pair in spec.split(","):
        ticker, value = pair.strip().split(":")
        out[ticker.strip()] = float(value)
    return out


def show_target_allocation(portfolio_id: str, total: float) -> None:
    """Print just the target allocation for a given total."""
    portfolios = load_portfolios()
    if portfolio_id not in portfolios:
        print(f"Error: portfolio '{portfolio_id}' not found.", file=sys.stderr)
        print(f"Available: {list(portfolios)}", file=sys.stderr)
        sys.exit(2)

    p = portfolios[portfolio_id]
    print(f"\n=== {p['name']} ===")
    print(f"Description: {p['description']}")
    print(f"Rebalance freq: {p['rebalance_freq']}")
    print()
    print(f"{'Ticker':<8} {'Weight':>8} {'Target $':>14}")
    print("-" * 32)
    for ticker, weight in p["assets"].items():
        print(f"{ticker:<8} {weight*100:>7.1f}% {total*weight:>13,.0f}")
    print()
    print(f"{'TOTAL':<8} {'100.0%':>8} {total:>13,.0f}")


def calc_rebalance(
    portfolio_id: str,
    current: Dict[str, float],
    check_only: bool = False,
) -> None:
    """Calculate orders needed to reach target allocation."""
    portfolios = load_portfolios()
    if portfolio_id not in portfolios:
        print(f"Error: portfolio '{portfolio_id}' not found.", file=sys.stderr)
        sys.exit(2)

    p = portfolios[portfolio_id]
    target_assets = p["assets"]
    drift_threshold = p.get("drift_threshold_pct", 5.0) / 100.0

    # Validate that current positions match portfolio definition
    extra = set(current) - set(target_assets)
    if extra:
        print(f"Warning: current positions {extra} not in target portfolio. Ignoring.")

    # Total current value
    total = sum(current.get(t, 0) for t in target_assets)
    if total <= 0:
        print(f"Error: no current value. Use --total instead.", file=sys.stderr)
        sys.exit(2)

    # Compute drift per asset
    print(f"\n=== {p['name']} — Rebalance Check ===")
    print(f"Total current value: ${total:,.0f}")
    print(f"Drift threshold:     {drift_threshold*100:.1f}%")
    print()
    print(f"{'Ticker':<8} {'Current':>12} {'Target':>12} {'Drift':>9} {'Action':>14}")
    print("-" * 60)

    any_rebalance_needed = False
    orders = []
    for ticker, target_weight in target_assets.items():
        current_value = current.get(ticker, 0)
        current_weight = current_value / total
        drift = current_weight - target_weight
        target_value = total * target_weight
        delta = target_value - current_value

        if abs(drift) > drift_threshold:
            any_rebalance_needed = True

        action_str = ""
        if delta > 1.0:
            action_str = f"BUY ${delta:,.0f}"
            orders.append(("BUY", ticker, delta))
        elif delta < -1.0:
            action_str = f"SELL ${-delta:,.0f}"
            orders.append(("SELL", ticker, -delta))
        else:
            action_str = "—"

        marker = " *" if abs(drift) > drift_threshold else ""
        print(
            f"{ticker:<8} {current_value:>11,.0f} {target_value:>11,.0f} "
            f"{drift*100:>+8.1f}% {action_str:>14}{marker}"
        )

    print()
    if check_only:
        if any_rebalance_needed:
            print("STATUS: REBALANCE NEEDED (at least one position exceeds drift threshold)")
            sys.exit(1)
        else:
            print("STATUS: no rebalance needed (all positions within threshold)")
            sys.exit(0)

    if not orders:
        print("No orders needed — portfolio already at target allocation.")
        return

    print("Execute these orders in your broker:")
    for action, ticker, amount in orders:
        print(f"  {action:<4} {ticker}  ${amount:,.0f}")
    print()
    print("After execution, save current state for next quarter (e.g. in a journal).")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Passive portfolio rebalance calculator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--portfolio", "-p", help="Portfolio id from portfolios.yaml")
    parser.add_argument("--total", type=float, help="Total capital to allocate (when starting fresh)")
    parser.add_argument(
        "--current",
        help="Current positions as 'TICKER:value,TICKER:value' (e.g. 'SPY:60000,TLT:42000')",
    )
    parser.add_argument("--check-only", action="store_true",
                        help="Exit code 0 if no rebalance needed, 1 otherwise (good for cron)")
    parser.add_argument("--list", action="store_true", help="List available portfolios")
    args = parser.parse_args()

    if args.list:
        portfolios = load_portfolios()
        print("Available portfolios:")
        for pid, p in portfolios.items():
            print(f"  {pid:<20} — {p['name']}: {p['description']}")
        return 0

    if not args.portfolio:
        parser.error("--portfolio is required (or use --list)")

    if args.current:
        current = parse_current(args.current)
        calc_rebalance(args.portfolio, current, check_only=args.check_only)
    elif args.total:
        show_target_allocation(args.portfolio, args.total)
    else:
        parser.error("Either --current or --total must be provided")

    return 0


if __name__ == "__main__":
    sys.exit(main())
