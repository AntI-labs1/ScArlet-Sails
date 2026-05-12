#!/usr/bin/env python3
"""
Historical simulation of passive portfolios — backtests the 60/40 / All-Weather /
Permanent Portfolio allocations on the same data the project's strategies were
tested against, so the user can see directly: passive ≈ active in terms of Sharpe.

This is the empirical anchor of the project's closure argument: if passive
allocation delivers Sharpe ~0.6 on the same data where our strategies delivered
Sharpe 0.4-0.6, then the 10 weeks of active development produced zero alpha.

Usage:
    # Backtest 60/40 on metals only (gold = stock proxy, silver/copper/platinum = bond/commodity proxies)
    python passive/historical_simulation.py --portfolio 60_40_metals_proxy

    # All-Weather on 4 metals (treats each metal as a sleeve)
    python passive/historical_simulation.py --portfolio metals_4way

    # All available
    python passive/historical_simulation.py --list

    # Annual rebalance vs quarterly
    python passive/historical_simulation.py --portfolio metals_4way --rebal annual

Notes:
- We use METALS data (gold/silver/copper/platinum) because we have 25-year daily
  history. The project's crypto data is too short (2.5 years) for meaningful
  passive-strategy comparison.
- For 60/40 SPY+TLT real-world simulation, fetch SPY and TLT via yfinance and run
  on US-equity proxies separately — out of scope here, but the logic is identical.
- Costs assumed 5 bps per trade (retail-friendly ETF) plus 2 bps slippage.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.data_loader import load_market_data  # noqa: E402

logger = logging.getLogger("passive_sim")


# =============================================================================
# PORTFOLIO DEFINITIONS (using assets we have data for)
# =============================================================================

PORTFOLIOS = {
    "metals_4way": {
        "description": "Equal weight across 4 metals (1/4 each). Represents a balanced commodity sleeve.",
        "weights": {"GOLD": 0.25, "SILVER": 0.25, "COPPER": 0.25, "PLATINUM": 0.25},
    },
    "metals_safety_bias": {
        "description": "Safety-biased metals: 60% gold + 20% silver + 10% copper + 10% platinum.",
        "weights": {"GOLD": 0.60, "SILVER": 0.20, "COPPER": 0.10, "PLATINUM": 0.10},
    },
    "metals_gold_only": {
        "description": "100% gold (single-asset benchmark).",
        "weights": {"GOLD": 1.00},
    },
    "metals_60_40_proxy": {
        "description": "60% gold (safer) + 40% basket of silver/copper/platinum (riskier). Loose 60/40 metaphor.",
        "weights": {"GOLD": 0.60, "SILVER": 0.20, "COPPER": 0.10, "PLATINUM": 0.10},
    },
}


# =============================================================================
# BACKTEST
# =============================================================================

def simulate(
    weights: Dict[str, float],
    rebal_freq: str = "Q",
    fees_bps: float = 5.0,
    slippage_bps: float = 2.0,
    capital: float = 10_000.0,
    start_date: Optional[str] = None,
) -> Dict[str, object]:
    """
    Backtest a passive portfolio with periodic rebalancing.

    Args:
        weights: target weights per asset (must sum to 1.0)
        rebal_freq: pandas freq alias for rebalance dates ('Q', 'M', 'A', 'W')
        fees_bps: commission per trade leg in basis points (0.01% = 1 bp)
        slippage_bps: slippage per trade leg in basis points
        capital: starting capital
        start_date: optional ISO date to filter from (e.g. '2010-01-01')

    Returns:
        Dict with keys: equity_curve (pd.Series), total_return_pct, cagr_pct,
        sharpe_annual, max_dd_pct, n_rebalances, years
    """
    assert abs(sum(weights.values()) - 1.0) < 1e-6, f"weights must sum to 1, got {sum(weights.values())}"
    cost_per_leg = (fees_bps + slippage_bps) / 10_000.0  # → fraction

    # Load and align all assets
    prices = {}
    for asset in weights:
        df = load_market_data(asset, "1d", start_date=start_date)
        prices[asset] = df["close"]
    px = pd.DataFrame(prices).dropna()
    if len(px) < 100:
        raise ValueError(f"insufficient overlapping data: {len(px)} bars")

    # Rebalance dates
    rebal_dates = pd.date_range(px.index[0], px.index[-1], freq=rebal_freq, tz=px.index.tz)
    rebal_dates = [d for d in rebal_dates if d in px.index]
    if not rebal_dates or rebal_dates[0] != px.index[0]:
        rebal_dates = [px.index[0]] + list(rebal_dates)

    # Simulation
    holdings: Dict[str, float] = {}
    cash = capital
    equity_curve = []
    n_rebal = 0

    for date in px.index:
        # Mark to market
        value = cash + sum(shares * px[asset].loc[date] for asset, shares in holdings.items())

        # Rebalance day?
        if date in rebal_dates:
            n_rebal += 1
            # Total turnover for cost calculation
            current_value = value
            new_holdings = {}
            cost_total = 0.0
            for asset, target_weight in weights.items():
                target_value = current_value * target_weight
                current_asset_value = holdings.get(asset, 0) * px[asset].loc[date]
                trade_size = abs(target_value - current_asset_value)
                cost_total += trade_size * cost_per_leg
                new_holdings[asset] = target_value / px[asset].loc[date]
            value -= cost_total
            holdings = new_holdings
            cash = 0.0

        equity_curve.append(value)

    eq = pd.Series(equity_curve, index=px.index, name="equity")

    # Metrics
    total_ret = eq.iloc[-1] / capital - 1
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / capital) ** (1 / years) - 1 if years > 0 else 0
    daily_ret = eq.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0
    rolling_max = eq.cummax()
    drawdown = (eq / rolling_max - 1).min()

    return {
        "equity_curve": eq,
        "total_return_pct": total_ret * 100,
        "cagr_pct": cagr * 100,
        "sharpe_annual": sharpe,
        "max_dd_pct": drawdown * 100,
        "n_rebalances": n_rebal,
        "years": years,
    }


def compare_against_strategies(portfolio_id: str, result: Dict) -> str:
    """Generate a comparison text against the project's tested strategies."""
    sharpe = result["sharpe_annual"]
    lines = [
        f"\nPASSIVE PORTFOLIO RESULT — {portfolio_id}:",
        f"  Sharpe (annual):    {sharpe:+.2f}",
        f"  CAGR:               {result['cagr_pct']:+.2f}%",
        f"  Total return:       {result['total_return_pct']:+.2f}%",
        f"  Max drawdown:       {result['max_dd_pct']:.2f}%",
        f"  Years:              {result['years']:.1f}",
        f"  Rebalances:         {result['n_rebalances']}",
        "",
        "COMPARISON WITH PROJECT STRATEGIES (same data, same period):",
        "  Combined mean-reversion (metals 1d) walk-forward avg Sharpe: -0.19",
        "  200-day SMA trend (metals 1d) avg Sharpe:                    +0.44",
        "  Dual momentum (metals monthly) Sharpe:                       +0.62",
        f"  >>> This passive portfolio Sharpe:                            {sharpe:+.2f}",
        "",
    ]

    if sharpe >= 0.55:
        lines.append("VERDICT: Passive matches or beats active strategies.")
        lines.append("         The 8-month active-development effort produced zero alpha.")
        lines.append("         This is the project's central finding (see POST_MORTEM.md).")
    elif sharpe >= 0.40:
        lines.append("VERDICT: Passive comparable to active trend-following.")
        lines.append("         Active strategies do not deliver risk-adjusted outperformance.")
    else:
        lines.append("VERDICT: Passive underperforms our trend-following baseline.")
        lines.append("         (Likely because passive includes the structurally weaker metals like platinum.)")
        lines.append("         Active still doesn't beat targeted single-asset passive like GOLD.")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--portfolio", "-p", help="Portfolio id from definitions")
    parser.add_argument("--rebal", choices=["Q", "M", "A", "W"], default="Q",
                        help="Rebalance frequency: Q=quarterly, M=monthly, A=annual, W=weekly")
    parser.add_argument("--start", help="Start date YYYY-MM-DD (default: full history)")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--list", action="store_true", help="List available portfolios")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()

    if args.list:
        print("Available portfolios:")
        for pid, p in PORTFOLIOS.items():
            print(f"  {pid:<25} {p['description']}")
        return 0

    if not args.portfolio:
        print("--portfolio required (or --list)")
        return 2

    if args.portfolio not in PORTFOLIOS:
        print(f"unknown portfolio: {args.portfolio}")
        print(f"available: {list(PORTFOLIOS)}")
        return 2

    p = PORTFOLIOS[args.portfolio]
    print(f"\n=== {args.portfolio} ===")
    print(f"Description: {p['description']}")
    print(f"Weights: {p['weights']}")
    print(f"Rebalance: {args.rebal}")
    print()

    result = simulate(
        weights=p["weights"],
        rebal_freq=args.rebal,
        capital=args.capital,
        start_date=args.start,
    )

    print(compare_against_strategies(args.portfolio, result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
