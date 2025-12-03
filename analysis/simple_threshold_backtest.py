"""Simple threshold-based backtest utilities.

This module evaluates probabilistic signals by applying fixed thresholds and
computing trade-level metrics such as Sharpe ratio. It is intentionally light
weight compared to HonestBacktestV2 and is meant for quick threshold sweeps.
"""

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def calculate_sharpe_ratio(
    returns: np.ndarray,
    periods_per_year: Optional[float] = None,
) -> float:
    """Compute Sharpe ratio with optional annualisation.

    If ``periods_per_year`` is ``None`` the ratio is returned without scaling.
    When provided, the value is scaled by ``sqrt(periods_per_year)``. This is
    designed for trade-level returns where the effective frequency is unknown
    until runtime.
    """

    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    sharpe = mean_return / std_return

    if periods_per_year is not None:
        sharpe = sharpe * np.sqrt(periods_per_year)

    return float(sharpe)


def evaluate_threshold(
    probabilities: np.ndarray, df: pd.DataFrame, threshold: float
) -> Dict:
    """Evaluate performance for a given threshold on trade-level returns."""

    signals = probabilities >= threshold
    entries = df.loc[signals]

    if "fee_ret" in entries:
        trade_returns = entries["fee_ret"]
    elif "raw_ret" in entries:
        trade_returns = entries["raw_ret"]
    else:
        raise ValueError("DataFrame must contain fee_ret or raw_ret for evaluation")

    trade_returns = trade_returns.to_numpy()
    n_trades = len(trade_returns)

    total_return = trade_returns.sum()
    mean_return = trade_returns.mean() if n_trades > 0 else 0.0
    win_rate = (trade_returns > 0).mean() if n_trades > 0 else 0.0
    max_dd = float(entries.get("max_drawdown_pct", pd.Series([0.0])).max())

    trades_per_year: Optional[float]
    if isinstance(df.index, pd.DatetimeIndex) and n_trades > 1:
        days = (df.index.max() - df.index.min()).days
        days = days if days > 0 else 1
        trades_per_year = n_trades * (365.0 / days)
    else:
        trades_per_year = float(n_trades) if n_trades > 0 else None

    sharpe_plain = calculate_sharpe_ratio(trade_returns, periods_per_year=None)
    sharpe_annual = calculate_sharpe_ratio(trade_returns, periods_per_year=trades_per_year)

    return {
        "threshold": threshold,
        "n_trades": int(n_trades),
        "total_return": float(total_return),
        "mean_return": float(mean_return),
        "sharpe_ratio": float(sharpe_annual),
        "sharpe_ratio_plain": float(sharpe_plain),
        "max_drawdown_pct": max_dd,
        "win_rate": float(win_rate),
    }


def evaluate_thresholds(
    probabilities: np.ndarray, df: pd.DataFrame, thresholds: Iterable[float]
) -> Dict[float, Dict]:
    """Evaluate a grid of thresholds and return per-threshold metrics."""

    results: Dict[float, Dict] = {}
    for th in thresholds:
        results[th] = evaluate_threshold(probabilities, df, th)
    return results


def select_optimal_threshold(backtest_results: Dict[float, Dict], max_dd_limit: float) -> Dict:
    """Select the threshold that maximises Sharpe under a drawdown cap."""

    best_threshold = None
    best_sharpe = -np.inf

    for th, metrics in backtest_results.items():
        if metrics.get("max_drawdown_pct", 0) <= max_dd_limit and metrics.get("sharpe_ratio", -np.inf) > best_sharpe:
            best_sharpe = metrics["sharpe_ratio"]
            best_threshold = th

    all_sharpes = [m.get("sharpe_ratio", -np.inf) for m in backtest_results.values()]
    no_profitable = all(s <= 0 for s in all_sharpes)

    return {
        "threshold": best_threshold,
        "sharpe": best_sharpe,
        "backtest_metrics": backtest_results.get(best_threshold, {}),
        "no_profitable_threshold": no_profitable,
    }

