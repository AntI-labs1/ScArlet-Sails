"""
Simple threshold backtest utilities for Model 2.

- По умолчанию считаем НЕ годовой Sharpe (raw), если periods_per_year=None.
- MaxDD считаем по equity curve в процентах (от пика до минимума).
"""

from typing import Iterable, List, Dict, Optional, Sequence

import numpy as np
import pandas as pd


def calculate_sharpe_ratio(
    returns: np.ndarray,
    periods_per_year: Optional[float] = None,
) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: массив доходностей сделок (или баров)
        periods_per_year:
            - None  → вернуть "raw" Sharpe = mean/std
            - число → умножить raw Sharpe на sqrt(periods_per_year)

    """
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0

    std = r.std(ddof=1)
    if std == 0:
        return 0.0

    mean = r.mean()
    raw = mean / std

    if periods_per_year is None:
        return float(raw)

    return float(raw * np.sqrt(periods_per_year))


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Max drawdown по equity curve в процентах.

    Args:
        equity_curve: массив значений equity (пример: [1.0, 1.1, 0.9, ...])

    Returns:
        Минимальный drawdown в процентах (отрицательное число, если просадка была).
    """
    eq = np.asarray(equity_curve, dtype=float)
    if eq.size == 0:
        return 0.0

    running_max = np.maximum.accumulate(eq)
    drawdowns = eq / running_max - 1.0  # от пика
    return float(drawdowns.min() * 100.0)


def evaluate_threshold(
    df: pd.DataFrame,
    proba_col: str,
    fee_ret_col: str,
    threshold: float,
) -> Dict:
    """
    Посчитать метрики по одному порогу.

    df: DataFrame с колонками:
        - proba_col  (например, 'P_ml')
        - fee_ret_col (например, 'fee_ret')
    """
    df = df.copy()

    mask = df[proba_col] >= threshold
    trades = df.loc[mask]

    n_trades = int(len(trades))
    if n_trades == 0:
        return {
            "threshold": float(threshold),
            "n_trades": 0,
            "total_return": 0.0,
            "mean_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate": 0.0,
        }

    trade_returns = trades[fee_ret_col].to_numpy(dtype=float)

    total_ret = float(trade_returns.sum())
    mean_ret = float(trade_returns.mean())
    win_rate = float((trade_returns > 0).mean() * 100.0)

    # Equity curve от 1.0
    equity = np.cumprod(1.0 + trade_returns)
    max_dd_pct = calculate_max_drawdown(equity)

    # Оценим "периодов в год":
    # берём календарный диапазон и считаем, сколько сделок приходится на год.
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 1:
        period_days = (idx.max() - idx.min()).total_seconds() / 86400.0
        if period_days > 0:
            trades_per_year = n_trades * (365.0 / period_days)
        else:
            trades_per_year = float(n_trades)
    else:
        trades_per_year = float(n_trades)

    sharpe = calculate_sharpe_ratio(trade_returns, periods_per_year=trades_per_year)

    return {
        "threshold": float(threshold),
        "n_trades": n_trades,
        "total_return": total_ret,
        "mean_return": mean_ret,
        "sharpe_ratio": float(sharpe),
        "max_drawdown_pct": float(max_dd_pct),
        "win_rate": win_rate,
    }


def evaluate_thresholds(
    df: pd.DataFrame,
    proba_col: str,
    fee_ret_col: str,
    thresholds: Iterable[float],
) -> List[Dict]:
    """Просто прогнать evaluate_threshold по сетке порогов."""
    return [
        evaluate_threshold(df, proba_col, fee_ret_col, t)
        for t in thresholds
    ]


def select_optimal_threshold(
    threshold_results: Sequence[Dict],
    *,
    max_dd_limit: float = 20.0,
    min_trades: int = 10,
) -> Dict:
    """
    Выбор "лучшего" порога:

    1) Отфильтровываем по:
       - n_trades >= min_trades
       - max_drawdown_pct >= -max_dd_limit  (т.е. не хуже -20%, если лимит=20)
    2) Из оставшихся берём с максимальным Sharpe.
    3) Если кандидатов нет — берём максимум Sharpe без ограничений.
    """
    if not threshold_results:
        raise ValueError("select_optimal_threshold: empty threshold_results")

    candidates = [
        r for r in threshold_results
        if r["n_trades"] >= min_trades
        and r["max_drawdown_pct"] >= -max_dd_limit
    ]

    if not candidates:
        candidates = list(threshold_results)

    best = max(candidates, key=lambda r: r["sharpe_ratio"])

    return {
        "threshold": best["threshold"],
        "sharpe": best["sharpe_ratio"],
        "backtest_metrics": best,
    }
