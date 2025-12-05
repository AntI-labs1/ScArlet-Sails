import numpy as np
import pandas as pd
import pytest

from analysis.simple_threshold_backtest import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    evaluate_threshold,
)


def test_sharpe_scaling_by_trades_per_year():
    trade_returns = np.array([-0.01, 0.0, 0.005, -0.007, 0.002])
    mean = trade_returns.mean()
    std = trade_returns.std(ddof=1)

    raw_expected = mean / std
    annual_expected = raw_expected * np.sqrt(76)

    # По умолчанию — raw Sharpe
    assert calculate_sharpe_ratio(trade_returns, periods_per_year=None) == pytest.approx(
        raw_expected
    )
    # С periods_per_year — аннуализированный
    assert calculate_sharpe_ratio(trade_returns, periods_per_year=76) == pytest.approx(
        annual_expected
    )


def test_max_drawdown_uses_equity_curve():
    equity_curve = np.array([1.0, 1.1, 0.9, 1.2, 0.8])
    max_dd = calculate_max_drawdown(equity_curve)
    assert max_dd == pytest.approx(-33.3333, rel=1e-3)


def test_evaluate_threshold_empty_trades():
    df = pd.DataFrame(
        {
            "proba": [0.1, 0.2, 0.3],
            "fee_ret": [0.01, -0.02, 0.03],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )

    metrics = evaluate_threshold(df=df, proba_col="proba", fee_ret_col="fee_ret", threshold=0.9)

    assert metrics["n_trades"] == 0
    assert metrics["total_return"] == 0.0
    assert metrics["sharpe_ratio"] == 0.0
    assert metrics["max_drawdown_pct"] == 0.0
