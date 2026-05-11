"""
Unified backtest engine on vectorbt.

Заменяет 7 разных ручных backtest-циклов (honest_backtest, honest_backtest_v2,
backtest_pjs_framework, analysis/backtest_dynamic_sizing, backtest_real_prb,
backtest_ood_comparison, simple_threshold_backtest, core/backtest_engine).
Единая аннуализация через `core.metrics_calculator.bars_per_year`.

Контракт стратегии:
    strategy.generate_signals(df: pd.DataFrame) -> pd.Series
    где df имеет колонки ['open','high','low','close','volume'] и DatetimeIndex,
    а возвращаемая серия принимает значения {1: вход в long, -1: выход, 0: hold}.

Использование:
    from backtesting.vbt_engine import VBTBacktestEngine, VBTBacktestConfig

    cfg = VBTBacktestConfig(initial_capital=10_000, commission=0.001)
    engine = VBTBacktestEngine(cfg)
    result = engine.run(strategy, coin="BTC", timeframe="15m")
    print(result.summary())
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from core.data_loader import load_market_data
from core.metrics_calculator import bars_per_year

try:
    import vectorbt as vbt  # type: ignore
except ImportError as e:  # pragma: no cover — only triggered without vbt installed
    raise ImportError(
        "vectorbt is required for the unified backtest engine. "
        "Install via `pip install vectorbt>=0.26.0` (already in requirements.txt)."
    ) from e

logger = logging.getLogger(__name__)


# Pandas frequency strings keyed by timeframe — used by vectorbt to annualize.
_TIMEFRAME_TO_FREQ: Dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
}


@dataclass
class VBTBacktestConfig:
    """Static config shared across runs of a single engine."""

    initial_capital: float = 10_000.0
    commission: float = 0.001       # 0.1% per trade
    slippage: float = 0.0005        # 0.05% per trade
    size_fraction: float = 0.95     # fraction of equity per entry
    stop_loss_pct: Optional[float] = None  # e.g. 0.02 for 2%; None disables
    take_profit_pct: Optional[float] = None  # e.g. 0.03 for 3%; None disables
    direction: str = "longonly"     # 'longonly' | 'shortonly' | 'both'


@dataclass
class VBTBacktestResult:
    """Outcome of a single backtest run."""

    coin: str
    timeframe: str
    strategy_name: str
    portfolio: Any                  # vbt.Portfolio
    metrics: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        lines = [
            f"Strategy: {self.strategy_name}",
            f"Coin/TF:  {self.coin}/{self.timeframe}",
            f"Total return:     {m.get('total_return_pct', 0):+8.2f}%",
            f"Annualized return:{m.get('annualized_return_pct', 0):+8.2f}%",
            f"Sharpe (ann):     {m.get('sharpe', 0):8.3f}",
            f"Sortino (ann):    {m.get('sortino', 0):8.3f}",
            f"Calmar:           {m.get('calmar', 0):8.3f}",
            f"Max drawdown:     {m.get('max_drawdown_pct', 0):8.2f}%",
            f"Win rate:         {m.get('win_rate_pct', 0):8.2f}%",
            f"Trades:           {m.get('num_trades', 0):8d}",
            f"Profit factor:    {m.get('profit_factor', 0):8.3f}",
        ]
        return "\n".join(lines)


class VBTBacktestEngine:
    """Wraps vectorbt's Portfolio.from_signals into the project's signal contract."""

    def __init__(self, config: Optional[VBTBacktestConfig] = None) -> None:
        self.config = config or VBTBacktestConfig()

    def run(
        self,
        strategy: Any,
        coin: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        data_dir: str = "data/raw",
    ) -> VBTBacktestResult:
        """Run a single backtest and return metrics + the underlying Portfolio."""
        if timeframe not in _TIMEFRAME_TO_FREQ:
            raise ValueError(
                f"Unsupported timeframe {timeframe!r}; expected one of "
                f"{list(_TIMEFRAME_TO_FREQ)}"
            )

        df = load_market_data(
            coin,
            timeframe,
            start_date=start_date,
            end_date=end_date,
            data_dir=data_dir,
        )
        if df.empty:
            raise ValueError(f"No OHLCV bars loaded for {coin}/{timeframe}")

        signals = strategy.generate_signals(df)
        if not isinstance(signals, pd.Series):
            raise TypeError(
                f"{type(strategy).__name__}.generate_signals must return pd.Series, "
                f"got {type(signals).__name__}"
            )
        signals = signals.reindex(df.index).fillna(0).astype(int)

        entries = signals == 1
        exits = signals == -1

        freq = _TIMEFRAME_TO_FREQ[timeframe]

        portfolio = vbt.Portfolio.from_signals(
            close=df["close"],
            entries=entries,
            exits=exits,
            init_cash=self.config.initial_capital,
            fees=self.config.commission,
            slippage=self.config.slippage,
            size=self.config.size_fraction,
            size_type="percent",
            sl_stop=self.config.stop_loss_pct,
            tp_stop=self.config.take_profit_pct,
            direction=self.config.direction,
            freq=freq,
        )

        metrics = self._collect_metrics(portfolio, timeframe)
        strategy_name = getattr(strategy, "name", type(strategy).__name__)

        logger.info(
            "VBT backtest %s/%s/%s: ret=%.2f%% sharpe=%.2f trades=%d",
            strategy_name,
            coin,
            timeframe,
            metrics["total_return_pct"],
            metrics["sharpe"],
            metrics["num_trades"],
        )

        return VBTBacktestResult(
            coin=coin,
            timeframe=timeframe,
            strategy_name=strategy_name,
            portfolio=portfolio,
            metrics=metrics,
        )

    @staticmethod
    def _collect_metrics(portfolio: Any, timeframe: str) -> Dict[str, float]:
        """Extract the standard metric set with consistent crypto-24/7 annualization."""
        bpy = bars_per_year(timeframe)

        # vectorbt's `total_return` is a fraction; *100 for percent.
        total_return = float(portfolio.total_return()) * 100.0

        try:
            sharpe = float(portfolio.sharpe_ratio())
        except Exception:  # noqa: BLE001 — vbt raises on zero variance
            sharpe = 0.0

        try:
            sortino = float(portfolio.sortino_ratio())
        except Exception:  # noqa: BLE001
            sortino = 0.0

        try:
            calmar = float(portfolio.calmar_ratio())
        except Exception:  # noqa: BLE001
            calmar = 0.0

        max_dd = float(portfolio.max_drawdown()) * 100.0

        trades = portfolio.trades
        num_trades = int(trades.count())
        if num_trades > 0:
            win_rate = float(trades.win_rate()) * 100.0
            profit_factor = (
                float(trades.profit_factor())
                if np.isfinite(trades.profit_factor())
                else 0.0
            )
        else:
            win_rate = 0.0
            profit_factor = 0.0

        # Annualized return: compound total over horizon → annual.
        n_bars = len(portfolio.wrapper.index)
        years = n_bars / bpy if bpy > 0 else 0.0
        if years > 0 and total_return > -100.0:
            annualized = (((1.0 + total_return / 100.0) ** (1.0 / years)) - 1.0) * 100.0
        else:
            annualized = 0.0

        return {
            "total_return_pct": total_return,
            "annualized_return_pct": annualized,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown_pct": max_dd,
            "win_rate_pct": win_rate,
            "num_trades": num_trades,
            "profit_factor": profit_factor,
            "bars_per_year": float(bpy),
        }


def run_multi_asset(
    engine: VBTBacktestEngine,
    strategy_factory,
    coins: list[str],
    timeframe: str,
    **run_kwargs,
) -> Dict[str, VBTBacktestResult]:
    """Convenience: backtest a strategy across many coins.

    `strategy_factory` is a zero-arg callable that returns a fresh strategy
    instance per coin (so stateful strategies don't bleed signals across runs).
    """
    results: Dict[str, VBTBacktestResult] = {}
    for coin in coins:
        strategy = strategy_factory()
        try:
            results[coin] = engine.run(strategy, coin=coin, timeframe=timeframe, **run_kwargs)
        except FileNotFoundError as e:
            logger.warning("Skip %s: %s", coin, e)
    return results
