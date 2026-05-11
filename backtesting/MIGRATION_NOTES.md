# Backtest Engines — Migration to vectorbt

С 2026-05 каноничный backtest-движок в проекте — **`backtesting/vbt_engine.py`** на
[vectorbt](https://vectorbt.dev/). Старые движки оставлены как research-артефакты,
но **не должны использоваться в новой работе**.

## Канон
- `backtesting/vbt_engine.py` — `VBTBacktestEngine`, `VBTBacktestConfig`, `VBTBacktestResult`.
- CLI: `python run_backtest.py --strategy ... --coin ... --timeframe ...`.
- Аннуализация: `core.metrics_calculator.bars_per_year(timeframe)` (365 дней крипты, не 252).

## Что задеплекейчено

| Файл | Размер | Замена |
|---|---|---|
| `backtesting/honest_backtest.py` | 262 LoC | `vbt_engine` |
| `backtesting/honest_backtest_v2.py` | 367 LoC | `vbt_engine` |
| `backtesting/backtest_pjs_framework.py` | 748 LoC | `vbt_engine` + PJS-фичи нужно переподнять как сигналы |
| `core/backtest_engine.py` | ~ | `vbt_engine` |
| `analysis/full_integration_backtest.py` | — | `vbt_engine` + multi-asset |
| `analysis/backtest_real_prb.py` | — | `vbt_engine` со стратегией `rule_based` |
| `analysis/backtest_dynamic_sizing.py` | — | `vbt_engine` (size_fraction параметр) |
| `analysis/backtest_ood_comparison.py` | — | `vbt_engine` + сравнение двух запусков |
| `analysis/backtest_dispersion.py` | — | `vbt_engine` + dispersion как post-hoc метрика |
| `analysis/simple_threshold_backtest.py` | — | `vbt_engine` со стратегией `combined` |

## Зачем заменили

- 7+ параллельных циклов с расходящейся Sharpe-аннуализацией (`sqrt(252*96)`,
  `sqrt(252*24*4)`, `sqrt(252*24)` — все живут в разных файлах).
- Ручной MtM, ручной TP/SL, ручные комиссии — баги переписывались отдельно в каждом.
- vectorbt векторизованный → walk-forward и multi-asset работают в десятки раз быстрее.

## Как мигрировать кастомный backtest

Если у вас есть скрипт типа `analysis/foo_backtest.py`, делающий собственный
цикл по барам:

1. Превратите всю логику входа/выхода в стратегию с методом
   `generate_signals(df) -> pd.Series` (значения 1/-1/0).
2. Если стратегия нужна с состоянием (cooldown, флаги) — храните его внутри
   объекта стратегии, vectorbt принимает любую серию.
3. Получите результат через `VBTBacktestEngine(...).run(strategy, coin, tf)`.
4. Из `result.metrics` достаньте Sharpe/Calmar/MDD; для трейд-уровня — из
   `result.portfolio.trades`.
