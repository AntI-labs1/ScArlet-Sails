# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [calendar versioning](https://calver.org/) (`YYYY-MM`).

---

## [v1.0-research-baseline] — 2026-05-11

The "honest closure" revision. Eight months of project work consolidated
into a reproducible research baseline plus a passive capital-allocation
framework. **Project is closed as an actively-managed trading system.**

### Added
- `POST_MORTEM.md` — primary project closure document
- `paper/drafts/main.md` — academic paper draft (5500 words, 19 references)
- `paper/drafts/medium.md` + `medium_ru.md` — Medium-adapted versions
- `paper/drafts/executive_summary.md` — one-page overview
- `paper/notebooks/stats.py` — Deflated Sharpe Ratio + PBO implementation (pure numpy)
- `paper/notebooks/missing_backtests.ipynb` — reproducible Kaggle notebook
- `paper/notebooks/figures.ipynb` — 6 publication figures generator
- `paper/results/` — 6 JSON files with real backtest data
- `paper/build.sh` — pandoc pipeline (md → PDF/LaTeX/HTML)
- `passive/README.md` — capital allocation framework
- `passive/portfolios.yaml` — 8 portfolio definitions (60/40, All Weather, Permanent, RU/EU/UAE variants)
- `passive/rebalance.py` — quarterly rebalance CLI
- `backtesting/vbt_engine.py` — unified vectorbt-based backtest engine
- `backtesting/MIGRATION_NOTES.md` — migration from 7 self-rolled engines
- `scripts/fetch_binance_klines.py` — free crypto OHLCV (Binance Vision)
- `scripts/fetch_metals.py` — free metals OHLCV (yfinance)
- `tests/conftest.py` — synthetic OHLCV bootstrap for CI/Kaggle
- `tests/test_stats.py` — 22 test cases for stats.py
- `kaggle/smoke_test.ipynb` — cloud reproduction notebook
- `LICENSE` — MIT
- `CITATION.cff` — GitHub citation metadata
- `REPRODUCING.md` — full reproduction recipe (3 paths)
- `CHANGELOG.md` — this file

### Fixed (critical infrastructure bugs discovered during audit)
- **B1**: Look-ahead bias in `core/feature_engine_v2.py:_normalize_features`. StandardScaler silently fitted on combined train+test data. Now requires explicit `fit_scaler=True` parameter; raises RuntimeError otherwise.
- **B2**: Hardcoded `current_drawdown=0.0` in `scripts/run_council.py:523` disabled killswitch. Added `DecisionLogger.get_current_drawdown()` that computes from closed-trade log.
- **B3**: Inverted dispersion-to-position-size formula in `tests/test_rolling_dispersion.py` and `tests/test_dispersion_inverted.py`. Source was correct (agreement → high multiplier); tests asserted wrong direction. Tests corrected.
- **B4**: Three inconsistent Sharpe annualization factors across modules (`sqrt(252*96)`, `sqrt(252*24*4)`, `sqrt(252*24)`). Unified through `core/metrics_calculator.bars_per_year(timeframe, asset_class)`.
- **B5**: Data file naming mismatch `BTC_USDT_15m.parquet` vs `BTCUSDT_15m.parquet`. Loader now accepts both naming conventions.
- **Binance Vision timestamp format change** (ms → µs around 2025-01-01) handled in fetcher; per-frame unit detection with `1e14` threshold and sanity-check on resulting year range.

### Changed
- `README.md` — fully rewritten to reflect closed-project state, with reading guide, reproduction paths, and reusable-component catalog
- `core/data_loader.py` — accepts both `BTC_USDT` and `BTCUSDT` naming; supports metals tickers (GOLD, SILVER, COPPER, PLATINUM)
- `core/metrics_calculator.py` — canonical `bars_per_year()` helper; supports both 252-day (TradFi) and 365-day (24/7 crypto) annualization
- `requirements.txt` — removed unused deps (gym, torchvision, ccxt, langchain); added vectorbt, quantstats, yfinance
- `pyproject.toml` — synchronized Python version (>=3.10) with requirements
- `.gitignore` — whitelisted `kaggle/*.ipynb` and `paper/notebooks/*.ipynb` (notebooks are version-controlled docs, not scratchpads)

### Deprecated
- All strategies in `strategies/` other than `simple_strategies.py` are research-only artefacts and should not be considered production code
- `council/`, `rag/`, `ood/`, `garch/`, `q_learning/` — research scaffolding never connected to backtests; documented in `POST_MORTEM.md` "Architecture astronaut phase"
- 7 self-rolled backtest engines (`backtesting/honest_backtest*.py`, `backtesting/backtest_pjs_framework.py`, `core/backtest_engine.py`, `analysis/backtest_*.py`) — superseded by `backtesting/vbt_engine.py`; see `backtesting/MIGRATION_NOTES.md`

### Removed
- `data/raw/` legacy DVC pointers without configured remote (the actual raw data was never accessible from DVC; project relies on Binance Vision public archive instead)
- Several stale README claims (e.g., "Sharpe 2.91 walk-forward validation") replaced with honest realistic estimates after bug-fix audit

### Empirical findings (negative result baseline)
- **Crypto mean-reversion**: 14 coins × 4h × 8 walk-forward windows = 108 points. Average Sharpe **−0.70**, positive windows **35%** (worse than coin-flip)
- **Crypto 15m mean-reversion**: 13/14 coins net negative within 2.5 years; commission drag (~405%) overwhelms signal
- **Metals daily mean-reversion**: 4 metals × 8 windows = 23 points. Average Sharpe **−0.19** (statistically indistinguishable from zero)
- **Metals 200-day SMA trend**: average Sharpe **+0.44**, all 4 metals positive — matches literature baseline [Hurst/Ooi/Pedersen 2017], does **not** exceed buy-and-hold
- **Dual momentum** (4 metals, monthly rebalance, 12m lookback, top-2): Sharpe **0.62 vs B&H equal-weight 0.63** — no meaningful alpha
- **Gold/Silver ratio mean reversion**: silver-side Sharpe 0.61, gold-side 0.27; combined captures ~23% of buy-and-hold return
- **Cost sensitivity**: Sharpe invariant to position size (engine correctness verified post-fix); negative-Sharpe strategies stay negative at any leverage

### Industry benchmark context
- AQR Style Premia Alternative Fund (QSPIX) realized Sharpe: **0.41** since inception (target 0.70)
- SG CTA Index long-run Sharpe: **~0.56**
- Passive 60/40 SPY+TLT Sharpe: **~0.6-0.7** with zero work
- Our retail trend-following on metals: **0.4-0.6** (sits within institutional band)

### Not implemented (intentional scope decision)
- LLM Council / RAG / human-in-the-loop interface
- Live or paper trading executor
- Multi-factor (carry / value / cross-sectional) extension — recommended in paper §7 but estimated to add only +0.1-0.3 Sharpe with high overfitting risk for retail
- Forks of `pysystemtrade` or other production trend-following frameworks — recommended in paper §7 for users who want to continue active management

---

## [Pre-revision: Phase 1-3 development, 2024-09 through 2026-04]

Pre-audit history. The project grew incrementally over 8 months with the
following architectural ambition:

- Phase 1: Foundation (feature engine, rule-based strategy, XGBoost ML)
- Phase 2 (claimed): Advanced Risk & RL — OOD detection, regime detection,
  dynamic position sizing, hybrid Q-learner, walk-forward validation
- Phase 3 (planned): Council & Human-in-Loop — canonical state, Council agents,
  RAG retrieval, human interface, dispersion analysis

**The 2026-05 revision (v1.0-research-baseline) honestly assesses that Phase 2
results were inflated by the bugs documented above, that Phase 3 components
were never connected to actual backtests, and that the underlying strategy
class (rule-based technical) does not have generalizable edge regardless of
how much scaffolding is built around it.**

The pre-revision README claimed Walk-forward Sharpe 2.91 and Calmar 8.98.
After bug fixes and honest multi-market walk-forward, realistic estimates for
the same strategies are Sharpe 0.4-0.6 and Calmar 0.5-1.5 — a factor-of-six
discrepancy attributable entirely to the documented bugs.

---

## Conventions used in this changelog

- **Added**: new features or files
- **Fixed**: bug fixes
- **Changed**: changes in existing functionality
- **Deprecated**: features marked for future removal
- **Removed**: features removed in this version
- **Security**: vulnerability fixes (none in this revision)
