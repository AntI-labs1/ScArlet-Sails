# ScArlet-Sails

> **Status (May 2026)**: Closed as an actively-managed algorithmic trading project. Repurposed as (a) a reproducible research baseline supporting an honest-negative-result paper, and (b) a passive capital-allocation framework. See [`POST_MORTEM.md`](POST_MORTEM.md) for the full story.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Status: Research Baseline](https://img.shields.io/badge/Status-Research%20Baseline-orange.svg)](POST_MORTEM.md)

---

## TL;DR

This repository contains the artifacts of an 8-month retail algorithmic-trading research project. After honest multi-market walk-forward auditing — **131 walk-forward windows across 14 cryptocurrencies and 4 precious metals** — the project was closed as an active-trading effort with the following empirical conclusion:

**Rule-based technical trading strategies do not have generalizable out-of-sample edge in our tested universe.** Trend-following on metals matches the literature baseline (Sharpe 0.4–0.6) but does not exceed passive buy-and-hold portfolios. The project's prior in-sample claim of Sharpe 2.91 was traced to four infrastructure bugs documented in [`POST_MORTEM.md`](POST_MORTEM.md).

The repository now serves three purposes:

1. **Research baseline** — methodology and code for honest backtesting (`backtesting/`, `core/`, `paper/notebooks/`)
2. **Academic paper draft** — technical report of the negative-result audit (`paper/drafts/`)
3. **Passive capital framework** — quarterly-rebalance script for actual money (`passive/`)

---

## Reading guide

If you are here for...

- **Why an 8-month project closed**: start with [`POST_MORTEM.md`](POST_MORTEM.md)
- **The academic paper draft**: see [`paper/drafts/main.md`](paper/drafts/main.md) (5500 words, 19 refs)
- **A casual blog version**: see [`paper/drafts/medium.md`](paper/drafts/medium.md) (English) or [`paper/drafts/medium_ru.md`](paper/drafts/medium_ru.md) (Russian)
- **Quick overview for reviewers/recruiters**: [`paper/drafts/executive_summary.md`](paper/drafts/executive_summary.md)
- **Reusable backtest code**: [`backtesting/vbt_engine.py`](backtesting/vbt_engine.py)
- **Deflated Sharpe Ratio + PBO implementation**: [`paper/notebooks/stats.py`](paper/notebooks/stats.py)
- **Run everything yourself on Kaggle**: [`kaggle/smoke_test.ipynb`](kaggle/smoke_test.ipynb) + [`paper/notebooks/missing_backtests.ipynb`](paper/notebooks/missing_backtests.ipynb)
- **Quarterly passive rebalance**: [`passive/rebalance.py`](passive/rebalance.py)

---

## Key empirical results

| Strategy class | Market | Walk-forward Sharpe | vs Buy-and-Hold |
|---|---|---:|---|
| Mean-reversion (RSI + BB + MA) | 14 crypto pairs, 4h | **−0.70 avg** (35% positive windows) | Catastrophic: 13/14 net negative |
| Mean-reversion combined | 4 metals, 1d | −0.19 avg (52% positive — random) | Marginally negative |
| **200-day SMA trend-following** | 4 metals, 1d | **+0.44 avg (4/4 positive)** | Underperforms B&H, lower DD |
| Dual momentum (Antonacci) | 4 metals, monthly | **+0.62** | ≈ B&H equal-weight (+0.63) — no alpha |
| Gold/Silver ratio mean-reversion | gold-silver pair | +0.27 / +0.61 | Combined +373% vs B&H +1630% |

**Industry context**: AQR Style Premia Fund (institutional multi-factor) realizes Sharpe 0.41 since inception (target was 0.70). SG CTA Index averages 0.56. Passive 60/40 SPY+TLT realizes ~0.6–0.7 with zero work. Our retail results sit within or below this institutional band, indicating that the bottleneck is the strategy class, not the implementation.

---

## Repository structure

```
ScArlet-Sails/
├── POST_MORTEM.md                ← Project closure document (read first)
├── README.md                     ← This file
│
├── paper/                        ← Academic Track C
│   ├── README.md                   ← Paper outline + target venues
│   ├── drafts/
│   │   ├── main.md                   ← Full paper (5500 words)
│   │   ├── medium.md                 ← Medium-adapted version (English)
│   │   ├── medium_ru.md              ← Medium-adapted version (Russian)
│   │   ├── executive_summary.md      ← 1-page overview
│   │   └── references.bib            ← 19 academic references
│   ├── notebooks/
│   │   ├── stats.py                  ← Deflated Sharpe + PBO (open-source impl)
│   │   ├── missing_backtests.ipynb   ← Run all backtests on Kaggle (~30 min)
│   │   └── figures.ipynb             ← Generate 6 publication figures
│   ├── results/                    ← JSON: empirical data from this project
│   ├── figures/                    ← PNG/PDF: matplotlib output
│   └── build.sh                    ← pandoc: md → PDF/LaTeX/HTML
│
├── passive/                      ← Capital-allocation Track D
│   ├── README.md                   ← 3 portfolios × 4 broker contexts
│   ├── portfolios.yaml             ← 8 portfolio definitions
│   └── rebalance.py                ← Quarterly rebalance CLI
│
├── backtesting/                  ← Reusable backtest infrastructure
│   ├── vbt_engine.py               ← Canonical vectorbt-based engine
│   ├── MIGRATION_NOTES.md          ← Why and how we unified 7 engines
│   └── (deprecated/* — research artefacts only)
│
├── core/                         ← Reusable utilities
│   ├── data_loader.py              ← OHLCV loader (Binance/yfinance compatible)
│   ├── metrics_calculator.py       ← Canonical bars_per_year + metrics
│   ├── feature_engine_v2.py        ← Multi-timeframe feature engine
│   └── (other modules; some research-only)
│
├── scripts/                      ← Data fetchers
│   ├── fetch_binance_klines.py     ← Crypto OHLCV (Binance Vision, free)
│   ├── fetch_metals.py             ← Metals OHLCV (yfinance, free)
│   └── (other scripts)
│
├── kaggle/                       ← Cloud-runnable replication notebooks
│   ├── README.md
│   └── smoke_test.ipynb            ← Run pytest + vbt smoke in Kaggle
│
├── tests/                        ← pytest suite (synthetic data via conftest)
│
└── (research artefacts: strategies/, council/, rag/, analysis/, etc.
    — REFERENCE ONLY, not part of the production path; see POST_MORTEM.md)
```

---

## Quick start

### Option 1 — Replicate paper results on Kaggle (no local setup)

1. Create Kaggle notebook with Internet=ON
2. **File → Import Notebook → URL**:
   ```
   https://raw.githubusercontent.com/StarDust1508/ScArlet-Sails/main/paper/notebooks/missing_backtests.ipynb
   ```
3. Run All — ~30 minutes
4. Results saved to `paper/results/*.json` and `paper/figures/*.pdf`

### Option 2 — Local replication

```bash
git clone https://github.com/StarDust1508/ScArlet-Sails.git
cd ScArlet-Sails
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Fetch real data (crypto: ~15min, metals: ~30sec)
python scripts/fetch_binance_klines.py --start 2023-01
python scripts/fetch_metals.py

# Run backtests
python run_backtest.py --strategy combined --coins BTC ETH SOL --timeframe 4h

# Run paper notebooks
jupyter notebook paper/notebooks/missing_backtests.ipynb
jupyter notebook paper/notebooks/figures.ipynb

# Build paper
./paper/build.sh all  # requires pandoc + xelatex
```

### Option 3 — Just use the passive portfolio framework

```bash
python passive/rebalance.py --list
python passive/rebalance.py --portfolio 60_40_us --total 10000
python passive/rebalance.py --portfolio 60_40_us --current "SPY:6000,TLT:4500"
```

---

## What you can re-use from this repository

If you are building your own honest research project, the following components are MIT-licensed and designed to be reused:

| Component | What it does | Lines |
|---|---|---:|
| [`backtesting/vbt_engine.py`](backtesting/vbt_engine.py) | Single-file vectorbt wrapper with canonical Sharpe annualization | 240 |
| [`scripts/fetch_binance_klines.py`](scripts/fetch_binance_klines.py) | Crypto OHLCV from Binance Vision public archive (free, no key) | 220 |
| [`scripts/fetch_metals.py`](scripts/fetch_metals.py) | Metals OHLCV from yfinance futures | 160 |
| [`paper/notebooks/stats.py`](paper/notebooks/stats.py) | Deflated Sharpe Ratio + Probability of Backtest Overfitting | 280 |
| [`core/metrics_calculator.py`](core/metrics_calculator.py) | Canonical bars-per-year per timeframe | 350 |
| [`tests/conftest.py`](tests/conftest.py) | Synthetic OHLCV bootstrap for CI/Kaggle without real data | 80 |
| [`passive/rebalance.py`](passive/rebalance.py) | Portfolio rebalance CLI with drift threshold checks | 180 |

The patterns most worth copying:

- **Per-frame timestamp unit detection** (`scripts/fetch_binance_klines.py`): handles silent Binance Vision format migration from ms to µs around 2025-01-01.
- **Position-size invariance assertion** (`paper/results/cost_sensitivity_sol.json` discussion): verifies the backtest engine doesn't have bug 4 (Sharpe-annualization drift).
- **Honest negative-result reporting structure** (`paper/drafts/main.md` §6): inventory of bugs, their detection method, fix, and net effect on prior claims.

---

## Citation

If you use this work in academic context:

```bibtex
@misc{scarlet_sails_2026,
  author       = {Bubble3 et al.},
  title        = {{ScArlet-Sails}: A Multi-Market Walk-Forward Audit of Retail Technical Trading Strategies},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/StarDust1508/ScArlet-Sails}},
  note         = {Tag v1.0-research-baseline}
}
```

---

## Acknowledgments

Extensive collaboration with Anthropic Claude (Opus 4.7) throughout the May 2026 audit and closure. The same model that helped construct the original over-engineered architecture also conducted the audit that compelled its closure — see [`POST_MORTEM.md`](POST_MORTEM.md) §"AI-assisted construction paired with AI-assisted auditing" for reflection on this pattern.

---

## License

MIT — see [LICENSE](LICENSE) if present; otherwise standard MIT terms apply to all code and data in this repository. Paper draft is CC-BY upon publication.
