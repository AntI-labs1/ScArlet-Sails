# Reproducing the Paper

This document gives reviewers, peers, and the curious reader a complete recipe for reproducing every empirical claim in the accompanying paper (`paper/drafts/main.md`).

## Prerequisites

- **Python**: 3.10 or higher
- **Disk**: ~2 GB for full data (mostly the metals 25-year history)
- **Time**: 30 minutes on Kaggle CPU, longer locally first time
- **Network**: required for initial data fetches (Binance Vision + yfinance)

Optional for paper build:
- pandoc (`brew install pandoc` / `apt-get install pandoc`)
- xelatex (`brew install --cask mactex` / `apt-get install texlive-xetex`)

---

## Path 1 — Cloud reproduction (Kaggle, recommended)

This is the easiest path. No local installation required.

### Step 1: clone via Kaggle notebook

In a new Kaggle notebook with Internet=ON:

```python
%cd /kaggle/working
!rm -rf ScArlet-Sails
!git clone --depth 1 https://github.com/StarDust1508/ScArlet-Sails.git
!pip install -q -r ScArlet-Sails/requirements.txt 2>&1 | tail -3
```

### Step 2: run all backtests

```python
%cd /kaggle/working/ScArlet-Sails
%run paper/notebooks/missing_backtests.ipynb
```

This will:
1. Fetch real crypto OHLCV via `scripts/fetch_binance_klines.py` (~15 min)
2. Fetch metals OHLCV via `scripts/fetch_metals.py` (~30 sec)
3. Run 5 backtests covering all paper Section 5 claims
4. Save 6 JSON files to `paper/results/`

### Step 3: generate figures

```python
%run paper/notebooks/figures.ipynb
```

Produces 6 PDF + 6 PNG files in `paper/figures/`.

### Step 4: download artifacts

After execution, download `paper/results/*.json` and `paper/figures/*.{png,pdf}` from Kaggle. Commit them to a fork of the repo if you want a permanent record.

---

## Path 2 — Local reproduction

For users who want full control or to extend the analysis.

```bash
# Clone and setup
git clone https://github.com/StarDust1508/ScArlet-Sails.git
cd ScArlet-Sails
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Fetch data (one-time, ~15 min for crypto)
python scripts/fetch_binance_klines.py --start 2023-01
python scripts/fetch_metals.py

# Run a single quick smoke
python run_backtest.py --strategy combined --coin BTC --timeframe 4h

# Run the pytest suite
pytest tests/ -q

# Run the full paper notebooks
jupyter notebook paper/notebooks/missing_backtests.ipynb
jupyter notebook paper/notebooks/figures.ipynb
```

### Local synthetic-data fallback

If you don't have network access, the pytest suite's `tests/conftest.py` generates deterministic synthetic OHLCV for a subset of assets. This is sufficient to verify that the pipeline runs end-to-end, but **does not** reproduce paper-quality results (those require real market data).

```bash
pytest tests/ -q  # uses conftest.py synthetic data automatically
```

---

## Path 3 — Read the paper without running anything

If you just want to read the empirical evidence:

- All JSON results are committed in [`paper/results/`](paper/results/) — these are the actual numbers the paper cites
- All figures will be in [`paper/figures/`](paper/figures/) after a Kaggle run
- Paper draft is in [`paper/drafts/main.md`](paper/drafts/main.md)

Numerical claims you can verify:

| Claim in paper | Source JSON | Key value |
|---|---|---|
| Crypto walk-forward avg Sharpe −0.70 | `walk_forward_crypto_combined.json` | `summary.avg_sharpe` |
| 35% positive windows | same | `summary.positive_pct` |
| Metals SMA200 avg Sharpe 0.44 | `metals_strategies.json` | `sma200_trend_following.summary.avg_sharpe` |
| Dual momentum Sharpe 0.62 | same | `dual_momentum_portfolio.results.strategy.sharpe` |
| BTC 15m −98.43% loss | `crypto_combined_full_period.json` | `results[0].total_return_pct` |
| 4 documented infrastructure bugs | `critical_bugs.json` | `bugs[*]` |

---

## Verifying the bug fixes

The paper claims four critical infrastructure bugs were discovered and fixed. To verify:

```bash
# Bug 1: feature_engine fit_scaler now required
python -c "
from core.feature_engine_v2 import FeatureEngine
import pandas as pd, numpy as np
df = pd.DataFrame({'open':[1.0]*100, 'high':[1.0]*100, 'low':[1.0]*100,
                   'close':[1.0]*100, 'volume':[1.0]*100},
                  index=pd.date_range('2024-01-01', periods=100, freq='15min'))
eng = FeatureEngine({'features': {'normalize': True}, 'base_timeframe': '15m'})
try:
    eng.calculate_features(df)  # should raise — no scaler loaded, no explicit fit
    print('BUG STILL PRESENT')
except RuntimeError as e:
    print(f'BUG FIXED: {e}')
"

# Bug 4: canonical bars-per-year
python -c "
from core.metrics_calculator import bars_per_year
print('15m crypto:', bars_per_year('15m'))  # should be 35040 (365*24*4)
print('1d crypto:', bars_per_year('1d'))    # should be 365
"
```

---

## Cross-referencing institutional benchmarks

The paper cites several external benchmarks. To verify these are current:

- AQR Style Premia Alternative Fund (QSPIX): https://funds.aqr.com/funds/aqr-style-premia-alternative-fund — check "Since Inception" Sharpe; we cited 0.41 in May 2026.
- SG CTA Index: https://www.sgmarkets.com — search for CTA Index methodology; we cited 0.56 long-run.
- 60/40 SPY+TLT Sharpe: any portfolio backtest tool (Portfolio Visualizer, QuantReturns) will reproduce ~0.6 over 2000-2025.

---

## What the paper does NOT reproduce

- The original "Sharpe 2.91" claim from the project's pre-audit README. This number was an artifact of the four bugs documented in `critical_bugs.json`. After fixes, no honest backtest produces it. The paper documents this explicitly in Section 6.
- Single-coin success on SOL (Sharpe 1.20). This exists but was demonstrated to be a non-generalizable outlier via walk-forward analysis. The paper presents it as a cautionary example.

---

## Contact

For questions about reproduction issues, open a GitHub issue at https://github.com/StarDust1508/ScArlet-Sails/issues.

For methodology questions, the paper's References section lists primary academic sources. The most relevant for replicators:
- Bailey & López de Prado (2014) — Deflated Sharpe Ratio paper
- Bailey et al. (2014) — Probability of Backtest Overfitting paper
- López de Prado (2018) — *Advances in Financial Machine Learning* — Chapter 11 on backtest statistics

---

## License & attribution

All code: MIT License.
Paper draft and figures: CC-BY 4.0 upon arXiv publication.
Cite as in `paper/drafts/executive_summary.md` final section.
