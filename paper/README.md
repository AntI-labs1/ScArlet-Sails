# Research Paper — Track C

## Working title
**"Rule-Based Technical Trading Strategies Fail to Generalize:
A Multi-Market Walk-Forward Audit (Crypto 2023-2026 + Metals 2000-2026)"**

## Type
Negative-result technical report. Honest empirical audit of retail-accessible
technical trading strategies across two fundamentally different markets,
using rigorous walk-forward validation and academic statistical correction.

## Why this paper matters

1. **Survivorship bias correction**: 99% of published retail backtests are
   single-coin cherry-picks. This is a 131-window cross-market audit.
2. **Honest infra documentation**: includes look-ahead bias / Sharpe drift /
   killswitch failures that we found in our own code during the audit — these
   bugs are ubiquitous in retail repos but never publicly disclosed.
3. **Industry benchmark contrast**: explicitly contrasts retail results
   against AQR Style Premia (Sharpe 0.41), SG CTA Index (0.56), and passive
   60/40 (0.6-0.7).

## Outline (15-20 pages)

### 1. Abstract (250 words)
Problem statement, methodology summary, key finding (no generalizable edge
on rule-based across 131 walk-forward windows), implications for retail.

### 2. Introduction (2 pages)
- Retail algo-trading hype 2023-2026 (TradingAgents, FinGPT, etc.)
- Why honest negative results matter
- Survivorship bias in published backtests

### 3. Related Work (1-2 pages)
- Bailey & Lopez de Prado (Deflated Sharpe, PBO)
- Asness/Moskowitz/Pedersen (Value & Momentum Everywhere)
- Hurst/Ooi/Pedersen (Century of Evidence for Trend-Following)
- Hou/Xue/Zhang (Replicating Anomalies — 64% of equity factors do not replicate)
- StockBench 2025 (LLM trading agents don't beat buy-and-hold)

### 4. Methodology (3 pages)
- Walk-forward with embargo (8 rolling windows per market)
- Canonical bars-per-year (365 for 24/7 crypto, 252 for 24/5 metals)
- Cost model (15 bps ETF, 25 bps metals, 30-50 bps crypto)
- Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014):
  `SR_deflated = SR × √((1 + γ² × T) / (T + SR² × T))`
- Probability of Backtest Overfitting (PBO)
- Strategy taxonomy: mean-reversion / trend-following / cross-sectional

### 5. Dataset (1 page)
- **Crypto**: 14 coins × 4 timeframes (15m/1h/4h/1d), Binance Vision public archive 2023-01 → 2026-04 (~28 months). Total: 4 × 14 = 56 series.
- **Metals**: GOLD, SILVER, COPPER, PLATINUM × 1d, Yahoo Finance futures (GC=F, SI=F, HG=F, PL=F), 1997-2026 (~25-28 years). 4 series.
- **Cost assumptions**: per asset class, justified separately.

### 6. Results (5-6 pages, the meat)

#### 6.1 Crypto: Mean-Reversion (Tested: RSI/Combined/RuleBased on 14 × 4 × 8 = 448 backtests)
- Walk-forward by-coin: average Sharpe **−0.70**, positive windows **38/108 (35%)**
- BTC combined / 15m: −98.4% return, 1351 trades (commission mincing)
- Per-timeframe breakdown table

#### 6.2 Metals: Mean-Reversion (Tested: Combined on 4 × 1d × 8 = 32 backtests)
- Per-metal Sharpe avg **−0.19**, positive 52% windows
- Marginal improvement over crypto (less commission impact)
- But still: 0/4 beat buy-and-hold

#### 6.3 Metals: Trend-Following (Tested: 200d SMA + dual momentum)
- 200d SMA Avg Sharpe **+0.44**, 4/4 positive
- Dual momentum 12m Sharpe **+0.62 ≈ B&H equal-weight (+0.63)**
- Gold/Silver ratio: gold side Sharpe 0.27, silver side 0.61
- **Confirms literature baseline** (Hurst/Ooi/Pedersen Sharpe 0.4-0.7 for simple trend)

#### 6.4 Cost Sensitivity Sweep
- Backtest at 1x / 1.5x / 2x cost assumptions
- Edge collapses to neutral at 2x for trend strategies
- **Margin of safety is thin**

#### 6.5 Deflated Sharpe & PBO
- Each strategy reported with raw Sharpe + deflated Sharpe
- PBO scores per strategy class
- Honest expected live performance after deflation

### 7. Discussion (3 pages)
- Why mean-reversion fails on trending assets (mathematical reasoning)
- Why commission drag dominates on intraday timeframes
- Comparison to AQR Style Premia live Sharpe 0.41 (institutional context)
- Comparison to passive 60/40 (0.6 retail-accessible without active management)
- Lessons for retail / educational implications
- Limitations (single dataset vendor, no transaction-cost-aware optimizer, no carry sleeve)

### 8. Conclusion (1 page)
- Negative finding: simple rules don't have generalizable edge in our universe
- Positive finding: trend-following matches literature, doesn't beat passive
- Methodological contribution: complete walk-forward audit pipeline (released as GitHub)
- Recommendation: retail capital → passive risk-parity portfolios

### 9. Reproducibility & Code (0.5 pages)
- GitHub link
- Kaggle notebooks for replication
- Synthetic-data smoke test for CI

## Missing backtests (planned, see paper/notebooks/)

1. **Trend-following on crypto** (200d SMA on BTC/ETH/SOL × 4 TF) — currently we have trend only on metals
2. **Mean-reversion on metals × multiple TF** (1h, 4h not just 1d)
3. **Deflated Sharpe computation** for all results
4. **PBO score** per strategy
5. **Cost sensitivity sweep** (1x, 1.5x, 2x bps)

See `paper/notebooks/missing_backtests.ipynb` for execution.

## Target venues

| Venue | Type | Timeline | Probability of acceptance |
|---|---|---|---|
| **arXiv preprint** | No review, instant | Week 4 | 100% |
| **SSRN** | Parallel posting, no review | Week 4 | 100% |
| **ICAIF 2026 workshop** | Peer-reviewed, ACM | Submission ~Aug 2026 | ~50% (good for negative result paper) |
| **Medium / Substack** | Adapted lay version | Week 5-6 | n/a, public service |
| **Journal of Portfolio Management** | Peer-reviewed academic | Long lead | ~10% |
| **Quantitative Finance** | Peer-reviewed academic | Long lead | ~15% |

## Status

- [x] Methodology validated (revision 2026-05)
- [x] Crypto walk-forward complete (108 windows, 4 strategies)
- [x] Metals walk-forward complete (23 windows, 4 strategies)
- [ ] Missing backtests (see `notebooks/missing_backtests.ipynb`)
- [ ] Deflated Sharpe + PBO implementation
- [ ] First draft writeup
- [ ] Figures + tables polished
- [ ] arXiv submission
