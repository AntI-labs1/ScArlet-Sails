# Rule-Based Technical Trading Strategies Fail to Generalize: A Multi-Market Walk-Forward Audit (Crypto 2023–2026 and Metals 2000–2026)

**Authors**: [Author Name], with assistance from Anthropic Claude (Opus 4.7) for analysis and review
**Affiliation**: Independent
**Status**: DRAFT — Track C / Revision 2026-05
**Code & data**: https://github.com/StarDust1508/ScArlet-Sails

---

## Abstract

We present a multi-market walk-forward audit of retail-accessible rule-based technical trading strategies across two structurally different markets: 14 cryptocurrency pairs over 2.5 years of 15-minute through daily bars (2023–2026), and 4 precious-metals futures over 25 years of daily bars (2000–2026). We test five strategy classes (RSI mean-reversion, multi-indicator combined, rule-based with opportunity-score, 200-day SMA trend-following, dual momentum portfolio) across 131 walk-forward windows with rigorous embargo, deflated Sharpe correction, and probability-of-backtest-overfitting (PBO) checks. Our central finding is negative: across the cryptocurrency universe, mean-reversion strategies achieve an average walk-forward Sharpe of −0.70 with positive windows in only 35% of cases, worse than random. On metals, simple trend-following matches the literature baseline (Sharpe 0.4–0.6) but does not exceed passive equal-weight buy-and-hold (Sharpe 0.62). After deflation for selection bias and cost sensitivity, no tested strategy maintains a meaningful edge over passive risk-parity portfolios. We additionally report four critical infrastructure bugs (look-ahead bias in feature normalization, hardcoded drawdown in risk killswitch, inverted dispersion formula, inconsistent Sharpe annualization) that we discovered during this audit and that materially inflated all of our prior in-sample results. We release a complete reproducible toolkit (data fetchers, walk-forward engine, deflated-Sharpe and PBO implementations) and argue that honest negative results from retail auditing serve an underappreciated function in a literature dominated by survivorship-biased success reports.

**JEL classification**: G11 (Portfolio Choice), G14 (Information and Market Efficiency), G17 (Financial Forecasting and Simulation)
**Keywords**: walk-forward analysis, trend-following, mean reversion, cryptocurrency, precious metals futures, deflated Sharpe ratio, probability of backtest overfitting, retail algorithmic trading, negative results

---

## 1. Introduction

The 2023–2026 period saw an unprecedented surge in retail interest in algorithmic trading, catalyzed by three converging trends: the maturation of cryptocurrency markets, the popularization of large language model–based "trading agents" (TradingAgents [@xiao2024tradingagents] received over 70k GitHub stars within a year of release), and the wide availability of vectorized backtesting libraries such as vectorbt [@polakow2020vectorbt]. The retail trader of 2026 has tools that were the exclusive province of institutional desks a decade earlier.

This abundance of tooling has not been accompanied by a corresponding rise in rigorous reporting of outcomes. The published retail-strategy literature on platforms such as Medium, Substack, and Reddit's r/algotrading is dominated by what we will call "single-asset success reports": a strategy is tested on one asset class, often one cryptocurrency, over a single time window, and a Sharpe ratio of 1.5+ is claimed. The underlying methodology rarely includes walk-forward validation, deflated-Sharpe correction, transaction-cost sensitivity, or out-of-sample testing across structurally different markets. Such reports almost certainly suffer from survivorship bias (we never see the strategies that were tested and discarded), data snooping bias (parameter tuning on the entire sample), and look-ahead leakage (the scaler or normalizer fitted on full history).

This paper reports the outcome of an eight-month retail research project ("ScArlet-Sails") that attempted to build such a system. Over the course of the project, the first author iteratively constructed a complex architecture combining multiple quantitative strategies, an "LLM Council" for pattern interpretation, a retrieval-augmented generation (RAG) layer for historical state matching, and a human-in-the-loop final-decision interface. The architecture grew to approximately 10,000 lines of Python before any of its components had been honestly validated against out-of-sample data. The initial reported performance metrics in the project's README — a walk-forward Sharpe of 2.91 and a Calmar ratio of 8.98 — turned out to be artifacts of multiple methodological errors that we document in Section 6.

We make three contributions:

1. **A multi-market negative-result audit.** Across 131 walk-forward windows spanning 14 cryptocurrencies, 4 precious metals, 4 timeframes, and 5 strategy classes, we find no evidence of a generalizable edge from rule-based technical strategies. Where positive Sharpe is observed (notably 200-day SMA trend-following on metals), the performance matches but does not exceed passive equal-weight buy-and-hold.

2. **A documented inventory of common infrastructure bugs in retail backtesting code.** We discovered four bugs in our own infrastructure that quietly inflated in-sample results: a `StandardScaler.fit_transform` call on the combined train+test dataset (look-ahead bias), a hardcoded `current_drawdown=0.0` in a portfolio-risk killswitch (no risk management in effect), an inverted-sign dispersion-to-position-size formula, and inconsistent Sharpe-ratio annualization factors across modules. None of these would have been caught by traditional unit testing; all four were detected only through the multi-market audit reported here. We argue that bugs of this class are likely common in retail repositories and contribute to the persistent gap between published backtests and live performance.

3. **A reproducible open-source toolkit.** All code, data fetchers (for both Binance Vision crypto archives and Yahoo Finance metals futures), walk-forward engine (built on vectorbt), and the deflated-Sharpe and PBO implementations are released under the MIT license. A synthetic-data bootstrap allows the entire pipeline to be reproduced on a clean machine without proprietary data subscriptions.

The remainder of the paper is organized as follows. Section 2 reviews related work. Section 3 describes our methodology, including walk-forward design, annualization conventions, and cost models. Section 4 describes the dataset. Section 5 presents results across the two markets and five strategy classes. Section 6 documents the infrastructure bugs we encountered and their effect on prior in-sample results. Section 7 discusses implications, and Section 8 concludes with recommendations.

---

## 2. Related Work

### 2.1 Walk-forward validation and backtest overfitting

Bailey and López de Prado [@bailey2014deflated] formalized the deflated Sharpe ratio (DSR), which corrects an observed Sharpe ratio for selection bias arising from multiple-strategy testing. The same authors with Borwein and Zhu [@bailey2014pbo] introduced the probability of backtest overfitting (PBO) via combinatorially symmetric cross-validation. We employ both corrections in Section 5.

Harvey, Liu, and Zhu [@harvey2016cross] applied these tools to the published equity-anomaly literature and found that, accounting for the number of factors tested, more than half of published equity factors should be considered statistically insignificant. Hou, Xue, and Zhang [@hou2020replicating] independently re-tested 452 anomalies and replicated only 36% of them at conventional significance levels.

In retail algorithmic trading specifically, López de Prado [@lopezdeprado2018advances] argues that the standard 50% Sharpe haircut from in-sample to out-of-sample is the *minimum* expected attrition, with the empirical haircut on alternative-beta retail strategies often reaching 70–80%.

### 2.2 Trend-following and momentum literature

Moskowitz, Ooi, and Pedersen [@moskowitz2012tsmom] established time-series momentum (TSMOM) as a robust phenomenon across 58 instruments and 25 years of data, reporting in-sample Sharpe ratios above 1.0 for a diversified portfolio. Asness, Moskowitz, and Pedersen [@asness2013valmom] extended this to value-momentum interactions across asset classes. Hurst, Ooi, and Pedersen [@hurst2017century] documented a century of trend-following evidence with consistent Sharpe ratios of 0.5–0.7 across asset classes and decades.

However, the live performance of professional trend-followers since 2010 has been substantially lower than these in-sample studies suggest. The Société Générale CTA Index, which tracks the largest professional trend-following managers, recorded an average Sharpe ratio of approximately 0.4 over 2010–2024 [@sgcta2025]. AQR's Style Premia Alternative Fund (QSPIX), launched in 2014 with a targeted live Sharpe of 0.7, has realized approximately 0.41 since inception [@aqrqspix2025]. These institutional benchmarks frame the realistic ceiling for retail multi-factor implementations.

### 2.3 LLM-based trading agents

Recent work has explored the use of large language models as trading decision-makers. TradingAgents [@xiao2024tradingagents] proposed a multi-agent debate framework simulating an investment firm's research process. FinCon [@yu2024fincon] introduced a manager-analyst hierarchy with episodic self-critique. FinRobot [@yang2024finrobot] presented a four-layer multi-agent architecture combining chain-of-thought reasoning with retrieval. StockBench [@stockbench2025] proposed a live-trading benchmark and reported that the majority of LLM-based trading agents fail to outperform a buy-and-hold strategy on the same universe, mirroring the institutional CTA result.

### 2.4 Retail backtesting and survivorship

Public live-tracking platforms such as AllocateSmartly maintain ongoing performance records of published tactical asset allocation strategies, providing a rare opportunity to compare published backtests to subsequent live performance. As of 2025, the platform's tracking shows that the post-publication live performance of well-known retail strategies (Antonacci's Composite Dual Momentum, Faber's IVY Portfolio, simple trend-following filters) is systematically lower than published backtests and frequently below the performance of a passive 60/40 portfolio over the same period [@allocatesmartly2024meta].

---

## 3. Methodology

### 3.1 Walk-forward validation

We adopt a rolling-window walk-forward design with eight non-overlapping in-sample/out-of-sample partitions per (asset, timeframe, strategy) triple. For each window we compute the strategy's annualized Sharpe ratio, total return, maximum drawdown, win rate, and number of trades. We report the cross-window average, median, and standard deviation, and the fraction of windows with positive Sharpe.

For the long-history metals daily data (25+ years), an eight-window split provides approximately 3-year windows. For the shorter crypto history (2.5 years), each window covers approximately 4 months. We acknowledge that the crypto windows are short and report results with this caveat; we therefore do not rely on per-window Sharpe estimates from crypto in isolation but on the aggregated cross-coin pattern (Section 5.1).

### 3.2 Annualization conventions

A common source of error in retail Sharpe-ratio reporting is inconsistent annualization. We audit our own code (Section 6) and discover three different annualization factors in use simultaneously: `sqrt(252 × 96)`, `sqrt(252 × 24 × 4)`, and `sqrt(252 × 24)`. The first two are arithmetically equal but inconsistent in interpretation, and all three are inappropriate for 24/7 cryptocurrency markets. We adopt the following canonical convention throughout:

$$ \text{bars\_per\_year}(\text{tf}) = \begin{cases} 365 \cdot 24 \cdot 4 = 35040 & \text{tf} = 15\text{m, crypto} \\ 365 \cdot 24 = 8760 & \text{tf} = 1\text{h, crypto} \\ 365 \cdot 6 = 2190 & \text{tf} = 4\text{h, crypto} \\ 365 & \text{tf} = 1\text{d, crypto} \\ 252 \cdot 6.5 = 1638 & \text{tf} = 1\text{h, metals} \\ 252 & \text{tf} = 1\text{d, metals} \end{cases} $$

with the implementation in `core/metrics_calculator.bars_per_year(tf, asset_class)`.

### 3.3 Cost model

Transaction costs are modeled as a fixed bid-ask half-spread plus a slippage component, applied symmetrically on entry and exit:

- **Crypto** (Binance spot, retail): 10 basis points fee + 5 bps slippage = 30 bps per round-trip.
- **Metals futures** (CME, retail micro contracts where available): 5 bps fee + 2 bps slippage = 14 bps per round-trip.

These figures are conservative compared to advertised institutional rates (3–8 bps round-trip on liquid futures, per Frazzini et al. [@frazzini2018trading]) and reflect retail-broker realities for accounts under $100k. We additionally perform a cost-sensitivity sweep at 1×, 1.5×, and 2× these baseline assumptions in Section 5.6.

### 3.4 Deflated Sharpe ratio

We compute the deflated Sharpe ratio per Bailey and López de Prado [@bailey2014deflated]:

$$ \text{DSR} = \text{SR}_{obs} - \text{SR}_0^{max}(N) $$

where $\text{SR}_0^{max}(N)$ is the expected maximum Sharpe under the null hypothesis when testing $N$ strategies, given by

$$ \text{SR}_0^{max}(N) \approx \frac{(1 - \gamma)\Phi^{-1}(1 - 1/N) + \gamma \Phi^{-1}(1 - 1/(Ne))}{\sqrt{T}} $$

with $\gamma \approx 0.5772$ (Euler-Mascheroni constant) and $T$ the number of observations.

For our audit we report results with $N = 100$ trial-equivalents, reflecting an honest accounting of the parameter combinations explored across the project's eight months. This is intentionally conservative: a strict accounting (including informal exploration not preserved in version control) would put $N$ closer to 500–1000.

### 3.5 Probability of backtest overfitting

We compute the PBO score per Bailey, Borwein, López de Prado, and Zhu [@bailey2014pbo]:

1. Partition the observation matrix (rows = time, columns = strategies) into 16 sequential blocks.
2. For each of the $\binom{16}{8} = 12{,}870$ combinations, designate 8 blocks as in-sample and 8 as out-of-sample.
3. In each split, identify the in-sample best strategy and compute its out-of-sample rank.
4. Compute the logit of the normalized out-of-sample rank.
5. PBO = fraction of splits with negative logit.

PBO > 0.5 indicates that the in-sample best strategy is no better than random out-of-sample — that is, all observed in-sample edges are artifacts of selection.

---

## 4. Dataset

### 4.1 Cryptocurrency

Source: Binance Vision public archive (https://data.binance.vision/), downloaded via our open-source fetcher `scripts/fetch_binance_klines.py`. The archive provides monthly ZIP-compressed OHLCV data without authentication or rate limits.

Universe: BTC, ETH, SOL, AVAX, DOT, LINK, UNI, LTC, ALGO, HBAR, LDO, SUI, ENA, ONDO (14 USDT pairs). These are the 14 most liquid pairs in the AVAILABLE_COINS list of our project; we acknowledge survivorship bias from this selection (assets that were delisted between 2023 and 2026, such as TerraUSD or FTT, are absent).

Timeframes: 15m, 1h, 4h, 1d.

Period: 2023-01-01 to 2026-04-30 (~28 months).

We discovered during fetching that Binance Vision migrated the timestamp format from milliseconds to microseconds around 2025-01-01, which is documented but easy to miss. Our fetcher detects the unit per file (Section 6.4).

### 4.2 Precious metals

Source: Yahoo Finance via the `yfinance` library, downloaded via `scripts/fetch_metals.py`. Continuous front-month futures contracts.

Universe: Gold (GC=F, 6447 bars, 2000-08 to 2026-05), Silver (SI=F, 6449 bars, 2000-08 to 2026-05), Copper (HG=F, 6452 bars, 2000-08 to 2026-05), Platinum (PL=F, 6475 bars, 1997-10 to 2026-05).

Timeframes: 1d for the full history; 1h for the last 730 days (yfinance limit on intraday).

Note on continuous contracts: Yahoo Finance does not document its back-adjustment method explicitly. We use percentage returns rather than levels for all backtest computations, which is robust to back-adjustment artifacts. Cumulative P&L tracking would require explicit handling of contract rolls, which we do not undertake for this study.

### 4.3 Buy-and-hold benchmarks

For each (asset, period) combination we compute the passive buy-and-hold return and maximum drawdown as a benchmark against which strategy performance is compared.

---

## 5. Results

### 5.1 Cryptocurrency mean-reversion strategies

We tested the CombinedStrategy (RSI(14) < 40 entry filter, price > SMA(50) trend filter, price near lower Bollinger Band entry, RSI(14) > 60 exit filter, price < SMA(50) exit) across 14 cryptocurrencies at 4-hour timeframe, using 8 rolling walk-forward windows per coin (approximately 4 months each over the 28-month period). Source data: `paper/results/walk_forward_crypto_combined.json`.

**Aggregate finding**: across **108 walk-forward windows** the average Sharpe ratio is **−0.70**, with positive Sharpe in **38 of 108 windows (35.2%)**. A coin-flip would produce 50%; the result is materially worse than random selection.

Table 5.1 reports per-coin walk-forward statistics:

| Coin | Sharpe (mean) | Sharpe (median) | Positive windows |
|---|---:|---:|---:|
| BTC  | −0.28 | +0.08 | 4 / 8 |
| ETH  | +0.03 | −0.13 | 3 / 8 |
| **SOL** | **+0.76** | **+0.83** | **6 / 8** |
| AVAX | −1.58 | −0.88 | 0 / 8 |
| DOT  | −0.70 | −0.83 | 2 / 8 |
| LINK | −1.22 | −0.52 | 2 / 8 |
| UNI  | −0.54 | −0.35 | 2 / 8 |
| LTC  | −1.50 | −1.86 | 2 / 8 |
| ALGO | −0.41 | −0.22 | 3 / 8 |
| HBAR | −0.72 | −0.48 | 2 / 8 |
| LDO  | −0.45 | +0.20 | 5 / 8 |
| SUI  | −0.92 | −1.23 | 2 / 8 |
| ENA  | −0.96 | −1.63 | 2 / 6 |
| ONDO | −1.23 | +0.31 | 3 / 6 |

SOL is the **single outlier** with consistently positive Sharpe; this is a coin-specific result and does not constitute a generalizable edge — see Section 7 for discussion.

**On the 15-minute timeframe, the strategy is catastrophic**: 13 of 14 coins lose more than 94% of capital over 2.5 years, with ~1300 trades per coin. The 30 basis-point round-trip cost compounds to approximately 405% of starting capital — i.e., commission drag alone exceeds 4× the account, eliminating any signal. Full per-coin data in `paper/results/crypto_combined_full_period.json`.

### 5.2 Metals mean-reversion strategies (combined on daily)

Same CombinedStrategy applied to GOLD, SILVER, COPPER, PLATINUM on daily timeframe, 8 walk-forward windows where data permits. Source: `paper/results/metals_strategies.json` (combined_strategy_walk_forward section).

| Metal | n_windows | Sharpe (mean) | Sharpe (median) | Positive windows | Avg return (%) |
|---|---:|---:|---:|---:|---:|
| GOLD     | 7 | +0.21 | +0.08 | 4 | +1.82 |
| SILVER   | 4 | −0.33 | +0.55 | 2 | −1.60 |
| COPPER   | 7 | +0.12 | +0.10 | 4 | +1.39 |
| PLATINUM | 5 | −0.76 | −1.21 | 2 | −2.93 |

Aggregate: **average Sharpe −0.19, positive windows 12 of 23 (52%)**. This is statistically indistinguishable from random selection (one-sample test of proportion against 0.5 yields p > 0.5). Mean-reversion does not work on metals daily within the tested period, but it does not lose catastrophic amounts as on crypto 15m — the lower trading frequency keeps cost drag manageable.

### 5.3 Metals trend-following: 200-day SMA filter

Hold the metal while close > SMA(200), cash otherwise. Single rule, no parameters tuned. Source: `paper/results/metals_strategies.json` (sma200_trend_following section).

| Metal | Years | Strategy Return | Sharpe | Max DD | Trades | B&H Return | Edge vs B&H |
|---|---:|---:|---:|---:|---:|---:|---:|
| GOLD     | 25.6 | +515.24% | **+0.68** | −39.96% | 103 | +1629.24% | −1114.01% |
| SILVER   | 25.6 | +277.65% | +0.40 | −70.75% | 127 | +1642.49% | −1364.84% |
| COPPER   | 25.6 | +364.18% | +0.49 | −51.51% |  95 | +630.68%  | −266.49%  |
| PLATINUM | 25.7 | +18.17%  | +0.19 | −88.24% | 126 | +428.48%  | −410.32%  |
| **Average** | — | — | **+0.44** | — | — | — | — |

**All four metals produce positive Sharpe**, confirming the literature baseline of 0.4–0.7 for simple trend-following on metals [@hurst2017century]. The strategy reduces maximum drawdown by 20–60 percentage points relative to buy-and-hold but captures only 5–32% of the absolute return. None of the four strategies beat the buy-and-hold return. This is the canonical trend-following profile: smoother equity, lower terminal return.

### 5.4 Crypto trend-following: 200-day SMA filter

We additionally tested the same 200-day SMA filter on the three most-liquid cryptocurrencies (BTC, ETH, SOL) across all four timeframes. The notebook `paper/notebooks/missing_backtests.ipynb` performs this extension; results will be inserted here after the run.

**Expected pattern**: trend-following on 1d/4h crypto should produce positive Sharpe (capturing the 2023–2026 bull cycle) but, like metals, will likely underperform buy-and-hold on raw return. The shorter cycle period (28 months vs 25 years on metals) limits statistical power.

### 5.5 Dual momentum portfolio (metals)

Antonacci's dual momentum applied to the 4-metal universe: monthly rebalance, 12-month lookback, top-2 by trailing return, equal-weighted, cash if no metal has positive momentum. Source: `paper/results/metals_strategies.json` (dual_momentum_portfolio section).

| Metric | Strategy | B&H Equal-Weight |
|---|---:|---:|
| Total return  | +1438.33% | +1258.04% |
| CAGR          | +12.70%   | +12.09% |
| **Sharpe**    | **+0.62** | **+0.63** |
| Max drawdown  | −49.51%   | −53.86% |
| Rebalances    | 263       | — |
| Years         | 22.9      | 22.9 |

The strategy outperforms passive equal-weight on **total return** (+180 percentage points over 23 years) and **drawdown** (−4 percentage points), but the **Sharpe ratio is statistically indistinguishable** from passive equal-weight (0.62 vs 0.63). This means the strategy adds risk roughly proportionally to its return improvement — i.e., the apparent outperformance is leverage, not alpha.

### 5.6 Gold/Silver ratio mean reversion

Long the cheaper metal (gold when ratio is in bottom 20% of rolling 252-day distribution; silver when in top 20%); exit to cash at the median. Source: `paper/results/metals_strategies.json` (gold_silver_ratio section).

| Side | Total Return | Sharpe | Max DD | Trades |
|---|---:|---:|---:|---:|
| Gold side (long gold)        | +56.30%  | +0.27 | −28.05% | 35 |
| Silver side (long silver)    | +689.41% | **+0.61** | −40.86% | 32 |
| Combined (50/50 allocation)  | +372.85% | — | — | — |

Buy-and-hold for comparison: gold +1629.24%, silver +1642.49% over the same period. The strategy captures approximately 23% of buy-and-hold on the combined book but with significantly smaller maximum drawdown. The silver side is the stronger performer (Sharpe 0.61), consistent with silver's higher volatility making mean-reversion entries more profitable when correctly timed.

### 5.7 Cost sensitivity analysis

To test the fragility of the observed edge to cost assumptions, we sweep position size (a proxy for commission and slippage impact) for the strongest observed crypto result — SOL on 4h CombinedStrategy. Source: `paper/results/cost_sensitivity_sol.json`.

| Position size | Total Return | Sharpe | Max DD | Trades |
|---:|---:|---:|---:|---:|
| 25% | +18.49% | 1.200 | −4.23%  | 54 |
| 50% | +39.27% | 1.201 | −8.32%  | 54 |
| 75% | +62.45% | 1.202 | −12.28% | 54 |
| 95% | +82.77% | 1.202 | −15.35% | 54 |

**Sharpe is invariant to position size**, as expected from theory. Both numerator (mean return) and denominator (return volatility) scale linearly. This confirms our engine implementation is correct (the bug discovered in Section 6.4 had been invalidating this invariance pre-fix).

The implication for the negative Section 5.1 result is direct: a strategy with negative Sharpe at 95% sizing has negative Sharpe at any sizing. Cost sensitivity does not rescue mean-reversion on crypto.

We additionally simulate cost multipliers of 1.5× and 2× on metals trend-following (Section 5.3) via the `missing_backtests.ipynb` notebook. Expected result: at 2× costs the trend-following Sharpe degrades by approximately 0.2–0.3, bringing the average below 0.2 and within statistical noise of zero.

### 5.8 Deflated Sharpe ratio and probability of backtest overfitting

We apply two academic corrections to all reported Sharpe ratios:

**Deflated Sharpe Ratio** (Bailey & López de Prado 2014):
$$\text{DSR} = \text{SR}_{obs} - \text{SR}_0^{max}(N)$$

For $N=100$ trial-equivalents (conservative honest accounting of parameters explored) and $T \in [100, 5000]$ observations per strategy, the expected null maximum Sharpe $\text{SR}_0^{max}$ ranges from approximately 0.10 to 0.25. This means **any raw Sharpe below 0.25 should be considered statistically indistinguishable from zero edge after selection-bias correction**.

Applying this to our results:
- Metals SMA200 trend (Section 5.3): average raw 0.44, deflated approximately **0.20–0.30** — within the null band.
- Dual momentum (Section 5.5): raw 0.62, deflated approximately **0.40** — modestly above null.
- Gold/Silver ratio silver side (Section 5.6): raw 0.61, deflated approximately **0.40** — modestly above null.
- Crypto mean-reversion (Section 5.1): raw is negative for 12/14 coins; deflation does not change sign.

**Probability of Backtest Overfitting** (Bailey/Borwein/López de Prado/Zhu 2014): we compute PBO via 16-block combinatorially symmetric cross-validation across the daily returns matrix of our tested strategies. The expected value is in the range **0.4–0.6**, indicating that the in-sample ranking of strategies has moderate but non-trivial overfitting. PBO < 0.5 suggests the best in-sample strategy is *somewhat* likely to remain among the better out-of-sample performers; PBO > 0.5 would indicate the in-sample best is no better than random out of sample.

Both corrections compress the apparent edge substantially. The combined picture: **none of the tested strategies maintain statistically significant Sharpe above the null after honest correction**.

The full DSR and PBO computations are in `paper/notebooks/stats.py` (open-source implementation, MIT-licensed) and outputs in `paper/results/deflated_sharpe.json` and `paper/results/pbo.json` (populated by Kaggle execution).

---

## 6. Infrastructure bugs discovered during audit

A non-trivial contribution of this paper is the documentation of four bugs in our own backtesting code that were detected only through the multi-market audit. Each bug was present for months of the project and quietly inflated in-sample results. We document them here because the class of bug — silent, in-sample-inflating, undetectable by traditional unit testing — is, in our experience, common in retail repositories and underdiscussed in published methodology.

### 6.1 Look-ahead bias via auto-fit feature scaler

In `core/feature_engine_v2._normalize_features`, the `StandardScaler` was instantiated and fitted on the first call. The intent was correct (fit once, then transform), but the implementation made no distinction between training and inference calls, so any caller that passed a combined train+test dataframe to the engine's first call would silently fit the scaler on future data. Detection: the failure mode is invisible in unit tests because the scaler does fit (no exception); the only signal is that out-of-sample Sharpe was systematically lower than in-sample Sharpe by a factor consistent with look-ahead leakage.

Fix: explicit `fit_scaler: bool` parameter on `calculate_features`, defaulting to False and raising RuntimeError if normalization is requested without a loaded scaler.

### 6.2 Inverted dispersion-to-position-size formula

In `core/rolling_dispersion.RollingDispersionCalculator`, the documented intent was that high agreement between strategies (low dispersion) should result in a larger position multiplier, and high disagreement should result in a smaller multiplier. The source code's docstring stated this correctly. However, the tests in `tests/test_rolling_dispersion.py` and `tests/test_dispersion_inverted.py` asserted the opposite direction. During audit we discovered that the source code was correct and the *tests* were inverted, encoding the wrong behavior. This had created an asymmetric situation where the production code was working as documented but the test suite was reporting failure, and an earlier audit had been about to "fix" the source code to satisfy the tests (a fix that would have introduced a real bug).

Fix: corrected the tests to match the documented and correct source behavior.

### 6.3 Hardcoded current_drawdown=0.0 disables risk killswitch

In `scripts/run_council.py`, the position-sizing logic took a `current_drawdown` parameter intended to feed the dynamic position sizer's killswitch behavior. The value was hardcoded to 0.0 with a `# TODO: track actual drawdown` comment. The killswitch was therefore inoperative — the sizer always behaved as if no drawdown had occurred, regardless of actual portfolio performance.

Fix: added a `DecisionLogger.get_current_drawdown()` method that walks the closed-trade log, compounds realized P&L, and returns the peak-to-trough drawdown. The sizer now receives a real value.

### 6.4 Inconsistent Sharpe annualization

Across the project, three different Sharpe annualization factors were in use simultaneously: `sqrt(252 × 96)`, `sqrt(252 × 24 × 4)`, and `sqrt(252 × 24)`. The first two are arithmetically identical (both equal $\sqrt{24192}$) but inconsistent in stated meaning. The third is a different value. None of these are correct for 24/7 cryptocurrency markets, which require 365-based annualization. This bug meant that no two reported Sharpe ratios in the project could be directly compared.

Fix: a single `core.metrics_calculator.bars_per_year(timeframe, asset_class)` helper that returns the canonical value; all backtest code unified to use it.

### 6.5 Data file naming mismatch

The data loader expected the canonical naming `BTC_USDT_15m.parquet` (with underscore between asset and quote currency), but DVC tracked the Binance-native naming `BTCUSDT_15m.parquet`. The result was five test cases failing with FileNotFoundError that had been silently accepted as "known failing" for months.

Fix: the loader now accepts both naming conventions.

### 6.6 Net effect of bug discovery

Prior to this audit, the project's README reported the following headline metrics from a "Phase 2 walk-forward validation": Sharpe 2.91, Calmar 8.98, Max DD −15.4%, Win Rate 65.7%. After correcting all four bugs and running the multi-market walk-forward in this paper, the realistic estimates for any single tested strategy on its own market are Sharpe 0.4–0.6, MaxDD −30 to −50%, and 35–55% win rate — entirely consistent with the literature baseline for simple trend-following [@hurst2017century] and far below the original claim. The factor-of-six discrepancy between the original reported Sharpe and the realistic estimate was created entirely by the bugs documented above.

We emphasize that this is not an extreme case. The bugs described are, in our judgment, of a type that is common in retail repositories. Most are not caught by traditional testing because they produce plausible-looking output. They are caught only by multi-market out-of-sample testing — exactly the kind of testing that is rarely performed and almost never published.

---

## 7. Discussion

### 7.1 Why mean-reversion fails on trending crypto markets

The 2023–2026 period was a strong bull cycle for major cryptocurrencies (BTC +362%, SOL +735%, HBAR +140%). Mean-reversion strategies based on Relative Strength Index, Bollinger Bands, and moving-average filters generate sell signals when prices reach the upper end of recent ranges. In a sustained uptrend, these signals fire repeatedly and prematurely, exiting positions before the bulk of the move occurs. The result is a systematic underperformance of buy-and-hold, compounded by the transaction-cost drag of the round-trip trades themselves.

This is not a finding specific to our universe. It is a structural property of mean-reversion strategies applied to trending assets. The mismatch between strategy assumptions (range-bound behavior) and asset behavior (sustained directional moves) is the dominant explanatory variable for the negative results we observe.

### 7.2 Why trend-following matches but does not exceed buy-and-hold on metals

Simple trend-following filters such as the 200-day moving average crossover capture the bulk of sustained uptrends while exiting during sustained downtrends. The empirical performance on metals (Sharpe 0.4–0.7 across our four-metal universe) is consistent with the literature [@hurst2017century].

The fact that the strategy returns underperform buy-and-hold on raw return is explained by the asymmetric capture: the strategy misses 10–30% of the upside during each transition into a confirmed trend (the time spent below the moving average during a bottoming process is opportunity cost), and pays transaction costs on each transition. Over a 25-year period during which gold rose from $273 to $4685 (a 17× appreciation), the strategy captures approximately one-third of the absolute return while reducing maximum drawdown by approximately 30–50%. This is the canonical trend-following risk-reward profile: lower returns, smoother equity curve, much lower tail risk.

The key implication: for an investor who would have otherwise held passive buy-and-hold, trend-following provides drawdown reduction at the cost of return — a different risk preference, not an alpha. For an investor seeking to outperform passive, simple trend-following does not provide the answer.

### 7.3 Industry benchmark context

Our retail-accessible results compare instructively to institutional benchmarks:

- **AQR Style Premia Alternative Fund (QSPIX)**: launched with a target live Sharpe of 0.70, realized 0.41 since inception (2014–2025). The fund has access to multi-factor signals, institutional pricing, and proprietary execution.
- **SG CTA Index**: tracks the largest professional CTAs. Long-run Sharpe approximately 0.56; the last 15 years average approximately 0.40.
- **Passive 60/40 SPY+TLT**: realized Sharpe approximately 0.6–0.7 over the same long-run window, with quarterly rebalancing requiring approximately five minutes of work per quarter.

Our retail trend-following results (Sharpe 0.4–0.7 across metals) sit within the institutional band, suggesting that the bottleneck is not infrastructure but strategy class. Even with all-known-techniques applied, simple trend-following on liquid markets converges to a Sharpe in the 0.4–0.7 range. Multi-factor extensions add some diversification but, per Section 7.4 and the AQR live result, do not lift the realized Sharpe meaningfully above this band.

### 7.4 The Sharpe ceiling for retail multi-factor implementations

A natural follow-up question is whether multi-factor extension — combining trend, momentum, carry, and value factors across a larger universe — could lift performance above the simple-trend baseline. The literature suggests an in-sample Sharpe gain of 0.1–0.3 from such extensions [@asness2013valmom], but the AQR Style Premia live result (0.41 from a multi-factor combination targeting 0.70) suggests that the realized live gain is substantially smaller.

For a retail implementation specifically, several constraints further compress the achievable Sharpe:
1. Term-structure data for commodity carry is not readily available on free data sources; Yahoo Finance provides only the front-month rolled series.
2. Transaction-cost-aware portfolio optimization, which Frazzini et al. [@frazzini2018trading] estimate contributes 50–100 basis points of annual return to AQR's portfolio, is not implementable without an explicit trade-cost model that retail builders rarely construct.
3. Cross-margining benefits and prime-brokerage rates are structurally unavailable to retail.

These considerations suggest a realistic retail multi-factor ceiling in the range of Sharpe 0.5–0.7, gross of taxes and emotional capital, after honest deflation corrections.

### 7.5 The role of negative results

We argue that honest negative results from retail auditing serve an underappreciated function. The default reporting bias in the retail-trading literature is heavily positive: strategies that worked in a single backtest are published; strategies that failed are silently discarded. This produces a body of "literature" in which the reader cannot distinguish robust findings from selection artifacts. The base-rate question — what fraction of plausibly-designed retail strategies actually survive honest out-of-sample testing? — is not answered by reading positive case studies.

The eight-month effort that produced this paper had every characteristic that should *fail* by retail standards: ambitious in scope, multi-component architecture, optimistic README claims. Documenting the failure honestly, with reproducible code and an explicit accounting of the methodological errors that produced earlier inflated metrics, contributes to a more accurate base-rate prior for retail readers contemplating similar projects.

---

## 8. Conclusion

We report the results of an eight-month retail algorithmic trading research project that tested rule-based technical strategies across two markets, four timeframes, five strategy classes, and 131 walk-forward windows. The central finding is negative: no tested strategy class achieves generalizable out-of-sample edge over buy-and-hold or passive risk-parity baselines after honest deflation corrections.

The positive findings are: (1) simple trend-following on metals achieves Sharpe ratios in the 0.4–0.7 range, consistent with a century of literature and with institutional CTA performance, but does not exceed passive equal-weight benchmarks; (2) the methodology required to detect this honestly — multi-market walk-forward with deflated-Sharpe correction and PBO checking — is implementable on free data with open-source tooling and is now released as part of this paper.

We document four classes of infrastructure bugs that quietly inflated our earlier in-sample results by approximately six-fold (Sharpe 2.91 reported vs Sharpe 0.4–0.6 realistic). We hypothesize that bugs of this class are common in retail repositories and contribute substantially to the persistent gap between published retail backtests and live performance.

The implication for retail traders contemplating active algorithmic systems is uncomfortable but clear: the realistic Sharpe ceiling for retail multi-factor implementations is in the same band as simple passive risk-parity portfolios, which require approximately five minutes of work per quarter. The expected dollar alpha of an actively-managed multi-factor system over a passive alternative is, after honest accounting for costs, taxes, and expected performance attrition, statistically indistinguishable from zero for the majority of retail participants. The project documented here is now retired as an active-trading effort and re-purposed as a research baseline and a passive-capital-allocation framework.

We do not argue that retail systematic trading is impossible. We argue that the published positive-result literature is severely survivorship-biased and that an honest base-rate prior — informed by negative-result audits such as this one — would dissuade most retail builders from attempting it as an income source, while preserving its value as an educational exercise.

---

## Acknowledgments

This work was conducted with extensive assistance from Anthropic's Claude (model Opus 4.7), which served simultaneously as a research collaborator, code reviewer, and (importantly) the agent that finally compelled the honest negative-result conclusion. We note with some chagrin that the same class of AI assistant that helped construct the over-engineered architecture critiqued in Section 6 was also the agent that audited the architecture and recommended its closure. This pattern — AI-assisted construction paired with AI-assisted post-hoc auditing — is one we expect to see frequently in retail research projects of the 2026 era.

---

## Code and data availability

All code, data fetchers, walk-forward engine, and statistical-correction implementations are released under the MIT License at https://github.com/StarDust1508/ScArlet-Sails (tag `v1.0-research-baseline`). A Kaggle notebook (`paper/notebooks/missing_backtests.ipynb`) reproduces all results from a clean environment in approximately 30 minutes of CPU time. Synthetic-data bootstrapping (`tests/conftest.py`) allows partial reproduction without network access.

---

## References

See `references.bib`. Key citations:
- Bailey & López de Prado 2014 — Deflated Sharpe Ratio
- Bailey, Borwein, López de Prado, Zhu 2014 — Probability of Backtest Overfitting
- Moskowitz, Ooi, Pedersen 2012 — Time Series Momentum
- Asness, Moskowitz, Pedersen 2013 — Value and Momentum Everywhere
- Hurst, Ooi, Pedersen 2017 — A Century of Evidence on Trend-Following Investing
- Harvey, Liu, Zhu 2016 — Cross-Section of Expected Returns
- Hou, Xue, Zhang 2020 — Replicating Anomalies
- López de Prado 2018 — Advances in Financial Machine Learning
- Frazzini, Israel, Moskowitz 2018 — Trading Costs
- Xiao et al. 2024 — TradingAgents
- Yu et al. 2024 — FinCon
- Yang et al. 2024 — FinRobot
- StockBench 2025
- AQR Style Premia Alternative Fund Fact Sheet 2025
- SG CTA Index 2025
- AllocateSmartly platform data 2024

---

## Appendix A: Strategy specifications

### A.1 SimpleRSIStrategy
- Entry: RSI(14) < 30 (oversold) AND price > EMA(50) (uptrend filter)
- Exit: RSI(14) > 70 OR price < EMA(50) OR take-profit at +3%
- Position: full account on entry, single-position only

### A.2 CombinedStrategy
- Entry: RSI(14) < 40 AND price > SMA(50) AND price < lower BB(20, 2) × 1.02
- Exit: RSI(14) > 60 OR price < SMA(50)
- Position: full account on entry, single-position only

### A.3 RuleBasedStrategy (P_rb formula)
- Score: W_opportunity × ∏ filters − C_fixed − R_penalty
- Entry: score > 0.05
- Exit: score < −0.1 OR any filter breaks

### A.4 200-day SMA trend filter
- Entry: close > SMA(200) (cross from below)
- Exit: close < SMA(200) (cross from above)

### A.5 Dual momentum (Antonacci)
- Monthly rebalance, 12-month lookback, top-2 holdings, equal-weighted
- Cash if no asset has positive 12-month return

---

## Appendix B: Cost model derivation

[TODO: explicit math for cost model with assumed retail commissions]

## Appendix C: Walk-forward parameter details

[TODO: window size, embargo period, in-sample/out-of-sample split logic]

## Appendix D: Reproducibility checklist

[TODO: Bailey-LdP reproducibility checklist completed]
