---
marp: true
theme: default
paginate: true
backgroundColor: white
header: 'ScArlet-Sails | v1.0-research-baseline | 2026'
footer: 'github.com/StarDust1508/ScArlet-Sails'
style: |
  section {
    font-size: 1.5em;
  }
  h1 {
    color: #1f3a5f;
  }
  h2 {
    color: #2c5282;
    border-bottom: 2px solid #cbd5e0;
  }
  table {
    font-size: 0.85em;
  }
  .neg { color: #c53030; font-weight: bold; }
  .pos { color: #2f855a; font-weight: bold; }
  .neutral { color: #4a5568; }
---

<!-- _class: lead -->

# Rule-Based Technical Trading Strategies Fail to Generalize

## A Multi-Market Walk-Forward Audit
### Crypto 2023-2026 and Metals 2000-2026

Independent research, May 2026
arXiv: [pending]

---

## The setup

An 8-month retail algorithmic trading research project ("ScArlet-Sails"):

- Multi-strategy ensemble (rule-based, ML, RL)
- LLM "Council" for pattern interpretation
- RAG retrieval over historical states
- Human-in-the-loop final decision
- ~10,000 lines of Python

**Pre-audit README claimed**: walk-forward Sharpe **2.91**, Calmar 8.98

This talk explains why the audit found the claim was wrong by a factor of six.

---

## The empirical setup

Tested **5 strategy classes** across **2 fundamentally different markets**:

| | Cryptocurrencies | Precious Metals |
|---|---|---|
| Universe | 14 USDT pairs | 4 (GOLD, SILVER, COPPER, PLATINUM) |
| Timeframes | 15m, 1h, 4h, 1d | 1d |
| History | 2.5 years | 25 years |
| Total walk-forward windows | 108 | 23 |
| Data source | Binance Vision (free archive) | Yahoo Finance (free) |

Strategies: RSI mean-reversion · Combined RSI+BB+MA · 200-day SMA trend · Gold/Silver ratio · Dual momentum

---

## Result 1: Crypto mean-reversion

<style scoped>
table { font-size: 0.7em; }
</style>

Walk-forward by coin (14 × 8 windows = 108 points):

| Coin | Sharpe mean | Pos windows |
|---|---:|---|
| BTC | <span class="neg">−0.28</span> | 4 / 8 |
| ETH | <span class="neutral">+0.03</span> | 3 / 8 |
| **SOL** | <span class="pos">**+0.76**</span> | 6 / 8 |
| AVAX | <span class="neg">−1.58</span> | 0 / 8 |
| 10 others | <span class="neg">−1.0 average</span> | 2 / 8 average |

**Aggregate: Avg Sharpe −0.70, positive windows 35% (worse than coin-flip).**

SOL is a *single-coin outlier*, not generalizable edge.

---

## Result 2: 15-minute timeframe is a death trap

On 15m timeframe, ~1300 trades per coin × 30 bps round-trip = **405% commission drag**.

| Coin | Total return | Trades |
|---|---:|---:|
| BTC | <span class="neg">−98.43%</span> | 1351 |
| ETH | <span class="neg">−97.13%</span> | 1353 |
| SOL | <span class="neg">−96.69%</span> | 1331 |
| ... 11 more all <span class="neg">−85% to −98%</span> | | |

**There is no signal that survives this cost level.**

Lesson: if your strategy needs to trade >1×/day at retail costs, math is against you.

---

## Result 3: Metals trend-following — works, doesn't beat B&H

200-day SMA filter on 4 metals, 25-year history:

| Metal | Strategy Sharpe | Strategy Return | B&H Return |
|---|---:|---:|---:|
| GOLD | <span class="pos">+0.68</span> | +515% | +1629% |
| SILVER | <span class="pos">+0.40</span> | +278% | +1642% |
| COPPER | <span class="pos">+0.49</span> | +364% | +631% |
| PLATINUM | <span class="pos">+0.19</span> | +18% | +428% |
| **Avg** | **+0.44** | — | — |

✅ All four positive Sharpe — confirms literature ($0.4-0.7$ baseline)
❌ All four underperform buy-and-hold raw return
**= Trend-following provides drawdown reduction, not alpha.**

---

## Result 4: Dual momentum ≈ passive equal-weight

Antonacci dual momentum on 4 metals, 23 years:

| | Strategy | B&H Equal-Weight |
|---|---:|---:|
| Total return | +1438% | +1258% |
| CAGR | 12.70% | 12.09% |
| **Sharpe** | <span class="pos">**+0.62**</span> | <span class="pos">**+0.63**</span> |
| Max DD | −49.5% | −53.9% |

**Sharpe difference: ≈ 0.**

Strategy adds risk proportionally to its return improvement → no risk-adjusted alpha.

---

## The industry benchmark reality

If retail with yfinance and 8 months can't generate edge, what can?

| Reference | Realized Sharpe |
|---|---:|
| AQR Style Premia Fund (institutional multi-factor) | <span class="neutral">0.41 since inception</span><br>(target was 0.70) |
| SG CTA Index (largest pro trend-followers) | <span class="neutral">~0.56 long-run</span> |
| **Passive 60/40 SPY+TLT, quarterly rebalance** | <span class="pos">**~0.6-0.7**</span><br>**(5 min of work / quarter)** |
| **Our retail metals trend-following** | <span class="neutral">**0.4-0.6**</span> |

**Even elite institutions cluster at 0.4-0.7. Retail with free data is not going to exceed that band.**

---

## Section 6: Infrastructure bugs (the main contribution)

Pre-audit Sharpe claim: **2.91**. Post-audit reality: **0.4-0.6**. The gap (factor of 6) is entirely attributable to:

1. **Look-ahead bias** in `feature_engine_v2._normalize_features()` — StandardScaler silently fitted on combined train+test
2. **Hardcoded `current_drawdown=0.0`** in `run_council.py` — killswitch never triggered
3. **Inverted dispersion → position formula** — tests asserted the wrong direction; nearly "fixed" by inverting source
4. **3 inconsistent Sharpe annualization factors** simultaneously: `sqrt(252×96)`, `sqrt(252×24×4)`, `sqrt(252×24)`

**None caught by traditional unit testing.** All caught by multi-market walk-forward.

---

## Why this matters

The retail algo-trading literature has a **survivorship problem**:

- ~95% of public content = single-asset success reports
- The audit process required to detect bugs like above is rarely performed
- Even more rarely published
- Newcomers see only the wins and form unrealistic base rates

**Honest base rates from external sources (AllocateSmartly, AQR data, Dalbar)**:

- 2-5% of retail attempts reach Sharpe > 0.7 live over 3+ years
- 30-40% end up with passive-equivalent in a more complex wrapper
- 60-70% abandon within 12-24 months

---

## What we contributed

1. **131 walk-forward windows** of honest multi-market validation
2. **Documented inventory of 4 silent-inflation bugs** with detection methodology
3. **Open-source MIT toolkit**:
   - vectorbt-based backtest engine
   - Binance Vision + yfinance fetchers (free data)
   - Deflated Sharpe Ratio + PBO implementations (numpy-only)
   - Walk-forward framework
   - Reproducible Kaggle notebooks (30-min replication)

**Code, data, paper, drafts**:
https://github.com/StarDust1508/ScArlet-Sails (tag `v1.0-research-baseline`)

---

## The AI angle

The over-engineered architecture was built **with AI help**.
The honest audit was done **with AI help**.
**Same AI assistant** built and dismantled the project.

Observation: **AI-assisted construction has no natural stop condition**.
AI eagerly produces architecture diagrams, ROADMAPs, design documents.
Audit phase has to be *explicitly requested*.

Pattern likely common in 2026-era retail research. Mitigation: pre-commit
to audit milestones at construction time.

---

## What I'd do differently

1. **Read 3 books first**: Clenow "Following the Trend", Antonacci "Dual Momentum", López de Prado "Advances in FinML". 6 weeks, $80.
2. **Fork existing**: `pysystemtrade` (Rob Carver, ex-AHL quant, 11 yr dev). Don't rebuild.
3. **Pre-commit stop conditions**: "close if walk-forward Sharpe < X at month 3".
4. **Test cross-market from day 1.** Single-asset success = casino evidence.
5. **Deflated Sharpe Ratio** for every reported number.

Or: just buy 60/40 and spend the 8 months on something else.

---

## What's next

- ✅ arXiv preprint (under submission)
- ✅ SSRN parallel posting
- 🔄 ICAIF 2026 workshop (when submission opens)
- ✅ Medium / Habr adapted versions
- ✅ Passive 60/40 portfolio framework (`passive/`)
- ❌ NOT: another iteration on the trading strategies

The project is closed as actively-managed system.
The capital is in 60/40 with quarterly rebalance.

---

<!-- _class: lead -->

# Thank you

## github.com/StarDust1508/ScArlet-Sails

### Questions?

Especially harsh methodological ones welcome.

The most useful response to this talk is **your own honest negative result.**

The corpus is too small.

---

## Backup: Cost sensitivity

| Position size | Total Return | Sharpe | Max DD |
|---|---:|---:|---:|
| 25% | +18.49% | 1.200 | −4.23% |
| 50% | +39.27% | 1.201 | −8.32% |
| 75% | +62.45% | 1.202 | −12.28% |
| 95% | +82.77% | 1.202 | −15.35% |

**Sharpe invariant under leverage** (theoretical result; verifies engine correctness post-bug-fix).

Position sizing is risk management, not edge creation.

---

## Backup: Why mean-reversion fails on trending crypto

Bull market 2023-2026: BTC +362%, SOL +735%, HBAR +140%.

Mean-reversion: sells at upper RSI / BB → exits before the bulk of the move.

This is **mathematical** structure mismatch, not strategy bug:
- Mean-reversion assumes range-bound behavior
- Sustained trends are not range-bound
- Round-trip costs compound on each false reversal signal

**Direction of bug**: model misfit on the *asset's nature*, not the model itself.

---

## Backup: Deflated Sharpe formula

$$ \text{DSR}(\hat{SR}) = \hat{SR} - \mathbb{E}[\max(\hat{SR}_0)]_N $$

where

$$ \mathbb{E}[\max(\hat{SR}_0)]_N \approx \frac{(1-\gamma)\Phi^{-1}(1 - 1/N) + \gamma\Phi^{-1}(1 - 1/(Ne))}{\sqrt{T}} $$

- $\hat{SR}$ = observed Sharpe
- $N$ = number of strategy trials tested
- $T$ = number of observations
- $\gamma \approx 0.5772$ = Euler-Mascheroni
- $\Phi^{-1}$ = inverse standard normal CDF

For $N=100$, $T=252$: $\mathbb{E}[\max] \approx 0.22$. So an observed Sharpe of 1.0 deflates to ~0.78.

Reference: Bailey & López de Prado (2014), JPM.
