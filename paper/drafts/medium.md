# Why I closed my 8-month algorithmic trading project: an honest negative result

*Or: how a $0 passive portfolio quietly outperformed everything I built*

---

## TL;DR

I spent 8 months building a retail algorithmic trading system. It had 10,000+ lines of Python, multi-strategy ensembles, an "LLM Council" for pattern recognition, RAG retrieval over historical states, and a vectorized backtest engine claiming a Sharpe ratio of 2.91.

I ran an honest multi-market audit. The 2.91 was wrong by a factor of six.

After fixing four infrastructure bugs that were silently inflating every backtest, the true Sharpe across 14 cryptocurrencies and 4 precious metals — 131 walk-forward windows total — was **−0.70 for mean-reversion strategies** and **+0.4 to +0.6 for trend-following**. The trend-following matched the literature baseline, but **did not exceed a passive 60/40 portfolio**.

I'm publishing the code, the data, and the failure honestly because the retail trading literature has a survivorship problem and a base-rate problem: we see the wins, never the losses, and people lose 8 months to projects they don't realize are doomed by design.

---

## What I built (the architecture astronaut phase)

In early 2025, inspired by a wave of "AI trading agent" releases, I started building. The plan was modular:

- **Three quant strategies**: rule-based (RSI/Bollinger/MA filters), XGBoost ML, hybrid Q-learning
- **LLM Council**: multiple AI agents debating each trading decision
- **RAG layer**: retrieval over historical market states for analogical reasoning
- **Dynamic position sizing**: regime-aware, OOD-detection-aware, dispersion-aware
- **Human-in-the-loop**: final accept/modify/reject by me
- **GARCH volatility models, CVaR risk penalties, Mahalanobis OOD detection**

I had architecture diagrams. I had a 13-page ROADMAP. I had a README listing performance metrics: Walk-forward Sharpe 2.91, Calmar 8.98, Max DD −15.4%.

What I didn't have was an honest backtest.

---

## The first audit (where the bottom dropped out)

In April 2026 I sat down with Anthropic's Claude (Opus 4.7) and asked it to audit the code base. Within an hour it had identified four bugs that, between them, accounted for the entire claimed performance:

### Bug 1: Look-ahead bias in feature scaling

The `StandardScaler` (a standard scikit-learn tool for normalizing features) was being fitted on the entire dataset — including future bars — and then used to "predict" the past. This is the textbook example of look-ahead bias. Every backtest was implicitly cheating by using information that wouldn't have been available at the time of the trade.

### Bug 2: Hardcoded `current_drawdown=0.0` in the killswitch

The risk killswitch — meant to reduce position size as the portfolio drew down — was reading `current_drawdown=0.0` from a hardcoded constant. Risk management wasn't running. The portfolio could lose 90% and the system would size positions as if everything was fine.

### Bug 3: Inverted dispersion-to-position formula

The system was *meant* to take smaller positions when strategies disagreed (high uncertainty) and larger when they agreed. The unit tests asserted exactly the opposite. The tests had been "failing" for months, and I'd been about to "fix" the source code to satisfy the wrong tests.

### Bug 4: Three different Sharpe annualization factors

Different modules used `sqrt(252 × 96)`, `sqrt(252 × 24 × 4)`, and `sqrt(252 × 24)`. The first two are arithmetically identical (24192) but conceptually inconsistent. None of these are correct for 24/7 cryptocurrency markets, which require 365-based annualization.

None of these would have been caught by typical unit testing. They were caught only by running the system on out-of-sample data across multiple markets — exactly the kind of test most retail systems never receive.

---

## The honest audit (where the dream died)

After the bug fixes, I ran the system properly:

**Crypto, 14 coins × 4 timeframes × 8 walk-forward windows = 108 data points:**
- Average Sharpe: **−0.70**
- Positive windows: **38 out of 108 (35%, worse than a coin flip)**
- On 15-minute timeframes, the combined strategy lost **97% of capital** through commission drag

**Metals, 4 metals × 25 years of daily data = 23 walk-forward windows:**
- Average Sharpe for the simple combined strategy: **−0.19** (essentially zero)
- 200-day SMA trend-following: **+0.44** — matching the literature
- Dual momentum portfolio: **+0.62** — but **identical to passive equal-weight buy-and-hold (+0.63)**

The trend-following on metals "worked" in the sense that it produced positive Sharpe consistent with a century of academic literature. But it **didn't beat buy-and-hold**. It just captured drawdown reduction at the cost of return.

---

## The institutional reality check

Here's what makes the result so quietly devastating:

| | Live Sharpe |
|---|---|
| **AQR Style Premia Alternative Fund (QSPIX)** — institutional multi-factor with proprietary data, billion-dollar AUM, PhD quants | **0.41 since inception** (target was 0.70) |
| **SG CTA Index** — the largest professional trend-following managers in the world | **~0.56 long-run** |
| **Passive 60/40 SPY+TLT, rebalanced quarterly** — five minutes of work per quarter | **~0.6–0.7** |
| **My 8-month retail multi-factor system** | **Inconclusive at best; matches passive at best-case** |

If elite hedge funds with billions in AUM, proprietary execution, and PhD quants average Sharpe 0.4–0.7 on this exact strategy class, **a retail trader with yfinance is not going to hit 1.0+**. It's not a question of effort. It's a physical limit of the strategy class on these markets.

And here's the kicker: a passive 60/40 portfolio gives you the **same Sharpe** with **zero work**. The math says my 8 months were spent rebuilding what already existed.

---

## What I learned (the painful version)

### 1. AI assistants are biased toward construction, not stopping

I built the over-engineered architecture *with* AI help. The AI was enthusiastic. It generated architecture diagrams, ROADMAPs, design documents. It never said "this is already solved better by pysystemtrade" or "the literature predicts your Sharpe ceiling is 0.5."

The same AI later helped me audit and close the project. It identified the bugs, ran the honest backtest, and forced the negative-result conversation.

**Pattern**: AI-assisted construction paired with AI-assisted auditing. The construction phase has no natural stopping mechanism. The auditing phase has to be requested explicitly. If you don't request the audit, your project grows forever in the wrong direction.

### 2. Single-coin backtests are casino-grade evidence

One of the coins in my universe (SOL) gave a Sharpe of 1.20 on a single backtest. For three days I told myself this was "my edge." Then I ran walk-forward across all 14 coins and found SOL was a statistical outlier — the same strategy averaged Sharpe −0.7 across the other 13. SOL was lucky, not edged.

**Rule**: if your strategy works on one asset but you haven't tested it on 10+ unrelated assets, you haven't tested anything.

### 3. Costs dominate edge on short timeframes

On 15-minute crypto with 0.15% round-trip costs, after 1300 trades over 2.5 years, you've paid 405% in commissions. There is no edge that survives that. Trading frequency is a hidden tax that retail almost universally underestimates.

**Rule**: if your strategy needs to trade more than once per day to capture its edge, ask yourself whether you've calculated round-trip costs honestly.

### 4. The literature is right, you just don't want to believe it

Andreas Clenow ("Following the Trend"), Cliff Asness (AQR), Hurst/Ooi/Pedersen ("Century of Evidence") — they've all said this for decades. Simple trend-following on diversified futures, after costs, gives Sharpe 0.4–0.7. That's the ceiling for retail without proprietary data or execution.

I read these papers. I thought I'd be different. I wasn't.

---

## What I did instead

I closed the actively-managed trading project. The repository now serves two purposes:

1. **A reproducible research baseline.** All code, walk-forward methodology, deflated-Sharpe and PBO implementations, and data fetchers (Binance Vision for crypto, yfinance for metals) are MIT-licensed at https://github.com/StarDust1508/ScArlet-Sails. The accompanying academic paper documents the methodology and the negative result.

2. **A passive capital allocation framework.** A 60/40 (or All-Weather, or Permanent Portfolio) split with a quarterly rebalance script. Five minutes per quarter. Sharpe ~0.6. Done.

The 10 weeks I was about to spend on "multi-factor build" went to writing this paper instead. The capital that was going to back-test in a paper-trade simulator went straight into a 60/40 split.

I'm sleeping better.

---

## If you're considering a similar project

I'm not going to tell you don't do it. The educational value of building, validating, and honestly failing is enormous. I learned more in this 8 months than I did in years of finance reading.

But if you go in, go in with **honest base rates**:

- **2–5%** of retail traders attempting multi-factor systems achieve Sharpe > 0.7 live over 3+ years
- **30–40%** end up with something equivalent to 60/40 in a more complex wrapper (no dollar alpha vs passive)
- **60–70%** abandon the project within 12–24 months

If you're in the median bucket, your honest expected dollar alpha vs passive is **zero**. Sometimes negative after taxes and transaction costs. You're paying 10 weeks of life for the educational value, which is real, but you should know that's what you're paying for.

If you want to compete with the professionals, accept that the professionals (AQR, Man AHL, Winton) are themselves running at Sharpe 0.4–0.7. They don't have a magical edge over passive; they have **scale**, **leverage access**, **execution costs**, and **diversification across dozens of weakly-correlated alphas**. None of those are individually replicable for retail.

---

## What I'd do differently

If I started over with what I know now, here's the order:

1. **Read three books first**: Clenow's "Following the Trend", Antonacci's "Dual Momentum", López de Prado's "Advances in Financial Machine Learning". Cost: 6 weeks of reading, $80 total.
2. **Fork an existing system**: `pysystemtrade` by Rob Carver (former AHL quant, 11 years of active development). Don't rebuild what already exists.
3. **Define a stop-loss for the project itself**: "I will close this if walk-forward Sharpe after 3 months of work is below X."
4. **Test on multiple unrelated markets from day one**: not just BTC. Not just one coin. Cross-asset is the only honest validation.
5. **Use deflated Sharpe ratio for any reported number**: if you can't say "deflated Sharpe X.X over Y trials," you don't have a result, you have an in-sample artifact.

---

## The data and code

Everything is on GitHub: https://github.com/StarDust1508/ScArlet-Sails

If you're considering a similar project, look at the `POST_MORTEM.md` first. If you're writing one of your own, the `paper/notebooks/stats.py` has clean implementations of Deflated Sharpe Ratio and Probability of Backtest Overfitting that you can drop into your own work.

The academic paper version is at [arXiv link when published] for those who want the full methodology.

---

**Final thought**: the most useful thing I produced in 8 months was the honest negative result. That's the value the retail trading literature doesn't publish. If you've gotten this far and are still considering algorithmic trading, please at least test honestly before reporting "my strategy works." Most don't. Mine didn't. The literature has been telling us this for thirty years.

The 60/40 portfolio is boring. The 60/40 portfolio also wins.

---

*If this resonates, I'd love to hear about your own honest negative results — they're harder to find than success stories, but more useful.*
