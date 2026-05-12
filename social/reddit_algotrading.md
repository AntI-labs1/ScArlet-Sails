# r/algotrading Post

## Subreddit

`/r/algotrading` (primary) — and after 48h, cross-post to:
- `/r/quant` — more academic angle, lead with paper draft
- `/r/SecurityAnalysis` — value-investing-leaning audience, may resonate with "passive is enough"
- `/r/wallstreetbets` — DON'T post, audience is wrong fit
- `/r/cryptocurrency` — only if framing is "why my crypto bot failed"

## Title

```
[Negative result] 8-month algo project audit: rule-based has no edge across crypto + metals (131 walk-forward windows, code & paper inside)
```

Alternatives:
- `My algo trading project found no edge — here's the honest write-up`
- `What 131 walk-forward backtests taught me: simple rules don't work`

## Body

```markdown
**TL;DR**: I built a retail algo trading system over 8 months. Audited it honestly across 14 cryptocurrencies and 4 precious metals (131 walk-forward windows total). Result: no generalizable out-of-sample edge from rule-based technical strategies. The literature was right; I was wrong.

Posting because this sub has a survivorship problem — we see 100 success posts for every honest failure post, which gives newcomers a wildly inflated base rate for what's achievable at retail.

# What I built

Multi-strategy ensemble (RSI/Bollinger/MA), LLM "Council" for pattern interpretation, RAG retrieval, dynamic position sizing, OOD detection, GARCH risk penalty. About 10k lines of Python. Targeted crypto, briefly pivoted to metals during validation.

Pre-audit README claimed walk-forward Sharpe 2.91, Calmar 8.98, MaxDD -15.4%.

# What honest audit found

Four infrastructure bugs that silently inflated every backtest:

1. **Look-ahead bias** via StandardScaler.fit_transform() on combined train+test data
2. **Hardcoded current_drawdown=0.0** — killswitch never fired
3. **Inverted dispersion-to-position formula** — unit tests asserted wrong direction
4. **Three different Sharpe annualization factors** across modules (none correct for 24/7 crypto)

None caught by typical unit tests. All four detected only through multi-market out-of-sample audit.

# Empirical results (after bug fixes)

**Crypto mean-reversion (combined strategy on 14 coins × 4h × 8 walk-forward windows = 108 points):**
- Average Sharpe: **−0.70**
- Positive Sharpe windows: **38 of 108 (35%, worse than coin flip)**
- On 15m timeframe: 13/14 coins lost >94% capital in 2.5 years (commission drag)

**Metals trend-following (200-day SMA on 4 metals × 25 years):**
- Average Sharpe: **+0.44** (4/4 positive)
- Matches literature [Hurst/Ooi/Pedersen 2017] baseline
- BUT: does NOT beat buy-and-hold (gold strategy 515% vs B&H 1629%)

**Dual momentum (4 metals, monthly, top-2):**
- Strategy: CAGR 12.70%, Sharpe 0.62, MaxDD -49.5%
- B&H equal-weight: CAGR 12.09%, Sharpe **0.63**, MaxDD -53.9%
- Net Sharpe difference: essentially zero

# Industry context

- AQR Style Premia Alternative Fund (QSPIX): institutional multi-factor with proprietary data, PhD quants, $billions AUM. Realized Sharpe **0.41 since inception** vs targeted 0.70
- SG CTA Index (largest professional trend-followers): ~0.56 long-run
- Passive 60/40 SPY+TLT, quarterly rebalance: ~0.6-0.7 with zero work

If elite funds with everything we don't have realize Sharpe 0.4-0.7, retail with yfinance is mathematically not going to achieve 1.0+. It's a class-of-strategy ceiling, not an implementation issue.

# What I did instead

Closed the active-trading project. Repo now serves as:
- Reproducible research baseline (vectorbt-based engine, walk-forward, Deflated Sharpe + PBO impls)
- Source for an academic paper on the negative result
- Passive portfolio framework (60/40 / All-Weather / Permanent options with rebalance CLI)

Real capital is going into passive 60/40. 10 weeks of "multi-factor build" effort went into the paper instead.

# Code, paper, data

All MIT-licensed: https://github.com/StarDust1508/ScArlet-Sails

- Full post-mortem: POST_MORTEM.md in repo root
- Academic paper draft: paper/drafts/main.md (5500 words)
- Casual Medium version: paper/drafts/medium.md
- Reproduction recipe: REPRODUCING.md (Kaggle one-click + local)
- Open-source Deflated Sharpe + PBO implementations: paper/notebooks/stats.py

# Open questions for the sub

1. How many of you have done multi-market walk-forward on your "working" strategies? If you haven't, your edge might be a single-asset survivorship artifact (mine was — SOL gave Sharpe 1.20 in one backtest before I tested it across all 14 coins).

2. How many of you can produce a Deflated Sharpe Ratio for your strategies? If not, your in-sample Sharpe is roughly 2x your real edge.

3. Anyone running a strategy for 3+ years live with verified Sharpe > 0.7? Would love to hear what specifically separates you from the median outcome. Genuinely — not asking to be contrarian, asking because the academic data says you're top 2-5%.

# Honest take

I'm not saying don't do this. I learned more about quant finance in 8 months than in years of reading. But go in with honest base rates (~2-5% of retail attempts reach Sharpe >0.7 live), pre-commit to stop conditions, and test cross-market from day one.

Happy to take harsh methodology critique in comments.
```

## Posting tips

- Post Sunday evening 8-10 PM EST for max early visibility
- Be active in first 4 hours answering comments — Reddit algorithm boosts engaged threads
- Don't repost on rejection; just delete and try the alt title 2 weeks later
- Mod-flair: "Education" or "Research" depending on sub flair options

## Don't

- Promise to share live results (sets you up for a follow-up failure narrative)
- Engage with "why didn't you try X" without first asking "how would you walk-forward validate X?"
- Brag about negative result being "actually a win" — let the readers conclude that
