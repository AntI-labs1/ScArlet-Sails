# LinkedIn Post

Professional-network framing. Recruiters and peers should read this and conclude: this person can recognize a failed project, stop it on time, and produce a publishable artifact from the experience.

## Post

```
I closed an 8-month side project last week and learned more from the closure than I did during construction.

The project: a retail algorithmic trading system combining multiple
quantitative strategies, LLM-based pattern recognition, and a vectorized
backtest engine. Initial reported metrics looked impressive — walk-forward
Sharpe ratio of 2.91, Calmar ratio of 8.98.

The audit: a rigorous multi-market walk-forward validation across 14
cryptocurrencies and 4 precious metals — 131 out-of-sample windows total,
with Deflated Sharpe Ratio (Bailey & López de Prado 2014) and Probability
of Backtest Overfitting corrections applied.

The finding: four infrastructure bugs (look-ahead bias in feature scaling,
disabled risk killswitch, inverted dispersion formula, inconsistent Sharpe
annualization across modules) silently inflated every backtest by
approximately 6x. After fixes, realistic Sharpe was 0.4-0.6 — matching
the literature baseline for simple trend-following but not exceeding
passive risk-parity portfolios.

The honest reckoning: AQR's Style Premia Fund — institutional multi-factor
with proprietary data, billion-dollar AUM, PhD quants — realizes Sharpe
0.41 since inception. SG CTA Index averages 0.56. Passive 60/40 SPY+TLT
delivers 0.6-0.7 with zero ongoing work. The retail algorithmic ceiling
is the same band, with significantly more operational complexity.

The decision: closed the project as an actively-managed system; transitioned
the capital to a passive risk-parity portfolio; converted the 10 weeks
that would have gone into "multi-factor extension" into writing an honest
negative-result paper.

Why I'm sharing publicly: the retail algorithmic trading literature has
a survivorship problem. ~95% of public content is single-asset success
reports. The base-rate truth (2-5% of retail attempts reach Sharpe >0.7
over 3+ years) gets lost in selection bias. Documenting failures
contributes more to the public corpus than another buy-side success story.

What's in the open-source repository:
• Reproducible walk-forward backtesting infrastructure
• Free-data fetchers for crypto (Binance Vision) and metals (yfinance)
• Open implementations of Deflated Sharpe Ratio and PBO
• Full academic paper draft (5500 words) + Medium-adapted version
• Bug inventory and fix documentation
• Passive portfolio rebalance framework (no active management)

I think the most useful thing for early-career quant researchers and
retail builders is to see what a closed-out project looks like — not
just the wins. If you're considering a similar project, the honest base
rates and the bug-inventory section may save you 6+ months.

Repository: https://github.com/StarDust1508/ScArlet-Sails
Paper: [arxiv link when published]

Happy to discuss methodology, the closure decision-process, or career
implications of publicly documenting a negative result. Comments and
critiques welcome — especially harsh methodological ones.

#QuantitativeFinance #AlgorithmicTrading #ResearchTransparency
#NegativeResults #OpenSource #DataScience
```

## Pre-launch checklist

- [ ] Update profile headline if relevant (e.g., "Independent researcher | open-source quantitative finance")
- [ ] Pin the post for 7 days post-launch
- [ ] Tag 3-5 quant-finance people if you know them personally (no spray-and-pray)
- [ ] Have a 1-paragraph DM response prepared for recruiters who reach out
- [ ] Have a "questions I get" doc ready (link to relevant repo sections)

## DM response template for recruiter reach-outs

```
Hi [name],

Thanks for reaching out. The ScArlet-Sails project closure was deliberate
— I'm not currently looking for [trading desk / algo seat / quant role],
but I am [open to / actively considering] [research / engineering /
independent] roles where rigorous empirical validation matters more than
strategy generation.

If that sounds like a fit, my background and current focus are summarized
at [your-personal-page / GitHub profile]. Happy to chat further.

[Your name]
```

(Adapt as appropriate for your actual situation.)

## What NOT to say in comments

- Promises about "next project" beyond what's actually planned
- Specific capital amounts
- Claims that "passive doesn't work for everyone"
- Implication that other people who do algo-trading are deluded

The post is strong because it lets the data speak. Don't undermine it with editorializing in replies.
