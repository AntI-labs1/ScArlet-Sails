# Twitter / X Thread

10-tweet thread. Designed for quant-fin Twitter (followers like @macrocephalopod, @jasoncstrasser, @TheBigToddler, @LucaDellanna).

---

## 1/10 — hook

```
8 months. 10k lines of Python. "Walk-forward Sharpe 2.91" in the README.

Audited my own retail algo trading project last week. Found 4 bugs that, together, accounted for the entire claimed performance.

Real Sharpe after fixes: 0.4-0.6. Same as passive 60/40.

🧵
```

## 2/10 — the bugs in plain language

```
The bugs:
1. StandardScaler fit on entire dataset (look-ahead leak)
2. current_drawdown hardcoded to 0.0 — killswitch never fired
3. Inverted dispersion→position formula (tests asserted wrong direction)
4. THREE different Sharpe annualization factors in one project

None caught by unit tests. All caught by multi-market walk-forward.
```

## 3/10 — the empirical result

```
After fixes, ran 131 walk-forward windows:

- 14 crypto coins × 4 timeframes × 8 windows = 108 points
  → Average Sharpe -0.70, positive in 35% of windows
- 4 precious metals × 25 years × 8 windows = 23 points
  → Average Sharpe -0.19 for mean-reversion

Worse than coin flip. Rule-based mean-reversion does not generalize.
```

## 4/10 — the bright spot that wasn't

```
Trend-following on metals matched the literature:
+0.44 avg Sharpe across 4 metals via 200-day SMA filter.

But: did NOT beat buy-and-hold.
Gold strategy: +515% return
Gold B&H: +1629%

Trend-following = drawdown reduction, not alpha.
```

## 5/10 — the institutional reality check

```
Industry benchmarks for comparison:

AQR Style Premia Fund: realized Sharpe 0.41 since inception (target was 0.70)
SG CTA Index long-run: ~0.56
Passive 60/40 SPY+TLT: ~0.6-0.7 with zero work

Elite funds with proprietary data, PhDs, billions AUM — Sharpe 0.4-0.7.

Retail with yfinance is not going to hit 1.0+.
```

## 6/10 — the AI co-author observation

```
Built the over-engineered version WITH AI help. Generated diagrams, ROADMAPs, design docs. AI never said "stop, this is solved better by pysystemtrade."

Closed the project WITH AI help too. AI did the audit, found the bugs, ran honest multi-market backtest.

AI-built, AI-audited. The construction phase has no natural stop mechanism. The audit phase has to be requested.
```

## 7/10 — the survivorship problem

```
~95% of retail algo-trading content is survivorship-biased success reports.

Base rates from honest sources (AllocateSmartly, Dalbar, AQR data):
- 2-5% of retail traders achieve Sharpe > 0.7 live over 3+ years
- 30-40% end up with passive-equivalent in a more complex wrapper
- 60-70% abandon within 12-24 months

If you're median, your expected dollar alpha = 0.
```

## 8/10 — what I'd do differently

```
If I started over:

1. Read Clenow, Antonacci, López de Prado first. 6 weeks, $80 in books.
2. Fork pysystemtrade (Rob Carver, ex-AHL quant, 11yr development) — don't rebuild
3. Pre-commit to stop condition ("close if Sharpe < X at month 3")
4. Test cross-market from day 1
5. Deflated Sharpe Ratio for every reported number
```

## 9/10 — what I did instead of "trying one more thing"

```
Closed the active trading project.
Wrote a paper documenting the negative result.
Moved capital to passive 60/40.
Set quarterly rebalance reminder.

Total active work going forward: 5 min/quarter.
Expected dollar alpha vs my 10-week multi-factor build: 0.

Sleeping much better.
```

## 10/10 — the goods

```
Everything is open-source MIT:

📄 Paper draft: [arxiv link when published]
🔧 Code: https://github.com/StarDust1508/ScArlet-Sails
📓 Medium version (lay): [medium link]
📊 Deflated Sharpe + PBO impls: paper/notebooks/stats.py

If this resonates, share your honest negative results. The corpus is way too small.

/end thread
```

---

## Notes

- Post tweets 90s-2min apart so each gets independent algorithmic reach
- Best post time: 9-11 AM ET on Tuesday/Wednesday for quant-fin Twitter
- Quote-tweet (don't reply) to good responses to give them visibility
- Don't argue with rage-replies — block early and often
- Pin the thread for 1 week post-launch
- Include 1-2 figures (boxplot from paper, edge vs B&H heatmap) as image attachments on tweets 3 and 4

## Image suggestions

- Tweet 3: walk-forward Sharpe boxplot (crypto vs metals)
- Tweet 4: edge vs B&H heatmap
- Tweet 7: base-rate triangle (2-5% / 30-40% / 60-70%)
- Last tweet: GitHub repo screenshot or paper title page
