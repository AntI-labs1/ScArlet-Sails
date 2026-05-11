# Hacker News Post

## Title (use this exact text)

```
Show HN: An honest negative-result audit of my 8-month algo-trading project
```

Alternative titles (A/B test on different days):

- `Why I closed my 8-month algorithmic trading project`
- `An 8-month retail algo trading project, audited honestly`
- `My algo-trading project: 4 bugs that inflated Sharpe 6x`

## URL field

```
https://github.com/StarDust1508/ScArlet-Sails
```

(Or substitute the Medium link if preferring the narrative version on first impression.)

## First comment (post immediately as author, gives context)

```
Author here. Quick context for HN:

I spent 8 months building a retail algo trading system. RSI/Bollinger/MA
strategies, an LLM "Council" for pattern recognition, RAG retrieval,
~10k lines of Python. README claimed walk-forward Sharpe 2.91.

Last week I audited it properly. Found 4 infrastructure bugs that
between them accounted for the entire claimed performance:

1. StandardScaler.fit_transform() on combined train+test data (textbook
   look-ahead bias, invisible in unit tests)
2. Hardcoded `current_drawdown=0.0` in the risk killswitch — the
   killswitch never fired regardless of actual losses
3. Inverted dispersion-to-position-size formula — unit tests asserted
   the wrong direction and I was about to "fix" the source to match them
4. Three different Sharpe annualization factors used simultaneously
   across modules, none of them correct for 24/7 crypto

After fixes + multi-market walk-forward (108 windows on 14 crypto pairs,
23 windows on 4 metals), realistic Sharpe is 0.4-0.6 — matching the
literature baseline for simple trend-following, NOT beating passive 60/40
(which is also Sharpe ~0.6 with zero work).

Industry context: AQR Style Premia (institutional multi-factor) realized
Sharpe is 0.41 since inception, vs their 0.70 target. SG CTA Index averages
~0.56 long-run. If elite funds with PhDs and billion-dollar AUMs hit 0.4-0.7,
retail with yfinance is not going to hit 1.0+.

I published the code, the paper draft, the bug inventory, and a passive-
allocation framework I'm transitioning to. Goal is to add to the (sparse)
public corpus of honest negative results from retail trading attempts,
since 95% of what gets published is survivorship-biased success stories.

Paper draft (academic): paper/drafts/main.md in the repo
Medium version (lay): paper/drafts/medium.md
Full post-mortem: POST_MORTEM.md

Happy to take hard questions, especially methodology critiques.
```

## Expected discussion patterns

- **Skeptics**: "Could be your specific strategies were bad" → answer: yes, that's the point; we documented 5 strategy classes failing similarly. The base-rate claim is from external sources (AQR fund, SG CTA Index), not from this project alone.
- **Implementation defenders**: "Did you try X parameter?" → redirect: the paper Section 7 discusses why parameter tuning is exactly the wrong response (data dredging).
- **Survivors**: "But trend-following works for me at Sharpe 1.5" → engage politely; ask if they've done multi-market walk-forward with deflated Sharpe.
- **Other negative-result stories**: encourage them to share publicly; the corpus is too small.

## Don't engage with

- Crypto cheerleaders ("just hold BTC bro")
- Demands to disclose actual capital amounts
- Comments asking for trading advice (deflect to disclaimer in LICENSE)
- Politics around AI/crypto markets

Aim for ~10-20 substantive replies, then let the thread drift naturally.
