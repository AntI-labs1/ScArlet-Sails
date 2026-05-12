# Executive Summary — ScArlet-Sails Project

**One-page overview for reviewers, peers, and recruiters.**

---

## Project

An eight-month retail algorithmic trading research effort attempting to combine quantitative strategies (rule-based, ML, RL), an LLM-based "Council" for pattern interpretation, retrieval-augmented historical state matching, and human-in-the-loop decision-making into a unified trading system. Approximately 10,000 lines of Python; targeted cryptocurrency markets initially, pivoted briefly to precious metals during honest evaluation.

## Outcome

**Closed as actively-managed trading system after honest multi-market walk-forward audit yielded no generalizable edge.** The project's prior in-sample claim of Sharpe 2.91 was traced to four infrastructure bugs (look-ahead bias, disabled killswitch, inverted dispersion formula, inconsistent Sharpe annualization) that silently inflated all reported metrics. Realistic Sharpe estimates after bug fixes and walk-forward validation: −0.70 for mean-reversion strategies on cryptocurrency, +0.4 to +0.6 for trend-following on metals (matching literature but not exceeding passive buy-and-hold).

## Key empirical results

- **131 walk-forward windows** tested across 14 cryptocurrencies and 4 precious metals
- **Crypto mean-reversion**: 35% positive Sharpe windows (worse than coin-flip)
- **Metals trend-following**: 100% positive Sharpe, average +0.44 — confirming literature
- **Dual momentum (Antonacci)**: Sharpe 0.62, identical to passive equal-weight (0.63)
- **Cost sensitivity**: Sharpe invariant to position size (engine verified correct); negative-Sharpe strategies remain negative at any leverage
- **Single coin outlier (SOL)** showed Sharpe 1.20 in isolated backtest; non-generalizable per walk-forward

## Industry benchmark context

| Reference point | Live Sharpe |
|---|---:|
| AQR Style Premia Alternative Fund (institutional multi-factor) | 0.41 |
| SG CTA Index (largest professional trend-followers) | 0.56 |
| Passive 60/40 SPY+TLT (zero-work baseline) | 0.6–0.7 |
| **Our retail trend-following on metals** | **0.4–0.6** |

The retail result matches the institutional band, indicating the bottleneck is the strategy class (not infrastructure). Multi-factor extension within retail constraints cannot reasonably target above Sharpe 0.7.

## Contributions

1. **Negative-result multi-market audit** (131 walk-forward points): rare in published retail-trading literature.
2. **Documented infrastructure-bug inventory** with detection and remediation: four bugs that silently inflated in-sample backtest by approximately 6× the realistic estimate.
3. **Open-source toolkit** (MIT): vectorbt-based backtest engine, Binance Vision and yfinance fetchers, walk-forward framework, Deflated Sharpe Ratio and Probability of Backtest Overfitting implementations. Reproducible from clean machine via Kaggle notebooks.

## Deliverables

| Artifact | Description | Location |
|---|---|---|
| Academic paper draft | 5500 words, 19 references, methodology + results + discussion | `paper/drafts/main.md` |
| Russian-language Medium version | 1850 words, retail audience | `paper/drafts/medium_ru.md` |
| Statistical corrections library | Deflated Sharpe + PBO, pure numpy | `paper/notebooks/stats.py` |
| Backtest results | 6 JSON files with all empirical data | `paper/results/*.json` |
| Reproducibility notebook | One-click rerun on Kaggle | `paper/notebooks/missing_backtests.ipynb` |
| Figures generator | 6 publication-quality matplotlib figures | `paper/notebooks/figures.ipynb` |
| Build pipeline | pandoc PDF / LaTeX / HTML / Medium HTML | `paper/build.sh` |
| Project post-mortem | Lessons learned, honest accounting | `POST_MORTEM.md` |
| Passive portfolio framework | 3 portfolio options × 4 broker contexts | `passive/` |

## Target venues for paper

- arXiv preprint (immediate)
- SSRN parallel posting
- ICAIF 2026 workshop (peer-reviewed, August submission)
- Medium / Substack adapted version

## What this project demonstrates

For an academic/research reviewer:
- Honest negative-result methodology execution
- Rigorous walk-forward validation across multiple markets
- Application of recent academic corrections (Bailey/López de Prado 2014) to retail backtests
- Open and reproducible code, data, and results

For a recruiting context:
- Ability to recognize and stop a non-working project against personal sunk cost
- Statistical literacy (DSR, PBO, walk-forward, multi-market validation)
- Software engineering (vectorized backtesting, data pipelines, CI-ready test suite)
- Technical writing (academic paper + retail-adapted Medium version)
- Self-criticism and post-mortem capability

## What this project is NOT

- A production trading system
- A tradeable strategy with positive expected dollar alpha
- A demonstration that algorithmic trading works for retail
- A demonstration that algorithmic trading doesn't work for anyone — only that the specific strategy class tested, on the specific markets tested, within retail constraints, did not produce statistically significant edge.

## Status

Currently completing paper writeup (Track C); transitioning capital to passive 60/40 portfolio (Track D). Project repository is public, MIT-licensed, and serves as a research baseline and reusable infrastructure for future projects.

---

**Contact**: see GitHub repository — https://github.com/StarDust1508/ScArlet-Sails
**License**: MIT (code) / CC-BY (paper, when published)
