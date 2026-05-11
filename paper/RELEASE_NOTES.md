# GitHub Release Notes — v1.0-research-baseline

To be used when creating the GitHub release at:
https://github.com/StarDust1508/ScArlet-Sails/releases/new

## Tag name

```
v1.0-research-baseline
```

## Release title

```
v1.0 — Research Baseline (Project Closure & Negative-Result Paper)
```

## Description (paste into release notes field)

```markdown
## What this release is

This is the **closure release** of the ScArlet-Sails 8-month retail algorithmic
trading research project. The repository transitions from an active-development
trading-system codebase to a **research baseline** and **reusable toolkit**.

## What's included

- **Full empirical audit**: 131 walk-forward windows across 14 cryptocurrencies
  and 4 precious metals. Average Sharpe -0.70 for mean-reversion on crypto,
  +0.44 for trend-following on metals. No generalizable edge after honest
  Deflated Sharpe Ratio correction.

- **Academic paper draft** (5500 words, 19 references): see
  [`paper/drafts/main.md`](paper/drafts/main.md). Submitted to arXiv as
  [arXiv:XXXX.XXXXX] *(link to be added after submission)*.

- **Documented infrastructure bugs**: 4 critical bugs (look-ahead bias,
  disabled killswitch, inverted dispersion formula, inconsistent Sharpe
  annualization) that silently inflated all prior in-sample results by
  approximately 6x. Each bug includes detection method and fix.
  See [`POST_MORTEM.md`](POST_MORTEM.md) §"Critical infrastructure bugs".

- **Reusable toolkit** (MIT):
  - `backtesting/vbt_engine.py` — unified vectorbt-based backtest engine
  - `scripts/fetch_binance_klines.py` — free crypto data fetcher
  - `scripts/fetch_metals.py` — free metals data fetcher
  - `paper/notebooks/stats.py` — Deflated Sharpe Ratio and PBO (open-source numpy implementations)
  - `passive/rebalance.py` — quarterly rebalance CLI for passive portfolios

- **Reproducible artifacts**:
  - `paper/notebooks/missing_backtests.ipynb` — Kaggle-ready notebook reproducing all results in ~30 minutes
  - `paper/notebooks/figures.ipynb` — generates 6 publication-quality figures
  - `paper/results/*.json` — exact numerical values cited in the paper
  - `paper/build.sh` — pandoc pipeline (Markdown → PDF / LaTeX / HTML / Medium-HTML)

## What this release is NOT

- Not a production trading system
- Not a working strategy with positive expected dollar alpha
- Not a "look at all my features" repository

The project explicitly closes the active-trading thread. See
[`POST_MORTEM.md`](POST_MORTEM.md) for the full reasoning.

## How to use

### Read the findings
- Start: [`POST_MORTEM.md`](POST_MORTEM.md)
- Academic version: [`paper/drafts/main.md`](paper/drafts/main.md)
- Casual blog version: [`paper/drafts/medium.md`](paper/drafts/medium.md) (EN) /
  [`paper/drafts/medium_ru.md`](paper/drafts/medium_ru.md) (RU)
- One-pager: [`paper/drafts/executive_summary.md`](paper/drafts/executive_summary.md)

### Reproduce the results
- See [`REPRODUCING.md`](REPRODUCING.md) — three paths (Kaggle, local, read-only)

### Re-use the code
- Pick from the catalog in [`README.md`](README.md) §"What you can re-use"

## Cite

```bibtex
@misc{scarlet_sails_2026,
  author       = {Bubble3 et al.},
  title        = {{ScArlet-Sails}: A Multi-Market Walk-Forward Audit of Retail
                 Technical Trading Strategies},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/StarDust1508/ScArlet-Sails}},
  note         = {Tag v1.0-research-baseline}
}
```

Or use the GitHub "Cite this repository" button (powered by [`CITATION.cff`](CITATION.cff)).

## Changelog

Full changelog in [`CHANGELOG.md`](CHANGELOG.md). Key items:

- Added: POST_MORTEM, full paper draft, reproducible Kaggle notebooks, social media drafts
- Fixed: 5 critical infrastructure bugs documented
- Removed: stale "Sharpe 2.91" claim from old README; archive of deprecated code

## Acknowledgments

Extensive collaboration with Anthropic Claude (Opus 4.7) throughout the May 2026
audit. See `POST_MORTEM.md` §Acknowledgments for reflection on the
AI-construction + AI-audit pattern.

## License

- Code: [MIT License](LICENSE)
- Paper drafts: CC BY 4.0 upon arXiv publication
- Disclaimer on financial content: see LICENSE

---

**This is a closure release.** No further actively-managed trading development
is planned. The repository will continue receive maintenance updates only:
documentation improvements, bug fixes in the reusable toolkit, paper revisions
in response to peer review. Pull requests on these grounds are welcome; pull
requests adding new strategies or features will likely be declined.
```

## How to create this release after pushing the tag

```bash
# On your mac, after final push:
cd "/Users/bubble3/Desktop/Проекты_Х/Scarlet Sails /ScArlet-Sails/.claude/worktrees/quizzical-raman-434cfb"

# Create annotated tag locally
git tag -a v1.0-research-baseline -m "Project closure release; see paper/RELEASE_NOTES.md"

# Push tag to GitHub
git push origin v1.0-research-baseline

# Then go to GitHub → Releases → Draft a new release
# Choose the tag v1.0-research-baseline
# Paste the contents of paper/RELEASE_NOTES.md "Description" section above
# Mark as "Latest release"
# Publish
```

## Why a tagged release matters

- GitHub's "Cite this repository" widget pulls from CITATION.cff at the latest release
- arXiv accepts GitHub release URLs as code references
- ICAIF and similar venues require persistent versioned code references
- Future readers can pin to this exact state regardless of subsequent commits
- Zenodo (if you connect GitHub→Zenodo) issues a DOI per release for academic citation
