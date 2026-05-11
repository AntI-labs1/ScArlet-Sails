# Paper Drafts

## Status

- `main.md` — primary paper draft (~5500 words, 382 lines)
  - Sections **complete**: Abstract, Introduction, Related Work, Methodology, Dataset,
    Infrastructure Bugs, Discussion, Conclusion, Acknowledgments, Code Availability
  - Sections **with TODO placeholders**: Results 5.1–5.8 (awaiting JSON output from
    `paper/notebooks/missing_backtests.ipynb`), Appendices B/C/D
- `references.bib` — 19 academic references with arXiv/SSRN/journal links
- `figures/` — placeholder for matplotlib output (currently empty)

## Workflow

1. **Run** `paper/notebooks/missing_backtests.ipynb` on Kaggle → outputs JSON to `paper/results/`
2. **Integrate** JSON into Section 5 of `main.md` (manual table insertion + commentary)
3. **Generate figures** (suggested):
   - Walk-forward Sharpe boxplots per (asset_class, strategy)
   - Cost sensitivity sweep curves
   - Deflated-vs-raw Sharpe scatter
   - PBO interpretation diagram
4. **Convert to PDF**: pandoc `main.md` → `main.pdf` with bibliography processing
   ```bash
   pandoc paper/drafts/main.md \
     --bibliography=paper/drafts/references.bib \
     --citeproc \
     --pdf-engine=xelatex \
     -o paper/drafts/main.pdf
   ```
5. **arXiv submission**: requires LaTeX source. Convert markdown to LaTeX via pandoc,
   tidy up math, submit through https://arxiv.org/submit.

## Estimated time to camera-ready

- Integrate Kaggle results into Section 5: 4 hours
- Generate 8 figures: 4 hours
- Polish prose, add cross-references, fix bibliography format: 4 hours
- Peer-review by 1-2 quant readers: 1 week
- Final revisions: 4 hours
- Pandoc→LaTeX conversion + arXiv submission: 2 hours

**Total active work**: ~20 hours over 2-3 weeks.

## Target venues

| Venue | Format | Lead time | Acceptance prob |
|---|---|---|---|
| arXiv preprint | LaTeX | Instant | 100% |
| SSRN | PDF | Instant | 100% |
| ICAIF 2026 workshop | LaTeX (ACM template) | Submit ~Aug 2026 | ~50% |
| Medium / Substack | Markdown | Same-day | n/a |
| Journal of Investment Strategies | LaTeX | 3-9 months | ~20% |

## Notes for future revisions

- If submitting to a peer-reviewed venue, expect requests for:
  - **More baselines**: 60/40, all-weather portfolio backtests explicitly included
  - **Longer crypto history**: BTC pre-2020 (would require additional data fetcher work)
  - **Robustness checks**: alternative timeframes, alternative strategy parameters,
    bootstrap confidence intervals on Sharpe
  - **Connection to live data**: a live-tracking element showing strategies trading
    out-of-sample post-paper publication

- For Medium/Substack version, simplify:
  - Remove formal hypothesis-testing math
  - Focus on the **bugs section** (most visceral for retail audience)
  - Highlight the practical conclusion: "10 weeks of work ≈ 5 minutes of 60/40 rebalance"
