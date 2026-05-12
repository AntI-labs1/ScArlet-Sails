# Paper Submission Checklist & Cover Notes

This document tracks submission state across all target venues and contains
ready-to-use cover letters / submission notes for each.

---

## Submission state

| Venue | Format | Status | URL |
|---|---|---|---|
| arXiv (q-fin.PM and q-fin.ST) | LaTeX | ⬜ Not submitted | https://arxiv.org/submit |
| SSRN | PDF | ⬜ Not submitted | https://www.ssrn.com/index.cfm/en/ |
| ICAIF 2026 workshop | LaTeX (ACM template) | ⬜ Submission window not open yet (typically Aug-Sep) | https://ai-finance.org/ |
| Medium (English) | Markdown | ⬜ Not submitted | https://medium.com/ |
| Habr / VC.ru (Russian) | Markdown | ⬜ Not submitted | https://habr.com/ |
| Personal blog / Substack | Markdown | ⬜ Optional | — |

---

## arXiv submission

### Categories

**Primary**: `q-fin.PM` (Portfolio Management)
**Secondary**: `q-fin.ST` (Statistical Finance), `q-fin.CP` (Computational Finance)

### Title

```
Rule-Based Technical Trading Strategies Fail to Generalize:
A Multi-Market Walk-Forward Audit (Crypto 2023-2026 and Metals 2000-2026)
```

### Abstract (paste into arXiv form, max ~1920 characters)

```
We present a multi-market walk-forward audit of retail-accessible rule-based
technical trading strategies across 14 cryptocurrency pairs (2023-2026, 15-minute
through daily bars) and 4 precious-metals futures (2000-2026, daily bars). Five
strategy classes (RSI mean-reversion, multi-indicator combined, rule-based with
opportunity scoring, 200-day SMA trend-following, dual momentum) are tested across
131 walk-forward windows with embargo, deflated Sharpe correction, and
probability-of-backtest-overfitting checks. Central finding is negative: across
the cryptocurrency universe, mean-reversion strategies achieve an average
walk-forward Sharpe of -0.70 with positive windows in only 35% of cases. On
metals, simple trend-following matches the literature baseline (Sharpe 0.4-0.6)
but does not exceed passive equal-weight buy-and-hold (Sharpe 0.62). After
deflation for selection bias and cost sensitivity, no tested strategy maintains
edge over passive risk-parity portfolios. We additionally report four critical
infrastructure bugs (look-ahead in feature normalization, hardcoded zero
drawdown in risk killswitch, inverted dispersion-to-position-size formula,
inconsistent Sharpe annualization) discovered during the audit that
materially inflated all prior in-sample results by approximately 6x. We release
a reproducible toolkit (data fetchers, walk-forward engine, deflated-Sharpe and
PBO implementations) and argue that honest negative results from retail
auditing serve an underappreciated function in a literature dominated by
survivorship-biased success reports.
```

### Comments field

```
24 pages, 6 figures, 3 tables. Code and data at github.com/StarDust1508/ScArlet-Sails (MIT-licensed). Companion software release including reproducible Kaggle notebooks. Findings are intentionally negative; no edge claimed.
```

### License (arXiv)

`CC BY 4.0` (allows commercial reuse with attribution; preferred for negative-result transparency papers).

### Required files for upload

1. `main.tex` (generated from `./paper/build.sh tex`)
2. `references.bib`
3. All `paper/figures/*.pdf` files
4. (Optional) `arxiv_supplementary.pdf` with code listings

### Compilation check

```bash
cd paper/build
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
# Check that no missing references warnings appear
```

If pdflatex fails on Cyrillic or special characters in the markdown, use xelatex via the build.sh `pdf` target.

---

## SSRN submission

SSRN allows simultaneous posting with arXiv. Workflow:

1. Generate PDF via `./paper/build.sh pdf`
2. Go to https://hq.ssrn.com/submissions/CreatePaperSubmission.cfm
3. Network: **Capital Markets eJournal** + **Risk Management eJournal**
4. Classification:
   - JEL: G11, G14, G17
   - Topic: Quantitative Methods; Asset Pricing
5. Title and abstract: same as arXiv
6. Keywords: same as CITATION.cff
7. Reference paper to arXiv version once arXiv ID is assigned

### SSRN-specific note

In the "Suggested Reviewers" field (optional): leave blank. We don't have institutional connections to suggest reviewers, and SSRN doesn't require them.

---

## ICAIF 2026 Workshop (peer-reviewed)

Submission window: typically **August-September 2026** for the November conference.

Watch: https://ai-finance.org/ for call-for-papers.

### Cover letter draft

```
Dear ICAIF 2026 Workshop Chairs,

I submit "Rule-Based Technical Trading Strategies Fail to Generalize" for
consideration in the workshop track.

The paper reports a multi-market negative-result audit conducted by an
independent retail researcher. It documents:

1. A rigorous walk-forward validation across 131 out-of-sample windows
   spanning two structurally different markets (cryptocurrency and
   precious metals).

2. An honest inventory of four infrastructure bugs in the author's own
   trading code that silently inflated all in-sample results by
   approximately six-fold, with detection methodology suitable for
   replication in other retail repositories.

3. An open-source reproducible toolkit (vectorbt-based engine, free-data
   fetchers, Deflated Sharpe Ratio and PBO implementations) released
   under MIT license.

The paper deliberately reports negative findings. We believe such findings,
when methodologically rigorous, contribute usefully to a literature in
which positive results are over-represented due to survivorship bias.

Reproducibility: full source code, data, and reproduction notebooks are at
https://github.com/StarDust1508/ScArlet-Sails. All numerical claims can be
verified in approximately 30 minutes of compute on free-tier Kaggle.

Submission category: empirical / educational.

[Author name]
[Affiliation]
[Email]
```

### ACM template

ICAIF uses ACM submission template. Use `acmart` LaTeX class with `[sigconf,anonymous,review]` options for double-blind review.

To convert main.md → ACM-compatible LaTeX:

```bash
pandoc paper/drafts/main.md \
    --bibliography=paper/drafts/references.bib \
    --natbib \
    --template=acm.template.tex \
    --standalone \
    -o paper/build/acm_main.tex
# Manual edit: change \documentclass to acmart with sigconf option
```

---

## Medium (English version)

Use `paper/drafts/medium.md` directly.

### Pre-publication checklist

- [ ] Create Medium account or use existing
- [ ] Choose 5 tags max — recommended: `algorithmic-trading`, `quantitative-finance`, `python`, `negative-results`, `open-source`
- [ ] Upload 2-3 key figures as images at high resolution
- [ ] Add a 1-paragraph author bio at bottom
- [ ] Use medium-friendly headers (## not #)

### Publication settings

- **Listing**: public
- **Distribution**: submit to "Better Programming" or "Towards Data Science" publication
- **Eligibility**: Medium Partner Program OK; honest content qualifies
- **Title**: identical to medium.md heading

### Cross-link

After publishing, add a redirect to medium.md:

```markdown
*This post was originally published at https://medium.com/[your-link]*
```

---

## Habr / VC.ru (Russian version)

Use `paper/drafts/medium_ru.md`.

Habr is preferred over VC.ru for technical content. Cross-post to VC.ru 48h later if Habr engagement is good.

### Habr-specific notes

- Markdown syntax differs slightly — Habr uses some custom shortcodes; review `social/habr.md` for posting tips
- Cyrillic in tags works fine
- Best time to post: weekday 10-11 AM MSK

---

## Twitter / X thread

Use `social/twitter_thread.md`. Post 10-tweet thread with figures attached on key tweets.

---

## Self-checks before any submission

- [ ] All `[arxiv link]`, `[medium link]`, `[author surname]` placeholders replaced
- [ ] Author affiliation specified (or "Independent" if applicable)
- [ ] Real ORCID iD if you have one (for arXiv)
- [ ] GitHub repo is public (not internal)
- [ ] All paths in code/notebooks use relative imports
- [ ] LICENSE file is at repo root
- [ ] CITATION.cff is at repo root
- [ ] Final PDF compiles without errors
- [ ] All 6 figures present and high-resolution
- [ ] References format is consistent (BibTeX → IEEE or APA)
- [ ] Spell-check on Markdown sources
- [ ] At least one trusted reader reviewed final draft

---

## Post-submission

### What to do in the first 48 hours after arXiv assignment

1. **Update CITATION.cff** with the arXiv ID
2. **Update README.md** with the arXiv link
3. **Update social media drafts** to replace `[arxiv link]` with real URL
4. **Post LinkedIn first** (most professional, lowest viral surface area — good to test reception)
5. **Wait 24-48h before Hacker News / Reddit** posting (controversial reception can damage if too many channels at once)

### Monitoring

- arXiv view/download stats: visible on paper landing page
- GitHub stars/forks: monitor for surge correlation with posts
- Citation tracking: Google Scholar will index within 1-3 months

### Response readiness

Prepare 1-2 paragraph stock responses for:
- "Did you try parameter X?"
- "Why didn't you include Y asset?"
- "Your costs are too conservative"
- "AQR's Sharpe 0.41 is wrong because [...]"

Most criticisms will be variants of these.

---

## If a peer-reviewed venue requests revisions

The most likely revision requests:
1. **More baselines** — add 60/40 and All-Weather backtest results explicitly (we have data for this in `paper/results/`)
2. **Longer crypto history** — BTC pre-2020 data is available via additional fetcher work
3. **Bootstrap confidence intervals** on Sharpe values
4. **Anonymization** if Habr/Medium versions are already public

All revision-likely changes are tracked in `paper/drafts/README.md` "Notes for future revisions".

---

## License for submitted versions

| Component | License |
|---|---|
| Code in repo | MIT |
| Paper (arXiv submission) | CC BY 4.0 |
| Paper (peer-reviewed acceptance) | per journal terms — typically CC BY or CC BY-NC |
| Medium / Habr versions | CC BY 4.0 |
| Twitter thread | Implicit fair-use of own work |
| Social drafts in repo (`social/`) | MIT (treated as code) |
