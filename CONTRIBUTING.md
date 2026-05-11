# Contributing to ScArlet-Sails

> **Status note**: This project was closed as an actively-developed trading
> system in May 2026 (see [`POST_MORTEM.md`](POST_MORTEM.md)). It is now
> maintained as a research baseline and reusable toolkit. The contribution
> guidelines below reflect this maintenance-only mode.

---

## What contributions are welcome

✅ **Bug fixes** in reusable infrastructure:
- `backtesting/vbt_engine.py`
- `core/data_loader.py`, `core/metrics_calculator.py`
- `scripts/fetch_binance_klines.py`, `scripts/fetch_metals.py`
- `paper/notebooks/stats.py` (Deflated Sharpe / PBO)
- `passive/rebalance.py`
- Tests (`tests/`)

✅ **Documentation improvements**:
- Typo fixes
- Clarifications in README, REPRODUCING, POST_MORTEM
- Better examples in code docstrings

✅ **Paper revisions** in response to peer-review feedback (after submission):
- Additional baselines requested by reviewers
- Methodology clarifications
- Notation improvements

✅ **Reproducibility improvements**:
- Better synthetic-data fallback in `tests/conftest.py`
- CI configuration improvements
- Docker / Codespaces support

✅ **Translations** of `paper/drafts/medium.md` into other languages.

## What contributions are NOT welcome

❌ **New trading strategies.** The project's empirical conclusion is that
   the strategy class tested does not generalize. Adding more variants of
   the same class will be declined. (You're welcome to fork.)

❌ **Performance optimizations to non-active modules.** Modules in
   `strategies/` (except `simple_strategies.py`), `council/`, `rag/` are
   research artefacts kept for historical reference only. Optimizing them
   is not value-add.

❌ **"Sharpe 2.91 was correct" arguments.** Section 6 of the paper
   documents specifically why the original claim was wrong. If you believe
   the bugs were not bugs, please open an issue explaining your reasoning
   with reproducible code rather than a PR.

❌ **Live trading integrations** (broker APIs, real-money executors).
   The project explicitly closed this direction; integration code is out of
   scope.

❌ **AI-agent / LLM extensions** of the existing Council/RAG scaffolding.
   These components were never connected to actual backtests and remain
   research-only; they will be removed in a future cleanup pass.

---

## How to contribute

### Reporting a bug

1. Open an issue at https://github.com/StarDust1508/ScArlet-Sails/issues
2. Include:
   - Affected file and function
   - Steps to reproduce (ideally Kaggle notebook URL or local commands)
   - Expected vs actual behavior
   - Python version, OS, relevant package versions
3. If you found a bug similar to the 4 documented in `POST_MORTEM.md` §6:
   add the tag `bug:silent-inflation` so it can be tracked under the
   same theme.

### Submitting a fix

1. Fork the repository
2. Create a branch: `git checkout -b fix/short-description`
3. Make your change
4. **Add or update tests** in `tests/`. PRs without tests are unlikely to merge for code changes.
5. Verify locally:
   ```bash
   pytest tests/ -q
   python -c "import ast; ast.parse(open('your_changed_file.py').read())"
   ```
6. For changes to `paper/notebooks/stats.py`: also verify the existing
   `tests/test_stats.py` still passes — that file is the regression test
   suite for the academic correctness claims.
7. Commit with a clear message:
   ```
   fix(data_loader): handle edge case where DVC pointers exist but parquets don't

   <body explaining what and why>

   Fixes #123
   ```
8. Push and open a PR against `main`
9. CI will run automatically (see `.github/workflows/test.yml`)
10. Maintainer review

### Style

- **Python**: PEP 8 with 100-char lines (relaxed from PEP 8's 79). No
  formatter enforced, but follow surrounding conventions.
- **Imports**: standard → third-party → local, separated by blank lines.
  Within each group, alphabetized.
- **Type hints**: encouraged on new public functions. Not required to
  add them to existing functions in a PR.
- **Tests**: pytest. Group related tests in classes. Use descriptive
  test names (`test_loader_accepts_underscore_form` not `test_1`).
- **Docstrings**: short summary line + (optional) longer explanation +
  Args / Returns / Raises sections for non-trivial functions.

### Commits

- Prefix with type: `fix`, `feat`, `docs`, `chore`, `test`, `refactor`
- Imperative mood: "fix bug" not "fixed bug"
- Reference issues with `Fixes #N` / `Closes #N`

### Review timeline

This project is maintained part-time. Typical response times:
- Issues: 1-7 days for first response
- PRs (small, well-tested): 1-2 weeks for review
- PRs (large or controversial): may take longer, especially if they
  conflict with the project's "closure" status

---

## Code of conduct

See [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

In short: be respectful, focus on the code/argument not the person,
and remember that the project's central finding is **negative** — please
don't take critiques of trading strategies personally.

---

## Recognition

Contributors are listed in commit history. Significant contributors will
be added to:
- `CITATION.cff` author list (if they contribute to academic-paper content)
- A `CONTRIBUTORS.md` file (created when the first non-author contribution lands)

---

## License

By contributing, you agree that your contributions will be licensed under
the same terms as the project:
- Code under MIT License (see [`LICENSE`](LICENSE))
- Paper content under CC BY 4.0 upon publication

---

## Questions

If something is unclear, open an issue with the `question` label. Vague
questions are fine. The maintainer would rather answer a "stupid" question
than have a contributor go in a wrong direction.
