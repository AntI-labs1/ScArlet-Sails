# Security Policy

## Supported versions

This is a research / educational project, not a production trading system.
Security updates are applied on a best-effort basis to the `main` branch only.

| Branch / tag | Status |
|---|---|
| `main` | Active maintenance |
| `v1.0-research-baseline` | Frozen release; security fixes via new patch release if needed |
| Feature / experimental branches | Not maintained |

## What this project IS NOT

Before reporting a "security issue", please note: **this is not a financial
service**. The project does not:

- Hold user funds
- Execute real trades
- Authenticate users
- Process payments
- Store personally-identifiable information

It is a **research codebase** that runs locally / on Kaggle and produces
backtest results from public market data.

Therefore, classes of issues that would normally be security-critical in
financial software are **not in scope** here:

- ❌ Backtest results are wrong (this is a research-quality issue, not a security issue)
- ❌ Trading strategy could lose money (the entire paper documents that
  the strategies don't have edge; this is the central finding, not a bug)
- ❌ Data fetcher could be slow under load (research code, no SLA)

## What IS in scope

✅ **Code execution vulnerabilities** in:
- `scripts/fetch_binance_klines.py`, `scripts/fetch_metals.py` — these
  download untrusted data from external sources and parse it
- `passive/rebalance.py` — reads user-supplied portfolio configurations
- `core/data_loader.py` — reads parquet files which could theoretically
  contain malicious content (pyarrow CVEs)

✅ **Dependency vulnerabilities**:
- Any CVE in `requirements.txt` packages
- Particularly: `vectorbt`, `yfinance`, `pandas`, `numpy`, `pyarrow`,
  `requests`, `joblib`, `xgboost`

✅ **Supply chain issues**:
- Compromised package versions
- Typo-squatted dependencies
- Pinning recommendations

✅ **Information disclosure**:
- If any code accidentally leaks API keys, file paths, system info
  (we don't think it does, but report if you find such)

## Reporting a vulnerability

For security issues that fall within scope above:

**Do not open a public issue.** Instead:

1. Send a private report via GitHub Security Advisories:
   https://github.com/StarDust1508/ScArlet-Sails/security/advisories/new

2. Include:
   - Affected file(s) and line(s)
   - Description of the vulnerability
   - Steps to reproduce
   - Suggested fix if you have one
   - Your name / handle if you want credit

3. Response timeline:
   - Acknowledgment within 7 days
   - Assessment and fix plan within 14 days
   - Fix release within 30 days (depending on severity)

4. Coordinated disclosure: please don't publish details until a fix is
   released or 90 days have passed, whichever is sooner.

## Out-of-scope reports

The following types of reports will be politely declined:

- Reports that the strategies don't make money (read `POST_MORTEM.md`)
- Reports that backtests are "wrong" because they don't match a different
  data source (we use specific data with specific caveats, documented in
  the paper)
- Reports based on running the code on production accounts and losing
  money (please don't do this; the LICENSE has an explicit financial
  disclaimer)
- Reports that AI-generated code is inherently insecure (we acknowledge
  AI was used; please point to specific issues, not the category)

## Acknowledgments

If you report a valid vulnerability and we ship a fix, you will be
credited in:
- The release notes for the fix release
- A `SECURITY_ACKNOWLEDGMENTS.md` file (created when first credit is due)

## Public-key contact

For sensitive coordination, GitHub Security Advisories is preferred over
email. If you need PGP, request the public key via Advisory.
