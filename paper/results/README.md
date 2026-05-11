# Paper Results

JSON outputs from `paper/notebooks/missing_backtests.ipynb`.

After running the Kaggle notebook these files will appear here:

- `crypto_trend_sma200.json` — 200d SMA trend on BTC/ETH/SOL × 4 TF (12 backtests)
- `metals_combined_multi_tf.json` — combined strategy on metals × 1h+1d (8 backtests)
- `deflated_sharpe.json` — Deflated Sharpe Ratio for all asset-strategy combos
- `pbo.json` — Probability of Backtest Overfitting (single score for whole experiment)
- `cost_sensitivity.json` — Sharpe at 1×/1.5×/2× cost assumptions (14 backtests)
- `paper_summary.json` — consolidated everything-in-one

These are inputs for paper tables/figures.
