#!/usr/bin/env python3
"""
TEST: P_j(S) Framework - VERSION 3
===================================

Интегрирует Filters в V2

Stage 3 (V3):
  - Rule-Based strategy
  - costs (from V1)
  - OpportunityScorer (from V2)
  - + Filters (NEW) - crisis, regime, correlation
  - Компаривает результаты V1 vs V2 vs V3

Вопрос: Как Filters улучшают risk-adjusted returns?
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
from backtesting.backtest_pjs_framework import (
    PjSBacktestEngine, BacktestConfig, RuleBasedStrategy
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    print("\n" + "="*80)
    print("P_j(S) FRAMEWORK TEST - VERSION 3 (V2 + Filters)")
    print("="*80 + "\n")

    # Setup
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "raw"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    # =========================================================================
    # STEP 1: Load data
    # =========================================================================
    print("[1/5] Loading BTC data...")
    btc_file = data_dir / "BTC_USDT_15m.parquet"
    if not btc_file.exists():
        logger.error(f"Data not found: {btc_file}")
        return

    ohlcv = pd.read_parquet(btc_file)
    logger.info(f"Loaded {len(ohlcv)} bars")

    # =========================================================================
    # STEP 2: Generate signals
    # =========================================================================
    print("[2/5] Generating Rule-Based signals (RSI < 30)...")
    strategy = RuleBasedStrategy(rsi_threshold=30, period=14)
    signals = strategy.generate_signals(ohlcv)
    signal_count = np.sum(signals)
    logger.info(f"Generated {signal_count} signals")

    ml_scores = signals.astype(float) * 0.7

    # =========================================================================
    # STEP 3: Run V1, V2, V3
    # =========================================================================
    results_all = {}

    configs = {
        'V1': BacktestConfig(
            initial_capital=100000, position_size_pct=0.95,
            take_profit=0.02, stop_loss=0.01, max_hold_bars=288,
            commission=0.001, slippage=0.0005, cooldown_bars=10,
            ml_enabled=False, filters_enabled=False,
            opportunity_enabled=False, cost_enabled=True,
            risk_penalty_enabled=False, ml_threshold=0.5
        ),
        'V2': BacktestConfig(
            initial_capital=100000, position_size_pct=0.95,
            take_profit=0.02, stop_loss=0.01, max_hold_bars=288,
            commission=0.001, slippage=0.0005, cooldown_bars=10,
            ml_enabled=False, filters_enabled=False,
            opportunity_enabled=True, cost_enabled=True,
            risk_penalty_enabled=False, ml_threshold=0.5
        ),
        'V3': BacktestConfig(
            initial_capital=100000, position_size_pct=0.95,
            take_profit=0.02, stop_loss=0.01, max_hold_bars=288,
            commission=0.001, slippage=0.0005, cooldown_bars=10,
            ml_enabled=False, filters_enabled=True,  # NEW
            opportunity_enabled=True, cost_enabled=True,
            risk_penalty_enabled=False, ml_threshold=0.5
        ),
    }

    for version, config in configs.items():
        print(f"[3/{len(configs)+1}] Running {version}...")
        backtest = PjSBacktestEngine(config)
        results_all[version] = backtest.run(
            ohlcv=ohlcv,
            raw_signals=signals,
            ml_scores=ml_scores
        )

    # =========================================================================
    # RESULTS COMPARISON
    # =========================================================================
    print("\n" + "="*80)
    print("RESULTS COMPARISON: V1 vs V2 vs V3")
    print("="*80)

    print(f"\n{'Metric':<25} {'V1':<15} {'V2':<15} {'V3':<15}")
    print("-" * 70)

    # Trades
    print(f"{'Trades':<25} {results_all['V1']['trades']:<15} {results_all['V2']['trades']:<15} {results_all['V3']['trades']:<15}")

    # Win Rate
    print(f"{'Win Rate':<25} {results_all['V1']['win_rate']:>6.1f}%     {results_all['V2']['win_rate']:>6.1f}%     {results_all['V3']['win_rate']:>6.1f}%")

    # Profit Factor
    print(f"{'Profit Factor':<25} {results_all['V1']['profit_factor']:>6.2f}       {results_all['V2']['profit_factor']:>6.2f}       {results_all['V3']['profit_factor']:>6.2f}")

    # P&L
    print(f"{'Final Capital':<25} ${results_all['V1']['final_capital']:>12,.0f} ${results_all['V2']['final_capital']:>12,.0f} ${results_all['V3']['final_capital']:>12,.0f}")
    print(f"{'Total P&L':<25} ${results_all['V1']['total_pnl']:>12,.0f} ${results_all['V2']['total_pnl']:>12,.0f} ${results_all['V3']['total_pnl']:>12,.0f}")

    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print("\n" + "="*80)
    print("ANALYSIS: Incremental Impact")
    print("="*80)

    # V2 impact
    print("\nOpportunityScorer (V1 → V2):")
    trade_change = results_all['V2']['trades'] - results_all['V1']['trades']
    print(f"  Trades: {trade_change:+d}")
    print(f"  Win Rate: {results_all['V2']['win_rate']-results_all['V1']['win_rate']:+.1f}pp")

    # V3 impact
    print("\nFilters (V2 → V3):")
    trade_change = results_all['V3']['trades'] - results_all['V2']['trades']
    print(f"  Trades: {trade_change:+d}")
    print(f"  Win Rate: {results_all['V3']['win_rate']-results_all['V2']['win_rate']:+.1f}pp")
    print(f"  Profit Factor: {results_all['V3']['profit_factor']-results_all['V2']['profit_factor']:+.2f}")

    # Overall
    print("\nOverall (V1 → V3):")
    trade_change = results_all['V3']['trades'] - results_all['V1']['trades']
    pct = trade_change / results_all['V1']['trades'] * 100 if results_all['V1']['trades'] > 0 else 0
    print(f"  Trades: {trade_change:+d} ({pct:+.1f}%)")
    print(f"  Win Rate: {results_all['V3']['win_rate']-results_all['V1']['win_rate']:+.1f}pp")
    print(f"  Final Capital: ${results_all['V3']['final_capital']-results_all['V1']['final_capital']:+,.0f}")

    # =========================================================================
    # NEXT STEPS
    # =========================================================================
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
✅ V1 DONE: Rule-Based with costs
✅ V2 DONE: + OpportunityScorer
✅ V3 DONE: + Filters
🟡 V4 TODO: + RiskPenalty
🟡 V5 TODO: Full integration + ML

Component Impact Summary:
  - OpportunityScorer: Filters by volume ratio (may not be effective with synthetic data)
  - Filters: Crisis detection and regime adjustment
  - Next: Add RiskPenalty for volatility-based risk management

Note: With synthetic data, some filters may not have visible impact.
      Real data with regime changes would show clearer differences.
    """)

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    report_file = reports_dir / "pjs_framework_v3_comparison.txt"
    with open(report_file, 'w') as f:
        for version in ['V1', 'V2', 'V3']:
            f.write(f"\n{version}:\n")
            f.write(f"  Trades: {results_all[version]['trades']}\n")
            f.write(f"  Win Rate: {results_all[version]['win_rate']:.1f}%\n")
            f.write(f"  Profit Factor: {results_all[version]['profit_factor']:.2f}\n")
            f.write(f"  Total P&L: ${results_all[version]['total_pnl']:,.0f}\n")

    logger.info(f"Results saved to {report_file}")
    print(f"\n✅ Test complete! Results saved to {report_file}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
