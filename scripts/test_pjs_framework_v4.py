#!/usr/bin/env python3
"""
TEST: P_j(S) Framework - VERSION 4
===================================

Интегрирует RiskPenalty в V3

Stage 4 (V4):
  - Rule-Based strategy
  - costs (V1) + OpportunityScorer (V2) + Filters (V3)
  - + RiskPenalty (NEW)
  - Полная P_j(S) формула (без ML)

Вопрос: Как RiskPenalty влияет на risk-adjusted returns?
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
    print("P_j(S) FRAMEWORK TEST - VERSION 4 (V3 + RiskPenalty)")
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
    # STEP 3: Run all versions
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
            ml_enabled=False, filters_enabled=True,
            opportunity_enabled=True, cost_enabled=True,
            risk_penalty_enabled=False, ml_threshold=0.5
        ),
        'V4': BacktestConfig(
            initial_capital=100000, position_size_pct=0.95,
            take_profit=0.02, stop_loss=0.01, max_hold_bars=288,
            commission=0.001, slippage=0.0005, cooldown_bars=10,
            ml_enabled=False, filters_enabled=True,
            opportunity_enabled=True, cost_enabled=True,
            risk_penalty_enabled=True,  # NEW
            ml_threshold=0.5
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
    print("RESULTS COMPARISON: V1 vs V2 vs V3 vs V4")
    print("="*80)

    print(f"\n{'Metric':<25} {'V1':<15} {'V2':<15} {'V3':<15} {'V4':<15}")
    print("-" * 85)

    # Trades
    print(f"{'Trades':<25} {results_all['V1']['trades']:<15} {results_all['V2']['trades']:<15} {results_all['V3']['trades']:<15} {results_all['V4']['trades']:<15}")

    # Win Rate
    wr_str = "Win Rate"
    print(f"{wr_str:<25} {results_all['V1']['win_rate']:>6.1f}%     {results_all['V2']['win_rate']:>6.1f}%     {results_all['V3']['win_rate']:>6.1f}%     {results_all['V4']['win_rate']:>6.1f}%")

    # Profit Factor
    pf_str = "Profit Factor"
    print(f"{pf_str:<25} {results_all['V1']['profit_factor']:>6.2f}       {results_all['V2']['profit_factor']:>6.2f}       {results_all['V3']['profit_factor']:>6.2f}       {results_all['V4']['profit_factor']:>6.2f}")

    # P&L
    cap_str = "Final Capital"
    print(f"{cap_str:<25} ${results_all['V1']['final_capital']:>12,.0f} ${results_all['V2']['final_capital']:>12,.0f} ${results_all['V3']['final_capital']:>12,.0f} ${results_all['V4']['final_capital']:>12,.0f}")
    pnl_str = "Total P&L"
    print(f"{pnl_str:<25} ${results_all['V1']['total_pnl']:>12,.0f} ${results_all['V2']['total_pnl']:>12,.0f} ${results_all['V3']['total_pnl']:>12,.0f} ${results_all['V4']['total_pnl']:>12,.0f}")

    # =========================================================================
    # COMPONENT IMPACT ANALYSIS
    # =========================================================================
    print("\n" + "="*80)
    print("COMPONENT IMPACT ANALYSIS")
    print("="*80)

    print("\nV4 Impact (Risk Penalty added):")
    print(f"  Trades affected: {results_all['V3']['trades']} → {results_all['V4']['trades']} ({results_all['V4']['trades']-results_all['V3']['trades']:+d})")
    print(f"  Win Rate: {results_all['V3']['win_rate']:.1f}% → {results_all['V4']['win_rate']:.1f}% ({results_all['V4']['win_rate']-results_all['V3']['win_rate']:+.1f}pp)")
    print(f"  Profit Factor: {results_all['V3']['profit_factor']:.2f} → {results_all['V4']['profit_factor']:.2f} ({results_all['V4']['profit_factor']-results_all['V3']['profit_factor']:+.2f})")

    if results_all['V4']['trades'] > 0 and results_all['V4']['trades'] < results_all['V3']['trades']:
        print(f"\n  ✓ Risk Penalty filtered {results_all['V3']['trades'] - results_all['V4']['trades']} high-risk trades")
        print(f"    This shows the penalty is working to avoid volatile conditions")

    # =========================================================================
    # CUMULATIVE IMPACT
    # =========================================================================
    print("\nCumulative Impact (V1 → V4):")
    trade_delta = results_all['V4']['trades'] - results_all['V1']['trades']
    pct = trade_delta / results_all['V1']['trades'] * 100 if results_all['V1']['trades'] > 0 else 0
    print(f"  Trades: {results_all['V1']['trades']} → {results_all['V4']['trades']} ({trade_delta:+d}, {pct:+.1f}%)")
    print(f"  Win Rate: {results_all['V1']['win_rate']:.1f}% → {results_all['V4']['win_rate']:.1f}% ({results_all['V4']['win_rate']-results_all['V1']['win_rate']:+.1f}pp)")
    print(f"  Final Capital: ${results_all['V1']['final_capital']:,.0f} → ${results_all['V4']['final_capital']:,.0f} ({results_all['V4']['final_capital']-results_all['V1']['final_capital']:+,.0f})")

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
✅ V4 DONE: + RiskPenalty
🟡 V5 TODO: Full integration + ML

P_j(S) Formula Complete (V4):
  P_j(S) = ML(1.0)                    [not yet enabled]
         × ∏I_k (filters)             [enabled]
         × opportunity(S)             [enabled]
         - costs(S)                   [enabled]
         - risk_penalty(S)            [enabled]

Next: Enable ML scoring and test with XGBoost models.
      This will be V5 - the full production implementation.
    """)

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    report_file = reports_dir / "pjs_framework_v4_comparison.txt"
    with open(report_file, 'w') as f:
        f.write("P_j(S) FRAMEWORK TEST - V4 COMPARISON\n")
        f.write("="*80 + "\n\n")

        for version in ['V1', 'V2', 'V3', 'V4']:
            f.write(f"{version}:\n")
            f.write(f"  Trades: {results_all[version]['trades']}\n")
            f.write(f"  Win Rate: {results_all[version]['win_rate']:.1f}%\n")
            f.write(f"  Profit Factor: {results_all[version]['profit_factor']:.2f}\n")
            f.write(f"  Total P&L: ${results_all[version]['total_pnl']:,.0f}\n\n")

    logger.info(f"Results saved to {report_file}")
    print(f"\n✅ Test complete! Results saved to {report_file}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
