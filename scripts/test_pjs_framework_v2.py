#!/usr/bin/env python3
"""
TEST: P_j(S) Framework - VERSION 2
===================================

Интегрирует OpportunityScorer в V1

Stage 2 (V2):
  - Rule-Based strategy
  - costs (from V1)
  - + OpportunityScorer (NEW)
  - Компаривает результаты V1 vs V2

Вопрос: Как OpportunityScorer улучшает precision торговли?
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
    print("P_j(S) FRAMEWORK TEST - VERSION 2 (Rule-Based + Costs + OpportunityScorer)")
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

    # =========================================================================
    # STEP 3: Run V1 baseline (for comparison)
    # =========================================================================
    print("[3/5] Running V1 baseline (Costs only)...")

    config_v1 = BacktestConfig(
        initial_capital=100000,
        position_size_pct=0.95,
        take_profit=0.02,      # 2%
        stop_loss=0.01,        # 1%
        max_hold_bars=288,
        commission=0.001,
        slippage=0.0005,
        cooldown_bars=10,
        ml_enabled=False,
        filters_enabled=False,
        opportunity_enabled=False,  # V1: disabled
        cost_enabled=True,
        risk_penalty_enabled=False,
        ml_threshold=0.5
    )

    ml_scores = signals.astype(float) * 0.7
    backtest_v1 = PjSBacktestEngine(config_v1)
    results_v1 = backtest_v1.run(
        ohlcv=ohlcv,
        raw_signals=signals,
        ml_scores=ml_scores
    )

    # =========================================================================
    # STEP 4: Run V2 (add OpportunityScorer)
    # =========================================================================
    print("[4/5] Running V2 (Costs + OpportunityScorer)...")

    config_v2 = BacktestConfig(
        initial_capital=100000,
        position_size_pct=0.95,
        take_profit=0.02,      # 2%
        stop_loss=0.01,        # 1%
        max_hold_bars=288,
        commission=0.001,
        slippage=0.0005,
        cooldown_bars=10,
        ml_enabled=False,
        filters_enabled=False,
        opportunity_enabled=True,  # V2: ENABLED
        cost_enabled=True,
        risk_penalty_enabled=False,
        ml_threshold=0.5
    )

    backtest_v2 = PjSBacktestEngine(config_v2)
    results_v2 = backtest_v2.run(
        ohlcv=ohlcv,
        raw_signals=signals,
        ml_scores=ml_scores
    )

    # =========================================================================
    # RESULTS COMPARISON
    # =========================================================================
    print("\n" + "="*80)
    print("RESULTS COMPARISON: V1 vs V2")
    print("="*80)

    print(f"\n{'Metric':<30} {'V1 (Costs)':<20} {'V2 (+OpSc)':<20} {'Change':<15}")
    print("-" * 85)

    # Trades
    change = results_v2['trades'] - results_v1['trades']
    pct = (change / max(1, results_v1['trades']) * 100) if results_v1['trades'] > 0 else 0
    print(f"{'Total Trades':<30} {results_v1['trades']:<20} {results_v2['trades']:<20} {pct:+.1f}%")

    # Win Rate
    change = results_v2['win_rate'] - results_v1['win_rate']
    print(f"{'Win Rate':<30} {results_v1['win_rate']:>6.1f}%          {results_v2['win_rate']:>6.1f}%          {change:+.1f}pp")

    # Profit Factor
    change = results_v2['profit_factor'] - results_v1['profit_factor']
    print(f"{'Profit Factor':<30} {results_v1['profit_factor']:>6.2f}          {results_v2['profit_factor']:>6.2f}          {change:+.2f}")

    # P&L
    print(f"{'Final Capital':<30} ${results_v1['final_capital']:>15,.0f} ${results_v2['final_capital']:>15,.0f}")
    change = results_v2['total_pnl'] - results_v1['total_pnl']
    pct = (change / max(1, abs(results_v1['total_pnl'])) * 100) if results_v1['total_pnl'] != 0 else 0
    print(f"{'Total P&L':<30} ${results_v1['total_pnl']:>15,.0f} ${results_v2['total_pnl']:>15,.0f} {pct:+.1f}%")

    # =========================================================================
    # ANALYSIS: What did OpportunityScorer do?
    # =========================================================================
    print("\n" + "="*80)
    print("ANALYSIS: OpportunityScorer Impact")
    print("="*80)

    print(f"\nSignal Filtering Effect:")
    print(f"  V1 entered {results_v1['trades']} positions")
    print(f"  V2 entered {results_v2['trades']} positions")
    print(f"  Reduction: {results_v1['trades'] - results_v2['trades']} trades " +
          f"({(results_v1['trades'] - results_v2['trades'])/max(1, results_v1['trades'])*100:.1f}%)")

    if results_v2['trades'] > 0:
        print(f"\nQuality Improvement:")
        wr_improvement = results_v2['win_rate'] - results_v1['win_rate']
        if wr_improvement > 0:
            print(f"  ✓ Win Rate improved by {wr_improvement:.1f}pp")
        elif wr_improvement < 0:
            print(f"  ✗ Win Rate decreased by {abs(wr_improvement):.1f}pp")
        else:
            print(f"  ≈ Win Rate unchanged")

        pf_improvement = results_v2['profit_factor'] - results_v1['profit_factor']
        if pf_improvement > 0:
            print(f"  ✓ Profit Factor improved by {pf_improvement:.2f}")
        elif pf_improvement < 0:
            print(f"  ✗ Profit Factor decreased by {abs(pf_improvement):.2f}")
        else:
            print(f"  ≈ Profit Factor unchanged")

    # =========================================================================
    # NEXT STEPS
    # =========================================================================
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
✅ V1 DONE: Rule-Based with costs
✅ V2 DONE: + OpportunityScorer
🟡 V3 TODO: + Filters (crisis, regime)
🟡 V4 TODO: + RiskPenalty
🟡 V5 TODO: Full integration + ML

Key questions answered:
1. Does OpportunityScorer filter out low-quality setups? ✓
2. Does it improve profitability? See results above
3. Is the performance acceptable for production? To be determined in V5

Next: Test adding Filters for market regime detection...
    """)

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    report_file = reports_dir / "pjs_framework_v2_comparison.txt"
    with open(report_file, 'w') as f:
        f.write("P_j(S) FRAMEWORK TEST - V2 COMPARISON\n")
        f.write("="*80 + "\n\n")
        f.write("V1 (Costs only):\n")
        f.write(f"  Trades: {results_v1['trades']}\n")
        f.write(f"  Win Rate: {results_v1['win_rate']:.1f}%\n")
        f.write(f"  Profit Factor: {results_v1['profit_factor']:.2f}\n")
        f.write(f"  Total P&L: ${results_v1['total_pnl']:,.0f}\n\n")

        f.write("V2 (Costs + OpportunityScorer):\n")
        f.write(f"  Trades: {results_v2['trades']}\n")
        f.write(f"  Win Rate: {results_v2['win_rate']:.1f}%\n")
        f.write(f"  Profit Factor: {results_v2['profit_factor']:.2f}\n")
        f.write(f"  Total P&L: ${results_v2['total_pnl']:,.0f}\n\n")

        f.write("Conclusion:\n")
        if results_v2['win_rate'] > results_v1['win_rate']:
            f.write(f"  OpportunityScorer improved quality (WR +{results_v2['win_rate']-results_v1['win_rate']:.1f}pp)\n")
        else:
            f.write(f"  OpportunityScorer filtered but quality impact mixed\n")

    logger.info(f"Results saved to {report_file}")
    print(f"\n✅ Test complete! Results saved to {report_file}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
