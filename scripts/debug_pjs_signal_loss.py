#!/usr/bin/env python3
"""
DEBUG: P_j(S) Framework - Find where signals disappear
=======================================================

Добавляет детальное логирование каждого шага:
- Сколько сигналов на вход
- Сколько прошли cooldown
- Сколько прошли P_j(S) фильтр
- Сколько вошли в позицию

Цель: Найти ГДЕ теряются сигналы
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

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def debug_backtest():
    """Backtest с детальным логированием"""

    print("\n" + "="*80)
    print("P_j(S) FRAMEWORK DEBUG - Find Signal Loss")
    print("="*80 + "\n")

    # Setup
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "raw"

    # Load data
    print("[1/5] Loading BTC data...")
    btc_file = data_dir / "BTC_USDT_15m.parquet"
    ohlcv = pd.read_parquet(btc_file)
    print(f"✓ Loaded {len(ohlcv)} bars\n")

    # Generate signals
    print("[2/5] Generating Rule-Based signals (RSI < 30)...")
    strategy = RuleBasedStrategy(rsi_threshold=30, period=14)
    signals = strategy.generate_signals(ohlcv)
    signal_count = np.sum(signals)
    print(f"✓ Generated {signal_count} signals\n")

    # Configure
    print("[3/5] Configuring backtest...")
    config = BacktestConfig(
        initial_capital=100000,
        position_size_pct=0.95,
        take_profit=0.02,
        stop_loss=0.01,
        max_hold_bars=288,
        commission=0.001,
        slippage=0.0005,
        cooldown_bars=10,
        ml_enabled=False,
        filters_enabled=False,
        opportunity_enabled=False,
        cost_enabled=True,
        risk_penalty_enabled=False,
        ml_threshold=0.5
    )
    print(f"✓ Config ready\n")

    # Create ml_scores
    print("[4/5] Creating ML scores...")
    ml_scores = signals.astype(float) * 0.7
    print(f"✓ ML scores: min={ml_scores.min():.3f}, max={ml_scores.max():.3f}, "
          f"non-zero={np.sum(ml_scores > 0)}\n")

    # Manual debug loop
    print("[5/5] Running backtest with debug info...\n")

    capital = config.initial_capital
    position = None
    last_exit_bar = -config.cooldown_bars
    trades = []
    signal_stats = {
        'total_signals': 0,
        'passed_cooldown': 0,
        'pjs_positive': 0,
        'entered_position': 0,
        'trades_closed': 0,
    }

    for i in range(len(ohlcv)):
        current_bar = ohlcv.iloc[i]

        # Check if signal
        if signals[i] > 0:
            signal_stats['total_signals'] += 1

            # Check cooldown
            if position is None:
                bars_since_exit = i - last_exit_bar
                if bars_since_exit >= config.cooldown_bars:
                    signal_stats['passed_cooldown'] += 1

                    # Calculate P_j(S) manually
                    ml_score = ml_scores[i]

                    # Check if ml_score is valid
                    if ml_score <= 0:
                        continue

                    # Calculate costs
                    costs = config.commission * 2 + config.slippage
                    if i >= 20:
                        recent_vol = ohlcv['volume'].iloc[max(0, i-20):i].mean()
                        if current_bar['volume'] < recent_vol * 0.5:
                            costs += 0.0005

                    # Calculate P_j(S)
                    pjs = (ml_score * 1.0 * 1.0  # ML * filters * opportunity
                           - costs
                           - 0.0)  # risk_penalty

                    if pjs > 0:
                        signal_stats['pjs_positive'] += 1

                        # Enter position
                        entry_price = current_bar['close']
                        position = {
                            'entry_bar': i,
                            'entry_price': entry_price,
                            'tp_level': entry_price * (1 + config.take_profit),
                            'sl_level': entry_price * (1 - config.stop_loss),
                        }
                        signal_stats['entered_position'] += 1

        # Check exit
        if position is not None:
            bars_held = i - position['entry_bar']
            exit_reason = None

            if current_bar['high'] >= position['tp_level']:
                exit_reason = 'tp'
            elif current_bar['low'] <= position['sl_level']:
                exit_reason = 'sl'
            elif bars_held >= config.max_hold_bars:
                exit_reason = 'time'

            if exit_reason:
                signal_stats['trades_closed'] += 1
                position = None
                last_exit_bar = i

    # Results
    print("\n" + "="*80)
    print("DEBUG RESULTS - Signal Flow")
    print("="*80 + "\n")

    print("Signal Flow Analysis:")
    print(f"  1. Total signals generated:      {signal_stats['total_signals']:,}")
    print(f"  2. Passed cooldown check:        {signal_stats['passed_cooldown']:,}")
    print(f"  3. P_j(S) > 0:                   {signal_stats['pjs_positive']:,}")
    print(f"  4. Entered position:             {signal_stats['entered_position']:,}")
    print(f"  5. Closed trades:                {signal_stats['trades_closed']:,}")

    print(f"\nLoss Points:")
    loss1 = signal_stats['total_signals'] - signal_stats['passed_cooldown']
    loss2 = signal_stats['passed_cooldown'] - signal_stats['pjs_positive']
    loss3 = signal_stats['pjs_positive'] - signal_stats['entered_position']

    print(f"  Lost in cooldown:     {loss1:,} ({loss1/max(1, signal_stats['total_signals'])*100:.1f}%)")
    print(f"  Lost in P_j(S) calc:  {loss2:,} ({loss2/max(1, signal_stats['passed_cooldown'])*100:.1f}%)")
    print(f"  Lost in entry logic:  {loss3:,} ({loss3/max(1, signal_stats['pjs_positive'])*100:.1f}%)")

    print(f"\n💡 DIAGNOSIS:")
    if signal_stats['entered_position'] == 0:
        if signal_stats['total_signals'] < 100:
            print("  ❌ PROBLEM: Very few signals generated (check RSI threshold)")
        elif signal_stats['passed_cooldown'] == 0:
            print("  ❌ PROBLEM: Cooldown is blocking ALL signals")
        elif signal_stats['pjs_positive'] == 0:
            print("  ❌ PROBLEM: All P_j(S) values are <= 0 (check costs/components)")
        elif signal_stats['entered_position'] == 0:
            print("  ❌ PROBLEM: Entry logic is broken")
    else:
        print(f"  ✓ {signal_stats['entered_position']} positions opened!")

    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    debug_backtest()
