#!/usr/bin/env python3
"""
Backtest для ImprovedRuleBasedStrategy

Сравнивает Original (только RSI<30) vs Improved (RSI + EMA + Volume + ATR фильтры)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.pjs_components import ImprovedRuleBasedStrategy

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Constants
ASSET = "BTC"
TIMEFRAME = "15m"
FORWARD_BARS = 96  # 24 hours for 15m
PROFIT_THRESHOLD = 0.01  # 1%

def load_and_prepare_data(asset: str, timeframe: str) -> pd.DataFrame:
    """Загружает и подготавливает данные"""

    file_path = DATA_DIR / f"{asset}_{timeframe}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp')

    # Calculate indicators
    print("Calculating indicators...")

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # EMA
    df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['close'].ewm(span=21, adjust=False).mean()

    # Volume
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)

    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.DataFrame({
        'HL': high_low,
        'HC': high_close,
        'LC': low_close
    }).max(axis=1)
    df['ATR_14'] = true_range.rolling(window=14).mean()
    df['ATR_pct'] = df['ATR_14'] / df['close']

    # Drop NaN
    df = df.dropna()

    print(f"✅ Loaded {len(df):,} bars")

    return df

def backtest_strategy(df: pd.DataFrame,
                     strategy: ImprovedRuleBasedStrategy,
                     name: str) -> Dict:
    """
    Backtesting стратегии

    Returns:
        Dict с результатами (trades, WR, PF, etc.)
    """

    print(f"\n{'='*70}")
    print(f"Backtesting: {name}")
    print(f"{'='*70}")

    trades = []

    for i in range(len(df) - FORWARD_BARS):
        bar = df.iloc[i]

        # Check entry condition
        should_enter = strategy.should_enter(
            rsi=bar['RSI_14'],
            ema_9=bar['EMA_9'],
            ema_21=bar['EMA_21'],
            volume_ratio=bar['volume_ratio'],
            atr_pct=bar['ATR_pct']
        )

        if not should_enter:
            continue

        # Enter trade
        entry_price = bar['close']
        entry_time = bar.name

        # Check exit (profit target or time limit)
        max_price = df.iloc[i:i+FORWARD_BARS]['close'].max()
        profit = (max_price - entry_price) / entry_price

        if profit >= PROFIT_THRESHOLD:
            result = 'WIN'
        else:
            result = 'LOSS'

        trades.append({
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_price': max_price,
            'profit': profit,
            'result': result
        })

    # Calculate metrics
    total_trades = len(trades)

    if total_trades == 0:
        return {
            'name': name,
            'trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_profit': 0.0,
            'avg_profit': 0.0
        }

    wins = sum(1 for t in trades if t['result'] == 'WIN')
    losses = total_trades - wins
    win_rate = wins / total_trades

    gross_profit = sum(t['profit'] for t in trades if t['result'] == 'WIN')
    gross_loss = abs(sum(t['profit'] for t in trades if t['result'] == 'LOSS'))

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    total_profit = gross_profit - gross_loss
    avg_profit = total_profit / total_trades

    results = {
        'name': name,
        'trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'total_profit': total_profit,
        'avg_profit': avg_profit,
        'config': {
            'rsi_threshold': strategy.rsi_threshold,
            'use_ema_filter': strategy.use_ema_filter,
            'use_volume_filter': strategy.use_volume_filter,
            'use_atr_filter': strategy.use_atr_filter
        }
    }

    print(f"  Trades: {total_trades:,}")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Total Profit: {total_profit:.2%}")

    return results

def main():
    """Main function"""

    print("="*70)
    print("IMPROVED RULE-BASED STRATEGY BACKTEST")
    print("="*70)
    print(f"Asset: {ASSET}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Forward window: {FORWARD_BARS} bars (24 hours)")
    print(f"Profit threshold: {PROFIT_THRESHOLD:.1%}")
    print("="*70)

    # Load data
    df = load_and_prepare_data(ASSET, TIMEFRAME)

    # Test configurations
    configs = [
        # Original (только RSI<30)
        {
            'name': 'Original (RSI only)',
            'rsi_threshold': 30.0,
            'use_ema_filter': False,
            'use_volume_filter': False,
            'use_atr_filter': False
        },
        # Improved v1 (RSI + EMA)
        {
            'name': 'Improved v1 (RSI + EMA)',
            'rsi_threshold': 30.0,
            'use_ema_filter': True,
            'use_volume_filter': False,
            'use_atr_filter': False
        },
        # Improved v2 (RSI + EMA + Volume)
        {
            'name': 'Improved v2 (RSI + EMA + Volume)',
            'rsi_threshold': 30.0,
            'use_ema_filter': True,
            'use_volume_filter': True,
            'use_atr_filter': False
        },
        # Improved v3 (RSI + EMA + Volume + ATR)
        {
            'name': 'Improved v3 (RSI + EMA + Volume + ATR)',
            'rsi_threshold': 30.0,
            'use_ema_filter': True,
            'use_volume_filter': True,
            'use_atr_filter': True
        }
    ]

    # Run backtests
    all_results = []

    for config in configs:
        strategy = ImprovedRuleBasedStrategy(
            rsi_threshold=config['rsi_threshold'],
            use_ema_filter=config['use_ema_filter'],
            use_volume_filter=config['use_volume_filter'],
            use_atr_filter=config['use_atr_filter']
        )

        results = backtest_strategy(df, strategy, config['name'])
        all_results.append(results)

    # Summary comparison
    print(f"\n{'='*70}")
    print("COMPARATIVE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Strategy':<35} {'Trades':>10} {'WR':>8} {'PF':>8}")
    print(f"{'-'*70}")

    for r in all_results:
        print(f"{r['name']:<35} {r['trades']:>10,} {r['win_rate']:>7.1%} {r['profit_factor']:>7.2f}")

    # Calculate improvements
    baseline = all_results[0]
    print(f"\n{'='*70}")
    print("IMPROVEMENTS vs ORIGINAL")
    print(f"{'='*70}")

    for r in all_results[1:]:
        wr_improvement = (r['win_rate'] - baseline['win_rate']) * 100
        pf_improvement = ((r['profit_factor'] - baseline['profit_factor']) / baseline['profit_factor']) * 100
        trades_change = ((r['trades'] - baseline['trades']) / baseline['trades']) * 100

        print(f"\n{r['name']}:")
        print(f"  Win Rate: {wr_improvement:+.1f} percentage points")
        print(f"  Profit Factor: {pf_improvement:+.1f}%")
        print(f"  Trades: {trades_change:+.1f}%")

    # Save results
    output_file = REPORTS_DIR / f"improved_rule_based_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'asset': ASSET,
            'timeframe': TIMEFRAME,
            'forward_bars': FORWARD_BARS,
            'profit_threshold': PROFIT_THRESHOLD,
            'results': all_results
        }, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")

    # Best configuration
    best = max(all_results, key=lambda x: x['win_rate'])
    print(f"\n{'='*70}")
    print(f"🏆 BEST CONFIGURATION: {best['name']}")
    print(f"{'='*70}")
    print(f"  Win Rate: {best['win_rate']:.1%}")
    print(f"  Profit Factor: {best['profit_factor']:.2f}")
    print(f"  Total Trades: {best['trades']:,}")
    print(f"  Total Profit: {best['total_profit']:.2%}")

    # Goal check
    print(f"\n{'='*70}")
    print("GOAL CHECK")
    print(f"{'='*70}")

    goal_wr = 0.50  # 50% Win Rate goal

    if best['win_rate'] >= goal_wr:
        print(f"✅ SUCCESS: Win Rate {best['win_rate']:.1%} >= {goal_wr:.0%}")
    else:
        gap = (goal_wr - best['win_rate']) * 100
        print(f"⚠️  Almost there: Need {gap:.1f} more percentage points to reach {goal_wr:.0%}")

    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print("1. If goal reached → Day 2 (OpportunityScorer integration)")
    print("2. If not → Tune parameters (RSI threshold, volume_min, etc.)")
    print("3. Consider adding more filters (Stochastic, MACD, etc.)")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
