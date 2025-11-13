#!/usr/bin/env python3
"""
REALISTIC Backtest для ImprovedRuleBasedStrategy

Отличия от простого backtest:
1. Реалистичный exit (TP/SL/Time, что произойдёт первым)
2. Честные издержки (0.3% round-trip)
3. Менее строгие фильтры
4. Детальная статистика (Sharpe, MDD, distribution)
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

# Realistic parameters
TAKE_PROFIT = 0.01    # 1.0% TP
STOP_LOSS = -0.005    # -0.5% SL
ENTRY_COST = 0.0015   # 0.15% (maker fee + slippage)
EXIT_COST = 0.0015    # 0.15%
TOTAL_COST = ENTRY_COST + EXIT_COST  # 0.30% round-trip

def load_and_prepare_data(asset: str, timeframe: str) -> pd.DataFrame:
    """Загружает и подготавливает данные"""

    file_path = DATA_DIR / f"{asset}_USDT_{timeframe}.parquet"

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    print(f"Loading {file_path}...")
    df = pd.read_parquet(file_path)

    # Ensure timestamp is datetime and set as index
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    elif df.index.name != 'timestamp':
        df.index = pd.to_datetime(df.index)

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

def backtest_realistic(df: pd.DataFrame,
                      strategy: ImprovedRuleBasedStrategy,
                      name: str) -> Dict:
    """
    REALISTIC Backtesting с TP/SL/Time exit и издержками

    Returns:
        Dict с результатами и детальной статистикой
    """

    print(f"\n{'='*70}")
    print(f"Backtesting: {name}")
    print(f"{'='*70}")

    trades = []
    equity_curve = [1.0]  # Start with 1.0 capital
    current_equity = 1.0

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

        # Calculate TP/SL levels (AFTER costs)
        # TP: need price to rise by (TP + costs) to get net TP profit
        tp_level = entry_price * (1 + TAKE_PROFIT + TOTAL_COST)
        # SL: if price drops by (SL + costs), we lose (SL + costs)
        sl_level = entry_price * (1 + STOP_LOSS - TOTAL_COST)

        # Simulate trade forward
        exit_bar = None
        exit_price = None
        exit_reason = None

        for j in range(i + 1, min(i + FORWARD_BARS + 1, len(df))):
            future_bar = df.iloc[j]

            # Check TP (high reached TP level)
            if future_bar['high'] >= tp_level:
                exit_bar = j - i
                exit_price = tp_level
                exit_reason = 'TP'
                break

            # Check SL (low reached SL level)
            if future_bar['low'] <= sl_level:
                exit_bar = j - i
                exit_price = sl_level
                exit_reason = 'SL'
                break

        # If no TP/SL hit, exit at time limit
        if exit_reason is None:
            exit_bar = FORWARD_BARS
            exit_price = df.iloc[i + FORWARD_BARS]['close']
            exit_reason = 'TIME'

        # Calculate P&L (including costs)
        gross_return = (exit_price - entry_price) / entry_price
        net_return = gross_return - TOTAL_COST

        # Update equity
        trade_pnl = net_return  # Assuming 1.0 position size
        current_equity *= (1 + trade_pnl)
        equity_curve.append(current_equity)

        # Classify result
        if net_return >= TAKE_PROFIT:
            result = 'WIN'
        elif net_return <= STOP_LOSS:
            result = 'LOSS'
        else:
            # Closed at time limit
            result = 'WIN' if net_return > 0 else 'LOSS'

        trades.append({
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_bar': exit_bar,
            'exit_reason': exit_reason,
            'gross_return': gross_return,
            'net_return': net_return,
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
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }

    wins = sum(1 for t in trades if t['result'] == 'WIN')
    losses = total_trades - wins
    win_rate = wins / total_trades

    # P&L statistics
    winning_trades = [t['net_return'] for t in trades if t['result'] == 'WIN']
    losing_trades = [t['net_return'] for t in trades if t['result'] == 'LOSS']

    gross_profit = sum(winning_trades) if winning_trades else 0
    gross_loss = abs(sum(losing_trades)) if losing_trades else 0

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    total_return = current_equity - 1.0  # Net profit/loss
    avg_return = np.mean([t['net_return'] for t in trades])
    std_return = np.std([t['net_return'] for t in trades])

    # Sharpe Ratio (annualized, assuming ~35,000 15m bars per year)
    # Sharpe = (mean_return / std_return) * sqrt(number_of_periods_per_year)
    periods_per_year = 35000
    sharpe_ratio = (avg_return / std_return) * np.sqrt(periods_per_year) if std_return > 0 else 0

    # Max Drawdown
    equity_array = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - running_max) / running_max
    max_drawdown = abs(drawdown.min())

    # Exit reason distribution
    exit_reasons = {}
    for t in trades:
        reason = t['exit_reason']
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    results = {
        'name': name,
        'trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'total_return': total_return,
        'final_equity': current_equity,
        'avg_return': avg_return,
        'std_return': std_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'exit_reasons': exit_reasons,
        'avg_win': np.mean(winning_trades) if winning_trades else 0,
        'avg_loss': np.mean(losing_trades) if losing_trades else 0,
        'config': {
            'rsi_threshold': strategy.rsi_threshold,
            'use_ema_filter': strategy.use_ema_filter,
            'use_volume_filter': strategy.use_volume_filter,
            'use_atr_filter': strategy.use_atr_filter
        }
    }

    # Print summary
    print(f"  Trades: {total_trades:,}")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Total Return: {total_return:.2%}")
    print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {max_drawdown:.2%}")
    print(f"  Exit reasons: TP={exit_reasons.get('TP', 0)}, SL={exit_reasons.get('SL', 0)}, TIME={exit_reasons.get('TIME', 0)}")

    return results

def main():
    """Main function"""

    print("="*70)
    print("REALISTIC IMPROVED RULE-BASED STRATEGY BACKTEST")
    print("="*70)
    print(f"Asset: {ASSET}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Forward window: {FORWARD_BARS} bars (24 hours)")
    print(f"Take Profit: {TAKE_PROFIT:.1%}")
    print(f"Stop Loss: {STOP_LOSS:.1%}")
    print(f"Trading costs: {TOTAL_COST:.2%} round-trip")
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
        # Improved v1 (RSI + EMA - less strict)
        {
            'name': 'Improved v1 (RSI + EMA trend)',
            'rsi_threshold': 30.0,
            'use_ema_filter': True,
            'use_volume_filter': False,
            'use_atr_filter': False,
            'ema_min_ratio': 0.99  # EMA_9 >= EMA_21 * 0.99 (less strict)
        },
        # Improved v2 (RSI + Volume moderate)
        {
            'name': 'Improved v2 (RSI + Volume)',
            'rsi_threshold': 30.0,
            'use_ema_filter': False,
            'use_volume_filter': True,
            'use_atr_filter': False,
            'volume_min': 1.2  # Less strict than 1.5
        },
        # Improved v3 (RSI + EMA + Volume)
        {
            'name': 'Improved v3 (RSI + EMA + Volume)',
            'rsi_threshold': 30.0,
            'use_ema_filter': True,
            'use_volume_filter': True,
            'use_atr_filter': False,
            'ema_min_ratio': 0.99,
            'volume_min': 1.2
        },
        # Improved v4 (RSI + EMA + Volume + ATR)
        {
            'name': 'Improved v4 (RSI + EMA + Volume + ATR)',
            'rsi_threshold': 30.0,
            'use_ema_filter': True,
            'use_volume_filter': True,
            'use_atr_filter': True,
            'ema_min_ratio': 0.99,
            'volume_min': 1.2,
            'atr_max': 0.06  # Less strict than 0.05
        }
    ]

    # Run backtests
    all_results = []

    for config in configs:
        # Extract custom params
        ema_min_ratio = config.pop('ema_min_ratio', None)
        volume_min = config.pop('volume_min', None)
        atr_max = config.pop('atr_max', None)

        # Create strategy with custom params
        strategy = ImprovedRuleBasedStrategy(
            rsi_threshold=config['rsi_threshold'],
            use_ema_filter=config['use_ema_filter'],
            use_volume_filter=config['use_volume_filter'],
            use_atr_filter=config['use_atr_filter']
        )

        # Override params if provided
        if ema_min_ratio is not None:
            strategy.ema_min_ratio = ema_min_ratio
        if volume_min is not None:
            strategy.volume_min = volume_min
        if atr_max is not None:
            strategy.atr_max = atr_max

        results = backtest_realistic(df, strategy, config['name'])
        all_results.append(results)

    # Summary comparison
    print(f"\n{'='*70}")
    print("COMPARATIVE SUMMARY (REALISTIC)")
    print(f"{'='*70}")
    print(f"{'Strategy':<40} {'Trades':>8} {'WR':>7} {'PF':>6} {'Return':>8} {'Sharpe':>7} {'MDD':>7}")
    print(f"{'-'*70}")

    for r in all_results:
        print(f"{r['name']:<40} {r['trades']:>8,} {r['win_rate']:>6.1%} {r['profit_factor']:>6.2f} {r['total_return']:>7.1%} {r['sharpe_ratio']:>7.2f} {r['max_drawdown']:>6.1%}")

    # Best configuration by Sharpe Ratio
    best_sharpe = max(all_results, key=lambda x: x['sharpe_ratio'])
    best_wr = max(all_results, key=lambda x: x['win_rate'])
    best_pf = max(all_results, key=lambda x: x['profit_factor'] if x['profit_factor'] != float('inf') else 0)

    print(f"\n{'='*70}")
    print("BEST CONFIGURATIONS")
    print(f"{'='*70}")
    print(f"🏆 Best Sharpe Ratio: {best_sharpe['name']} ({best_sharpe['sharpe_ratio']:.2f})")
    print(f"🎯 Best Win Rate: {best_wr['name']} ({best_wr['win_rate']:.1%})")
    print(f"💰 Best Profit Factor: {best_pf['name']} ({best_pf['profit_factor']:.2f})")

    # Save results
    output_file = REPORTS_DIR / f"realistic_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Convert to JSON-serializable format
    json_results = []
    for r in all_results:
        r_copy = r.copy()
        # Convert inf to string
        if r_copy['profit_factor'] == float('inf'):
            r_copy['profit_factor'] = 'inf'
        json_results.append(r_copy)

    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'asset': ASSET,
            'timeframe': TIMEFRAME,
            'parameters': {
                'take_profit': TAKE_PROFIT,
                'stop_loss': STOP_LOSS,
                'entry_cost': ENTRY_COST,
                'exit_cost': EXIT_COST,
                'total_cost': TOTAL_COST
            },
            'results': json_results
        }, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")

    # Recommendation
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")

    if best_sharpe['win_rate'] >= 0.50 and best_sharpe['sharpe_ratio'] >= 1.0:
        print(f"✅ EXCELLENT: {best_sharpe['name']}")
        print(f"   - Win Rate: {best_sharpe['win_rate']:.1%} (>= 50%)")
        print(f"   - Sharpe Ratio: {best_sharpe['sharpe_ratio']:.2f} (>= 1.0)")
        print(f"   - Max Drawdown: {best_sharpe['max_drawdown']:.1%}")
        print(f"\n   → Ready for Day 2 (OpportunityScorer integration)")
    else:
        print(f"⚠️  NEEDS IMPROVEMENT")
        print(f"   Current best: {best_sharpe['name']}")
        print(f"   - Win Rate: {best_sharpe['win_rate']:.1%}")
        print(f"   - Sharpe Ratio: {best_sharpe['sharpe_ratio']:.2f}")
        print(f"\n   → Consider parameter tuning or additional filters")

    print(f"{'='*70}")

if __name__ == "__main__":
    main()
