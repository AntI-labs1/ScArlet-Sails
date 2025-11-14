#!/usr/bin/env python3
"""
TP/SL Grid Search для ML XGBoost модели

Цель: Найти оптимальные параметры Take Profit и Stop Loss
Тестирует различные комбинации TP/SL для определения:
- Какое соотношение TP/SL дает лучший Sharpe Ratio
- Нужен ли широкий SL для лучших результатов
- Как TP/SL влияет на Win Rate и Profit Factor

Гипотеза: Фиксированный TP/SL не оптимален для всех рыночных условий
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime
import joblib
from itertools import product

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.xgboost_model import XGBoostModel
from features.multi_timeframe_extractor import MultiTimeframeFeatureExtractor

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Constants
ASSET = "BTC"
TIMEFRAME = "15m"
FORWARD_BARS = 96  # 24 hours for 15m
ML_THRESHOLD = 0.65  # Use best threshold from previous tests

# Trading costs (fixed)
ENTRY_COST = 0.0015   # 0.15%
EXIT_COST = 0.0015    # 0.15%
TOTAL_COST = ENTRY_COST + EXIT_COST  # 0.30%

# Grid search ranges
TP_OPTIONS = [0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]  # 0.3% to 5%
SL_OPTIONS = [-0.003, -0.005, -0.007, -0.01, -0.015, -0.02, -0.025, -0.03, -0.04, -0.05]  # -0.3% to -5%

def backtest_ml_with_tpsl(
    df: pd.DataFrame,
    all_tf: Dict[str, pd.DataFrame],
    extractor: MultiTimeframeFeatureExtractor,
    model: XGBoostModel,
    scaler,
    take_profit: float,
    stop_loss: float,
    ml_threshold: float
) -> Dict:
    """
    Backtesting ML модели с заданными TP/SL

    Args:
        df: Primary timeframe dataframe
        all_tf: All timeframes data
        extractor: Feature extractor
        model: Trained XGBoost model
        scaler: StandardScaler
        take_profit: TP level (e.g., 0.01 = 1%)
        stop_loss: SL level (e.g., -0.005 = -0.5%)
        ml_threshold: ML probability threshold

    Returns:
        Dict с результатами
    """

    trades = []
    equity_curve = [1.0]
    current_equity = 1.0

    ood_count = 0

    for i in range(len(df) - FORWARD_BARS):
        # Extract features
        features = extractor.extract_features_at_bar(all_tf, TIMEFRAME, i)

        if features is None:
            continue

        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))[0]

        # Check for OOD
        is_ood = np.any(np.abs(features_scaled) > 3.0)
        if is_ood:
            ood_count += 1
            continue

        # Predict probability
        prob = model.predict_proba(features.reshape(1, -1))[0]
        prob_up = prob[1]

        # Check ML threshold
        if prob_up < ml_threshold:
            continue

        # Enter trade
        bar = df.iloc[i]
        entry_price = bar['close']
        entry_time = bar.name

        # Calculate TP/SL levels (AFTER costs)
        tp_level = entry_price * (1 + take_profit + TOTAL_COST)
        sl_level = entry_price * (1 + stop_loss - TOTAL_COST)

        # Simulate trade forward
        exit_bar = None
        exit_price = None
        exit_reason = None

        for j in range(i + 1, min(i + FORWARD_BARS + 1, len(df))):
            future_bar = df.iloc[j]

            # Check TP
            if future_bar['high'] >= tp_level:
                exit_bar = j - i
                exit_price = tp_level
                exit_reason = 'TP'
                break

            # Check SL
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

        # Calculate P&L
        gross_return = (exit_price - entry_price) / entry_price
        net_return = gross_return - TOTAL_COST

        # Update equity
        trade_pnl = net_return
        current_equity *= (1 + trade_pnl)
        equity_curve.append(current_equity)

        # Classify result
        if net_return >= take_profit:
            result = 'WIN'
        elif net_return <= stop_loss:
            result = 'LOSS'
        else:
            result = 'WIN' if net_return > 0 else 'LOSS'

        trades.append({
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_bar': exit_bar,
            'exit_reason': exit_reason,
            'gross_return': gross_return,
            'net_return': net_return,
            'result': result,
            'ml_prob': prob_up
        })

    # Calculate metrics
    total_trades = len(trades)

    if total_trades == 0:
        return {
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'avg_return': 0.0,
            'tp_count': 0,
            'sl_count': 0,
            'time_count': 0
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

    total_return = current_equity - 1.0
    avg_return = np.mean([t['net_return'] for t in trades])
    std_return = np.std([t['net_return'] for t in trades])

    # Sharpe Ratio
    periods_per_year = 35000
    sharpe_ratio = (avg_return / std_return) * np.sqrt(periods_per_year) if std_return > 0 else 0

    # Max Drawdown
    equity_array = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - running_max) / running_max
    max_drawdown = abs(drawdown.min())

    # Exit reason distribution
    tp_count = sum(1 for t in trades if t['exit_reason'] == 'TP')
    sl_count = sum(1 for t in trades if t['exit_reason'] == 'SL')
    time_count = sum(1 for t in trades if t['exit_reason'] == 'TIME')

    return {
        'take_profit': float(take_profit),
        'stop_loss': float(stop_loss),
        'trades': int(total_trades),
        'wins': int(wins),
        'losses': int(losses),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor) if profit_factor != float('inf') else 'inf',
        'gross_profit': float(gross_profit),
        'gross_loss': float(gross_loss),
        'total_return': float(total_return),
        'final_equity': float(current_equity),
        'avg_return': float(avg_return),
        'std_return': float(std_return),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'tp_count': int(tp_count),
        'sl_count': int(sl_count),
        'time_count': int(time_count),
        'tp_sl_ratio': float(tp_count / sl_count) if sl_count > 0 else float('inf'),
        'ood_count': int(ood_count)
    }

def main():
    """Main function"""

    print("="*70)
    print("TP/SL GRID SEARCH FOR ML XGBOOST")
    print("="*70)
    print(f"Asset: {ASSET}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"ML threshold: {ML_THRESHOLD:.2f}")
    print(f"Trading costs: {TOTAL_COST:.2%} round-trip")
    print(f"\nTP options: {len(TP_OPTIONS)} values from {TP_OPTIONS[0]:.1%} to {TP_OPTIONS[-1]:.1%}")
    print(f"SL options: {len(SL_OPTIONS)} values from {SL_OPTIONS[0]:.1%} to {SL_OPTIONS[-1]:.1%}")
    print(f"Total combinations: {len(TP_OPTIONS) * len(SL_OPTIONS)}")
    print("="*70)

    # Load ML model
    model_path = MODELS_DIR / "xgboost_normalized_model.json"
    scaler_path = MODELS_DIR / "xgboost_normalized_scaler.pkl"

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return

    if not scaler_path.exists():
        print(f"❌ Scaler not found: {scaler_path}")
        return

    print(f"\nLoading ML model...")
    model = XGBoostModel()
    model.load(str(model_path))
    scaler = joblib.load(scaler_path)
    print(f"✅ Model loaded")

    # Load data
    print(f"\nLoading multi-timeframe data...")
    extractor = MultiTimeframeFeatureExtractor(data_dir=str(DATA_DIR))

    try:
        all_tf, primary_df = extractor.prepare_multi_timeframe_data(ASSET, TIMEFRAME)
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return

    print(f"✅ Loaded {len(primary_df):,} bars")

    # Grid search
    print(f"\n{'='*70}")
    print("RUNNING GRID SEARCH...")
    print(f"{'='*70}")

    all_results = []
    total_combinations = len(TP_OPTIONS) * len(SL_OPTIONS)

    for idx, (tp, sl) in enumerate(product(TP_OPTIONS, SL_OPTIONS), 1):
        print(f"[{idx}/{total_combinations}] Testing TP={tp:.2%}, SL={sl:.2%}...", end=' ')

        results = backtest_ml_with_tpsl(
            primary_df,
            all_tf,
            extractor,
            model,
            scaler,
            tp,
            sl,
            ML_THRESHOLD
        )

        all_results.append(results)

        # Quick summary
        print(f"Trades={results['trades']:,}, WR={results['win_rate']:.1%}, PF={results['profit_factor']}, Sharpe={results['sharpe_ratio']:.2f}")

    # Find best configurations
    print(f"\n{'='*70}")
    print("ANALYSIS: TOP CONFIGURATIONS")
    print(f"{'='*70}")

    # Filter profitable configs
    profitable = [r for r in all_results if r['profit_factor'] != 'inf' and r['profit_factor'] > 1.0 and r['trades'] > 0]

    if profitable:
        # Sort by different metrics
        best_sharpe = sorted(profitable, key=lambda x: x['sharpe_ratio'], reverse=True)[:5]
        best_wr = sorted(profitable, key=lambda x: x['win_rate'], reverse=True)[:5]
        best_pf = sorted(profitable, key=lambda x: x['profit_factor'], reverse=True)[:5]
        best_return = sorted(profitable, key=lambda x: x['total_return'], reverse=True)[:5]

        print(f"\n🏆 TOP 5 BY SHARPE RATIO:")
        print(f"{'TP':>6} {'SL':>7} {'Trades':>8} {'WR':>7} {'PF':>6} {'Return':>8} {'Sharpe':>7} {'TP/SL':>6}")
        print(f"{'-'*70}")
        for r in best_sharpe:
            tp_sl_ratio = r['tp_sl_ratio'] if r['tp_sl_ratio'] != float('inf') else 999
            print(f"{r['take_profit']:>5.1%} {r['stop_loss']:>6.1%} {r['trades']:>8,} {r['win_rate']:>6.1%} {r['profit_factor']:>6.2f} {r['total_return']:>7.1%} {r['sharpe_ratio']:>7.2f} {tp_sl_ratio:>6.2f}")

        print(f"\n🎯 TOP 5 BY WIN RATE:")
        print(f"{'TP':>6} {'SL':>7} {'Trades':>8} {'WR':>7} {'PF':>6} {'Return':>8} {'Sharpe':>7}")
        print(f"{'-'*70}")
        for r in best_wr:
            print(f"{r['take_profit']:>5.1%} {r['stop_loss']:>6.1%} {r['trades']:>8,} {r['win_rate']:>6.1%} {r['profit_factor']:>6.2f} {r['total_return']:>7.1%} {r['sharpe_ratio']:>7.2f}")

        print(f"\n💰 TOP 5 BY PROFIT FACTOR:")
        print(f"{'TP':>6} {'SL':>7} {'Trades':>8} {'WR':>7} {'PF':>6} {'Return':>8} {'Sharpe':>7}")
        print(f"{'-'*70}")
        for r in best_pf:
            print(f"{r['take_profit']:>5.1%} {r['stop_loss']:>6.1%} {r['trades']:>8,} {r['win_rate']:>6.1%} {r['profit_factor']:>6.2f} {r['total_return']:>7.1%} {r['sharpe_ratio']:>7.2f}")

        print(f"\n📈 TOP 5 BY TOTAL RETURN:")
        print(f"{'TP':>6} {'SL':>7} {'Trades':>8} {'WR':>7} {'PF':>6} {'Return':>8} {'Sharpe':>7}")
        print(f"{'-'*70}")
        for r in best_return:
            print(f"{r['take_profit']:>5.1%} {r['stop_loss']:>6.1%} {r['trades']:>8,} {r['win_rate']:>6.1%} {r['profit_factor']:>6.2f} {r['total_return']:>7.1%} {r['sharpe_ratio']:>7.2f}")

        # Identify patterns
        print(f"\n{'='*70}")
        print("PATTERN ANALYSIS")
        print(f"{'='*70}")

        avg_tp = np.mean([r['take_profit'] for r in profitable])
        avg_sl = np.mean([r['stop_loss'] for r in profitable])
        avg_ratio = avg_tp / abs(avg_sl)

        print(f"Profitable configs: {len(profitable)}/{len(all_results)} ({len(profitable)/len(all_results):.1%})")
        print(f"Average TP: {avg_tp:.2%}")
        print(f"Average SL: {avg_sl:.2%}")
        print(f"Average TP/SL ratio: {avg_ratio:.2f}")

        # Best overall
        best_overall = best_sharpe[0]
        print(f"\n{'='*70}")
        print("RECOMMENDED CONFIGURATION")
        print(f"{'='*70}")
        print(f"TP: {best_overall['take_profit']:.2%}")
        print(f"SL: {best_overall['stop_loss']:.2%}")
        print(f"Win Rate: {best_overall['win_rate']:.1%}")
        print(f"Profit Factor: {best_overall['profit_factor']:.2f}")
        print(f"Sharpe Ratio: {best_overall['sharpe_ratio']:.2f}")
        print(f"Total Return: {best_overall['total_return']:.1%}")
        print(f"Trades: {best_overall['trades']:,}")
        print(f"TP hits: {best_overall['tp_count']:,} ({best_overall['tp_count']/best_overall['trades']:.1%})")
        print(f"SL hits: {best_overall['sl_count']:,} ({best_overall['sl_count']/best_overall['trades']:.1%})")

    else:
        print(f"\n❌ NO PROFITABLE CONFIGURATIONS FOUND")
        print(f"All {len(all_results)} combinations resulted in PF <= 1.0")

        # Find least bad
        valid_results = [r for r in all_results if r['profit_factor'] != 'inf' and r['trades'] > 0]
        if valid_results:
            least_bad = max(valid_results, key=lambda x: x['profit_factor'])
            print(f"\nLeast bad configuration:")
            print(f"TP: {least_bad['take_profit']:.2%}, SL: {least_bad['stop_loss']:.2%}")
            print(f"WR: {least_bad['win_rate']:.1%}, PF: {least_bad['profit_factor']:.2f}")

    # Save results
    output_file = REPORTS_DIR / f"ml_tpsl_grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'asset': ASSET,
            'timeframe': TIMEFRAME,
            'ml_threshold': ML_THRESHOLD,
            'parameters': {
                'entry_cost': ENTRY_COST,
                'exit_cost': EXIT_COST,
                'total_cost': TOTAL_COST,
                'tp_options': TP_OPTIONS,
                'sl_options': SL_OPTIONS
            },
            'results': all_results,
            'summary': {
                'total_combinations': len(all_results),
                'profitable_count': len(profitable) if profitable else 0,
                'best_sharpe': best_sharpe[0] if profitable else None,
                'best_wr': best_wr[0] if profitable else None,
                'best_pf': best_pf[0] if profitable else None
            }
        }, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
