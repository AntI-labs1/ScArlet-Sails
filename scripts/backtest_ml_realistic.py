#!/usr/bin/env python3
"""
REALISTIC Backtest для ML XGBoost модели

Проверяет, остаётся ли ML profitable с реалистичными условиями:
- Take Profit: +1.0%
- Stop Loss: -0.5%
- Trading costs: 0.3% round-trip
- Exit при первом событии (TP/SL/Time)

Тестирует разные ML thresholds для оптимизации.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime
import joblib

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

# Realistic parameters
TAKE_PROFIT = 0.01    # 1.0% TP
STOP_LOSS = -0.005    # -0.5% SL
ENTRY_COST = 0.0015   # 0.15% (maker fee + slippage)
EXIT_COST = 0.0015    # 0.15%
TOTAL_COST = ENTRY_COST + EXIT_COST  # 0.30% round-trip

def backtest_ml_realistic(
    df: pd.DataFrame,
    all_tf: Dict[str, pd.DataFrame],
    extractor: MultiTimeframeFeatureExtractor,
    model: XGBoostModel,
    scaler,
    ml_threshold: float,
    name: str
) -> Dict:
    """
    REALISTIC Backtesting ML модели с TP/SL/Time exit и издержками

    Args:
        df: Primary timeframe dataframe
        all_tf: All timeframes data
        extractor: Feature extractor
        model: Trained XGBoost model
        scaler: StandardScaler
        ml_threshold: Probability threshold for entry (0-1)
        name: Strategy name

    Returns:
        Dict с результатами
    """

    print(f"\n{'='*70}")
    print(f"Backtesting: {name} (threshold={ml_threshold:.2f})")
    print(f"{'='*70}")

    trades = []
    equity_curve = [1.0]
    current_equity = 1.0

    ood_count = 0  # Out-of-distribution count

    for i in range(len(df) - FORWARD_BARS):
        # Extract features at this bar
        features = extractor.extract_features_at_bar(all_tf, TIMEFRAME, i)

        if features is None:
            continue

        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))[0]

        # Check for OOD (any feature > 3σ)
        is_ood = np.any(np.abs(features_scaled) > 3.0)
        if is_ood:
            ood_count += 1
            # Skip OOD signals
            continue

        # Predict probability
        prob = model.predict_proba(features.reshape(1, -1))[0]
        prob_up = prob[1]  # Probability of UP

        # Check ML threshold
        if prob_up < ml_threshold:
            continue

        # Enter trade
        bar = df.iloc[i]
        entry_price = bar['close']
        entry_time = bar.name

        # Calculate TP/SL levels (AFTER costs)
        tp_level = entry_price * (1 + TAKE_PROFIT + TOTAL_COST)
        sl_level = entry_price * (1 + STOP_LOSS - TOTAL_COST)

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

        # Calculate P&L (including costs)
        gross_return = (exit_price - entry_price) / entry_price
        net_return = gross_return - TOTAL_COST

        # Update equity
        trade_pnl = net_return
        current_equity *= (1 + trade_pnl)
        equity_curve.append(current_equity)

        # Classify result
        if net_return >= TAKE_PROFIT:
            result = 'WIN'
        elif net_return <= STOP_LOSS:
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
            'name': name,
            'ml_threshold': ml_threshold,
            'trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'ood_ratio': ood_count / len(df) if len(df) > 0 else 0.0
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
    exit_reasons = {}
    for t in trades:
        reason = t['exit_reason']
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    # ML probability stats
    ml_probs = [t['ml_prob'] for t in trades]
    avg_ml_prob = np.mean(ml_probs)

    results = {
        'name': name,
        'ml_threshold': ml_threshold,
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
        'avg_ml_prob': avg_ml_prob,
        'ood_ratio': ood_count / len(df) if len(df) > 0 else 0.0,
        'ood_count': ood_count
    }

    # Print summary
    print(f"  Trades: {total_trades:,}")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Total Return: {total_return:.2%}")
    print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {max_drawdown:.2%}")
    print(f"  Avg ML Prob: {avg_ml_prob:.1%}")
    print(f"  OOD Ratio: {ood_count / len(df):.1%}")
    print(f"  Exit reasons: TP={exit_reasons.get('TP', 0)}, SL={exit_reasons.get('SL', 0)}, TIME={exit_reasons.get('TIME', 0)}")

    return results

def main():
    """Main function"""

    print("="*70)
    print("REALISTIC ML XGBOOST BACKTEST")
    print("="*70)
    print(f"Asset: {ASSET}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Forward window: {FORWARD_BARS} bars (24 hours)")
    print(f"Take Profit: {TAKE_PROFIT:.1%}")
    print(f"Stop Loss: {STOP_LOSS:.1%}")
    print(f"Trading costs: {TOTAL_COST:.2%} round-trip")
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
    print(f"✅ Model loaded: {model_path.name}")

    # Load data with MultiTimeframeFeatureExtractor
    print(f"\nLoading multi-timeframe data...")
    extractor = MultiTimeframeFeatureExtractor(data_dir=str(DATA_DIR))

    try:
        all_tf, primary_df = extractor.prepare_multi_timeframe_data(ASSET, TIMEFRAME)
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return

    print(f"✅ Loaded {len(primary_df):,} bars")

    # Test different ML thresholds
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7]
    all_results = []

    for threshold in thresholds:
        name = f"ML (threshold={threshold:.2f})"
        results = backtest_ml_realistic(
            primary_df,
            all_tf,
            extractor,
            model,
            scaler,
            threshold,
            name
        )
        all_results.append(results)

    # Summary comparison
    print(f"\n{'='*70}")
    print("COMPARATIVE SUMMARY (REALISTIC ML)")
    print(f"{'='*70}")
    print(f"{'Strategy':<30} {'Thresh':>6} {'Trades':>8} {'WR':>7} {'PF':>6} {'Return':>8} {'Sharpe':>7} {'MDD':>7}")
    print(f"{'-'*70}")

    for r in all_results:
        print(f"{r['name']:<30} {r['ml_threshold']:>6.2f} {r['trades']:>8,} {r['win_rate']:>6.1%} {r['profit_factor']:>6.2f} {r['total_return']:>7.1%} {r['sharpe_ratio']:>7.2f} {r['max_drawdown']:>6.1%}")

    # Best configuration
    profitable_results = [r for r in all_results if r['profit_factor'] > 1.0 and r['trades'] > 0]

    if profitable_results:
        best_sharpe = max(profitable_results, key=lambda x: x['sharpe_ratio'])
        best_wr = max(profitable_results, key=lambda x: x['win_rate'])
        best_pf = max(profitable_results, key=lambda x: x['profit_factor'] if x['profit_factor'] != float('inf') else 0)

        print(f"\n{'='*70}")
        print("BEST PROFITABLE CONFIGURATIONS")
        print(f"{'='*70}")
        print(f"🏆 Best Sharpe Ratio: {best_sharpe['name']} ({best_sharpe['sharpe_ratio']:.2f})")
        print(f"🎯 Best Win Rate: {best_wr['name']} ({best_wr['win_rate']:.1%})")
        print(f"💰 Best Profit Factor: {best_pf['name']} ({best_pf['profit_factor']:.2f})")
    else:
        print(f"\n{'='*70}")
        print("⚠️  NO PROFITABLE CONFIGURATIONS")
        print(f"{'='*70}")
        print("All tested thresholds resulted in PF < 1.0")

        # Find least bad
        best_pf = max(all_results, key=lambda x: x['profit_factor'] if x['profit_factor'] != float('inf') else 0)
        print(f"\nLeast bad: {best_pf['name']} (PF={best_pf['profit_factor']:.2f}, WR={best_pf['win_rate']:.1%})")

    # Save results
    output_file = REPORTS_DIR / f"ml_realistic_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Convert to JSON-serializable
    json_results = []
    for r in all_results:
        r_copy = r.copy()
        # Handle infinity
        if r_copy['profit_factor'] == float('inf'):
            r_copy['profit_factor'] = 'inf'
        # Convert numpy types to Python native types
        for key, value in r_copy.items():
            if isinstance(value, np.floating):
                r_copy[key] = float(value)
            elif isinstance(value, np.integer):
                r_copy[key] = int(value)
            elif isinstance(value, dict):
                # Convert nested dict values
                for k, v in value.items():
                    if isinstance(v, (np.floating, np.integer)):
                        r_copy[key][k] = float(v) if isinstance(v, np.floating) else int(v)
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

    # Comparison with Rule-Based
    print(f"\n{'='*70}")
    print("COMPARISON WITH RULE-BASED")
    print(f"{'='*70}")
    print(f"Rule-Based (realistic): WR 36.8%, PF 0.52, Return -100%")

    if profitable_results:
        print(f"ML (best):              WR {best_sharpe['win_rate']:.1%}, PF {best_sharpe['profit_factor']:.2f}, Return {best_sharpe['total_return']:.1%}")

        if best_sharpe['win_rate'] > 0.368:
            print(f"\n✅ ML SIGNIFICANTLY BETTER than Rule-Based!")
            print(f"   Win Rate: +{(best_sharpe['win_rate'] - 0.368) * 100:.1f} percentage points")

        if best_sharpe['profit_factor'] > 1.0:
            print(f"✅ ML is PROFITABLE (PF > 1.0)")

        # Recommendation
        print(f"\n{'='*70}")
        print("RECOMMENDATION")
        print(f"{'='*70}")

        if best_sharpe['profit_factor'] > 1.2 and best_sharpe['win_rate'] > 0.45:
            print(f"✅ EXCELLENT: ML performs well with realistic conditions")
            print(f"   → Use ML with threshold={best_sharpe['ml_threshold']:.2f}")
            print(f"   → Proceed to Hybrid backtest (Day 2.2)")
        elif best_sharpe['profit_factor'] > 1.0:
            print(f"⚠️  ACCEPTABLE: ML is profitable but could be better")
            print(f"   → Consider OpportunityScorer for better entry selection")
            print(f"   → Add filters from mathematicians (Day 3)")
        else:
            print(f"⚠️  NEEDS IMPROVEMENT: ML barely profitable")
            print(f"   → OpportunityScorer integration critical")
            print(f"   → Filters from mathematicians essential")
    else:
        print(f"ML (best):              WR {best_pf['win_rate']:.1%}, PF {best_pf['profit_factor']:.2f}, Return {best_pf['total_return']:.1%}")
        print(f"\n❌ ML NOT PROFITABLE with current settings")
        print(f"   → OpportunityScorer integration CRITICAL")
        print(f"   → Filters from mathematicians ESSENTIAL")
        print(f"   → Consider adjusting TP/SL ratio")

    print(f"{'='*70}")

if __name__ == "__main__":
    main()
