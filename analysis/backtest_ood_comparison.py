# DEPRECATED 2026-05: используйте backtesting/vbt_engine.py (см. backtesting/MIGRATION_NOTES.md).
"""
Day 9: Backtest comparison - With vs Without OOD Penalty.
Shows how OOD detection improves risk-adjusted returns.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ood_detector import OODDetector


def load_data():
    """Load predictions and features."""
    pred = pd.read_parquet('models/xgboost_v3_btc_15m_predictions.parquet')
    feat = pd.read_parquet('data/features/BTC_USDT_15m_features.parquet')
    
    # Load feature names
    with open('models/xgboost_v3_btc_15m_metadata.json') as f:
        meta = json.load(f)
    feature_names = meta['feature_names']
    
    # Align
    n_pred = len(pred)
    feat_aligned = feat.iloc[-n_pred:].reset_index(drop=True)
    pred = pred.reset_index(drop=True)
    
    return pred, feat_aligned, feature_names


def calculate_metrics(returns, name="Strategy"):
    """Calculate performance metrics."""
    if len(returns) == 0 or returns.std() == 0:
        return {'name': name, 'total_return': 0, 'sharpe': 0, 'max_dd': 0, 'n_trades': 0}
    
    total_return = (1 + returns).prod() - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24 * 4)
    
    cum = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max
    max_dd = dd.min()
    
    n_trades = (returns != 0).sum()
    
    return {
        'name': name,
        'total_return': total_return * 100,
        'sharpe': sharpe,
        'max_dd': max_dd * 100,
        'n_trades': int(n_trades),
    }


def run_backtest(pred, feat, feature_names, threshold=0.70):
    """Compare with/without OOD penalty."""
    
    # Load OOD detector
    ood_detector = OODDetector()
    ood_path = Path('models/ood_detector_btc_15m.json')
    if ood_path.exists():
        ood_detector.load(str(ood_path))
        print(f"OOD Detector loaded, threshold: {ood_detector.threshold:.2f}")
    else:
        print("WARNING: OOD detector not found")
        return None
    
    n = len(pred)
    p_ml = pred['y_pred'].values
    returns = pred['returns'].values
    
    # Arrays for results
    returns_no_ood = np.zeros(n)
    returns_with_ood = np.zeros(n)
    
    ood_count = 0
    ood_penalties = []
    
    print(f"Running backtest on {n:,} bars...")
    
    for i in range(n):
        # Skip if below threshold
        if p_ml[i] < threshold:
            continue
        
        # Get features for this bar
        row = feat.iloc[i]
        features = row[feature_names].values
        
        # Skip if NaN
        if np.any(np.isnan(features)):
            continue
        
        # Base return (no OOD)
        returns_no_ood[i] = returns[i]
        
        # OOD check
        ood_state = ood_detector.detect(features)
        
        if ood_state.is_ood:
            ood_count += 1
            ood_penalties.append(ood_state.ood_penalty)
            # Reduce position by confidence multiplier
            returns_with_ood[i] = returns[i] * ood_state.confidence_multiplier
        else:
            returns_with_ood[i] = returns[i]
        
        if (i + 1) % 20000 == 0:
            print(f"  Processed {i+1:,}/{n:,}, OOD detected: {ood_count}")
    
    print(f"\nTotal OOD detections: {ood_count} ({ood_count/n*100:.2f}%)")
    if ood_penalties:
        print(f"Avg OOD penalty: {np.mean(ood_penalties):.3f}")
    
    # Calculate metrics
    metrics_no_ood = calculate_metrics(returns_no_ood, "Without OOD")
    metrics_with_ood = calculate_metrics(returns_with_ood, "With OOD")
    
    return {
        'no_ood': metrics_no_ood,
        'with_ood': metrics_with_ood,
        'ood_count': ood_count,
        'ood_rate': ood_count / n * 100,
    }


def main():
    print("=" * 60)
    print("DAY 9: BACKTEST WITH vs WITHOUT OOD PENALTY")
    print("=" * 60)
    
    pred, feat, feature_names = load_data()
    print(f"Data: {len(pred):,} rows, {len(feature_names)} features")
    
    print("\n" + "-" * 60)
    results = run_backtest(pred, feat, feature_names, threshold=0.70)
    
    if results is None:
        return
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print("\n{:<20} {:>15} {:>15}".format("Metric", "Without OOD", "With OOD"))
    print("-" * 50)
    
    for metric in ['total_return', 'sharpe', 'max_dd', 'n_trades']:
        no = results['no_ood'][metric]
        with_ood = results['with_ood'][metric]
        
        if metric == 'n_trades':
            print(f"{metric:<20} {int(no):>15,} {int(with_ood):>15,}")
        elif metric in ['total_return', 'max_dd']:
            print(f"{metric:<20} {no:>14.1f}% {with_ood:>14.1f}%")
        else:
            print(f"{metric:<20} {no:>15.2f} {with_ood:>15.2f}")
    
    print(f"\nOOD Detection Rate: {results['ood_rate']:.2f}%")
    
    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    no = results['no_ood']
    with_ood = results['with_ood']
    
    if with_ood['max_dd'] > no['max_dd']:  # Less negative = better
        print(f"✅ Max DD improved: {no['max_dd']:.1f}% → {with_ood['max_dd']:.1f}%")
    
    if with_ood['sharpe'] > no['sharpe']:
        print(f"✅ Sharpe improved: {no['sharpe']:.2f} → {with_ood['sharpe']:.2f}")
    elif with_ood['sharpe'] < no['sharpe']:
        sharpe_loss = (no['sharpe'] - with_ood['sharpe']) / no['sharpe'] * 100
        print(f"⚠️ Sharpe decreased: {no['sharpe']:.2f} → {with_ood['sharpe']:.2f} ({sharpe_loss:.1f}% loss)")
    
    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'threshold': 0.70,
        **results,
    }
    
    with open('analysis/backtest_ood_comparison.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✅ Saved: analysis/backtest_ood_comparison.json")


if __name__ == "__main__":
    main()