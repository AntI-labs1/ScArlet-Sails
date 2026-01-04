"""
Day 5: Walk-Forward Validation
Tests model on truly unseen data with rolling windows.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import json
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sanitize_features import sanitize_for_model, validate_features

# Embargo period: 5 days for 15-minute candles (5 * 24 * 4 = 480 bars)
EMBARGO_BARS = 480


def load_data():
    """Load feature data and model."""
    data_path = Path('data/features/BTC_USDT_15m_features.parquet')
    model_path = Path('models/xgboost_v3_btc_15m.json')
    meta_path = Path('models/xgboost_v3_btc_15m_metadata.json')
    
    df = pd.read_parquet(data_path)
    
    model = xgb.Booster()
    model.load_model(str(model_path))
    
    with open(meta_path) as f:
        meta = json.load(f)
    
    return df, model, meta


def calculate_metrics(returns, signals):
    """Calculate trading metrics."""
    strategy_returns = signals * returns
    n_trades = signals.sum()
    
    if n_trades == 0:
        return {"sharpe": 0, "total_return": 0, "win_rate": 0, "max_dd": 0, "n_trades": 0}
    
    # Sharpe (annualized 15m)
    sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(252 * 24 * 4)
    
    # Total return
    total_return = strategy_returns.sum()
    
    # Win rate
    wins = ((strategy_returns > 0) & (signals == 1)).sum()
    losses = ((strategy_returns < 0) & (signals == 1)).sum()
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    
    # Max drawdown
    cum_returns = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min() * 100
    
    return {
        "sharpe": sharpe,
        "total_return": total_return * 100,
        "win_rate": win_rate,
        "max_dd": max_dd,
        "n_trades": int(n_trades),
    }


def walk_forward_test(df, model, feature_names, threshold=0.70, n_folds=5, test_size=0.1):
    """
    Walk-forward validation with rolling windows.
    """
    n = len(df)
    fold_size = int(n * test_size)
    
    # Prepare features and target
    horizon = 4
    df = df.copy()
    df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
    df['target'] = (df['future_return'] > 0.001).astype(int)
    
    available_features = [f for f in feature_names if f in df.columns]
    
    results = []
    
    print(f"\nRunning {n_folds}-fold walk-forward validation...")
    print(f"Fold size: {fold_size:,} bars (~{fold_size//(24*4)} days)")
    print(f"Embargo period: {EMBARGO_BARS} bars (~{EMBARGO_BARS//(24*4)} days)")
    print("-" * 60)
    
    for fold in range(n_folds):
        # Define test window (original calculation)
        test_end_original = n - fold * fold_size
        test_start_original = test_end_original - fold_size
        
        # Apply embargo: test starts EMBARGO_BARS after training ends
        # Training implicitly ends at test_start_original
        train_end = test_start_original
        test_start = train_end + EMBARGO_BARS
        
        # Calculate new test_end to maintain fold_size (if possible)
        test_end = min(test_start + fold_size, test_end_original, n)
        
        # Check if we have enough data after embargo
        if test_start >= test_end or test_start >= n:
            break
        
        # Ensure minimum test size
        if (test_end - test_start) < 100:
            break
        
        # Get test data (after embargo period)
        test_df = df.iloc[test_start:test_end].copy()
        
        # CRITICAL: Sanitize features (same as training pipeline)
        test_df = sanitize_for_model(
            test_df, 
            available_features,
            drop_inf=True,
            drop_nan=True,
            clip_extreme=True,
            verbose=False
        )
        
        # Also remove NaN in future_return
        test_df = test_df[test_df['future_return'].notna()]
        
        if len(test_df) < 100:
            print(f"Fold {fold+1}: Skipped (only {len(test_df)} samples after cleaning)")
            continue
        
        # Validate before inference
        is_valid, stats = validate_features(test_df, available_features)
        if not is_valid:
            print(f"Fold {fold+1}: Skipped (invalid features: {stats})")
            continue
        
        # Predict
        X_test = test_df[available_features]
        dtest = xgb.DMatrix(X_test, feature_names=available_features)
        y_pred = model.predict(dtest)
        
        # Generate signals
        signals = (y_pred > threshold).astype(int)
        returns = test_df['future_return'].values
        
        # Calculate metrics
        metrics = calculate_metrics(returns, signals)
        
        # Get date ranges for output
        if hasattr(df.index, '__getitem__'):
            try:
                train_end_date = str(df.index[train_end])[:10] if train_end < len(df) else "N/A"
                test_start_date = str(test_df.index.min())[:10] if len(test_df) > 0 else "N/A"
                test_end_date = str(test_df.index.max())[:10] if len(test_df) > 0 else "N/A"
            except (IndexError, KeyError):
                train_end_date = f"Index {train_end}"
                test_start_date = f"Index {test_start}"
                test_end_date = f"Index {test_end}"
        else:
            train_end_date = f"Index {train_end}"
            test_start_date = f"Index {test_start}"
            test_end_date = f"Index {test_end}"
        
        # Calculate gap in bars
        gap_bars = test_start - train_end
        
        metrics['fold'] = fold + 1
        metrics['date_range'] = f"{test_start_date} to {test_end_date}"
        metrics['train_end_date'] = train_end_date
        metrics['test_start_date'] = test_start_date
        metrics['embargo_gap_bars'] = gap_bars
        metrics['n_samples'] = len(test_df)
        
        results.append(metrics)
        
        print(f"Fold {fold+1}:")
        print(f"  Training ends: {train_end_date} (index {train_end})")
        print(f"  Embargo gap: {gap_bars} bars (~{gap_bars//(24*4)} days)")
        print(f"  Test period: {test_start_date} to {test_end_date}")
        print(f"  Samples: {len(test_df):,} | Trades: {metrics['n_trades']}")
        print(f"  Sharpe: {metrics['sharpe']:.2f} | WR: {metrics['win_rate']:.1f}% | DD: {metrics['max_dd']:.1f}%")
    
    return results


def main():
    print("=" * 60)
    print("DAY 5: WALK-FORWARD VALIDATION")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    df, model, meta = load_data()
    feature_names = meta.get('feature_names', [])
    
    print(f"Data: {len(df):,} rows")
    print(f"Features: {len(feature_names)}")
    print(f"Threshold: 0.70")
    
    # Run walk-forward
    results = walk_forward_test(df, model, feature_names, threshold=0.70, n_folds=5)
    
    # Summary
    print()
    print("=" * 60)
    print("WALK-FORWARD SUMMARY")
    print("=" * 60)
    
    if results:
        avg_sharpe = np.mean([r['sharpe'] for r in results])
        avg_wr = np.mean([r['win_rate'] for r in results])
        avg_dd = np.mean([r['max_dd'] for r in results])
        total_trades = sum(r['n_trades'] for r in results)
        
        print(f"Folds completed: {len(results)}")
        print(f"Total trades: {total_trades}")
        print(f"Average Sharpe: {avg_sharpe:.2f}")
        print(f"Average Win Rate: {avg_wr:.1f}%")
        print(f"Average Max DD: {avg_dd:.1f}%")
        
        # Stability check
        sharpes = [r['sharpe'] for r in results]
        sharpe_std = np.std(sharpes)
        
        print()
        print("STABILITY ANALYSIS:")
        print(f"  Sharpe std: {sharpe_std:.2f}")
        print(f"  Sharpe range: [{min(sharpes):.2f}, {max(sharpes):.2f}]")
        
        if sharpe_std < 0.5 and avg_sharpe > 0.5:
            print("  ✅ Model is STABLE across time periods")
        elif avg_sharpe > 0:
            print("  ⚠️ Model is profitable but has variance")
        else:
            print("  ❌ Model performance is inconsistent")
        
        # Save results
        output = {
            "validation_type": "walk_forward",
            "n_folds": len(results),
            "threshold": 0.70,
            "folds": results,
            "summary": {
                "avg_sharpe": avg_sharpe,
                "avg_win_rate": avg_wr,
                "avg_max_dd": avg_dd,
                "sharpe_std": sharpe_std,
                "total_trades": total_trades,
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        with open('analysis/walk_forward_results.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to: analysis/walk_forward_results.json")
    else:
        print("No valid folds completed")


if __name__ == "__main__":
    main()