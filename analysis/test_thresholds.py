"""
Test different probability thresholds to optimize Sharpe ratio.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def load_predictions(model_dir='models'):
    """
    Load model predictions if available.
    NOTE: This assumes predictions are saved during training.
    If not, need to re-run prediction on test set.
    """
    # Try to find predictions file
    pred_files = list(Path(model_dir).glob('*predictions*.parquet'))
    if pred_files:
        return pd.read_parquet(pred_files[0])
    else:
        print("⚠️ No predictions file found!")
        print("   Need to generate predictions first.")
        print("   Run: python scripts/generate_predictions.py")
        return None

def calculate_metrics(y_true, y_pred, returns, threshold):
    """Calculate trading metrics for given threshold."""
    # Generate signals
    signals = (y_pred > threshold).astype(int)
    
    # Calculate returns
    strategy_returns = signals * returns
    
    # Metrics
    n_trades = signals.sum()
    if n_trades == 0:
        return None
    
    total_return = strategy_returns.sum()
    mean_return = strategy_returns[signals == 1].mean()
    sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(252 * 24)  # Annualized
    
    wins = (strategy_returns > 0).sum()
    losses = (strategy_returns < 0).sum()
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    
    # Max drawdown
    cum_returns = (1 + strategy_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min() * 100
    
    return {
        'threshold': threshold,
        'n_trades': int(n_trades),
        'total_return': total_return,
        'mean_return': mean_return,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'max_drawdown_pct': max_dd,
        'precision': (strategy_returns > 0).mean()  # Fraction of profitable trades
    }

def test_thresholds(thresholds=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]):
    """Test multiple thresholds and report results."""
    
    print('='*60)
    print('THRESHOLD ANALYSIS')
    print('='*60)
    print()
    
    # Load data
    df = load_predictions()
    if df is None:
        return
    
    results = []
    for t in thresholds:
        metrics = calculate_metrics(
            df['y_true'], 
            df['y_pred'], 
            df['returns'],
            threshold=t
        )
        if metrics:
            results.append(metrics)
    
    # Print results
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print()
    
    # Find best threshold
    best_idx = results_df['sharpe'].idxmax()
    best = results_df.iloc[best_idx]
    
    print('='*60)
    print(f'BEST THRESHOLD: {best["threshold"]:.2f}')
    print('='*60)
    print(f"Sharpe: {best['sharpe']:.4f}")
    print(f"Win rate: {best['win_rate']:.2f}%")
    print(f"Total trades: {best['n_trades']}")
    print(f"Mean return: {best['mean_return']:.6f}")
    print()
    
    # Save results
    results_df.to_csv('analysis/threshold_analysis_results.csv', index=False)
    print("Results saved to: analysis/threshold_analysis_results.csv")

if __name__ == '__main__':
    test_thresholds()
