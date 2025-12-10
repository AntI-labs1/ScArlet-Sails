"""
Test threshold 0.70 with ATR filter.
Goal: Increase trades while maintaining risk profile.
"""
import pandas as pd
import numpy as np
from pathlib import Path

def calculate_atr(df, period=14):
    """Calculate ATR as percentage of price."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(period).mean()
    atr_pct = atr / close  # ATR as % of price
    
    return atr_pct

def backtest_with_filter(predictions_df, data_df, threshold, atr_threshold):
    """Run backtest with ATR filter."""
    # Align data
    n_pred = len(predictions_df)
    data_aligned = data_df.iloc[-n_pred:].copy()
    data_aligned = data_aligned.reset_index(drop=True)
    
    # Calculate ATR
    atr_pct = calculate_atr(data_aligned)
    
    # Generate signals
    base_signal = predictions_df['y_pred'] > threshold
    atr_filter = atr_pct.values > atr_threshold
    
    # Combined signal
    signals = (base_signal.values & atr_filter).astype(int)
    
    returns = predictions_df['returns'].values
    strategy_returns = signals * returns
    
    # Metrics
    n_trades = signals.sum()
    if n_trades == 0:
        return None
    
    total_return = strategy_returns.sum()
    mean_return = strategy_returns[signals == 1].mean()
    sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(252 * 24 * 4)  # 15m bars
    
    wins = ((strategy_returns > 0) & (signals == 1)).sum()
    losses = ((strategy_returns < 0) & (signals == 1)).sum()
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    
    # Max drawdown
    cum_returns = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min() * 100
    
    # Profit factor
    gross_profit = strategy_returns[strategy_returns > 0].sum()
    gross_loss = abs(strategy_returns[strategy_returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    return {
        'threshold': threshold,
        'atr_filter': atr_threshold,
        'n_trades': int(n_trades),
        'total_return': total_return,
        'mean_return_pct': mean_return * 100,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'max_dd_pct': max_dd,
        'profit_factor': profit_factor
    }

def main():
    print('='*70)
    print('ATR FILTER ANALYSIS (Threshold = 0.70)')
    print('='*70)
    
    # Load predictions
    pred_path = Path('models/xgboost_v3_btc_15m_predictions.parquet')
    predictions = pd.read_parquet(pred_path)
    print(f"Predictions: {len(predictions):,}")
    
    # Load raw data for ATR calculation
    data_path = Path('data/features/BTC_USDT_15m_features.parquet')
    data = pd.read_parquet(data_path)
    print(f"Raw data: {len(data):,}")
    
    # Test configurations
    threshold = 0.70
    atr_levels = [0.000, 0.003, 0.005, 0.007, 0.010, 0.015]
    
    print()
    print(f"Testing ATR filters with threshold={threshold}")
    print('-'*70)
    
    results = []
    for atr in atr_levels:
        metrics = backtest_with_filter(predictions, data, threshold, atr)
        if metrics:
            results.append(metrics)
            print(f"ATR>{atr:.3f}: {metrics['n_trades']:>5} trades, "
                  f"Sharpe={metrics['sharpe']:>6.2f}, "
                  f"WR={metrics['win_rate']:>5.1f}%, "
                  f"DD={metrics['max_dd_pct']:>6.1f}%, "
                  f"PF={metrics['profit_factor']:>5.2f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('analysis/atr_filter_results.csv', index=False)
    
    print()
    print('='*70)
    print('RECOMMENDATION')
    print('='*70)
    
    # Find best risk-adjusted (Sharpe / DD ratio)
    results_df['risk_adj'] = results_df['sharpe'] / abs(results_df['max_dd_pct'])
    best_idx = results_df['risk_adj'].idxmax()
    best = results_df.iloc[best_idx]
    
    print(f"Best risk-adjusted config: ATR > {best['atr_filter']:.3f}")
    print(f"  Sharpe: {best['sharpe']:.2f}")
    print(f"  Max DD: {best['max_dd_pct']:.1f}%")
    print(f"  Trades: {best['n_trades']}")
    print(f"  Win Rate: {best['win_rate']:.1f}%")
    print(f"  Profit Factor: {best['profit_factor']:.2f}")
    
    # Also show threshold 0.60 baseline for comparison
    print()
    print('-'*70)
    print('COMPARISON: Threshold 0.60 (no filter) vs Best ATR config')
    print('-'*70)
    
    baseline = backtest_with_filter(predictions, data, 0.60, 0.000)
    if baseline:
        print(f"0.60 baseline: {baseline['n_trades']} trades, "
              f"Sharpe={baseline['sharpe']:.2f}, "
              f"DD={baseline['max_dd_pct']:.1f}%")
        print(f"0.70+ATR best: {best['n_trades']} trades, "
              f"Sharpe={best['sharpe']:.2f}, "
              f"DD={best['max_dd_pct']:.1f}%")

if __name__ == '__main__':
    main()