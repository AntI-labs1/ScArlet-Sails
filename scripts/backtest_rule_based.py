"""
Backtest Rule-Based Strategy.
Calculates P_rb(S) for each bar and evaluates trading performance.
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from strategies.rule_based_v2 import RuleBasedStrategy
from analysis.simple_threshold_backtest import (
    evaluate_threshold, 
    calculate_sharpe_ratio
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coin', required=True)
    parser.add_argument('--tf', required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    # Load data
    data_path = Path(f"data/features/{args.coin}_USDT_{args.tf}_features.parquet")
    if not data_path.exists():
        print(f"ERROR: {data_path} not found")
        return
    
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} rows from {data_path}")
    
    # Use same validation period as Model 2
    val_start = pd.Timestamp("2023-01-01", tz="UTC")
    val_end = pd.Timestamp("2024-01-01", tz="UTC")
    val_df = df[(df.index >= val_start) & (df.index < val_end)].copy()
    print(f"Validation period: {val_start} to {val_end}")
    print(f"Validation samples: {len(val_df)}")
    
    # Initialize strategy
    strategy = RuleBasedStrategy()
    
    # Generate signals
    print("\nGenerating P_rb signals...")
    signals_df = strategy.generate_signals(val_df)
    
    # Check if P_rb column exists
    if 'P_rb' not in signals_df.columns:
        # Try alternative column names
        possible_cols = ['signal', 'score', 'opportunity_score']
        for col in possible_cols:
            if col in signals_df.columns:
                signals_df['P_rb'] = signals_df[col]
                break
    
    if 'P_rb' not in signals_df.columns:
        print("ERROR: Could not find P_rb or equivalent signal column")
        print(f"Available columns: {signals_df.columns.tolist()}")
        return
    
    # Merge with fee_ret from original data
    if 'fee_ret' in val_df.columns:
        signals_df['fee_ret'] = val_df['fee_ret']
    else:
        print("WARNING: fee_ret not found, using close returns")
        signals_df['fee_ret'] = val_df['close'].pct_change().shift(-1)
    
    # Drop NaN
    signals_df = signals_df.dropna(subset=['P_rb', 'fee_ret'])
    
    # Evaluate thresholds
    print("\n" + "="*60)
    print("THRESHOLD ANALYSIS")
    print("="*60)
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for t in thresholds:
        metrics = evaluate_threshold(signals_df, 'P_rb', 'fee_ret', t)
        print(f"T={t:.1f}: trades={metrics['n_trades']:4d}, "
              f"sharpe={metrics['sharpe_ratio']:7.2f}, "
              f"win={metrics['win_rate']:5.1f}%, "
              f"ret={metrics['total_return']*100:6.2f}%")
    
    # Best threshold
    print("\n" + "="*60)
    print(f"RULE-BASED SUMMARY ({args.coin}/{args.tf})")
    print("="*60)
    
    best_t = args.threshold
    final_metrics = evaluate_threshold(signals_df, 'P_rb', 'fee_ret', best_t)
    
    print(f"Threshold: {best_t}")
    print(f"Trades: {final_metrics['n_trades']}")
    print(f"Sharpe: {final_metrics['sharpe_ratio']:.4f}")
    print(f"Win rate: {final_metrics['win_rate']:.2f}%")
    print(f"Total return: {final_metrics['total_return']*100:.2f}%")
    print(f"Max DD: {final_metrics['max_drawdown_pct']:.2f}%")
    
    # Save results
    output_path = f"reports/rule_based_backtest_{args.coin}_{args.tf}.json"
    import json
    with open(output_path, 'w') as f:
        json.dump({
            'coin': args.coin,
            'timeframe': args.tf,
            'val_period': f"{val_start} to {val_end}",
            'val_samples': len(val_df),
            'threshold': best_t,
            'metrics': final_metrics
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()