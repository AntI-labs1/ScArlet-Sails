"""
Diagnostic script for XGBoost model performance analysis.
Analyzes actual trade distribution and identifies issues.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

def load_metadata(model_path='models/xgboost_v3_btc_15m_metadata.json'):
    """Load model metadata."""
    with open(model_path) as f:
        return json.load(f)

def analyze_model_metadata(meta):
    """Print model configuration summary."""
    print('='*60)
    print('MODEL CONFIGURATION ANALYSIS')
    print('='*60)
    print(f"Target type: {meta.get('target_type', 'unknown')}")
    print(f"Horizon: {meta.get('horizon_bars', 'unknown')} bars")
    print(f"Features: {meta.get('n_features', 'unknown')}")
    print(f"Trained on: {meta.get('coin', 'unknown')} {meta.get('timeframe', 'unknown')}")
    print()
    
    # Test set performance
    test_metrics = meta.get('metrics', {}).get('test', {})
    print('='*60)
    print('TEST SET PERFORMANCE')
    print('='*60)
    print(f"AUC: {test_metrics.get('auc', 'N/A'):.4f}")
    print(f"Precision: {test_metrics.get('precision', 'N/A'):.4f}")
    print(f"Recall: {test_metrics.get('recall', 'N/A'):.4f}")
    print(f"F1: {test_metrics.get('f1', 'N/A'):.4f}")
    print(f"Accuracy: {test_metrics.get('accuracy', 'N/A'):.4f}")
    print(f"Class balance: {test_metrics.get('class_balance', 'N/A'):.4f}")
    print()
    
    # Backtest results
    bt = meta.get('optimal_threshold_trading', {}).get('backtest_metrics', {})
    print('='*60)
    print('BACKTEST RESULTS (Threshold=0.7)')
    print('='*60)
    print(f"Total trades: {bt.get('n_trades', 'N/A')}")
    print(f"Win rate: {bt.get('win_rate', 'N/A'):.2f}%")
    print(f"Total return: {bt.get('total_return', 'N/A'):.4f}")
    print(f"Mean return per trade: {bt.get('mean_return', 'N/A'):.6f}")
    print(f"Sharpe ratio: {bt.get('sharpe_ratio', 'N/A'):.4f}")
    print(f"Max drawdown: {bt.get('max_drawdown_pct', 'N/A'):.4f}%")
    print()

def diagnose_issues(meta):
    """Diagnose specific issues causing negative Sharpe."""
    print('='*60)
    print('ISSUE DIAGNOSIS')
    print('='*60)
    
    test = meta.get('metrics', {}).get('test', {})
    bt = meta.get('optimal_threshold_trading', {}).get('backtest_metrics', {})
    
    # Issue 1: Low precision
    precision = test.get('precision', 0)
    if precision < 0.4:
        print(f"🔴 ISSUE 1: Low Precision ({precision:.2%})")
        print(f"   → {int((1-precision)*100)} из 100 сигналов убыточны!")
        print(f"   → Recommendation: Increase threshold to 0.85-0.90")
        print()
    
    # Issue 2: Class imbalance
    class_balance = test.get('class_balance', 0)
    if class_balance < 0.25:
        print(f"🔴 ISSUE 2: Severe Class Imbalance ({class_balance:.2%} positive)")
        print(f"   → Model trained on too few positive examples")
        print(f"   → Recommendation: Use class_weight or SMOTE")
        print()
    
    # Issue 3: Win rate too low
    win_rate = bt.get('win_rate', 0)
    if win_rate < 50:
        print(f"🔴 ISSUE 3: Win Rate Below 50% ({win_rate:.1f}%)")
        mean_return = bt.get('mean_return', 0)
        print(f"   → Mean return per trade: {mean_return:.6f}")
        print(f"   → With {win_rate:.1f}% win rate, need win/loss ratio > {1/(win_rate/100) - 1:.2f}")
        print(f"   → Recommendation: Add stop-loss or increase threshold")
        print()
    
    # Issue 4: AUC vs Sharpe disconnect
    auc = test.get('auc', 0)
    sharpe = bt.get('sharpe_ratio', 0)
    if auc > 0.6 and sharpe < 0:
        print(f"🔴 ISSUE 4: AUC ≠ Trading Performance")
        print(f"   → Model AUC: {auc:.4f} (decent ranking)")
        print(f"   → Sharpe: {sharpe:.4f} (losing money)")
        print(f"   → Problem: Model optimized for classification, not profit")
        print(f"   → Recommendation: Use regression target (returns) instead of binary")
        print()

def recommend_actions(meta):
    """Provide concrete action items."""
    print('='*60)
    print('RECOMMENDED ACTIONS (Prioritized)')
    print('='*60)
    
    test = meta.get('metrics', {}).get('test', {})
    precision = test.get('precision', 0)
    
    print("1. [IMMEDIATE] Test higher thresholds:")
    print("   $ python analysis/test_thresholds.py --thresholds 0.75,0.80,0.85,0.90")
    print()
    
    print("2. [QUICK WIN] Add volatility filter:")
    print("   → Only trade when ATR > 0.5% (filters low-vol noise)")
    print("   $ python analysis/add_atr_filter.py")
    print()
    
    print("3. [MEDIUM] Retrain with regression target:")
    print("   → Change target from binary to actual returns")
    print("   → Objective: 'reg:squarederror' instead of 'binary:logistic'")
    print()
    
    if precision < 0.35:
        print("4. [LONG-TERM] Address class imbalance:")
        print("   → Option A: Use scale_pos_weight in XGBoost")
        print("   → Option B: SMOTE oversampling")
        print("   → Option C: Focal loss")
        print()

def main():
    # Load metadata
    meta = load_metadata()
    
    # Run analysis
    analyze_model_metadata(meta)
    diagnose_issues(meta)
    recommend_actions(meta)
    
    print('='*60)
    print('NEXT STEP: Run threshold analysis')
    print('='*60)
    print("$ python analysis/test_thresholds.py")

if __name__ == '__main__':
    main()
