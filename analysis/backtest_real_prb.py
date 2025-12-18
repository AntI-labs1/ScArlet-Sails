"""
Day 10: Integration Backtest with REAL P_rb.
No more simulation - using actual RuleBasedStrategy.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.rule_based_v2 import RuleBasedStrategy
from core.rolling_dispersion import RollingDispersionCalculator
from core.regime_detector import RegimeDetector, REGIME_POSITION_MULTIPLIER


def normalize_p_rb(p_rb: float) -> float:
    """Convert P_rb decision score to probability [0,1]."""
    return 1 / (1 + np.exp(-10 * (p_rb - 0.05)))


def calculate_metrics(returns, name):
    """Calculate performance metrics."""
    if len(returns) == 0 or returns.std() == 0:
        return {'name': name, 'sharpe': 0, 'calmar': 0, 'max_dd': 0, 'total_return': 0}
    
    total_return = ((1 + returns).prod() - 1) * 100
    sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24 * 4)
    
    cum = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max
    max_dd = dd.min() * 100
    
    calmar = total_return / abs(max_dd) if max_dd != 0 else 0
    
    return {
        'name': name,
        'total_return': round(total_return, 2),
        'sharpe': round(sharpe, 3),
        'max_dd': round(max_dd, 2),
        'calmar': round(calmar, 3),
    }


def main():
    print("=" * 70)
    print("DAY 10: BACKTEST WITH REAL P_rb")
    print("=" * 70)
    
    # Load data
    pred = pd.read_parquet('models/xgboost_v3_btc_15m_predictions.parquet')
    feat = pd.read_parquet('data/features/BTC_USDT_15m_features.parquet')
    
    # Align - use last N bars matching predictions
    n = len(pred)
    feat = feat.iloc[-n:].reset_index(drop=True)
    pred = pred.reset_index(drop=True)
    
    p_ml = pred['y_pred'].values
    returns = pred['returns'].values
    
    print(f"Data: {n:,} bars")
    
    # Generate real P_rb
    print("\nGenerating REAL P_rb signals...")
    rb_strategy = RuleBasedStrategy()
    
    # Use OHLCV from features
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    df_ohlcv = feat[ohlcv_cols].copy()
    
    # Generate signals (returns DataFrame with DatetimeIndex)
    rb_results = rb_strategy.generate_signals(df_ohlcv)
    
    # Extract P_rb values and align with predictions
    # Reset both to numeric index for alignment
    p_rb_raw = rb_results['P_rb'].values  # numpy array, same length as feat
    
    print(f"P_rb raw: non-nan={np.sum(~np.isnan(p_rb_raw))}/{len(p_rb_raw)}")
    print(f"P_rb raw (valid): min={np.nanmin(p_rb_raw):.4f}, max={np.nanmax(p_rb_raw):.4f}, mean={np.nanmean(p_rb_raw):.4f}")
    
    # Normalize to [0, 1]
    p_rb = np.array([normalize_p_rb(x) if not np.isnan(x) else np.nan for x in p_rb_raw])
    
    print(f"P_rb normalized (valid): min={np.nanmin(p_rb):.4f}, max={np.nanmax(p_rb):.4f}, mean={np.nanmean(p_rb):.4f}")
    
    # Check correlation (only on valid pairs)
    valid_mask = ~np.isnan(p_rb) & ~np.isnan(p_ml)
    if valid_mask.sum() > 100:
        corr = np.corrcoef(p_rb[valid_mask], p_ml[valid_mask])[0, 1]
        print(f"Correlation(P_rb, P_ml): {corr:.3f} (on {valid_mask.sum()} samples)")
    else:
        corr = 0
        print("WARNING: Not enough valid samples for correlation")
    
    # Initialize components
    dispersion_calc = RollingDispersionCalculator(window=100)
    regime_detector = RegimeDetector()
    
    # Configurations to test
    configs = [
        ("P_ml only (threshold 0.70)", True, False, False, False),
        ("P_ml + Regime", True, True, False, False),
        ("P_ml + Regime + Dispersion (real P_rb)", True, True, True, False),
        ("Full: P_hyb + Regime + Dispersion", True, True, True, True),
    ]
    
    results = []
    
    for name, use_threshold, use_regime, use_dispersion, use_hybrid in configs:
        print(f"\nRunning: {name}...")
        
        strategy_returns = np.zeros(n)
        dispersion_calc = RollingDispersionCalculator(window=100)
        regime_detector.reset()
        
        for i in range(100, n):
            # Get probabilities
            curr_p_ml = p_ml[i]
            curr_p_rb = p_rb[i] if not np.isnan(p_rb[i]) else 0.5
            
            # Hybrid probability
            if use_hybrid:
                p_signal = 0.5 * curr_p_rb + 0.5 * curr_p_ml
            else:
                p_signal = curr_p_ml
            
            # Threshold filter
            if use_threshold and p_signal < 0.70:
                continue
            
            position = 1.0
            
            # Regime adjustment
            if use_regime and 'open' in feat.columns:
                window_start = max(0, i - 100)
                ohlcv = feat.iloc[window_start:i+1][ohlcv_cols]
                if len(ohlcv) >= 20:
                    regime_state = regime_detector.detect(ohlcv)
                    position *= REGIME_POSITION_MULTIPLIER.get(regime_state.regime, 1.0)
            
            # Dispersion adjustment (with REAL P_rb!)
            if use_dispersion:
                p_hyb = 0.5 * curr_p_rb + 0.5 * curr_p_ml
                disp_state = dispersion_calc.update(curr_p_rb, curr_p_ml, p_hyb)
                if disp_state:
                    position *= disp_state.confidence_multiplier
            
            position = np.clip(position, 0.0, 1.5)
            strategy_returns[i] = returns[i] * position
        
        metrics = calculate_metrics(strategy_returns, name)
        results.append(metrics)
        print(f"  Sharpe: {metrics['sharpe']:.2f}, Calmar: {metrics['calmar']:.2f}, DD: {metrics['max_dd']:.1f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS: REAL P_rb INTEGRATION")
    print("=" * 70)
    
    print("\n{:<45} {:>10} {:>10} {:>10}".format("Configuration", "Sharpe", "Calmar", "MaxDD%"))
    print("-" * 75)
    
    for r in results:
        print("{:<45} {:>10.2f} {:>10.2f} {:>10.1f}".format(
            r['name'][:45], r['sharpe'], r['calmar'], r['max_dd']
        ))
    
    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'correlation_p_rb_p_ml': float(corr),
        'results': results,
    }
    
    with open('analysis/backtest_real_prb_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Saved: analysis/backtest_real_prb_results.json")


if __name__ == "__main__":
    main()