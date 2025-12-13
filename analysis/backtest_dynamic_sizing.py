"""
Day 6: Backtest comparison - Static vs Dynamic Position Sizing.
Validates that dynamic sizing improves risk-adjusted returns.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.regime_detector import RegimeDetector, RegimeState
from core.dynamic_position_sizer import DynamicPositionSizer, PositionSizingInput
from core.rolling_dispersion import RollingDispersionCalculator


def load_predictions():
    """Load model predictions."""
    path = Path('models/xgboost_v3_btc_15m_predictions.parquet')
    if not path.exists():
        raise FileNotFoundError(f"Predictions not found: {path}")
    return pd.read_parquet(path)


def load_features():
    """Load feature data for regime detection."""
    path = Path('data/features/BTC_USDT_15m_features.parquet')
    if not path.exists():
        raise FileNotFoundError(f"Features not found: {path}")
    return pd.read_parquet(path)


def simulate_signals(df_pred, df_feat, threshold=0.70):
    """
    Simulate strategy signals with P_rb approximation.
    """
    # Align indices
    common_idx = df_pred.index.intersection(df_feat.index)
    df_pred = df_pred.loc[common_idx]
    df_feat = df_feat.loc[common_idx]
    
    # P_ml from model
    p_ml = df_pred['y_pred_proba'].values
    
    # Approximate P_rb using RSI (if available)
    if 'rsi_14' in df_feat.columns:
        rsi = df_feat['rsi_14'].values
        p_rb = np.where(rsi < 30, 0.7, np.where(rsi > 70, 0.3, 0.5))
    else:
        # Random correlation with P_ml
        np.random.seed(42)
        noise = np.random.randn(len(p_ml)) * 0.1
        p_rb = np.clip(p_ml + noise, 0, 1)
    
    # P_hyb = weighted average
    p_hyb = 0.6 * p_ml + 0.4 * p_rb
    
    # Agreement
    agreement = 1 - np.abs(p_rb - p_ml)
    
    return pd.DataFrame({
        'p_rb': p_rb,
        'p_ml': p_ml,
        'p_hyb': p_hyb,
        'agreement': agreement,
        'y_true': df_pred['y_true'].values,
        'future_return': df_pred['future_return'].values if 'future_return' in df_pred else np.zeros(len(p_ml)),
    }, index=common_idx), df_feat


def calculate_metrics(returns, positions, name="Strategy"):
    """Calculate performance metrics."""
    strategy_returns = returns * positions
    
    # Basic metrics
    total_return = (1 + strategy_returns).prod() - 1
    
    # Sharpe
    if strategy_returns.std() > 0:
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252 * 24 * 4)
    else:
        sharpe = 0
    
    # Max drawdown
    cum_returns = (1 + strategy_returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()
    
    # Win rate (only on active trades)
    active = positions > 0
    if active.sum() > 0:
        wins = ((strategy_returns > 0) & active).sum()
        losses = ((strategy_returns < 0) & active).sum()
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    else:
        win_rate = 0
    
    # Calmar
    calmar = total_return / abs(max_dd) if max_dd != 0 else 0
    
    return {
        'name': name,
        'total_return': total_return * 100,
        'sharpe': sharpe,
        'max_dd': max_dd * 100,
        'win_rate': win_rate * 100,
        'calmar': calmar,
        'avg_position': positions[active].mean() if active.sum() > 0 else 0,
        'n_trades': active.sum(),
    }


def run_backtest(signals_df, features_df, threshold=0.70):
    """
    Run backtest comparing static vs dynamic sizing.
    """
    # Initialize components
    regime_detector = RegimeDetector()
    dispersion_calc = RollingDispersionCalculator(window=100)
    position_sizer = DynamicPositionSizer()
    
    n = len(signals_df)
    
    # Arrays for results
    static_positions = np.zeros(n)
    dynamic_positions = np.zeros(n)
    returns = signals_df['future_return'].values
    
    # Simulate drawdown tracking
    cumulative_return = 1.0
    peak = 1.0
    current_dd = 0.0
    
    print(f"Running backtest on {n:,} bars...")
    
    for i in range(n):
        p_ml = signals_df['p_ml'].iloc[i]
        p_rb = signals_df['p_rb'].iloc[i]
        p_hyb = signals_df['p_hyb'].iloc[i]
        agreement = signals_df['agreement'].iloc[i]
        
        # Only trade if above threshold
        if p_hyb < threshold:
            static_positions[i] = 0
            dynamic_positions[i] = 0
        else:
            # Static: fixed position
            static_positions[i] = 1.0
            
            # Dynamic: use all factors
            # Update dispersion
            disp_state = dispersion_calc.update(p_rb, p_ml, p_hyb)
            
            # Get regime (use rolling window of features)
            if i >= 50:
                window_start = max(0, i - 100)
                ohlcv_window = features_df.iloc[window_start:i+1][['open', 'high', 'low', 'close', 'volume']]
                regime_state = regime_detector.detect(ohlcv_window)
            else:
                regime_state = None
            
            # Calculate dynamic position
            inputs = PositionSizingInput(
                p_hyb=p_hyb,
                agreement=agreement,
                dispersion_state=disp_state,
                regime_state=regime_state,
                current_drawdown=current_dd,
            )
            output = position_sizer.calculate(inputs)
            dynamic_positions[i] = output.position_size
        
        # Update drawdown tracking
        if i > 0:
            ret = returns[i-1] * dynamic_positions[i-1]
            cumulative_return *= (1 + ret)
            peak = max(peak, cumulative_return)
            current_dd = (cumulative_return - peak) / peak
        
        # Progress
        if (i + 1) % 20000 == 0:
            print(f"  Processed {i+1:,}/{n:,} bars")
    
    print("Calculating metrics...")
    
    # Calculate metrics for both
    static_metrics = calculate_metrics(returns, static_positions, "Static (1.0)")
    dynamic_metrics = calculate_metrics(returns, dynamic_positions, "Dynamic")
    
    return {
        'static': static_metrics,
        'dynamic': dynamic_metrics,
        'positions': {
            'static_mean': static_positions[static_positions > 0].mean() if (static_positions > 0).sum() > 0 else 0,
            'dynamic_mean': dynamic_positions[dynamic_positions > 0].mean() if (dynamic_positions > 0).sum() > 0 else 0,
            'dynamic_min': dynamic_positions[dynamic_positions > 0].min() if (dynamic_positions > 0).sum() > 0 else 0,
            'dynamic_max': dynamic_positions[dynamic_positions > 0].max() if (dynamic_positions > 0).sum() > 0 else 0,
        }
    }


def main():
    print("=" * 60)
    print("DAY 6: STATIC vs DYNAMIC POSITION SIZING BACKTEST")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    df_pred = load_predictions()
    df_feat = load_features()
    print(f"Predictions: {len(df_pred):,} rows")
    print(f"Features: {len(df_feat):,} rows")
    
    # Simulate signals
    print("\nSimulating signals...")
    signals_df, features_df = simulate_signals(df_pred, df_feat)
    print(f"Aligned data: {len(signals_df):,} rows")
    
    # Run backtest
    print("\n" + "-" * 60)
    results = run_backtest(signals_df, features_df, threshold=0.70)
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print("\n{:<20} {:>12} {:>12}".format("Metric", "Static", "Dynamic"))
    print("-" * 44)
    
    metrics = ['total_return', 'sharpe', 'max_dd', 'win_rate', 'calmar', 'avg_position', 'n_trades']
    formats = ['.1f%', '.2f', '.1f%', '.1f%', '.2f', '.3f', ',']
    
    for metric, fmt in zip(metrics, formats):
        static_val = results['static'][metric]
        dynamic_val = results['dynamic'][metric]
        
        if fmt == ',':
            print(f"{metric:<20} {int(static_val):>12,} {int(dynamic_val):>12,}")
        elif '%' in fmt:
            print(f"{metric:<20} {static_val:>11.1f}% {dynamic_val:>11.1f}%")
        else:
            print(f"{metric:<20} {static_val:>12.2f} {dynamic_val:>12.2f}")
    
    # Position stats
    print("\n" + "-" * 44)
    print("POSITION SIZE DISTRIBUTION (Dynamic):")
    print(f"  Mean: {results['positions']['dynamic_mean']:.3f}")
    print(f"  Min:  {results['positions']['dynamic_min']:.3f}")
    print(f"  Max:  {results['positions']['dynamic_max']:.3f}")
    
    # Verdict
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    static = results['static']
    dynamic = results['dynamic']
    
    improvements = []
    if dynamic['sharpe'] > static['sharpe']:
        improvements.append(f"Sharpe: +{(dynamic['sharpe'] - static['sharpe']):.2f}")
    if dynamic['max_dd'] > static['max_dd']:  # Less negative = better
        improvements.append(f"Max DD: {(dynamic['max_dd'] - static['max_dd']):.1f}pp better")
    if dynamic['calmar'] > static['calmar']:
        improvements.append(f"Calmar: +{(dynamic['calmar'] - static['calmar']):.2f}")
    
    if improvements:
        print("✅ Dynamic sizing improvements:")
        for imp in improvements:
            print(f"   - {imp}")
    else:
        print("⚠️ Static sizing performed better in this period")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'threshold': 0.70,
        'static': results['static'],
        'dynamic': results['dynamic'],
        'positions': results['positions'],
    }
    
    with open('analysis/dynamic_sizing_backtest.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results saved to: analysis/dynamic_sizing_backtest.json")


if __name__ == "__main__":
    main()