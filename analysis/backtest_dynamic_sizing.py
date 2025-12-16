"""
Day 6: Backtest comparison - Static vs Dynamic Position Sizing.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.regime_detector import RegimeDetector
from core.dynamic_position_sizer import DynamicPositionSizer, PositionSizingInput
from core.rolling_dispersion import RollingDispersionCalculator


def load_data():
    """Load and align predictions with features."""
    pred_path = Path('models/xgboost_v3_btc_15m_predictions.parquet')
    feat_path = Path('data/features/BTC_USDT_15m_features.parquet')
    
    df_pred = pd.read_parquet(pred_path)
    df_feat = pd.read_parquet(feat_path)
    
    # Predictions are from test set (last 30% of data)
    n_pred = len(df_pred)
    n_feat = len(df_feat)
    
    # Align: take last n_pred rows from features
    df_feat_aligned = df_feat.iloc[-n_pred:].reset_index(drop=True)
    df_pred = df_pred.reset_index(drop=True)
    
    print(f"Predictions: {n_pred:,} rows")
    print(f"Features aligned: {len(df_feat_aligned):,} rows")
    
    return df_pred, df_feat_aligned


def calculate_metrics(returns, positions, name="Strategy"):
    """Calculate performance metrics."""
    strategy_returns = returns * positions
    n_active = (positions > 0).sum()
    
    if n_active == 0:
        return {
            'name': name, 'total_return': 0, 'sharpe': 0,
            'max_dd': 0, 'win_rate': 0, 'calmar': 0,
            'avg_position': 0, 'n_trades': 0,
        }
    
    # Total return
    total_return = (1 + strategy_returns).prod() - 1
    
    # Sharpe
    std = strategy_returns.std()
    sharpe = strategy_returns.mean() / std * np.sqrt(252 * 24 * 4) if std > 0 else 0
    
    # Max drawdown
    cum_returns = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()
    
    # Win rate
    active = positions > 0
    wins = ((strategy_returns > 0) & active).sum()
    losses = ((strategy_returns < 0) & active).sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    
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
        'n_trades': int(active.sum()),
    }


def run_backtest(df_pred, df_feat, threshold=0.70):
    """Run backtest comparing static vs dynamic sizing."""
    
    # Initialize components
    regime_detector = RegimeDetector()
    dispersion_calc = RollingDispersionCalculator(window=100)
    position_sizer = DynamicPositionSizer()
    
    n = len(df_pred)
    
    # Get data
    p_ml = df_pred['y_pred'].values
    returns = df_pred['returns'].values
    
    # Simulate P_rb from RSI if available
    if 'rsi_14' in df_feat.columns:
        rsi = df_feat['rsi_14'].values
        p_rb = np.where(rsi < 30, 0.7, np.where(rsi > 70, 0.3, 0.5))
    else:
        np.random.seed(42)
        p_rb = np.clip(p_ml + np.random.randn(n) * 0.1, 0, 1)
    
    # P_hyb
    p_hyb = 0.6 * p_ml + 0.4 * p_rb
    agreement = 1 - np.abs(p_rb - p_ml)
    
    # Arrays for results
    static_positions = np.zeros(n)
    dynamic_positions = np.zeros(n)
    
    # Drawdown tracking
    cumulative = 1.0
    peak = 1.0
    current_dd = 0.0
    
    print(f"Running backtest on {n:,} bars...")
    
    for i in range(n):
        if p_hyb[i] < threshold:
            static_positions[i] = 0
            dynamic_positions[i] = 0
        else:
            # Static
            static_positions[i] = 1.0
            
            # Dynamic
            disp_state = dispersion_calc.update(p_rb[i], p_ml[i], p_hyb[i])
            
            # Regime
            regime_state = None
            if i >= 50 and 'open' in df_feat.columns:
                window = df_feat.iloc[max(0, i-100):i+1][['open', 'high', 'low', 'close', 'volume']]
                if len(window) >= 20:
                    regime_state = regime_detector.detect(window)
            
            inputs = PositionSizingInput(
                p_hyb=p_hyb[i],
                agreement=agreement[i],
                dispersion_state=disp_state,
                regime_state=regime_state,
                current_drawdown=current_dd,
            )
            output = position_sizer.calculate(inputs)
            dynamic_positions[i] = output.position_size
        
        # Update drawdown
        if i > 0:
            ret = returns[i-1] * dynamic_positions[i-1]
            cumulative *= (1 + ret)
            peak = max(peak, cumulative)
            current_dd = (cumulative - peak) / peak if peak > 0 else 0
        
        if (i + 1) % 20000 == 0:
            print(f"  Processed {i+1:,}/{n:,}")
    
    print("Calculating metrics...")
    
    static_metrics = calculate_metrics(returns, static_positions, "Static (1.0)")
    dynamic_metrics = calculate_metrics(returns, dynamic_positions, "Dynamic")
    
    # Position stats
    dyn_active = dynamic_positions[dynamic_positions > 0]
    
    return {
        'static': static_metrics,
        'dynamic': dynamic_metrics,
        'positions': {
            'dynamic_mean': dyn_active.mean() if len(dyn_active) > 0 else 0,
            'dynamic_min': dyn_active.min() if len(dyn_active) > 0 else 0,
            'dynamic_max': dyn_active.max() if len(dyn_active) > 0 else 0,
        }
    }


def main():
    print("=" * 60)
    print("DAY 6: STATIC vs DYNAMIC POSITION SIZING BACKTEST")
    print("=" * 60)
    
    df_pred, df_feat = load_data()
    
    print("\n" + "-" * 60)
    results = run_backtest(df_pred, df_feat, threshold=0.70)
    
    # Display
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print("\n{:<20} {:>12} {:>12}".format("Metric", "Static", "Dynamic"))
    print("-" * 44)
    
    for metric in ['total_return', 'sharpe', 'max_dd', 'win_rate', 'calmar', 'avg_position', 'n_trades']:
        s = results['static'][metric]
        d = results['dynamic'][metric]
        if metric == 'n_trades':
            print(f"{metric:<20} {int(s):>12,} {int(d):>12,}")
        elif metric in ['total_return', 'max_dd', 'win_rate']:
            print(f"{metric:<20} {s:>11.1f}% {d:>11.1f}%")
        else:
            print(f"{metric:<20} {s:>12.2f} {d:>12.2f}")
    
    print("\nDynamic Position Distribution:")
    print(f"  Mean: {results['positions']['dynamic_mean']:.3f}")
    print(f"  Min:  {results['positions']['dynamic_min']:.3f}")
    print(f"  Max:  {results['positions']['dynamic_max']:.3f}")
    
    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'threshold': 0.70,
        **results
    }
    
    with open('analysis/dynamic_sizing_backtest.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✅ Saved: analysis/dynamic_sizing_backtest.json")


if __name__ == "__main__":
    main()