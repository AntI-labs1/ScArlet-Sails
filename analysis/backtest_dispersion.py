"""
Day 4: Backtest comparing fixed vs dispersion-based position sizing.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.rolling_dispersion import RollingDispersionCalculator


def load_predictions():
    """Load predictions from Day 2 analysis."""
    path = Path('models/xgboost_v3_btc_15m_predictions.parquet')
    return pd.read_parquet(path)


def simulate_signals(predictions, threshold=0.70):
    """Generate P_rb, P_ml, P_hyb signals for simulation."""
    n = len(predictions)
    np.random.seed(42)
    
    # P_ml = actual model predictions
    p_ml = predictions['y_pred'].values
    
    # P_rb = simulated rule-based (correlated but different)
    noise_rb = np.random.normal(0, 0.08, n)
    p_rb = np.clip(p_ml + noise_rb, 0, 1)
    
    # P_hyb = weighted average with noise
    noise_hyb = np.random.normal(0, 0.05, n)
    p_hyb = np.clip(0.6 * p_ml + 0.4 * p_rb + noise_hyb, 0, 1)
    
    return p_rb, p_ml, p_hyb


def backtest_fixed_sizing(predictions, threshold, fixed_size=1.0):
    """Backtest with fixed position sizing."""
    signals = (predictions['y_pred'] > threshold).astype(int)
    returns = predictions['returns'].values
    
    strategy_returns = signals * returns * fixed_size
    
    return calculate_metrics(strategy_returns, signals, "Fixed")


def backtest_dispersion_sizing(predictions, threshold, p_rb, p_ml, p_hyb):
    """Backtest with dispersion-based position sizing."""
    calc = RollingDispersionCalculator(window=100)
    
    signals = (predictions['y_pred'] > threshold).astype(int)
    returns = predictions['returns'].values
    
    position_sizes = []
    for i in range(len(predictions)):
        state = calc.update(
            p_rb=p_rb[i] if signals[i] else None,
            p_ml=p_ml[i] if signals[i] else None,
            p_hyb=p_hyb[i] if signals[i] else None,
        )
        
        # Base size from agreement
        values = [v for v in [p_rb[i], p_ml[i], p_hyb[i]] if v is not None]
        if len(values) >= 2:
            spread = max(values) - min(values)
            agreement = 1.0 - min(spread, 1.0)
        else:
            agreement = 1.0
        
        base_size = 0.25 + 0.75 * agreement
        adjusted_size = base_size * state.confidence_multiplier
        position_sizes.append(np.clip(adjusted_size, 0.1, 1.5))
    
    position_sizes = np.array(position_sizes)
    strategy_returns = signals * returns * position_sizes
    
    return calculate_metrics(strategy_returns, signals, "Dispersion", position_sizes)


def calculate_metrics(strategy_returns, signals, name, position_sizes=None):
    """Calculate comprehensive metrics."""
    n_trades = signals.sum()
    
    if n_trades == 0:
        return {"name": name, "n_trades": 0}
    
    total_return = strategy_returns.sum()
    mean_return = strategy_returns[signals == 1].mean()
    
    # Sharpe (annualized for 15m bars)
    sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(252 * 24 * 4)
    
    # Win rate
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
    
    # Calmar ratio
    calmar = (total_return / abs(max_dd / 100)) if max_dd != 0 else np.inf
    
    result = {
        "name": name,
        "n_trades": int(n_trades),
        "total_return_pct": total_return * 100,
        "mean_return_pct": mean_return * 100,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "max_dd_pct": max_dd,
        "profit_factor": profit_factor,
        "calmar": calmar,
    }
    
    if position_sizes is not None:
        result["avg_position"] = position_sizes[signals == 1].mean()
        result["min_position"] = position_sizes[signals == 1].min()
        result["max_position"] = position_sizes[signals == 1].max()
    
    return result


def main():
    print("=" * 70)
    print("DAY 4: DISPERSION SIZING BACKTEST")
    print("=" * 70)
    
    # Load data
    predictions = load_predictions()
    print(f"Loaded {len(predictions):,} predictions")
    
    threshold = 0.70
    print(f"Threshold: {threshold}")
    
    # Generate simulated strategy signals
    p_rb, p_ml, p_hyb = simulate_signals(predictions, threshold)
    print(f"Generated P_rb, P_ml, P_hyb signals")
    
    print()
    print("-" * 70)
    print("BACKTEST RESULTS")
    print("-" * 70)
    
    # Fixed sizing backtest
    fixed_results = backtest_fixed_sizing(predictions, threshold, fixed_size=1.0)
    
    # Dispersion sizing backtest
    disp_results = backtest_dispersion_sizing(predictions, threshold, p_rb, p_ml, p_hyb)
    
    # Print comparison
    print()
    print(f"{'Metric':<25} {'Fixed Size':>15} {'Dispersion':>15} {'Diff':>12}")
    print("-" * 70)
    
    metrics = ['n_trades', 'total_return_pct', 'sharpe', 'win_rate', 'max_dd_pct', 'profit_factor', 'calmar']
    
    for m in metrics:
        fixed_val = fixed_results.get(m, 0)
        disp_val = disp_results.get(m, 0)
        
        if isinstance(fixed_val, float):
            diff = disp_val - fixed_val
            diff_str = f"{diff:+.2f}"
            print(f"{m:<25} {fixed_val:>15.2f} {disp_val:>15.2f} {diff_str:>12}")
        else:
            print(f"{m:<25} {fixed_val:>15} {disp_val:>15} {'':>12}")
    
    # Position size stats (dispersion only)
    if 'avg_position' in disp_results:
        print()
        print("Position Sizing Statistics (Dispersion):")
        print(f"  Avg: {disp_results['avg_position']:.3f}")
        print(f"  Min: {disp_results['min_position']:.3f}")
        print(f"  Max: {disp_results['max_position']:.3f}")
    
    # Save results
    results = {
        "fixed": fixed_results,
        "dispersion": disp_results,
        "threshold": threshold,
    }
    
    with open('analysis/dispersion_backtest_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    sharpe_diff = disp_results['sharpe'] - fixed_results['sharpe']
    dd_diff = disp_results['max_dd_pct'] - fixed_results['max_dd_pct']
    
    if sharpe_diff > 0 and dd_diff > 0:  # Higher Sharpe, less negative DD
        print("✅ Dispersion sizing IMPROVES risk-adjusted returns")
    elif sharpe_diff > 0:
        print("⚠️ Dispersion sizing improves Sharpe but increases drawdown")
    elif dd_diff > 0:
        print("⚠️ Dispersion sizing reduces drawdown but lowers Sharpe")
    else:
        print("❌ Dispersion sizing does not improve metrics")
    
    print(f"\nResults saved to: analysis/dispersion_backtest_results.json")


if __name__ == "__main__":
    main()