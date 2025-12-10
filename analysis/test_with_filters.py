"""
Test trading strategy with ATR volatility filter.
Only trade when market volatility is above minimum threshold.
"""

import pandas as pd
import numpy as np
import talib

def add_atr_filter(df, atr_period=14, min_atr_pct=0.005):
    """
    Add ATR-based volatility filter.
    
    Args:
        df: DataFrame with OHLCV data
        atr_period: ATR calculation period
        min_atr_pct: Minimum ATR as % of price (default 0.5%)
    """
    # Calculate ATR
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)
    df['atr_pct'] = df['atr'] / df['close']
    
    # Volatility filter
    df['vol_filter'] = (df['atr_pct'] > min_atr_pct).astype(int)
    
    return df

def backtest_with_filters(
    y_pred,
    returns,
    vol_filter,
    threshold=0.85,
    min_atr_pct=0.005
):
    """Run backtest with probability threshold AND volatility filter."""
    
    # Combined filter
    signals = ((y_pred > threshold) & (vol_filter == 1)).astype(int)
    
    # Calculate returns
    strategy_returns = signals * returns
    
    # Metrics
    n_trades = signals.sum()
    if n_trades == 0:
        print("⚠️ No trades with these filters!")
        return None
    
    total_return = strategy_returns.sum()
    mean_return = strategy_returns[signals == 1].mean()
    sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(252 * 24)
    
    wins = (strategy_returns > 0).sum()
    win_rate = wins / n_trades * 100
    
    # Max drawdown
    cum_returns = (1 + strategy_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min() * 100
    
    return {
        'n_trades': int(n_trades),
        'total_return': total_return,
        'mean_return': mean_return,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'max_drawdown_pct': max_dd
    }

def main():
    print('='*60)
    print('BACKTEST WITH FILTERS')
    print('='*60)
    print()
    
    # Load data (you need to implement data loading)
    print("⚠️ TODO: Load your test data here")
    print("   df = pd.read_parquet('data/features/BTC_USDT_15m_features.parquet')")
    print("   Add ATR and run backtest")
