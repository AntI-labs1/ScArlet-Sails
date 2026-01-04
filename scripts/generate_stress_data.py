import pandas as pd
import numpy as np
from pathlib import Path
import os

def generate_crypto_data(symbol="BTC", days=365):
    print(f"⚡️ Generating stress data for {symbol}...")
    
    # Setup directories
    raw_dir = Path("data/raw")
    feat_dir = Path("data/features")
    raw_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    # Time index (15m)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days*96, freq="15min")
    n = len(dates)

    # Random walk with regime shifts (Normal -> Boom -> Crash)
    np.random.seed(42)
    returns = np.random.normal(0, 0.002, n)
    
    # Inject Volatility Shock (Last 20% of data = CRISIS)
    crisis_idx = int(n * 0.8)
    returns[crisis_idx:] *= 5.0  # 5x volatility spike
    
    # Inject Trend (Boom then Crash)
    trend = np.linspace(0, 0.0001, n)
    trend[crisis_idx:] = -0.0005 # Hard crash
    
    price_path = 50000 * np.exp(np.cumsum(returns + trend))
    
    # OHLCV generation
    df = pd.DataFrame(index=dates)
    df['close'] = price_path
    df['open'] = df['close'].shift(1).fillna(price_path[0])
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.005, n)))
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.005, n)))
    df['volume'] = np.random.lognormal(10, 1, n)
    
    # Volume spike during crisis
    df.iloc[crisis_idx:, 4] *= 3.0

    # Save RAW
    raw_path = raw_dir / f"{symbol}_USDT_15m.parquet"
    df.to_parquet(raw_path)
    print(f"✅ Saved RAW: {raw_path}")

    # Generate Features (Simplified for test)
    # In real system, feature_engine does this. Here we mock basic ones.
    feat_df = df.copy()
    feat_df['returns'] = feat_df['close'].pct_change()
    feat_df['volatility'] = feat_df['returns'].rolling(20).std()
    feat_df['rsi'] = 50 + np.random.normal(0, 10, n) # Mock RSI
    
    feat_path = feat_dir / f"{symbol}_USDT_15m_features.parquet"
    feat_df.to_parquet(feat_path)
    print(f"✅ Saved FEATURES: {feat_path}")

if __name__ == "__main__":
    generate_crypto_data("BTC")
    generate_crypto_data("ETH")
    generate_crypto_data("SOL")