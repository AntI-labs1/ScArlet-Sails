"""
Generate predictions on test data for threshold analysis.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import json
from pathlib import Path

def main():
    print('='*60)
    print('GENERATING PREDICTIONS FOR THRESHOLD ANALYSIS')
    print('='*60)
    
    # 1. Load model
    model_path = Path('models/xgboost_v3_btc_15m.json')
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return
    
    model = xgb.Booster()
    model.load_model(str(model_path))
    print(f"✅ Model loaded: {model_path.name}")
    
    # 2. Load metadata to get feature names
    meta_path = Path('models/xgboost_v3_btc_15m_metadata.json')
    with open(meta_path) as f:
        meta = json.load(f)
    
    feature_names = meta.get('feature_names', [])
    print(f"✅ Features: {len(feature_names)}")
    
    # 3. Load data
    data_path = Path('data/features/BTC_USDT_15m_features.parquet')
    if not data_path.exists():
        print(f"ERROR: Data not found at {data_path}")
        return
    
    df = pd.read_parquet(data_path)
    print(f"✅ Data loaded: {len(df):,} rows")
    
    # 4. Prepare features
    # Filter to only features that exist in data
    available_features = [f for f in feature_names if f in df.columns]
    missing_features = [f for f in feature_names if f not in df.columns]
    
    if missing_features:
        print(f"⚠️ Missing {len(missing_features)} features: {missing_features[:5]}...")
    
    print(f"✅ Using {len(available_features)} features")
    
    X = df[available_features].copy()
    
    # 5. Calculate returns for evaluation (4-bar forward return)
    horizon = meta.get('horizon', 4)
    df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
    
    # Fee threshold (from metadata or default)
    fee = 0.001  # 0.1% round-trip
    df['target'] = (df['future_return'] > fee).astype(int)
    
    # 6. Handle NaN
    valid_mask = X.notna().all(axis=1) & df['future_return'].notna()
    X_valid = X[valid_mask]
    returns_valid = df.loc[valid_mask, 'future_return']
    target_valid = df.loc[valid_mask, 'target']
    
    print(f"✅ Valid samples: {len(X_valid):,}")
    
    # 7. Split into train/val/test (same as training)
    # Use last 30% as test (matches metadata: 63.6k test samples)
    n = len(X_valid)
    test_start = int(n * 0.7)
    
    X_test = X_valid.iloc[test_start:]
    returns_test = returns_valid.iloc[test_start:]
    target_test = target_valid.iloc[test_start:]
    
    print(f"✅ Test set: {len(X_test):,} samples")
    
    # 8. Generate predictions
    dtest = xgb.DMatrix(X_test, feature_names=available_features)
    y_pred = model.predict(dtest)
    
    print(f"✅ Predictions generated")
    print(f"   Min: {y_pred.min():.4f}")
    print(f"   Max: {y_pred.max():.4f}")
    print(f"   Mean: {y_pred.mean():.4f}")
    
    # 9. Create output dataframe
    output = pd.DataFrame({
        'y_true': target_test.values,
        'y_pred': y_pred,
        'returns': returns_test.values
    })
    
    # 10. Save
    output_path = Path('models/xgboost_v3_btc_15m_predictions.parquet')
    output.to_parquet(output_path)
    print(f"✅ Saved: {output_path}")
    
    # 11. Quick stats
    print()
    print('='*60)
    print('QUICK ANALYSIS')
    print('='*60)
    
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        signals = (y_pred > threshold).astype(int)
        n_trades = signals.sum()
        if n_trades > 0:
            trade_returns = returns_test.values[signals == 1]
            win_rate = (trade_returns > 0).mean() * 100
            mean_ret = trade_returns.mean() * 100
            print(f"Threshold {threshold}: {n_trades:,} trades, WR={win_rate:.1f}%, Mean={mean_ret:.3f}%")
        else:
            print(f"Threshold {threshold}: 0 trades")

if __name__ == '__main__':
    main()