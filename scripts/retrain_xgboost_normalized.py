#!/usr/bin/env python3
"""
RETRAIN XGBOOST - NORMALIZED FEATURES VERSION
Решает проблему out-of-distribution используя нормализованные features

Вместо absolute prices ($107K) использует:
- Returns (% change)
- Price ratios (close / SMA)
- Normalized indicators

Это позволяет модели работать на любых ценах!
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.xgboost_model import XGBoostModel
from features.multi_timeframe_extractor import MultiTimeframeFeatureExtractor
from sklearn.preprocessing import StandardScaler

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

# Constants
ASSET = "BTC"
TARGET_TF = "15m"
FORWARD_BARS = 96  # 24 hours
PROFIT_THRESHOLD = 0.01  # 1%
MAX_SAMPLES = 50000


class NormalizedMultiTFExtractor_OLD:
    """
    Извлекает НОРМАЛИЗОВАННЫЕ multi-TF features
    Использует returns и ratios вместо absolute prices
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_all_timeframes(self, asset: str) -> Dict[str, pd.DataFrame]:
        """Load all 4 timeframes"""
        timeframes = {}

        for tf in ['15m', '1h', '4h', '1d']:
            file_path = self.data_dir / f"{asset}_USDT_{tf}.parquet"
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")

            df = pd.read_parquet(file_path)
            df.index = pd.to_datetime(df.index)
            timeframes[tf] = df

        return timeframes

    def calculate_normalized_indicators(self, df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
        """
        Calculate NORMALIZED indicators (no absolute prices!)
        """
        df = df.copy()

        # RSI (already 0-100, normalized)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df[f'{prefix}RSI_14'] = 100 - (100 / (1 + rs))

        # EMA & SMA (will use ratios, not absolute values)
        df[f'{prefix}EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df[f'{prefix}EMA_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df[f'{prefix}SMA_50'] = df['close'].rolling(window=50).mean()

        # Bollinger Bands (will use width ratio)
        bb_middle = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df[f'{prefix}BB_width_pct'] = (2 * bb_std / bb_middle)  # Normalized!

        # ATR (normalize by price)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=14).mean()
        df[f'{prefix}ATR_pct'] = atr / df['close']  # Normalized by price!

        # Returns (already normalized!)
        df[f'{prefix}returns_5'] = df['close'].pct_change(5)
        df[f'{prefix}returns_10'] = df['close'].pct_change(10)

        # Volume ratio (normalized)
        df[f'{prefix}volume_ratio_5'] = df['volume'] / df['volume'].rolling(5).mean()
        df[f'{prefix}volume_ratio_10'] = df['volume'] / df['volume'].rolling(10).mean()

        # Price ratios (normalized!)
        df[f'{prefix}price_to_EMA9'] = df['close'] / df[f'{prefix}EMA_9']
        df[f'{prefix}price_to_EMA21'] = df['close'] / df[f'{prefix}EMA_21']
        df[f'{prefix}price_to_SMA50'] = df['close'] / df[f'{prefix}SMA_50']

        return df

    def prepare_multi_timeframe_data(self, asset: str, target_tf: str) -> tuple:
        """Prepare multi-TF data with normalized indicators"""
        print(f"\n📊 Loading all timeframes for {asset}...")

        all_tf = self.load_all_timeframes(asset)
        primary_df = all_tf[target_tf].copy()

        print(f"✅ Loaded data:")
        for tf, df in all_tf.items():
            print(f"   {tf}: {len(df)} bars")

        # Calculate indicators for primary timeframe
        print(f"\n📈 Calculating normalized indicators for {target_tf}...")
        primary_df = self.calculate_normalized_indicators(primary_df, prefix=f'{target_tf}_')

        # Calculate for higher timeframes and resample
        for tf in ['1h', '4h', '1d']:
            if tf == target_tf:
                continue

            print(f"📈 Calculating normalized indicators for {tf}...")
            all_tf[tf] = self.calculate_normalized_indicators(all_tf[tf], prefix=f'{tf}_')

            # Resample to target timeframe
            resampled = all_tf[tf].resample(pd.Timedelta(target_tf)).ffill()
            resampled = resampled.reindex(primary_df.index, method='ffill')

            # Merge selected columns
            for col in [f'{tf}_RSI_14', f'{tf}_returns_5', f'{tf}_price_to_EMA9',
                       f'{tf}_price_to_EMA21', f'{tf}_price_to_SMA50', f'{tf}_ATR_pct']:
                if col in resampled.columns:
                    primary_df[col] = resampled[col]

        return all_tf, primary_df

    def extract_features_at_bar(self, primary_df: pd.DataFrame, bar_index: int) -> np.ndarray:
        """
        Extract NORMALIZED features (31 features)
        """
        if bar_index >= len(primary_df):
            return None

        features = []

        # Primary timeframe (15m) - 13 features
        tf = TARGET_TF
        for col in [f'{tf}_RSI_14', f'{tf}_price_to_EMA9', f'{tf}_price_to_EMA21', f'{tf}_price_to_SMA50',
                   f'{tf}_BB_width_pct', f'{tf}_ATR_pct',
                   f'{tf}_returns_5', f'{tf}_returns_10',
                   f'{tf}_volume_ratio_5', f'{tf}_volume_ratio_10']:
            if col not in primary_df.columns:
                return None
            val = primary_df[col].iloc[bar_index]
            if pd.isna(val):
                return None
            features.append(val)

        # Add EMA/SMA as ratios (additional 3 features for completeness)
        features.append(primary_df[f'{tf}_price_to_EMA9'].iloc[bar_index])  # duplicate but OK
        features.append(primary_df[f'{tf}_price_to_EMA21'].iloc[bar_index])
        features.append(primary_df[f'{tf}_price_to_SMA50'].iloc[bar_index])

        # Higher timeframes (1h, 4h, 1d) - 6 features each = 18 total
        for higher_tf in ['1h', '4h', '1d']:
            for col in [f'{higher_tf}_RSI_14', f'{higher_tf}_returns_5',
                       f'{higher_tf}_price_to_EMA9', f'{higher_tf}_price_to_EMA21',
                       f'{higher_tf}_price_to_SMA50', f'{higher_tf}_ATR_pct']:
                if col not in primary_df.columns:
                    return None
                val = primary_df[col].iloc[bar_index]
                if pd.isna(val):
                    return None
                features.append(val)

        return np.array(features)


def main():
    """Retrain XGBoost with normalized features"""

    print("="*100)
    print("RETRAIN XGBOOST - NORMALIZED FEATURES")
    print("="*100)
    print("\nThis solves out-of-distribution problem by using:")
    print("  - Returns (% change) instead of absolute prices")
    print("  - Price ratios (close/SMA) instead of absolute values")
    print("  - Normalized indicators (ATR/price, BB_width/price)")
    print("\n→ Model will work on ANY price level ($100 or $1,000,000)!")

    # Load data
    print(f"\n{'='*100}")
    print(f"STEP 1: Loading multi-timeframe data")
    print(f"{'='*100}")

    extractor = MultiTimeframeFeatureExtractor(data_dir=str(DATA_DIR))
    all_tf, primary_df = extractor.prepare_multi_timeframe_data(ASSET, TARGET_TF)

    print(f"✅ Loaded {len(primary_df)} bars")
    print(f"✅ Prepared {primary_df.shape[1]} columns")

    # Create labels
    print(f"\n{'='*100}")
    print(f"STEP 2: Creating labels")
    print(f"{'='*100}")

    close = primary_df['close'].values
    future_close = np.roll(close, -FORWARD_BARS)
    primary_df['future_return'] = (future_close - close) / close
    primary_df['label'] = (primary_df['future_return'] > PROFIT_THRESHOLD).astype(int)

    # Remove last FORWARD_BARS (no future data)
    primary_df = primary_df.iloc[:-FORWARD_BARS].copy()

    total_samples = len(primary_df)
    up_samples = (primary_df['label'] == 1).sum()
    down_samples = (primary_df['label'] == 0).sum()

    print(f"Total samples: {total_samples:,}")
    print(f"UP (1):        {up_samples:,} ({up_samples/total_samples*100:.1f}%)")
    print(f"DOWN (0):      {down_samples:,} ({down_samples/total_samples*100:.1f}%)")

    # Extract features
    print(f"\n{'='*100}")
    print(f"STEP 3: Extracting NORMALIZED features")
    print(f"{'='*100}")

    # Sample indices
    if total_samples > MAX_SAMPLES:
        print(f"⚠️  Limiting to {MAX_SAMPLES:,} samples (from {total_samples:,})")
        sample_indices = np.random.choice(total_samples, MAX_SAMPLES, replace=False)
        sample_indices.sort()
    else:
        sample_indices = np.arange(total_samples)

    print(f"Extracting features for {len(sample_indices):,} samples...")

    X_list = []
    y_list = []

    for i, bar_idx in enumerate(sample_indices):
        if i % 5000 == 0:
            print(f"   Progress: {i:,} / {len(sample_indices):,} ({i/len(sample_indices)*100:.1f}%)")

        features = extractor.extract_features_at_bar(all_tf, TARGET_TF, bar_idx)

        if features is not None:
            X_list.append(features)
            y_list.append(primary_df['label'].iloc[bar_idx])

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"\n✅ Extracted features for {len(X):,} samples")
    print(f"   Feature shape: {X.shape}")
    print(f"   Label shape: {y.shape}")

    # Distribution
    up_count = (y == 1).sum()
    down_count = (y == 0).sum()

    print(f"\nFinal distribution:")
    print(f"   UP (1):   {up_count:,} ({up_count/len(y)*100:.1f}%)")
    print(f"   DOWN (0): {down_count:,} ({down_count/len(y)*100:.1f}%)")

    # Split
    print(f"\n{'='*100}")
    print(f"STEP 4: Scaling NORMALIZED features")
    print(f"{'='*100}")

    split_idx = int(len(X) * 0.8)

    X_train_raw = X[:split_idx]
    y_train = y[:split_idx]
    X_test_raw = X[split_idx:]
    y_test = y[split_idx:]

    print(f"Train: {len(X_train_raw):,} samples")
    print(f"Test:  {len(X_test_raw):,} samples")

    # Scale (still useful even with normalized features!)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print(f"✅ Features scaled")
    print(f"   Mean: {X_train.mean():.4f}")
    print(f"   Std:  {X_train.std():.4f}")

    # Train
    print(f"\n{'='*100}")
    print(f"STEP 5: Training XGBoost")
    print(f"{'='*100}")

    scale_pos_weight = down_count / up_count
    print(f"Class weight ratio: {scale_pos_weight:.2f}")

    # Validation split
    val_split_idx = int(len(X_train) * 0.8)
    X_train_fit = X_train[:val_split_idx]
    y_train_fit = y_train[:val_split_idx]
    X_val = X_train[val_split_idx:]
    y_val = y_train[val_split_idx:]

    print(f"\nTrain fit: {len(X_train_fit):,}")
    print(f"Val:       {len(X_val):,}")

    model = XGBoostModel(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight
    )

    print(f"\nTraining...")
    model.fit(X_train_fit, y_train_fit, eval_set=(X_val, y_val))

    print(f"✅ Training complete")

    # Evaluate
    print(f"\n{'='*100}")
    print(f"STEP 6: Evaluation")
    print(f"{'='*100}")

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {accuracy:.4f}")

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['DOWN (0)', 'UP (1)']))

    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"              Predicted")
    print(f"              DOWN    UP")
    print(f"Actual DOWN    {cm[0][0]:<6}  {cm[0][1]:<6}")
    print(f"       UP      {cm[1][0]:<6}  {cm[1][1]:<6}")

    # Save
    print(f"\n{'='*100}")
    print(f"STEP 7: Saving model")
    print(f"{'='*100}")

    model_path = MODELS_DIR / "xgboost_normalized_model.json"
    scaler_path = MODELS_DIR / "xgboost_normalized_scaler.pkl"
    features_path = MODELS_DIR / "xgboost_normalized_features.json"

    model.save(str(model_path))
    joblib.dump(scaler, scaler_path)

    # Feature names
    feature_names = []
    tf = TARGET_TF
    for name in [f'{tf}_RSI_14', f'{tf}_price_to_EMA9', f'{tf}_price_to_EMA21', f'{tf}_price_to_SMA50',
                f'{tf}_BB_width_pct', f'{tf}_ATR_pct',
                f'{tf}_returns_5', f'{tf}_returns_10',
                f'{tf}_volume_ratio_5', f'{tf}_volume_ratio_10',
                f'{tf}_price_to_EMA9_dup', f'{tf}_price_to_EMA21_dup', f'{tf}_price_to_SMA50_dup']:
        feature_names.append(name)

    for higher_tf in ['1h', '4h', '1d']:
        for name in [f'{higher_tf}_RSI_14', f'{higher_tf}_returns_5',
                    f'{higher_tf}_price_to_EMA9', f'{higher_tf}_price_to_EMA21',
                    f'{higher_tf}_price_to_SMA50', f'{higher_tf}_ATR_pct']:
            feature_names.append(name)

    with open(features_path, 'w') as f:
        json.dump({
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'asset': ASSET,
            'target_tf': TARGET_TF,
            'normalized': True,
            'description': 'Normalized features (returns, ratios) - works on any price level'
        }, f, indent=2)

    print(f"✅ Model saved to {model_path}")
    print(f"✅ Scaler saved to {scaler_path}")
    print(f"✅ Feature names saved to {features_path}")

    print(f"\n{'='*100}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*100}")

    print(f"\nModel: XGBoost NORMALIZED (no out-of-distribution issues!)")
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Test accuracy: {accuracy:.4f}")

    print(f"\n📁 Files created:")
    print(f"   {model_path}")
    print(f"   {scaler_path}")
    print(f"   {features_path}")

    print(f"\n✅ READY TO USE!")
    print(f"\nThis model uses NORMALIZED features and should work on ANY BTC price!")
    print("="*100)


if __name__ == "__main__":
    main()
