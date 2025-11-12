#!/usr/bin/env python3
"""
QUICK TEST - Normalization Fix

Test if OOD problem is fixed after normalizing features.

Expected results:
- OOD should be <10% (not 100%!)
- Sigma values should be 0-3 (not millions!)
- ML should generate 20K-50K trades (not 266K!)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from features.multi_timeframe_extractor import MultiTimeframeFeatureExtractor
from models.xgboost_model import XGBoostModel

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*100)
print("QUICK TEST - NORMALIZATION FIX")
print("="*100)

# Load scaler
scaler_path = MODELS_DIR / "xgboost_normalized_scaler.pkl"
scaler = joblib.load(scaler_path)

# Load model
model_path = MODELS_DIR / "xgboost_normalized_model.json"
model = XGBoostModel()
model.load(str(model_path))

print(f"\n✅ Model & scaler loaded")

# Load data
extractor = MultiTimeframeFeatureExtractor(data_dir=str(DATA_DIR))
all_tf, primary_df = extractor.prepare_multi_timeframe_data("BTC", "15m")

# Apply date cutoff
cutoff_ts = pd.Timestamp("2025-10-01", tz='UTC')
primary_df = primary_df[primary_df.index < cutoff_ts]
for tf_key in all_tf:
    all_tf[tf_key] = all_tf[tf_key][all_tf[tf_key].index < cutoff_ts]

print(f"✅ BTC_15m data loaded: {len(primary_df)} bars")

# Test on 1000 bars only (quick test)
print(f"\n{'='*100}")
print("TESTING ON 1000 BARS (QUICK)")
print(f"{'='*100}")

test_bars = 1000
ood_count = 0
signals = 0
sigma_values = []

for i in range(test_bars):
    features = extractor.extract_features_at_bar(all_tf, "15m", i + 10000)

    if features is None:
        continue

    # Scale
    features_scaled = scaler.transform(features.reshape(1, -1))[0]

    # Check OOD
    max_sigma = np.max(np.abs(features_scaled))
    sigma_values.append(max_sigma)

    is_ood = np.any(np.abs(features_scaled) > 3.0)
    if is_ood:
        ood_count += 1

    # Predict
    prob = model.predict_proba(features.reshape(1, -1))[0]
    prob_up = prob[1]

    if prob_up >= 0.65:
        signals += 1

print(f"\n✅ Test complete!")
print(f"\n{'─'*100}")
print("RESULTS:")
print(f"{'─'*100}")

print(f"\n1. OOD RATIO:")
print(f"   OOD bars: {ood_count}/{test_bars} = {ood_count/test_bars*100:.1f}%")
if ood_count / test_bars < 0.10:
    print(f"   ✅ FIXED! OOD < 10%")
elif ood_count / test_bars < 0.50:
    print(f"   ⚠️  BETTER (was 100%), but still high")
else:
    print(f"   ❌ STILL BROKEN (>50% OOD)")

print(f"\n2. SIGMA VALUES:")
sigma_values = np.array(sigma_values)
print(f"   Mean: {sigma_values.mean():.2f}σ")
print(f"   Median: {np.median(sigma_values):.2f}σ")
print(f"   Max: {sigma_values.max():.2f}σ")

if sigma_values.max() < 10:
    print(f"   ✅ FIXED! Max sigma < 10")
elif sigma_values.max() < 100:
    print(f"   ⚠️  BETTER (was millions!), but still high")
else:
    print(f"   ❌ STILL BROKEN (sigma > 100)")

print(f"\n3. SIGNALS:")
print(f"   Signals generated: {signals}/{test_bars} = {signals/test_bars*100:.1f}%")
expected_ratio = signals / test_bars
if expected_ratio < 0.30:
    print(f"   ✅ REASONABLE (<30% signal rate)")
else:
    print(f"   ⚠️  HIGH signal rate (threshold may be too low)")

# Extrapolate to full dataset
full_bars = 266491
expected_full_signals = int(full_bars * expected_ratio)
print(f"\n4. EXTRAPOLATION TO FULL DATASET:")
print(f"   Expected signals on all bars: ~{expected_full_signals:,}")
if expected_full_signals < 100000:
    print(f"   ✅ REASONABLE (was 266K, now ~{expected_full_signals//1000}K)")
else:
    print(f"   ❌ STILL TOO MANY (>100K)")

# Final verdict
print(f"\n{'='*100}")
print("VERDICT:")
print(f"{'='*100}")

if (ood_count / test_bars < 0.10
    and sigma_values.max() < 10
    and expected_full_signals < 100000):
    print("\n✅ ✅ ✅ NORMALIZATION FIX WORKED!")
    print("\n   Next step: Run full audit on all 56 combinations")
    print("   Command: python scripts/comprehensive_model_audit.py")
elif (ood_count / test_bars < 0.50
      and sigma_values.max() < 100):
    print("\n⚠️  PARTIAL FIX - Better but not perfect")
    print("\n   Improvements:")
    print(f"     - OOD: 100% → {ood_count/test_bars*100:.1f}%")
    print(f"     - Max sigma: 2,000,000 → {sigma_values.max():.2f}")
    print(f"     - Signals: 266K → ~{expected_full_signals//1000}K")
    print("\n   Next step: Review feature order mismatch or retrain model")
else:
    print("\n❌ FIX DIDN'T WORK")
    print("\n   Pivot to VARIANT C: Focus on Rule-Based improvements")
    print("   ML will be fixed in Day 2")

print(f"\n{'='*100}")
