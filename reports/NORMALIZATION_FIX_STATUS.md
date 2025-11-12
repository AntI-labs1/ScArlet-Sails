# NORMALIZATION FIX - STATUS REPORT

**Date:** 2025-11-12
**Status:** 🟡 FIX APPLIED - TESTING PENDING
**Branch:** claude/debug-naive-strategy-performance-011CUx4E9miJvu4k8Sn2gBkH

---

## 🎯 PROBLEM IDENTIFIED

### Root Cause
**Feature mismatch between training and inference:**

**Training (retrain_xgboost_normalized.py):**
- Used normalized features: `price_to_EMA9` (ratio ~1.0), `ATR_pct` (ratio)
- Scaler expected values around ~1.0

**Inference (MultiTimeframeFeatureExtractor.extract_features_at_bar):**
- Used ABSOLUTE features: `EMA_9` ($100,000+), `SMA_50` ($100,000+), `ATR_14` ($2000+)
- Result: Scaler saw (100,000 - 1.0) / 0.005 = **2,000,000σ** (catastrophic!)

### Symptoms
```
OOD: 100% (every single bar!)
Sigma values: 2,000,000σ (should be 0-3σ)
Trades: 266,491 (should be ~20K-50K)
```

---

## ✅ FIX APPLIED

### Commit: 4d7c41f
**File:** `features/multi_timeframe_extractor.py`
**Method:** `extract_features_at_bar()`

**Changes:**
```python
# BEFORE (BROKEN):
features.append(df['EMA_9'])  # Absolute $100,000+
features.append(df['ATR_14'])  # Absolute $2000+
features.append(df['RSI_14'])  # 0-100 range

# AFTER (FIXED):
features.append(close / df['EMA_9'])  # Ratio ~1.0
features.append(df['ATR_14'] / close)  # Percentage ~0.02
features.append(df['RSI_14'] / 100.0)  # Normalized 0-1
```

**Result:** All 31 features now return normalized values (ratios, percentages) matching training expectations.

---

## 📝 ADDITIONAL CHANGES

### 1. Updated Training Script
**File:** `scripts/retrain_xgboost_normalized.py`

**Change:** Now uses actual `MultiTimeframeFeatureExtractor` instead of custom `NormalizedMultiTFExtractor`

**Why:** Ensures training and inference use IDENTICAL feature extraction logic

### 2. Created Test Script
**File:** `scripts/test_normalization_fix.py`

**Purpose:** Quick verification (1000 bars) to check:
- OOD ratio < 10% (was 100%)
- Sigma values < 10 (were millions)
- Signal rate < 30% (was 100%)
- Total signals < 100K (were 266K)

### 3. Created Verification Script
**File:** `scripts/verify_feature_normalization.py`

**Purpose:** Verify feature extraction returns normalized values WITHOUT needing the model

---

## 🔴 CANNOT TEST YET

### Problem: Missing Files

**This development environment lacks:**
1. Actual data files (only DVC pointers exist)
2. Trained model files (`xgboost_normalized_model.json`, `xgboost_normalized_scaler.pkl`)

**Available:**
- ✅ Code fix (committed)
- ✅ Test scripts (ready)
- ❌ Data files (need DVC pull or local files)
- ❌ Trained model (need to retrain)

---

## 📋 NEXT STEPS (User Local Machine)

### Step 1: Pull Code
```bash
cd /path/to/scarlet-sails
git fetch origin
git checkout claude/debug-naive-strategy-performance-011CUx4E9miJvu4k8Sn2gBkH
git pull
```

### Step 2: Retrain Model
```bash
# This will create:
# - models/xgboost_normalized_model.json
# - models/xgboost_normalized_scaler.pkl
python scripts/retrain_xgboost_normalized.py
```

**Expected output:**
- Training on ~50K samples
- Test accuracy ~55-60%
- Model saved successfully

**Time:** ~5-10 minutes

### Step 3: Quick Test
```bash
# Test on 1000 bars (fast)
python scripts/test_normalization_fix.py
```

**Expected output:**
```
✅ FIXED! OOD < 10%
✅ FIXED! Max sigma < 10
✅ REASONABLE (<30% signal rate)
✅ REASONABLE (was 266K, now ~20K-50K)
```

### Step 4: Full Audit
```bash
# Test all 56 combinations
python scripts/comprehensive_model_audit.py
```

**Expected output:**
```
ML Model:
- Available combos: 42-50 (not 0!)
- OOD ratio: <10% (not 98.3%)
- Avg win rate: ~50-55%
- Profit factor: ~1.5-2.0
```

---

## 🚦 SUCCESS CRITERIA

### ✅ Fix Successful If:
- OOD ratio < 10% (was 100%)
- Sigma values 0-3 (were millions)
- Total signals 20K-50K (were 266K)
- ML generates trades on 40+ combinations (was 0)
- Win rate ~50-55% (was 38%)

### ⚠️ Partial Fix If:
- OOD ratio 10-50%
- Sigma values 10-100
- Signals 50K-100K
- → Investigate feature order mismatch

### ❌ Fix Failed If:
- OOD still >50%
- Sigma still >100
- Still 200K+ signals
- → Pivot to Variant C (Rule-Based focus)

---

## 🔄 FALLBACK PLAN

**If fix doesn't work within 2 hours:**

### Variant C: Rule-Based Improvements
1. Enhanced RSI strategy (EMA + Volume + ATR filters)
2. Multi-timeframe confluence
3. Regime detection integration
4. Crisis classifier integration
5. Target: Win rate 55-60%, Profit factor 2.0+

**ML becomes Day 2 focus:**
- Review entire training pipeline
- Check for data leakage
- Consider different model architecture
- Collect more training data

---

## 📊 TECHNICAL DETAILS

### Feature Count
- **Total features:** 31
- **Primary timeframe (15m):** 13 features
  - RSI (normalized 0-1)
  - 3x price ratios (close/EMA9, close/EMA21, close/SMA50)
  - BB width % (normalized)
  - ATR % (normalized)
  - 2x returns (pct_change)
  - 2x volume ratios
  - 3x duplicate ratios (for compatibility)
- **Higher timeframes (1h, 4h, 1d):** 18 features (6 each)
  - RSI (normalized)
  - 1x return
  - 3x price ratios
  - ATR %

### All Features Are Now:
- ✅ Ratios (values ~1.0)
- ✅ Percentages (values ~0.01-0.10)
- ✅ Normalized indicators (values 0-1)
- ❌ NO absolute prices
- ❌ NO absolute EMAs/SMAs
- ❌ NO absolute ATR values

---

## 🎯 EXPECTED OUTCOMES

### Before Fix:
```
BTC_15m ML Strategy:
- Total trades: 266,491 (100% of bars!)
- OOD ratio: 98.3%
- Max sigma: 2,123,670
- Win rate: 38.2%
- Profit factor: 1.09
```

### After Fix (Predicted):
```
BTC_15m ML Strategy:
- Total trades: 25,000-40,000 (~10-15% of bars)
- OOD ratio: <10%
- Max sigma: <10
- Win rate: 50-55%
- Profit factor: 1.5-2.0
```

---

## 📁 FILES MODIFIED

### Code Changes:
1. ✅ `features/multi_timeframe_extractor.py` (commit 4d7c41f)
2. ✅ `scripts/retrain_xgboost_normalized.py` (this commit)

### New Scripts:
3. ✅ `scripts/test_normalization_fix.py` (this commit)
4. ✅ `scripts/verify_feature_normalization.py` (this commit)

### Documentation:
5. ✅ `reports/NORMALIZATION_FIX_STATUS.md` (this file)

---

## 💡 WHY THIS SHOULD WORK

### The Core Issue
ML models learn DISTRIBUTIONS, not absolute values. When training data has prices $6K-$20K (2018-2020) and test data has $80K-$108K (2024-2025), the model fails.

### The Solution
**Normalized features are SCALE-INVARIANT:**
- `close / EMA9` = 1.02 (works at $100 or $100,000)
- `ATR / close` = 0.025 (works at any price level)
- `returns` = 0.05 (same % regardless of absolute price)

### Result
Model trained on 2018 data ($6K BTC) will work on 2024 data ($100K BTC) because:
- All features are RELATIVE to current price
- Distribution shape stays similar
- OOD becomes <10% instead of 100%

---

**Status:** Ready for testing on user's local machine with full data

**Last updated:** 2025-11-12 (normalization fix applied)
