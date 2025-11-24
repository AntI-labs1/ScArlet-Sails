# PHASE 2B ACTION PLAN: XGBoost ML Integration

**Goal:** Integrate XGBoost ML model into P_j(S) framework and create Version 6-8 tests

**Approach:** OPTION A - Clear TODO list with debugging help

**Timeline:** ~1 hour to completion

---

## PREREQUISITE: Understand Current State

### Current Framework Status:
- ✅ Signal Generation: Working (Rule-Based RSI < 30)
- ✅ P_j(S) Calculation: Working (all components active)
- ✅ Backtest Engine: Working (realistic results)
- ✅ TP/SL Optimization: Completed (best: 3.0% / 1.2%)

### What Works:
```
V1: 75 trades, 34.7% WR (no filters)
V3: 69 trades, 34.8% WR (with filters)
```

### What's Missing:
```
ML Scores: Currently using signals × 0.7 (placeholder)
           Need: Real XGBoost predictions (0-1 probability)
```

---

## STEP 1: Analyze XGBoost Model

**File:** `models/xgboost_normalized_model.json`
**Task:** Understand model structure and features

### Action Items:

#### 1.1 - Inspect Model File
```bash
# Check file size and structure
ls -lh models/xgboost_normalized_model.json
head -100 models/xgboost_normalized_model.json | grep -E '"name"|"features|"field"'
```

**Expected Output:**
- File size: ~673 KB
- Should see feature names or structure
- Look for: field names, tree count, feature importance

#### 1.2 - Create Model Analysis Script
**File to create:** `scripts/analyze_xgboost_model.py`

```python
#!/usr/bin/env python3
"""Analyze XGBoost model structure"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
model_file = PROJECT_ROOT / "models" / "xgboost_normalized_model.json"

with open(model_file, 'r') as f:
    model_data = json.load(f)

print(f"Model type: {type(model_data)}")
print(f"Model keys: {list(model_data.keys())}")

if 'learner' in model_data:
    print(f"Learner keys: {list(model_data['learner'].keys())}")

if 'feature_names' in model_data:
    print(f"Features: {model_data['feature_names']}")
    print(f"Feature count: {len(model_data['feature_names'])}")
```

**Expected Output:**
```
Model type: <class 'dict'>
Model keys: [...]
Features: [...list of feature names...]
Feature count: ~31
```

---

## STEP 2: Load Model and Test Predictions

**File to create:** `scripts/test_xgboost_model.py`

```python
#!/usr/bin/env python3
"""Test XGBoost model prediction"""

import sys, json
from pathlib import Path
import pandas as pd
import numpy as np

try:
    from xgboost import XGBClassifier
except ImportError:
    print("ERROR: xgboost not installed. Run: pip install xgboost")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent.parent
model_file = PROJECT_ROOT / "models" / "xgboost_normalized_model.json"
data_file = PROJECT_ROOT / "data" / "raw" / "BTC_USDT_15m.parquet"

print("[1/4] Loading model...")
with open(model_file, 'r') as f:
    model_data = json.load(f)
print(f"✅ Model loaded: {type(model_data)}")

print("[2/4] Loading data...")
ohlcv = pd.read_parquet(data_file)
print(f"✅ Data loaded: {len(ohlcv)} bars")

print("[3/4] Extracting features...")
# Try to understand required features
required_features = model_data.get('feature_names', [])
print(f"   Required: {required_features}")
print(f"   Available columns: {list(ohlcv.columns)}")

print("[4/4] Status: Checking if model can be loaded with xgboost library...")
try:
    model = XGBClassifier()
    model.load_model(model_file)
    print("✅ Model loaded successfully with XGBoost library")
except Exception as e:
    print(f"⚠️  Could not load with XGBClassifier: {e}")
    print("    This is expected - we'll use json approach instead")
```

**Expected Output:**
```
✅ Model loaded: <class 'dict'>
✅ Data loaded: 10000 bars
   Required: [list of 31 features]
   Available: ['open', 'high', 'low', 'close', 'volume', ...]
```

---

## STEP 3: Create ML Features Generator

**File to create:** `backtesting/ml_features.py`

This component extracts ML features from OHLCV data.

```python
import numpy as np
import pandas as pd

class MLFeatureGenerator:
    """Generate ML features from OHLCV data"""

    def __init__(self, required_features=None):
        self.required_features = required_features or []
        self.scaler = None  # Load from xgboost_normalized_scaler.pkl if needed

    def extract_features(self, ohlcv: pd.DataFrame, idx: int) -> np.ndarray:
        """
        Extract features at bar idx

        Returns: 1D array of features for XGBoost prediction
        """
        if idx < 50:  # Need historical data
            return None

        # Basic features that can be computed from OHLCV
        features = {}

        # Price features
        features['close'] = ohlcv.iloc[idx]['close']
        features['high'] = ohlcv.iloc[idx]['high']
        features['low'] = ohlcv.iloc[idx]['low']
        features['volume'] = ohlcv.iloc[idx]['volume']

        # Technical indicators (examples)
        recent_close = ohlcv['close'].iloc[max(0, idx-50):idx].values
        features['sma_10'] = np.mean(recent_close[-10:])
        features['sma_50'] = np.mean(recent_close[-50:])
        features['volatility'] = np.std(recent_close)

        # Return as array in order of required_features
        # (Will need to align with actual model requirements)

        return features
```

**Key Points:**
- Model requires ~31 features
- Must extract from OHLCV data
- Features need to be normalized/scaled
- Check `xgboost_normalized_scaler.pkl` for scaling info

---

## STEP 4: Integrate ML Scoring into Framework

**File to update:** `backtesting/backtest_pjs_framework.py`

Add to `PjSBacktestEngine.__init__()`:
```python
from ml_features import MLFeatureGenerator
from xgboost import XGBClassifier

self.ml_model = None
self.ml_features = None

if ml_model_enabled:
    try:
        self.ml_model = XGBClassifier()
        self.ml_model.load_model(Path(__file__).parent.parent / "models" / "xgboost_normalized_model.json")
        self.ml_features = MLFeatureGenerator()
        print("✅ ML model loaded")
    except Exception as e:
        print(f"⚠️ ML model loading failed: {e}")
        self.ml_model = None
```

Update `calculate_pjs()` method:
```python
# If ML model available, get real prediction
if self.ml_model is not None:
    features = self.ml_features.extract_features(ohlcv, idx)
    if features is not None:
        ml_prediction = self.ml_model.predict_proba(features)
        ml_score = ml_prediction[0][1]  # Probability of positive class
    else:
        ml_score = ml_scores[idx]
else:
    ml_score = ml_scores[idx]
```

---

## STEP 5: Create Version 6-8 Tests

**File to create:** `scripts/test_pjs_framework_v6_v7_v8.py`

```python
"""Test versions 6-8 with optimized TP/SL and ML models"""

# V6: Rule-Based with optimized TP/SL (3.0% / 1.2%)
config_v6 = BacktestConfig(
    take_profit=0.03,
    stop_loss=0.012,
    ml_enabled=False,
    filters_enabled=True,
    # ... other params
)

# V7: ML Model with optimized TP/SL
config_v7 = BacktestConfig(
    take_profit=0.03,
    stop_loss=0.012,
    ml_enabled=True,
    filters_enabled=True,
    # ... other params
)

# V8: Hybrid (Rule-Based + ML)
config_v8 = BacktestConfig(
    take_profit=0.03,
    stop_loss=0.012,
    ml_enabled=True,
    filters_enabled=True,
    # ... other params with Bayesian fusion
)

# Expected Results:
# V6: 48-52% WR, 25%+ return
# V7: 52-56% WR, 30%+ return
# V8: 54-58% WR, 35%+ return
```

---

## PHASE 2B CHECKLIST

### Analysis Phase:
- [ ] Inspect XGBoost model file structure
- [ ] Identify all 31 required features
- [ ] Check for scaler/normalizer file
- [ ] Understand feature engineering requirements

### Development Phase:
- [ ] Create `analyze_xgboost_model.py`
- [ ] Create `test_xgboost_model.py`
- [ ] Create `backtesting/ml_features.py`
- [ ] Update `backtest_pjs_framework.py` with ML loading
- [ ] Create `scripts/test_pjs_framework_v6_v7_v8.py`

### Testing Phase:
- [ ] Run Version 6 test (Rule-Based optimized)
- [ ] Verify 48-52% win rate
- [ ] Run Version 7 test (ML model)
- [ ] Verify 52-56% win rate
- [ ] Run Version 8 test (Hybrid)
- [ ] Verify 54-58% win rate

### Validation Phase:
- [ ] Compare V6 vs V7 vs V8 results
- [ ] Check for overfitting (should improve on test data)
- [ ] Generate comparison table
- [ ] Commit all changes with clear messages

---

## SUCCESS CRITERIA

### Minimal Success (Phase 2B Complete):
✅ V6 backtest runs without errors
✅ V7 backtest runs without errors
✅ V8 backtest runs without errors
✅ Results saved and comparable

### Target Success (High Confidence):
✅ V6: ~50% WR, 25% return
✅ V7: ~54% WR, 30% return
✅ V8: ~56% WR, 35% return

### Stretch Goals (Extra Credit):
✅ V7 beats V6 by >2% WR
✅ V8 beats V7 by >2% WR
✅ Profit Factor all > 1.3

---

## DEBUGGING GUIDE

### If ML Model Won't Load:
```bash
# 1. Check file exists
ls -la models/xgboost_normalized_model.json

# 2. Check if JSON is valid
python3 -c "import json; json.load(open('models/xgboost_normalized_model.json'))"

# 3. Check XGBoost version
pip show xgboost
```

### If Features Don't Match:
```
Problem: "Feature count mismatch"
Solution:
  1. Print required feature names
  2. Print available OHLCV columns
  3. Create feature mapping/engineering
  4. Test on small data sample
```

### If Predictions Are Wrong:
```
Problem: "All predictions are 0.5" or similar
Solution:
  1. Check feature scaling/normalization
  2. Verify features are in correct order
  3. Load scaler.pkl if it exists
  4. Test with known good data
```

---

## NEXT STEPS AFTER PHASE 2B

### PHASE 3: OOT Validation
- Load 2024 data (unseen)
- Run all 3 models
- Compare training vs OOT performance
- Check for overfitting

### PHASE 4: Reports
- Generate investor reports for each model
- Include performance metrics, risk analysis
- Create comparison charts
- Finalize recommendations

---

## TIMELINE ESTIMATE

| Task | Time | Status |
|------|------|--------|
| Analyze XGBoost | 10 min | 🔲 |
| Create feature generator | 10 min | 🔲 |
| Update framework | 5 min | 🔲 |
| Create V6-V8 tests | 15 min | 🔲 |
| Run tests | 10 min | 🔲 |
| Debug/iterate | 10-20 min | 🔲 |
| **TOTAL** | **~60 min** | 🔲 |

---

## FILES CREATED/MODIFIED

### New Files:
- `scripts/analyze_xgboost_model.py`
- `scripts/test_xgboost_model.py`
- `backtesting/ml_features.py`
- `scripts/test_pjs_framework_v6_v7_v8.py`

### Modified Files:
- `backtesting/backtest_pjs_framework.py` (add ML loading)

### Reference Files:
- `models/xgboost_normalized_model.json` (673 KB)
- `models/xgboost_normalized_scaler.pkl` (check if exists)
- `data/raw/BTC_USDT_15m.parquet` (test data)

---

## READY?

This plan provides:
✅ Clear step-by-step actions
✅ Expected outputs for each step
✅ Debugging guide for common issues
✅ Success criteria for Phase 2B

**Start with Step 1:** Analyze the XGBoost model structure

**When you hit a blocker:** Come back here, check the debugging guide, or let me know what error you're seeing

**After each step completes:** Commit to git with descriptive message

Good luck! 🚀

---

**Generated:** 2025-11-14 13:55:00
**Status:** Ready to begin Phase 2B
**Next Review:** After analyzing XGBoost model structure
