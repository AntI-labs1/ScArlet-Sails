# HYBRID IMPLEMENTATION PLAN

**Date:** 2025-11-10
**Timeline:** Week 2-3 (10 days)
**Goal:** Объединить Rule-based + ML в working Hybrid систему

---

## 📋 OVERVIEW

**What we're building:**
```
Rule-based signals (7,464)
    ↓
ML filter (reduce to ~3,000)
    ↓
Crisis gate (halt during crashes)
    ↓
Execute trade (HybridPositionManager)
```

**Expected result:**
- Win rate: 47% → 55-60%
- Profit factor: 1.17 → 1.8-2.0
- Crisis protection: ✅
- Annual return: 50% → 70-80% (realistic)

---

## 🗓️ WEEK 2: INTEGRATION (Days 1-5)

### DAY 1: Create Unified Entry System

**File to create:** `models/hybrid_entry_system.py`

**Tasks:**
1. [ ] Extract rule-based signal logic from master_audit
2. [ ] Create HybridEntrySystem class
3. [ ] Implement 3-layer filtering (Rules → ML → Crisis)
4. [ ] Unit tests

**Code skeleton:**
```python
# models/hybrid_entry_system.py
from models.xgboost_model import XGBoostModel
from features.crisis_detection import CrisisDetector
from models.regime_detector import SimpleRegimeDetector
import numpy as np

class HybridEntrySystem:
    """
    3-Layer entry system:
    Layer 1: Rule-based (RSI < 30)
    Layer 2: ML filter (XGBoost)
    Layer 3: Crisis gate
    """

    def __init__(self, ml_threshold=0.6, crisis_sensitivity='medium'):
        # Load components
        self.ml_model = XGBoostModel.load('models/xgboost_model.json')
        self.crisis_detector = CrisisDetector()
        self.regime_detector = SimpleRegimeDetector()

        # Config
        self.ml_threshold = ml_threshold
        self.rsi_threshold = 30
        self.min_bars_between_signals = 24

    def should_enter(self, df, bar_index):
        """
        Check if we should enter at this bar

        Returns:
            (bool, str): (should_enter, reason)
        """
        # Layer 1: Rule check (RSI < 30)
        if not self._check_rule_signal(df, bar_index):
            return False, "RSI >= 30"

        # Layer 2: ML filter
        ml_score, ml_pass = self._check_ml_filter(df, bar_index)
        if not ml_pass:
            return False, f"ML rejected (score: {ml_score:.2f})"

        # Layer 3: Crisis gate
        if self._check_crisis(df, bar_index):
            return False, "Crisis detected"

        return True, f"All passed (ML: {ml_score:.2f})"

    def _check_rule_signal(self, df, bar_index):
        """Layer 1: Check RSI < 30"""
        if bar_index < 200:
            return False

        rsi = df['rsi'].iloc[bar_index]
        return rsi < self.rsi_threshold

    def _check_ml_filter(self, df, bar_index):
        """Layer 2: ML filter"""
        # Extract features
        features = self._extract_features(df, bar_index)

        # Get ML prediction
        proba = self.ml_model.predict_proba(features.reshape(1, -1))[0][1]

        # Check threshold
        passed = proba >= self.ml_threshold

        return proba, passed

    def _check_crisis(self, df, bar_index):
        """Layer 3: Crisis detection"""
        return self.crisis_detector.detect_crisis(df, bar_index)

    def _extract_features(self, df, bar_index):
        """
        Extract features for ML model

        TODO: Match feature order expected by trained model!
        """
        features = []

        # Technical indicators
        features.append(df['rsi'].iloc[bar_index])
        features.append(df['atr'].iloc[bar_index])
        features.append(df['volume'].iloc[bar_index])

        # Add more features to match model input
        # ...

        return np.array(features)
```

**Testing:**
```python
# Test imports
from models.hybrid_entry_system import HybridEntrySystem

# Test instantiation
system = HybridEntrySystem(ml_threshold=0.6)

# Test on sample data
should_enter, reason = system.should_enter(df, 1000)
print(f"Enter: {should_enter}, Reason: {reason}")
```

**Time estimate:** 4-6 hours

---

### DAY 2: Feature Extraction

**File to create:** `features/feature_extractor.py`

**Tasks:**
1. [ ] Identify features used by trained XGBoost
2. [ ] Create unified extraction function
3. [ ] Verify feature order matches model
4. [ ] Test on sample data

**Key challenge:** Match features to what XGBoost was trained on!

**Code skeleton:**
```python
# features/feature_extractor.py
import numpy as np
import pandas as pd

def extract_features_for_ml(df, bar_index):
    """
    Extract features in exact format expected by XGBoost

    Args:
        df: DataFrame with OHLCV + indicators
        bar_index: Current bar index

    Returns:
        np.array: Features shape (n_features,)
    """
    if bar_index < 200:
        return None

    features = []

    # Feature 1-10: Technical indicators
    features.append(df['rsi'].iloc[bar_index])
    features.append(df['atr'].iloc[bar_index] / df['close'].iloc[bar_index])  # ATR ratio
    features.append(df['volume'].iloc[bar_index] / df['volume'].iloc[bar_index-20:bar_index].mean())  # Volume ratio

    # Feature 11-20: Moving averages
    ma20 = df['close'].iloc[bar_index-20:bar_index].mean()
    ma50 = df['close'].iloc[bar_index-50:bar_index].mean()
    features.append(df['close'].iloc[bar_index] / ma20)
    features.append(df['close'].iloc[bar_index] / ma50)

    # Feature 21-30: Volatility
    returns = df['close'].pct_change()
    vol_20 = returns.iloc[bar_index-20:bar_index].std()
    features.append(vol_20)

    # TODO: Add remaining features to match model!
    # Check models/xgboost_model.json to see what it was trained on

    return np.array(features)


def verify_feature_compatibility():
    """
    Verify features match what model expects

    Returns:
        (bool, str): (compatible, message)
    """
    # Load sample data
    import pandas as pd
    df = pd.read_parquet('data/raw/BTCUSDT_1h.parquet')

    # Extract features
    features = extract_features_for_ml(df, 500)

    if features is None:
        return False, "Could not extract features"

    # Load model and check
    from models.xgboost_model import XGBoostModel
    model = XGBoostModel()
    model.load('models/xgboost_model.json')

    try:
        pred = model.predict(features.reshape(1, -1))
        return True, f"Compatible! Shape: {features.shape}, Prediction: {pred[0]}"
    except Exception as e:
        return False, f"Incompatible: {e}"
```

**Time estimate:** 2-3 hours

---

### DAY 3: Create Hybrid Backtest

**File to create:** `scripts/hybrid_backtest.py`

**Tasks:**
1. [ ] Load BTC 1h data
2. [ ] Generate rule-based signals
3. [ ] Filter through ML
4. [ ] Filter through crisis gate
5. [ ] Execute with HybridPositionManager
6. [ ] Compare to baseline

**Code skeleton:**
```python
# scripts/hybrid_backtest.py
"""
Hybrid System Backtest

Compare:
1. Rule-based only (baseline)
2. Rule + ML filter
3. Rule + ML + Crisis gate (full hybrid)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.hybrid_entry_system import HybridEntrySystem
from models.hybrid_position_manager import HybridPositionManager

print("="*80)
print("HYBRID SYSTEM BACKTEST")
print("="*80)

# Load data
print("\n📂 Loading data...")
df = pd.read_parquet('data/raw/BTCUSDT_1h.parquet')
print(f"✅ Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

# Calculate indicators if needed
if 'rsi' not in df.columns:
    print("📊 Calculating indicators...")
    # Add RSI, ATR, etc.
    pass

# TEST 1: Rule-based only (baseline)
print("\n" + "="*80)
print("TEST 1: RULE-BASED ONLY (BASELINE)")
print("="*80)

rule_signals = []
for i in range(200, len(df)):
    if df['rsi'].iloc[i] < 30:
        if not rule_signals or (i - rule_signals[-1] > 24):
            rule_signals.append(i)

print(f"✅ Generated {len(rule_signals)} rule-based signals")

# Backtest rule-based
rule_trades = backtest_signals(df, rule_signals)
rule_stats = calculate_stats(rule_trades)

print(f"\nRule-based results:")
print(f"  Trades: {len(rule_trades)}")
print(f"  Win rate: {rule_stats['win_rate']:.1f}%")
print(f"  Profit factor: {rule_stats['profit_factor']:.2f}")
print(f"  Annual return: {rule_stats['annual_return']:.1f}%")

# TEST 2: Rule + ML filter
print("\n" + "="*80)
print("TEST 2: RULE + ML FILTER")
print("="*80)

entry_system = HybridEntrySystem(ml_threshold=0.6)

ml_filtered_signals = []
ml_rejections = []

for bar_idx in rule_signals:
    should_enter, reason = entry_system.should_enter(df, bar_idx)
    if should_enter:
        ml_filtered_signals.append(bar_idx)
    else:
        ml_rejections.append((bar_idx, reason))

reduction_pct = (1 - len(ml_filtered_signals)/len(rule_signals)) * 100
print(f"✅ ML filter kept {len(ml_filtered_signals)} signals ({reduction_pct:.1f}% reduction)")
print(f"   Rejections: {len(ml_rejections)}")
print(f"   Top reasons: ...")

# Backtest ML-filtered
ml_trades = backtest_signals(df, ml_filtered_signals)
ml_stats = calculate_stats(ml_trades)

print(f"\nML-filtered results:")
print(f"  Trades: {len(ml_trades)}")
print(f"  Win rate: {ml_stats['win_rate']:.1f}% (Δ{ml_stats['win_rate']-rule_stats['win_rate']:+.1f}%)")
print(f"  Profit factor: {ml_stats['profit_factor']:.2f} (Δ{ml_stats['profit_factor']-rule_stats['profit_factor']:+.2f})")
print(f"  Annual return: {ml_stats['annual_return']:.1f}%")

# TEST 3: Full hybrid (Rule + ML + Crisis)
print("\n" + "="*80)
print("TEST 3: FULL HYBRID (RULE + ML + CRISIS)")
print("="*80)

# TODO: Add crisis filtering
# ...

# COMPARISON
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

comparison = pd.DataFrame({
    'Metric': ['Trades', 'Win Rate', 'Profit Factor', 'Annual Return'],
    'Rule-based': [
        len(rule_trades),
        rule_stats['win_rate'],
        rule_stats['profit_factor'],
        rule_stats['annual_return']
    ],
    'ML-filtered': [
        len(ml_trades),
        ml_stats['win_rate'],
        ml_stats['profit_factor'],
        ml_stats['annual_return']
    ],
    'Improvement': [
        len(ml_trades) - len(rule_trades),
        ml_stats['win_rate'] - rule_stats['win_rate'],
        ml_stats['profit_factor'] - rule_stats['profit_factor'],
        ml_stats['annual_return'] - rule_stats['annual_return']
    ]
})

print(comparison.to_string(index=False))

# Save results
output_dir = Path('reports/hybrid_analysis')
output_dir.mkdir(exist_ok=True, parents=True)

results = {
    'rule_based': rule_stats,
    'ml_filtered': ml_stats,
    'comparison': comparison.to_dict()
}

with open(output_dir / 'hybrid_backtest_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to {output_dir}")

def backtest_signals(df, signals):
    """Simple backtest implementation"""
    # TODO: Implement
    pass

def calculate_stats(trades):
    """Calculate performance statistics"""
    # TODO: Implement
    pass
```

**Time estimate:** 4-6 hours

---

### DAY 4: Testing & Debugging

**Tasks:**
1. [ ] Run hybrid_backtest.py
2. [ ] Fix import errors
3. [ ] Fix feature mismatches
4. [ ] Verify results make sense
5. [ ] Document any issues

**Expected issues:**
- Feature count mismatch (trained model expects X features, we provide Y)
- Crisis detector API different than expected
- Performance not as expected

**Time estimate:** 4-8 hours

---

### DAY 5: Analysis & Documentation

**Tasks:**
1. [ ] Analyze hybrid vs baseline
2. [ ] Create comparison charts
3. [ ] Document findings
4. [ ] Commit all to GitHub

**Deliverables:**
```
reports/hybrid_analysis/
├── hybrid_backtest_results.json
├── comparison.txt
└── analysis.md
```

**Time estimate:** 2-4 hours

---

## 🗓️ WEEK 3: OPTIMIZATION (Days 6-10)

### DAY 6: Tune ML Threshold

**Goal:** Find optimal ML threshold

**Test thresholds:** 0.4, 0.5, 0.6, 0.7, 0.8

**Expected:**
- Lower threshold: More signals, lower quality
- Higher threshold: Fewer signals, higher quality

**Optimal:** Balance frequency vs quality

---

### DAY 7: Tune Crisis Sensitivity

**Goal:** Balance protection vs missed opportunities

**Test:**
- High sensitivity: Halt more often (safer but miss profits)
- Low sensitivity: Trade more (riskier but more profits)

---

### DAY 8-9: Multi-Asset Test

**Goal:** Verify hybrid works on other assets

**Test on:**
- BTC 1h (must work!)
- ETH 1h (must work!)
- ALGO 15m (should work)
- AVAX 15m (should work)
- SOL 15m (should work)

**Success criteria:**
- Works on at least 3/5 assets
- PF > 1.5 on each
- Win rate > 50% on each

---

### DAY 10: Final Report

**Create:** `HYBRID_FINAL_REPORT.md`

**Include:**
- Baseline vs Hybrid comparison
- Multi-asset results
- Crisis protection analysis
- Production readiness assessment
- Next steps

---

## ✅ SUCCESS CRITERIA

### Minimum (Must Achieve):

```
✅ Win rate > 50% (vs 47% baseline)
✅ Profit factor > 1.5 (vs 1.17 baseline)
✅ Crisis detection works (halt before crashes)
✅ 200+ trades (statistical confidence)
✅ Works on BTC and ETH
```

### Target (Should Achieve):

```
🎯 Win rate 55-60%
🎯 Profit factor 1.8-2.0
🎯 Max drawdown < 20%
🎯 Works on 5+ assets
```

---

## 📦 DELIVERABLES

### Code:

```
models/hybrid_entry_system.py          # NEW
features/feature_extractor.py          # NEW
scripts/hybrid_backtest.py             # NEW
configs/hybrid_config.yaml             # NEW (optional)
```

### Documentation:

```
HYBRID_SYSTEM_ARCHITECTURE.md          # ✅ DONE
HYBRID_COMPONENTS_STATUS.md            # ✅ DONE
HYBRID_IMPLEMENTATION_PLAN.md          # ✅ THIS FILE
HYBRID_FINAL_REPORT.md                 # Week 3
```

### Results:

```
reports/hybrid_analysis/
├── hybrid_backtest_results.json
├── comparison.txt
├── crisis_protection_analysis.txt
└── multi_asset_results/
    ├── BTC_1h.json
    ├── ETH_1h.json
    └── ...
```

---

## 🚨 RISK MITIGATION

### Risk 1: Feature Mismatch
**Impact:** ML model won't work
**Mitigation:**
- Document features used in training
- Create feature verification test
- Start with simple features, add complexity

### Risk 2: Performance Not Improved
**Impact:** Hybrid no better than baseline
**Mitigation:**
- Start with known-good ML threshold (0.6)
- Validate ML model works standalone first
- Can always fall back to rule-based

### Risk 3: Too Few Signals
**Impact:** Not enough trades for statistics
**Mitigation:**
- Tune ML threshold down if needed
- Test on longer time period
- Multi-asset to increase sample size

---

## 📝 DAILY CHECKLIST

### Each Day:

- [ ] Commit code at end of day
- [ ] Document any blockers
- [ ] Update progress in this file
- [ ] Test code runs without errors

---

**Status:** 📋 PLAN READY
**Estimated time:** 10 days (2 weeks)
**Confidence:** HIGH (all components proven to work)

**Ready to start implementation!**
