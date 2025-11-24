# HYBRID SYSTEM ARCHITECTURE - SCARLET SAILS

**Date:** 2025-11-10
**Status:** 🎯 PLAN - Ready for Implementation
**Decision:** Объединить Rule-based + ML-based в единую Hybrid систему

---

## 🎯 EXECUTIVE SUMMARY

У нас есть **ДВЕ работающие системы:**

1. **Rule-based** - много сделок (7,464), слабый edge (PF 1.17)
2. **ML-based** - мало сделок (46), сильный edge (PF 2.12)

**Hybrid подход:**
- Используем Rule-based для частоты входов
- Фильтруем через ML для качества
- Добавляем Crisis Detection для защиты
- Получаем: частота + качество + защита

---

## 📊 ТЕКУЩЕЕ СОСТОЯНИЕ

### System 1: Rule-Based (master_comprehensive_audit.py)

**Характеристики:**
```
Entry: RSI < 30 (fixed threshold)
Exit: ATR * multiplier (regime-based)
Regime: MA20 vs MA200

Результаты (8 лет, 7,464 сделок):
├─ Win rate: 47.3%
├─ Profit factor: 1.17
├─ Average win: +2.58%
├─ Average loss: -1.98%
├─ Bull regime: +0.82% avg ✅
├─ Bear regime: +0.28% avg ⚠️
└─ Sideways: +0.06% avg ❌
```

**Проблемы:**
- ❌ 74% trades hit stop (много ложных входов)
- ❌ Sideways = почти break-even (70% времени)
- ❌ Слабый Profit Factor (нужно 1.5+)

**Преимущества:**
- ✅ Много данных (7,464 trades)
- ✅ Протестировано через 2 краха
- ✅ Простая система
- ✅ Работает (PF > 1)

---

### System 2: ML-Based (XGBoost + Crisis Detection)

**Характеристики:**
```
Entry: RSI < 30 + XGBoost filter
Model: Trained XGBoost (147KB)
Crisis: Trained detector (98.4% halt rate)

Результаты (??? период, 46 сделок):
├─ Win rate: 60.9% ✅
├─ Profit factor: 2.12 ✅
├─ Crisis detection: 98.4% halt rate ✅
└─ COVID/Luna/FTX: All detected ✅
```

**Проблемы:**
- ❌ Мало сделок (46 trades - недостаточно!)
- ❌ Период неизвестен
- ❌ Нет статистической уверенности

**Преимущества:**
- ✅ Высокий PF (2.12)
- ✅ Высокий WR (60.9%)
- ✅ Crisis detection работает!
- ✅ Меньше ложных входов

---

## 🔄 HYBRID SYSTEM DESIGN

### Архитектура (3 слоя)

```
┌─────────────────────────────────────────────────────────────┐
│                   LAYER 1: SIGNAL GENERATION                │
│                         (Rule-based)                        │
│                                                             │
│  Entry: RSI < 30                                           │
│  Frequency: HIGH (7,464 signals over 8 years)              │
│  Quality: MIXED (47.3% WR)                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 2: ML FILTER                       │
│                       (XGBoost)                             │
│                                                             │
│  Input: [RSI, ATR, Volume, MA, Regime, ...]               │
│  Output: Probability (0-1)                                 │
│  Threshold: 0.6 (tune based on backtest)                   │
│                                                             │
│  IF ml_score > threshold:                                  │
│     PASS signal to Layer 3                                 │
│  ELSE:                                                      │
│     REJECT signal                                          │
│                                                             │
│  Expected: 60-70% reduction in signals                     │
│  Expected: Win rate 50% → 60%+                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  LAYER 3: CRISIS GATE                       │
│                   (Crisis Detector)                         │
│                                                             │
│  IF crisis_detected:                                       │
│     HALT all trading                                       │
│  ELSE:                                                      │
│     EXECUTE trade                                          │
│                                                             │
│  Protection: COVID, Luna, FTX, etc.                        │
│  Halt rate: 98.4% before crashes                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 4: EXECUTION                       │
│                  (HybridPositionManager)                    │
│                                                             │
│  Adaptive stop-loss (ATR-based)                            │
│  Trailing stop                                             │
│  Partial exits                                             │
│  Max holding time: 7 days                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📈 EXPECTED RESULTS

### Conservative Estimate:

**Baseline (Rule-based only):**
```
7,464 signals → 47.3% WR → PF 1.17
```

**After ML filter (60% reduction):**
```
~3,000 signals → 60% WR → PF 1.8-2.0 (expected)
```

**After crisis protection:**
```
During normal markets: Same as above
During crises: 0 trades (protected!)
```

**Annual Performance:**
```
Rule-based: 114% annual (backtest) → 50% (realistic)
Hybrid: 80-100% annual (expected after filters)

With crisis protection:
- Avoid -50% drawdowns (COVID, Luna, FTX)
- Smoother equity curve
- Lower max drawdown
```

---

## 🔧 IMPLEMENTATION COMPONENTS

### Existing (Already Works):

```
✅ models/hybrid_position_manager.py - Position management
✅ models/regime_detector.py - Regime detection
✅ models/xgboost_model.py - ML model wrapper
✅ models/xgboost_model.json - Trained model (147KB)
✅ features/crisis_detection.py - Crisis detector
✅ scripts/master_comprehensive_audit.py - Rule-based backtest
```

### Needs Creation:

```
📝 models/hybrid_strategy.py - Unified entry system
   - Combines RSI signal + ML filter + Crisis gate
   - Single interface for all three layers

📝 scripts/hybrid_backtest.py - Test hybrid system
   - Use rule-based signals
   - Apply ML filter
   - Apply crisis gate
   - Measure results

📝 configs/hybrid_config.yaml - Configuration
   - ML threshold
   - Crisis sensitivity
   - Position sizing
   - Risk limits
```

---

## 🗓️ IMPLEMENTATION ROADMAP

### Week 2: Integration (5 days)

**Day 1-2: Create Hybrid Strategy**
```python
# models/hybrid_strategy.py
class HybridStrategy:
    def __init__(self):
        self.rule_based = SimpleRules()  # RSI < 30
        self.ml_filter = XGBoostModel.load('xgboost_model.json')
        self.crisis_gate = CrisisDetector()

    def should_enter(self, df, bar_index):
        # Layer 1: Rule-based signal
        if not self.rule_based.check(df, bar_index):
            return False, "No rule signal"

        # Layer 2: ML filter
        features = extract_features(df, bar_index)
        ml_score = self.ml_filter.predict_proba(features)[1]
        if ml_score < ML_THRESHOLD:
            return False, f"ML rejected (score: {ml_score:.2f})"

        # Layer 3: Crisis gate
        if self.crisis_gate.is_crisis(df, bar_index):
            return False, "Crisis detected"

        return True, f"All checks passed (ML: {ml_score:.2f})"
```

**Day 3-4: Backtest Hybrid**
```python
# scripts/hybrid_backtest.py
# Test on 8 years of BTC data
# Measure:
# - How many signals pass all 3 layers?
# - Win rate improvement?
# - Profit factor improvement?
# - Crisis protection works?
```

**Day 5: Analysis & Documentation**
```
Compare:
- Rule-based only
- Rule-based + ML
- Rule-based + ML + Crisis
- Full Hybrid

Document results in reports/hybrid_analysis/
```

### Week 3: Optimization (5 days)

**Day 1: Tune ML Threshold**
```
Test thresholds: 0.4, 0.5, 0.6, 0.7, 0.8
Find optimal trade-off:
- Signal frequency vs quality
- Win rate vs trade count
```

**Day 2: Tune Crisis Sensitivity**
```
Too sensitive: Miss profitable periods
Too relaxed: Don't protect from crashes
Find balance
```

**Day 3-4: Multi-Asset Test**
```
Test hybrid on:
- BTC, ETH (must work!)
- ALGO, AVAX, SOL (should work)
- 5 assets × 2 timeframes = 10 combos
```

**Day 5: Final Report**
```
Create comprehensive report:
- Hybrid vs Rule-based comparison
- Crisis protection analysis
- Production readiness assessment
```

---

## 🎯 SUCCESS CRITERIA

### Minimum (Must Have):

```
✅ Win rate > 50% (vs 47.3% baseline)
✅ Profit factor > 1.5 (vs 1.17 baseline)
✅ Crisis detection works (0 trades during COVID/Luna/FTX)
✅ At least 200 trades over 8 years (for statistics)
✅ Works on BTC and ETH
```

### Target (Should Have):

```
🎯 Win rate 55-60%
🎯 Profit factor 1.8-2.0
🎯 Max drawdown < 20%
🎯 Sharpe ratio > 2.0
🎯 Annual return 60-80% (realistic)
```

### Stretch (Nice to Have):

```
🌟 Works on 5+ assets
🌟 Works on 2+ timeframes
🌟 Profit factor > 2.0
🌟 Crisis detection 100% accurate
```

---

## ⚠️ RISKS & MITIGATION

### Risk 1: ML Overfitting
**Problem:** ML trained on same data we test on
**Mitigation:**
- Use walk-forward testing
- Out-of-sample validation
- Test on different assets

### Risk 2: Too Few Signals
**Problem:** Filters too aggressive → no trades
**Mitigation:**
- Tune ML threshold (start at 0.5, adjust)
- Monitor signal rejection rate
- Target: 30-40% reduction, not 90%

### Risk 3: Crisis False Positives
**Problem:** Detector stops trading during normal volatility
**Mitigation:**
- Tune crisis sensitivity
- Use multiple indicators
- Allow override for "mild volatility"

### Risk 4: Integration Bugs
**Problem:** Components don't work together
**Mitigation:**
- Unit tests for each layer
- Integration tests
- Start simple, add complexity gradually

---

## 📁 FILE STRUCTURE

```
scarlet-sails/
├── models/
│   ├── hybrid_strategy.py          # NEW: Unified entry system
│   ├── hybrid_position_manager.py  # EXISTS
│   ├── regime_detector.py          # EXISTS
│   ├── xgboost_model.py            # EXISTS
│   └── xgboost_model.json          # EXISTS (trained)
│
├── features/
│   ├── crisis_detection.py         # EXISTS
│   └── feature_extractor.py        # NEW: Unified feature extraction
│
├── configs/
│   └── hybrid_config.yaml          # NEW: Configuration
│
├── scripts/
│   ├── hybrid_backtest.py          # NEW: Test hybrid system
│   ├── hybrid_optimization.py      # NEW: Tune parameters
│   └── master_comprehensive_audit.py  # EXISTS (baseline)
│
└── reports/
    └── hybrid_analysis/            # NEW: Hybrid results
        ├── comparison.txt          # Rule vs Hybrid
        ├── crisis_protection.txt   # Crisis analysis
        └── results.json            # Full results
```

---

## 🚀 NEXT STEPS (RIGHT NOW)

1. ✅ Create this architecture document
2. ⏭️ Create implementation plan
3. ⏭️ Create hybrid_strategy.py skeleton
4. ⏭️ Create hybrid_backtest.py
5. ⏭️ Test on small dataset (1 month BTC)
6. ⏭️ Full backtest (8 years)
7. ⏭️ Commit all to GitHub

---

## 💡 KEY INSIGHTS

### Why Hybrid > Pure ML?

**Pure ML problems:**
- Needs tons of data to train
- Risk of overfitting
- "Black box" decisions
- Hard to debug

**Pure Rule-based problems:**
- Lots of false signals
- Can't adapt to patterns
- Fixed thresholds

**Hybrid advantages:**
- Rules generate signals (fast, explainable)
- ML filters quality (pattern recognition)
- Crisis detector protects (safety layer)
- Best of both worlds

### Expected Improvement:

```
Metric              Rule-based    Hybrid      Improvement
─────────────────────────────────────────────────────────
Win Rate            47.3%         55-60%      +16-27%
Profit Factor       1.17          1.8-2.0     +54-71%
Trades              7,464         3,000       -60% (good!)
False Signals       74%           40-50%      -46%
Crisis Protection   ❌            ✅          Priceless
```

---

**Status:** 📋 PLAN READY
**Ready to implement:** ✅ YES
**Estimated time:** 10 days (Week 2-3)
**Confidence level:** HIGH (both components work separately)

---

*This architecture combines the best of both worlds: frequency from rules, quality from ML, and protection from crisis detection.*
