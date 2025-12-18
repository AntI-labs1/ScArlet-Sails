# DEPRECATED FILES
Date: 2024-12-18

## Moved to archive/

| File | Reason | New Location |
|------|--------|-------------|
| strategies/xgboost_ml.py | Replaced by v3 | archive/deprecated_strategies/ |
| strategies/xgboost_ml_v2.py | Replaced by v3 | archive/deprecated_strategies/ |
| strategies/hybrid_v2.py | Replaced by hybrid_q_learner | archive/deprecated_hybrid/ |
| tests/_old_test_integration.py | Outdated API | archive/tests_legacy/ |
| tests/_old_test_real_data.py | Outdated API | archive/tests_legacy/ |

## Legacy but still in use

| File | Reason | Action Plan |
|------|--------|-------------|
| core/feature_engine.py | Only used in main.py (line 108) | Migrate to feature_engine_v2.py, then archive |

## Active Versions

| Component | Active File |
|-----------|-------------|
| ML Strategy (P_ml) | strategies/xgboost_ml_v3.py |
| Rule-Based (P_rb) | strategies/rule_based_v2.py |
| Hybrid (P_hyb) | strategies/hybrid_q_learner.py |
| Feature Engine | core/feature_engine_v2.py |

---

## Cleanup Status

✅ Deprecated strategies moved to archive/deprecated_strategies/  
✅ Deprecated tests moved to archive/tests_legacy/  
⚠️ core/feature_engine.py pending migration (main.py dependency)  
✅ PROJECT_STRUCTURE.md created to reflect current state
