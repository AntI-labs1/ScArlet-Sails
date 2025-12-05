import pandas as pd
from core.feature_engine_v2 import FeatureSpecV3
from strategies.xgboost_ml_v3 import XGBoostMLStrategyV3


def test_feature_spec_enforce_reorders_and_drops_extra():
    cols = [f"f{i}" for i in range(74)]
    data = pd.DataFrame({c: range(5) for c in cols})
    data["target"] = 1

    spec = FeatureSpecV3.from_dataframe(data)
    shuffled = data[cols[:10] + ["target"] + cols[10:]].copy()
    shuffled["extra"] = 0

    enforced = spec.enforce(shuffled)

    assert enforced.columns.tolist() == spec.feature_names
    assert "extra" not in enforced.columns


def test_validate_feature_order_reports_missing_and_extra():
    spec = FeatureSpecV3(feature_names=["a", "b", "c"])
    report = spec.validate(["b", "c", "d"])

    assert "a" in report["missing"]
    assert "d" in report["extra"]


def test_strategy_validate_feature_order_uses_spec():
    strategy = XGBoostMLStrategyV3(model_path=None)
    strategy.feature_names = ["x", "y"]

    report = strategy.validate_feature_order(pd.DataFrame(columns=["y", "x"]))

    assert not report["missing"]
    assert report["reordered"]
