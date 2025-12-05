"""
Tests for sanitize_features_and_target function in train_xgboost_v3.py.
Verifies correct handling of inf/-inf/NaN values before XGBoost training.
"""

import logging
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.train_xgboost_v3 import sanitize_features_and_target


class TestSanitizeFeaturesAndTarget:
    """Test suite for sanitize_features_and_target function."""

    def test_sanitize_clean_data(self):
        """Test that clean data passes through unchanged."""
        X = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])

        # Для чистых данных допускаем высокий порог, чтобы не было ValueError
        X_clean, y_clean = sanitize_features_and_target(
            X, y, context="test", max_bad_ratio=1.0
        )

        assert len(X_clean) == len(X)
        assert len(y_clean) == len(y)
        pd.testing.assert_frame_equal(X_clean, X)
        pd.testing.assert_series_equal(y_clean, y)

    def test_sanitize_inf_in_features(self):
        """Test removal of rows with inf values in features."""
        X = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, np.inf, 4.0, 5.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])

        # 1/5 = 20% bad → выше дефолтных 10%, поэтому поднимаем порог
        X_clean, y_clean = sanitize_features_and_target(
            X, y, context="test", max_bad_ratio=1.0
        )

        assert len(X_clean) == 4
        assert len(y_clean) == 4
        assert not np.isinf(X_clean.values).any()

    def test_sanitize_negative_inf_in_features(self):
        """Test removal of rows with -inf values in features."""
        X = pd.DataFrame(
            {
                "feature1": [1.0, -np.inf, 3.0, 4.0, 5.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])

        X_clean, y_clean = sanitize_features_and_target(
            X, y, context="test", max_bad_ratio=1.0
        )

        assert len(X_clean) == 4
        assert not np.isinf(X_clean.values).any()

    def test_sanitize_nan_in_features(self):
        """Test removal of rows with NaN values in features."""
        X = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, np.nan, 5.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])

        X_clean, y_clean = sanitize_features_and_target(
            X, y, context="test", max_bad_ratio=1.0
        )

        assert len(X_clean) == 4
        assert not X_clean.isna().any().any()

    def test_sanitize_inf_in_target(self):
        """Test removal of rows with inf values in target."""
        X = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        y = pd.Series([0, 1, np.inf, 1, 0])

        X_clean, y_clean = sanitize_features_and_target(
            X, y, context="test", max_bad_ratio=1.0
        )

        assert len(X_clean) == 4
        assert len(y_clean) == 4
        assert not np.isinf(y_clean.values).any()

    def test_sanitize_nan_in_target(self):
        """Test removal of rows with NaN values in target."""
        X = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        y = pd.Series([0, np.nan, 0, 1, 0])

        X_clean, y_clean = sanitize_features_and_target(
            X, y, context="test", max_bad_ratio=1.0
        )

        assert len(X_clean) == 4
        assert not y_clean.isna().any()

    def test_sanitize_multiple_bad_rows(self):
        """Test removal of multiple rows with various non-finite values."""
        X = pd.DataFrame(
            {
                "feature1": [1.0, np.inf, 3.0, np.nan, 5.0, 6.0],
                "feature2": [10.0, 20.0, np.inf, 40.0, 50.0, 60.0],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0, 1])

        # 3/6 = 50% bad → дефолт 10% вызвал бы ValueError, поэтому поднимаем порог
        X_clean, y_clean = sanitize_features_and_target(
            X, y, context="test", max_bad_ratio=1.0
        )

        assert len(X_clean) == 3
        assert not np.isinf(X_clean.values).any()
        assert not X_clean.isna().any().any()

    def test_sanitize_exceeds_threshold(self):
        """Test that ValueError is raised when too many rows are bad."""
        # 50% of rows have bad values, exceeds default 10% threshold
        X = pd.DataFrame(
            {
                "feature1": [1.0, np.inf, 3.0, np.nan, 5.0, np.inf],
                "feature2": [10.0, 20.0, np.inf, 40.0, 50.0, 60.0],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0, 1])

        with pytest.raises(ValueError, match="exceeds the threshold"):
            sanitize_features_and_target(X, y, context="test", max_bad_ratio=0.1)

    def test_sanitize_custom_threshold(self):
        """Test that custom threshold works correctly."""
        # 33% of rows have bad values
        X = pd.DataFrame(
            {
                "feature1": [1.0, np.inf, 3.0, np.nan, 5.0, 6.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0, 1])

        # Should pass with 50% threshold
        X_clean, y_clean = sanitize_features_and_target(
            X, y, context="test", max_bad_ratio=0.5
        )
        assert len(X_clean) == 4

        # Should fail with 20% threshold
        with pytest.raises(ValueError):
            sanitize_features_and_target(X, y, context="test", max_bad_ratio=0.2)

    def test_sanitize_empty_dataset(self):
        """Test handling of empty dataset."""
        X = pd.DataFrame()
        y = pd.Series(dtype=float)

        X_clean, y_clean = sanitize_features_and_target(X, y, context="test")

        assert len(X_clean) == 0
        assert len(y_clean) == 0

    def test_sanitize_preserves_index(self):
        """Test that DataFrame index is preserved after sanitization."""
        index = pd.date_range("2020-01-01", periods=5, freq="1H")
        X = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, np.inf, 4.0, 5.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
            },
            index=index,
        )
        y = pd.Series([0, 1, 0, 1, 0], index=index)

        X_clean, y_clean = sanitize_features_and_target(
            X, y, context="test", max_bad_ratio=1.0
        )

        assert isinstance(X_clean.index, pd.DatetimeIndex)
        assert isinstance(y_clean.index, pd.DatetimeIndex)
        assert len(X_clean) == 4

    def test_sanitize_no_inf_or_nan_remain(self):
        """Test that absolutely no inf/nan remain after sanitization."""
        X = pd.DataFrame(
            {
                "feature1": [1.0, np.inf, 3.0, -np.inf, 5.0, np.nan],
                "feature2": [10.0, 20.0, np.inf, 40.0, np.nan, 60.0],
                "feature3": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0, 1])

        X_clean, y_clean = sanitize_features_and_target(
            X, y, context="test", max_bad_ratio=1.0
        )

        assert np.isfinite(X_clean.values).all()
        assert np.isfinite(y_clean.values).all()
        assert not X_clean.isna().any().any()
        assert not y_clean.isna().any()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Running sanitize_features_and_target tests...\n")
    print("=" * 60)

    test_suite = TestSanitizeFeaturesAndTarget()

    tests = [
        ("test_sanitize_clean_data", "clean data unchanged"),
        ("test_sanitize_inf_in_features", "remove inf in features"),
        ("test_sanitize_negative_inf_in_features", "remove -inf in features"),
        ("test_sanitize_nan_in_features", "remove NaN in features"),
        ("test_sanitize_inf_in_target", "remove inf in target"),
        ("test_sanitize_nan_in_target", "remove NaN in target"),
        ("test_sanitize_multiple_bad_rows", "remove multiple bad rows"),
        ("test_sanitize_exceeds_threshold", "error when exceeds threshold"),
        ("test_sanitize_custom_threshold", "custom threshold works"),
        ("test_sanitize_empty_dataset", "handle empty dataset"),
        ("test_sanitize_preserves_index", "preserve DataFrame index"),
        ("test_sanitize_no_inf_or_nan_remain", "no inf/nan remain"),
    ]

    passed = 0
    failed = 0

    for test_method, description in tests:
        try:
            method = getattr(test_suite, test_method)
            method()
            print(f"✓ {description}")
            passed += 1
        except Exception as e:
            print(f"✗ {description}")
            print(f"  Error: {e}")
            failed += 1

    print("=" * 60)
    print(f"\nResults: {passed} passed, {failed} failed")

    if failed == 0:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
        sys.exit(1)
