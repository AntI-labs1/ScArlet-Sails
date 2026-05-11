"""
Tests for canonical feature sanitization.

Test philosophy:
- Hard safety tests: Verify inf/NaN removal (critical)
- Soft hygiene tests: Verify clipping works on realistic distributions
- Robustness tests: Verify function doesn't crash on edge cases
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sanitize_features import sanitize_for_model, validate_features


class TestHardSafety:
    """Critical: These MUST pass for XGBoost to not crash."""
    
    def test_removes_positive_inf(self):
        """Rows with +inf are removed."""
        df = pd.DataFrame({'a': [1.0, np.inf, 3.0], 'b': [4.0, 5.0, 6.0]})
        result = sanitize_for_model(df, ['a', 'b'])
        assert len(result) == 2
        assert not np.isinf(result['a']).any()
    
    def test_removes_negative_inf(self):
        """Rows with -inf are removed."""
        df = pd.DataFrame({'a': [1.0, -np.inf, 3.0], 'b': [4.0, 5.0, 6.0]})
        result = sanitize_for_model(df, ['a', 'b'])
        assert len(result) == 2
    
    def test_removes_nan_values(self):
        """Rows with NaN are removed."""
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, 6.0]})
        result = sanitize_for_model(df, ['a', 'b'])
        assert len(result) == 2
        assert not result['a'].isna().any()
    
    def test_removes_mixed_bad_values(self):
        """Handles multiple bad value types."""
        df = pd.DataFrame({
            'a': [1.0, np.inf, np.nan, -np.inf, 5.0],
            'b': [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result = sanitize_for_model(df, ['a', 'b'])
        assert len(result) == 2  # Only rows 0 and 4 survive
        assert not np.isinf(result['a']).any()
        assert not result['a'].isna().any()


class TestSoftHygiene:
    """Clipping extreme values - nice to have, not critical."""
    
    def test_clips_extreme_in_normal_distribution(self):
        """Clips outliers in normally distributed data."""
        np.random.seed(42)
        n = 1000
        normal_data = np.random.randn(n) * 10 + 50
        normal_data[-1] = 10000.0  # Add outlier
        
        df = pd.DataFrame({'a': normal_data})
        result = sanitize_for_model(df, ['a'], clip_extreme=True, clip_percentile=99.0)
        
        assert result['a'].max() < 10000.0
    
    def test_clips_both_tails(self):
        """Clips both high and low extremes."""
        np.random.seed(42)
        n = 1000
        data = np.random.randn(n) * 10 + 50
        data[0] = -10000.0
        data[-1] = 10000.0
        
        df = pd.DataFrame({'a': data})
        result = sanitize_for_model(df, ['a'], clip_extreme=True, clip_percentile=99.0)
        
        assert result['a'].min() > -10000.0
        assert result['a'].max() < 10000.0
    
    def test_spike_detection_in_constant_column(self):
        """Near-constant columns with spikes are handled without crash."""
        n = 100
        df = pd.DataFrame({
            'a': np.append(np.ones(n - 1), 10000.0),
        })
        
        result = sanitize_for_model(df, ['a'], clip_extreme=True)
        
        # Function completes without error
        assert len(result) == n
        # No inf/nan introduced
        assert not np.isinf(result['a']).any()
        assert not result['a'].isna().any()
    
    def test_no_clipping_when_disabled(self):
        """Clipping can be disabled."""
        df = pd.DataFrame({'a': [1.0, 2.0, 10000.0]})
        result = sanitize_for_model(df, ['a'], clip_extreme=False)
        assert result['a'].max() == 10000.0


class TestRobustness:
    """Function doesn't crash on edge cases."""
    
    def test_handles_missing_features(self):
        """Missing features don't cause error."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0]})
        result = sanitize_for_model(df, ['a', 'b', 'c'])
        assert len(result) == 3
    
    def test_handles_empty_feature_list(self):
        """Empty feature list doesn't crash."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0]})
        result = sanitize_for_model(df, [])
        assert len(result) == 3
    
    def test_returns_copy(self):
        """Original DataFrame unchanged."""
        df = pd.DataFrame({'a': [1.0, np.inf, 3.0]})
        original_len = len(df)
        _ = sanitize_for_model(df, ['a'])
        assert len(df) == original_len
    
    def test_handles_mixed_dtypes(self):
        """Mixed dtypes don't crash, strings preserved."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.0, 2.0, 3.0, 4.0, 5.0],
            'str_col': ['a', 'b', 'c', 'd', 'e'],
        })
        result = sanitize_for_model(df, ['int_col', 'float_col', 'str_col'])
        
        assert len(result) == 5
        assert not result.select_dtypes(include=[np.number]).isna().any().any()
        assert list(result['str_col']) == ['a', 'b', 'c', 'd', 'e']
    
    def test_handles_all_same_values(self):
        """Constant columns don't crash."""
        df = pd.DataFrame({'a': [5.0] * 100})
        result = sanitize_for_model(df, ['a'])
        assert len(result) == 100
        assert (result['a'] == 5.0).all()


class TestValidateFeatures:
    """Validation function tests."""
    
    def test_valid_data_passes(self):
        """Clean data validates."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
        is_valid, stats = validate_features(df, ['a', 'b'])
        assert is_valid
        assert stats['inf_values'] == 0
        assert stats['nan_values'] == 0
    
    def test_detects_inf(self):
        """Detects inf."""
        df = pd.DataFrame({'a': [1.0, np.inf, 3.0]})
        is_valid, stats = validate_features(df, ['a'])
        assert not is_valid
        assert stats['inf_values'] > 0
    
    def test_detects_nan(self):
        """Detects NaN."""
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0]})
        is_valid, stats = validate_features(df, ['a'])
        assert not is_valid
        assert stats['nan_values'] > 0


class TestRealData:
    """Integration with real feature files."""
    
    def test_real_features_sanitize_cleanly(self):
        """Real features can be sanitized."""
        path = Path('data/features/BTC_USDT_15m_features.parquet')
        meta_path = Path('models/xgboost_v3_btc_15m_metadata.json')
        if not path.exists():
            pytest.skip("Real data not available")
        if not meta_path.exists():
            pytest.skip("Model metadata not available")

        import json
        with open(meta_path) as f:
            meta = json.load(f)
        feature_names = meta.get('feature_names', [])

        df = pd.read_parquet(path)
        missing = [f for f in feature_names if f not in df.columns]
        if missing:
            pytest.skip(
                f"feature parquet is missing {len(missing)} columns the model expects; "
                "regenerate via feature_engine_v2 (stale on-disk artefact)."
            )

        # Test multiple slices
        for start in [0, 100000, len(df) - 1000]:
            sample = df.iloc[start:start + 1000]
            result = sanitize_for_model(sample, feature_names)
            is_valid, stats = validate_features(result, feature_names)

            assert is_valid, f"Slice {start} failed: {stats}"
            assert len(result) > 900, f"Slice {start}: too many rows dropped"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])