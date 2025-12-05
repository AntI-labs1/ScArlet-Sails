"""
Tests for temporal_split function with timezone handling.
Verifies correct behavior with both tz-aware and tz-naive datetime indices.
"""

import pandas as pd
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.train_xgboost_v3 import temporal_split


class TestTemporalSplitTimezone:
    """Test suite for temporal_split timezone handling."""

    def create_sample_df(self, start: str, end: str, freq: str = "1H", tz=None) -> pd.DataFrame:
        """Create a sample DataFrame with datetime index."""
        date_range = pd.date_range(start=start, end=end, freq=freq, tz=tz)
        df = pd.DataFrame({
            "feature1": range(len(date_range)),
            "feature2": range(len(date_range), 2 * len(date_range)),
            "target": [i % 2 for i in range(len(date_range))],
        }, index=date_range)
        return df

    def test_temporal_split_tz_aware_utc(self):
        """Test temporal_split with tz-aware (UTC) datetime index."""
        # Create DataFrame with UTC timezone
        df = self.create_sample_df(
            start="2020-01-01",
            end="2023-12-31",
            freq="1D",
            tz="UTC"
        )

        # Define split boundaries
        train_start = "2020-01-01"
        train_end = "2021-01-01"
        val_end = "2022-01-01"
        test_end = "2023-01-01"

        # Perform split
        train_df, val_df, test_df = temporal_split(
            df, train_start, train_end, val_end, test_end
        )

        # Verify no errors occurred
        assert train_df is not None
        assert val_df is not None
        assert test_df is not None

        # Verify splits are not empty
        assert len(train_df) > 0, "Train set should not be empty"
        assert len(val_df) > 0, "Val set should not be empty"
        assert len(test_df) > 0, "Test set should not be empty"

        # Verify temporal ordering
        assert train_df.index.max() < val_df.index.min(), "Train should end before val starts"
        assert val_df.index.max() < test_df.index.min(), "Val should end before test starts"

        # Verify timezone is preserved
        assert train_df.index.tz is not None
        assert val_df.index.tz is not None
        assert test_df.index.tz is not None

    def test_temporal_split_tz_naive(self):
        """Test temporal_split with tz-naive datetime index."""
        # Create DataFrame without timezone
        df = self.create_sample_df(
            start="2020-01-01",
            end="2023-12-31",
            freq="1D",
            tz=None
        )

        # Define split boundaries
        train_start = "2020-01-01"
        train_end = "2021-01-01"
        val_end = "2022-01-01"
        test_end = "2023-01-01"

        # Perform split
        train_df, val_df, test_df = temporal_split(
            df, train_start, train_end, val_end, test_end
        )

        # Verify no errors occurred
        assert train_df is not None
        assert val_df is not None
        assert test_df is not None

        # Verify splits are not empty
        assert len(train_df) > 0, "Train set should not be empty"
        assert len(val_df) > 0, "Val set should not be empty"
        assert len(test_df) > 0, "Test set should not be empty"

        # Verify temporal ordering
        assert train_df.index.max() < val_df.index.min(), "Train should end before val starts"
        assert val_df.index.max() < test_df.index.min(), "Val should end before test starts"

        # Verify timezone is None (tz-naive)
        assert train_df.index.tz is None
        assert val_df.index.tz is None
        assert test_df.index.tz is None

    def test_temporal_split_no_test_end(self):
        """Test temporal_split with test_end=None (use all remaining data)."""
        df = self.create_sample_df(
            start="2020-01-01",
            end="2023-12-31",
            freq="1D",
            tz="UTC"
        )

        train_start = "2020-01-01"
        train_end = "2021-01-01"
        val_end = "2022-01-01"
        test_end = None  # Use all remaining data

        train_df, val_df, test_df = temporal_split(
            df, train_start, train_end, val_end, test_end
        )

        # Test set should include all data from val_end onwards
        assert len(test_df) > 0
        assert test_df.index.min() >= pd.Timestamp(val_end, tz="UTC")
        assert test_df.index.max() == df.index.max()

    def test_temporal_split_non_datetime_index(self):
        """Test that temporal_split raises TypeError for non-DatetimeIndex."""
        # Create DataFrame with integer index
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [10, 20, 30, 40, 50],
        })

        with pytest.raises(TypeError, match="temporal_split expects a DatetimeIndex"):
            temporal_split(df, "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01")

    def test_temporal_split_empty_splits_warning(self):
        """Test that temporal_split handles out-of-range dates gracefully."""
        df = self.create_sample_df(
            start="2020-01-01",
            end="2020-12-31",
            freq="1D",
            tz="UTC"
        )

        # Use dates outside the data range
        train_start = "2021-01-01"  # After data ends
        train_end = "2022-01-01"
        val_end = "2023-01-01"
        test_end = "2024-01-01"

        train_df, val_df, test_df = temporal_split(
            df, train_start, train_end, val_end, test_end
        )

        # All splits should be empty
        assert len(train_df) == 0
        assert len(val_df) == 0
        assert len(test_df) == 0

    def test_temporal_split_boundary_precision(self):
        """Test that split boundaries are precise (inclusive start, exclusive end)."""
        df = self.create_sample_df(
            start="2020-01-01 00:00",
            end="2020-01-05 00:00",
            freq="1H",
            tz="UTC"
        )

        train_start = "2020-01-01 00:00"
        train_end = "2020-01-02 00:00"  # Exactly 24 hours
        val_end = "2020-01-03 00:00"
        test_end = "2020-01-04 00:00"

        train_df, val_df, test_df = temporal_split(
            df, train_start, train_end, val_end, test_end
        )

        # Train should include 00:00 on Jan 1 but exclude 00:00 on Jan 2
        assert train_df.index.min() == pd.Timestamp("2020-01-01 00:00", tz="UTC")
        assert train_df.index.max() == pd.Timestamp("2020-01-01 23:00", tz="UTC")

        # Val should include 00:00 on Jan 2 but exclude 00:00 on Jan 3
        assert val_df.index.min() == pd.Timestamp("2020-01-02 00:00", tz="UTC")
        assert val_df.index.max() == pd.Timestamp("2020-01-02 23:00", tz="UTC")

        # Test should include 00:00 on Jan 3 but exclude 00:00 on Jan 4
        assert test_df.index.min() == pd.Timestamp("2020-01-03 00:00", tz="UTC")
        assert test_df.index.max() == pd.Timestamp("2020-01-03 23:00", tz="UTC")


if __name__ == "__main__":
    # Run tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Running temporal_split timezone tests...\n")
    print("="*60)
    
    test_suite = TestTemporalSplitTimezone()
    
    tests = [
        ("test_temporal_split_tz_aware_utc", "tz-aware (UTC) index"),
        ("test_temporal_split_tz_naive", "tz-naive index"),
        ("test_temporal_split_no_test_end", "no test_end specified"),
        ("test_temporal_split_non_datetime_index", "non-DatetimeIndex error"),
        ("test_temporal_split_empty_splits_warning", "out-of-range dates"),
        ("test_temporal_split_boundary_precision", "boundary precision"),
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
    
    print("="*60)
    print(f"\nResults: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
        sys.exit(1)
