"""
Tests for OOD Detector.
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.risk.ood_detector import OODDetector, OODState


class TestOODDetector:
    """Tests for OOD detection."""
    
    @pytest.fixture
    def fitted_detector(self):
        """Create fitted detector with known distribution."""
        np.random.seed(42)
        # Training data: standard normal
        X_train = np.random.randn(1000, 10)
        
        detector = OODDetector(threshold_sigma=3.0)
        detector.fit(X_train)
        return detector
    
    def test_initialization(self):
        """Detector initializes correctly."""
        detector = OODDetector()
        assert detector.threshold_sigma == 3.0
        assert detector.kappa == 0.1
        assert not detector._fitted
    
    def test_fit(self):
        """Fit computes statistics."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        
        detector = OODDetector()
        detector.fit(X)
        
        assert detector._fitted
        assert detector.mean is not None
        assert detector.cov_inv is not None
        assert detector.n_features == 5
    
    def test_normal_sample_not_ood(self, fitted_detector):
        """Normal sample is not detected as OOD."""
        np.random.seed(42)
        x_normal = np.random.randn(10)  # From same distribution
        
        state = fitted_detector.detect(x_normal)
        
        assert isinstance(state, OODState)
        assert not state.is_ood
        assert state.ood_penalty == 0.0
        assert state.confidence_multiplier == 1.0
    
    def test_extreme_sample_is_ood(self, fitted_detector):
        """Extreme sample is detected as OOD."""
        x_extreme = np.ones(10) * 10  # Far from normal
        
        state = fitted_detector.detect(x_extreme)
        
        assert state.is_ood
        assert state.ood_penalty > 0
        assert state.confidence_multiplier < 1.0
        assert state.mahalanobis_distance > fitted_detector.threshold
    
    def test_confidence_multiplier_range(self, fitted_detector):
        """Confidence multiplier stays in valid range."""
        test_cases = [
            np.zeros(10),
            np.ones(10) * 5,
            np.ones(10) * 10,
            np.ones(10) * 100,
        ]
        
        for x in test_cases:
            state = fitted_detector.detect(x)
            assert 0 < state.confidence_multiplier <= 1.0
    
    def test_percentile_ordering(self, fitted_detector):
        """More extreme samples have higher percentile."""
        x_normal = np.zeros(10)
        x_extreme = np.ones(10) * 5
        
        state_normal = fitted_detector.detect(x_normal)
        state_extreme = fitted_detector.detect(x_extreme)
        
        assert state_extreme.percentile > state_normal.percentile
    
    def test_save_and_load(self, fitted_detector):
        """Detector can be saved and loaded."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        
        fitted_detector.save(path)
        
        loaded = OODDetector()
        loaded.load(path)
        
        assert loaded._fitted
        assert loaded.n_features == fitted_detector.n_features
        np.testing.assert_array_almost_equal(loaded.mean, fitted_detector.mean)
        
        # Same detection results
        x = np.random.randn(10)
        state1 = fitted_detector.detect(x)
        state2 = loaded.detect(x)
        
        assert abs(state1.mahalanobis_distance - state2.mahalanobis_distance) < 1e-6
        
        Path(path).unlink()
    
    def test_unfitted_detector_returns_default(self):
        """Unfitted detector returns safe defaults."""
        detector = OODDetector()
        x = np.random.randn(10)
        
        state = detector.detect(x)
        
        assert not state.is_ood
        assert state.confidence_multiplier == 1.0
        assert state.ood_penalty == 0.0
    
    def test_feature_count_mismatch_warning(self, fitted_detector):
        """Warning on feature count mismatch."""
        x_wrong_size = np.random.randn(5)  # Wrong size
        
        with pytest.warns(UserWarning):
            state = fitted_detector.detect(x_wrong_size)
        
        assert state.confidence_multiplier == 1.0  # Safe default


class TestOODIntegration:
    """Integration tests with real-like data."""
    
    def test_regime_change_detection(self):
        """Detect regime change as OOD."""
        np.random.seed(42)
        
        # Training: calm market (low volatility)
        X_calm = np.random.randn(500, 10) * 0.5
        
        # Test: volatile market
        X_volatile = np.random.randn(100, 10) * 3.0
        
        detector = OODDetector()
        detector.fit(X_calm)
        
        # Calm samples should not be OOD
        calm_ood_count = sum(
            detector.detect(X_calm[i]).is_ood 
            for i in range(100)
        )
        
        # Volatile samples should be OOD
        volatile_ood_count = sum(
            detector.detect(X_volatile[i]).is_ood 
            for i in range(100)
        )
        
        assert volatile_ood_count > calm_ood_count
        assert volatile_ood_count > 0  # At least some volatile detected
    
    def test_gradual_shift_detection(self):
        """Detect gradual distribution shift."""
        np.random.seed(42)
        
        # Training: centered at 0
        X_train = np.random.randn(500, 5)
        
        detector = OODDetector()
        detector.fit(X_train)
        
        # Test samples with increasing shift
        distances = []
        for shift in [0, 1, 2, 3, 4, 5]:
            x = np.ones(5) * shift
            state = detector.detect(x)
            distances.append(state.mahalanobis_distance)
        
        # Distances should increase with shift
        assert all(distances[i] <= distances[i+1] for i in range(len(distances)-1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])