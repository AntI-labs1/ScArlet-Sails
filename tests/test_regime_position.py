"""
Tests for Regime Detector and Dynamic Position Sizer.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.risk.regime_detector import (
    RegimeDetector, 
    MarketRegime, 
    RegimeState,
    REGIME_POSITION_MULTIPLIER,
)
from core.risk.dynamic_position_sizer import (
    DynamicPositionSizer,
    PositionSizingInput,
    PositionSizingOutput,
)


class TestRegimeDetector:
    """Tests for market regime detection."""
    
    @pytest.fixture
    def detector(self):
        return RegimeDetector()
    
    @pytest.fixture
    def sample_ohlcv(self):
        """Generate sample OHLCV data."""
        np.random.seed(42)
        n = 100
        close = 50000 + np.cumsum(np.random.randn(n) * 100)
        return pd.DataFrame({
            'open': close * (1 + np.random.randn(n) * 0.001),
            'high': close * (1 + np.abs(np.random.randn(n)) * 0.002),
            'low': close * (1 - np.abs(np.random.randn(n)) * 0.002),
            'close': close,
            'volume': np.random.exponential(1000000, n),
        })
    
    def test_detector_initialization(self, detector):
        """Detector initializes correctly."""
        assert detector.atr_period == 14
        assert detector.lookback == 100
    
    def test_detect_returns_regime_state(self, detector, sample_ohlcv):
        """Detection returns RegimeState."""
        state = detector.detect(sample_ohlcv)
        
        assert isinstance(state, RegimeState)
        assert isinstance(state.regime, MarketRegime)
        assert 0 <= state.volatility_percentile <= 100
        assert -1 <= state.trend_strength <= 1
    
    def test_volatility_difference(self, detector):
        """High vol data has higher ATR than low vol."""
        np.random.seed(42)
        n = 50
        
        # Low volatility
        close_low = 50000 + np.cumsum(np.random.randn(n) * 10)
        df_low = pd.DataFrame({
            'open': close_low,
            'high': close_low * 1.0001,
            'low': close_low * 0.9999,
            'close': close_low,
            'volume': np.ones(n) * 1000000,
        })
        
        # High volatility
        close_high = 50000 + np.cumsum(np.random.randn(n) * 1000)
        df_high = pd.DataFrame({
            'open': close_high,
            'high': close_high * 1.05,
            'low': close_high * 0.95,
            'close': close_high,
            'volume': np.ones(n) * 1000000,
        })
        
        detector1 = RegimeDetector()
        detector2 = RegimeDetector()
        
        state_low = detector1.detect(df_low)
        state_high = detector2.detect(df_high)
        
        # High vol should have higher ATR value
        assert state_high.volatility > state_low.volatility
    
    def test_reset_clears_history(self, detector, sample_ohlcv):
        """Reset clears ATR history."""
        detector.detect(sample_ohlcv)
        assert len(detector._atr_history) > 0
        
        detector.reset()
        assert len(detector._atr_history) == 0
    
    def test_regime_multipliers_defined(self):
        """All regimes have position multipliers."""
        for regime in MarketRegime:
            assert regime in REGIME_POSITION_MULTIPLIER
    
    def test_all_regime_types_exist(self):
        """All expected regime types defined."""
        expected = ['LOW_VOL', 'NORMAL', 'HIGH_VOL', 'CRISIS']
        for name in expected:
            assert hasattr(MarketRegime, name)


class TestDynamicPositionSizer:
    """Tests for dynamic position sizing."""
    
    @pytest.fixture
    def sizer(self):
        return DynamicPositionSizer()
    
    def test_sizer_initialization(self, sizer):
        """Sizer initializes with defaults."""
        assert sizer.base_position == 0.25
        assert sizer.max_position == 1.5
    
    def test_basic_sizing(self, sizer):
        """Basic sizing without optional inputs."""
        inputs = PositionSizingInput(
            p_hyb=0.8,
            agreement=0.7,
        )
        output = sizer.calculate(inputs)
        
        assert isinstance(output, PositionSizingOutput)
        assert 0 < output.position_size <= sizer.max_position
        assert isinstance(output.reasoning, str)
    
    def test_high_conviction_larger_position(self, sizer):
        """High conviction → larger position."""
        low_conv = PositionSizingInput(p_hyb=0.55, agreement=0.5)
        high_conv = PositionSizingInput(p_hyb=0.95, agreement=0.9)
        
        out_low = sizer.calculate(low_conv)
        out_high = sizer.calculate(high_conv)
        
        assert out_high.position_size > out_low.position_size
    
    def test_drawdown_reduces_position(self, sizer):
        """Drawdown reduces position size."""
        no_dd = PositionSizingInput(p_hyb=0.8, agreement=0.7, current_drawdown=0.0)
        high_dd = PositionSizingInput(p_hyb=0.8, agreement=0.7, current_drawdown=-0.20)
        
        out_no_dd = sizer.calculate(no_dd)
        out_high_dd = sizer.calculate(high_dd)
        
        assert out_high_dd.position_size < out_no_dd.position_size
    
    def test_crisis_regime_minimal_position(self, sizer):
        """Crisis regime → minimal position."""
        crisis_state = RegimeState(
            regime=MarketRegime.CRISIS,
            volatility=5.0,
            volatility_percentile=98.0,
            trend_strength=0.0,
            confidence=1.0,
        )
        
        inputs = PositionSizingInput(
            p_hyb=0.9,
            agreement=0.9,
            regime_state=crisis_state,
        )
        output = sizer.calculate(inputs)
        
        assert output.position_size < 0.5
    
    def test_position_within_bounds(self, sizer):
        """Position always within [0, max_position]."""
        test_cases = [
            PositionSizingInput(p_hyb=0.0, agreement=0.0),
            PositionSizingInput(p_hyb=1.0, agreement=1.0),
            PositionSizingInput(p_hyb=0.5, agreement=0.5, current_drawdown=-0.35),
        ]
        
        for inputs in test_cases:
            output = sizer.calculate(inputs)
            assert 0.0 <= output.position_size <= sizer.max_position
    
    def test_output_has_components(self, sizer):
        """Output includes all components."""
        inputs = PositionSizingInput(p_hyb=0.8, agreement=0.7)
        output = sizer.calculate(inputs)
        
        expected_keys = ['base', 'conviction', 'agreement', 'signal_strength',
                        'dispersion_mult', 'regime_mult', 'dd_mult', 'final']
        
        for key in expected_keys:
            assert key in output.components
    
    def test_regime_affects_position(self, sizer):
        """Different regimes produce different positions."""
        base_inputs = {'p_hyb': 0.8, 'agreement': 0.7}
        
        normal = PositionSizingInput(
            **base_inputs,
            regime_state=RegimeState(
                regime=MarketRegime.NORMAL,
                volatility=1.0,
                volatility_percentile=50.0,
                trend_strength=0.0,
                confidence=1.0,
            )
        )
        
        crisis = PositionSizingInput(
            **base_inputs,
            regime_state=RegimeState(
                regime=MarketRegime.CRISIS,
                volatility=5.0,
                volatility_percentile=98.0,
                trend_strength=0.0,
                confidence=1.0,
            )
        )
        
        out_normal = sizer.calculate(normal)
        out_crisis = sizer.calculate(crisis)
        
        assert out_normal.position_size > out_crisis.position_size


class TestIntegration:
    """Integration tests for regime + position sizing."""
    
    def test_full_pipeline(self):
        """Full pipeline: OHLCV → Regime → Position."""
        np.random.seed(42)
        
        n = 100
        close = 50000 + np.cumsum(np.random.randn(n) * 100)
        df = pd.DataFrame({
            'open': close,
            'high': close * 1.001,
            'low': close * 0.999,
            'close': close,
            'volume': np.ones(n) * 1000000,
        })
        
        detector = RegimeDetector()
        regime_state = detector.detect(df)
        
        sizer = DynamicPositionSizer()
        inputs = PositionSizingInput(
            p_hyb=0.75,
            agreement=0.65,
            regime_state=regime_state,
        )
        output = sizer.calculate(inputs)
        
        assert output.position_size > 0
        assert len(output.reasoning) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])