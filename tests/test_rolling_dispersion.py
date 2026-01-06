"""
Tests for Rolling Dispersion Calculator.
"""
import pytest
import numpy as np
from core.risk.rolling_dispersion import (
    RollingDispersionCalculator,
    DispersionState,
    integrate_dispersion_with_position_sizing,
    create_dispersion_calculator,
)


class TestRollingDispersionCalculator:
    """Tests for RollingDispersionCalculator."""
    
    def test_initialization(self):
        """Calculator initializes correctly."""
        calc = RollingDispersionCalculator(window=50)
        assert calc.window == 50
        assert calc.get_state() is None
    
    def test_single_update(self):
        """Single update returns valid state."""
        calc = RollingDispersionCalculator()
        state = calc.update(p_rb=0.7, p_ml=0.65, p_hyb=0.72)
        
        assert isinstance(state, DispersionState)
        assert state.current_std > 0
        assert state.n_samples == 1
        assert 0.3 <= state.confidence_multiplier <= 1.5
    
    def test_no_signals_returns_neutral(self):
        """No signals returns neutral state."""
        calc = RollingDispersionCalculator()
        state = calc.update()
        
        assert state.current_std < 1e-10  
        assert state.confidence_multiplier == 1.0
        assert state.n_samples == 0
    
    def test_single_signal_no_dispersion(self):
        """Single signal has zero dispersion."""
        calc = RollingDispersionCalculator()
        state = calc.update(p_rb=0.7)
        
        assert state.current_std < 1e-10  # Machine precision zero
    
    def test_perfect_agreement_low_dispersion(self):
        """Perfect agreement results in low dispersion."""
        calc = RollingDispersionCalculator()
        state = calc.update(p_rb=0.7, p_ml=0.7, p_hyb=0.7)
        
        assert state.current_std < 1e-10  # Machine precision zero
        assert state.confidence_multiplier == calc.min_mult
    
    def test_high_disagreement_high_dispersion(self):
        """High disagreement results in high dispersion."""
        calc = RollingDispersionCalculator()
        state = calc.update(p_rb=0.2, p_ml=0.8, p_hyb=0.5)
        
        assert state.current_std > 0.2
        assert state.confidence_multiplier >= 1.0
    
    def test_rolling_window_fills(self):
        """Rolling window accumulates samples."""
        calc = RollingDispersionCalculator(window=10)
        
        for i in range(15):
            state = calc.update(p_rb=0.5 + i*0.01, p_ml=0.5, p_hyb=0.5)
        
        # Window should be capped at 10
        assert state.n_samples == 10
    
    def test_percentile_calculation(self):
        """Percentile calculated after min_samples."""
        calc = RollingDispersionCalculator(window=50, min_samples=10)
        
        # Fill with low dispersion
        for _ in range(15):
            calc.update(p_rb=0.5, p_ml=0.5, p_hyb=0.5)
        
        # Now add high dispersion - should be high percentile
        state = calc.update(p_rb=0.2, p_ml=0.8, p_hyb=0.5)
        
        assert state.rolling_percentile > 0.8  # Should be top percentile
    
    def test_reset(self):
        """Reset clears history."""
        calc = RollingDispersionCalculator()
        calc.update(p_rb=0.7, p_ml=0.6, p_hyb=0.65)
        calc.reset()
        
        assert calc.get_state() is None
        stats = calc.get_statistics()
        assert stats["n_samples"] == 0
    
    def test_statistics(self):
        """Statistics computed correctly."""
        calc = RollingDispersionCalculator(window=10)
        
        for i in range(10):
            calc.update(p_rb=0.5 + i*0.02, p_ml=0.5, p_hyb=0.5)
        
        stats = calc.get_statistics()
        
        assert stats["n_samples"] == 10
        assert "mean_std" in stats
        assert "median_std" in stats
        assert stats["min_std"] <= stats["mean_std"] <= stats["max_std"]


class TestIntegration:
    """Tests for integration function."""
    
    def test_integrate_with_base_position(self):
        """Integration adjusts base position."""
        state = DispersionState(
            current_std=0.1,
            rolling_mean_std=0.1,
            rolling_percentile=0.5,
            confidence_multiplier=1.0,
            n_samples=50,
        )
        
        adjusted, justification = integrate_dispersion_with_position_sizing(
            base_position=1.0,
            dispersion_state=state,
        )
        
        assert adjusted == 1.0  # multiplier is 1.0
        assert "Dispersion" in justification
    
    def test_integrate_with_agreement(self):
        """Integration uses agreement when provided."""
        state = DispersionState(
            current_std=0.1,
            rolling_mean_std=0.1,
            rolling_percentile=0.5,
            confidence_multiplier=1.0,
            n_samples=50,
        )
        
        # Low agreement should reduce position
        adjusted_low, _ = integrate_dispersion_with_position_sizing(
            base_position=1.0,
            dispersion_state=state,
            agreement=0.5,
        )
        
        # High agreement should not reduce much
        adjusted_high, _ = integrate_dispersion_with_position_sizing(
            base_position=1.0,
            dispersion_state=state,
            agreement=1.0,
        )
        
        assert adjusted_low < adjusted_high


class TestFactory:
    """Tests for factory function."""
    
    def test_create_default(self):
        """Default calculator created."""
        calc = create_dispersion_calculator()
        assert calc.min_mult == 0.3
        assert calc.max_mult == 1.5
    
    def test_create_conservative(self):
        """Conservative calculator has tighter range."""
        calc = create_dispersion_calculator(conservative=True)
        assert calc.min_mult == 0.5
        assert calc.max_mult == 1.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])