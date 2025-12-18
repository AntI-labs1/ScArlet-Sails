"""
Test that dispersion multiplier is INVERTED:
- High dispersion → High multiplier
- Low dispersion → Low multiplier
"""
import pytest
import numpy as np

from core.rolling_dispersion import RollingDispersionCalculator


class TestDispersionInvertedLogic:
    """Verify inverted dispersion logic."""
    
    def test_high_dispersion_gives_high_multiplier(self):
        """High dispersion should give higher multiplier."""
        calc = RollingDispersionCalculator(window=50, min_samples=10)
        
        # Fill with low dispersion history
        for _ in range(30):
            calc.update(p_rb=0.5, p_ml=0.5, p_hyb=0.5)
        
        # Low dispersion state
        state_low = calc.update(p_rb=0.5, p_ml=0.5, p_hyb=0.5)
        
        # High dispersion state
        state_high = calc.update(p_rb=0.2, p_ml=0.8, p_hyb=0.5)
        
        # High dispersion should give HIGHER multiplier
        assert state_high.confidence_multiplier >= state_low.confidence_multiplier, \
            f"High disp mult {state_high.confidence_multiplier} should be > low disp mult {state_low.confidence_multiplier}"
    
    def test_multiplier_increases_with_dispersion(self):
        """Multiplier should increase as dispersion increases."""
        calc = RollingDispersionCalculator(window=50, min_samples=10)
        
        # Warmup
        for _ in range(30):
            calc.update(p_rb=0.5, p_ml=0.5, p_hyb=0.5)
        
        # Test increasing dispersion
        dispersions = [
            (0.50, 0.50, 0.50),  # Low
            (0.40, 0.60, 0.50),  # Medium
            (0.30, 0.70, 0.50),  # High
            (0.20, 0.80, 0.50),  # Very high
        ]
        
        multipliers = []
        for p_rb, p_ml, p_hyb in dispersions:
            state = calc.update(p_rb=p_rb, p_ml=p_ml, p_hyb=p_hyb)
            multipliers.append(state.confidence_multiplier)
        
        # Should be monotonically increasing
        for i in range(len(multipliers) - 1):
            assert multipliers[i] <= multipliers[i+1], \
                f"Multiplier should increase: {multipliers}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


