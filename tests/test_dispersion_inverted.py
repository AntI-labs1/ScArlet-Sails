"""
Confidence-multiplier logic in RollingDispersionCalculator:

- Low dispersion  (agreement) → HIGH multiplier (high confidence → larger position)
- High dispersion (chaos)     → LOW multiplier  (low confidence → smaller position)

Historical note: an earlier version of this file asserted the opposite direction
and was kept failing as a documented bug in REMAINING_TEST_ISSUES.md. The source
was already correct; tests were the ones encoding the inverted (wrong) rule.
"""
import pytest

from core.rolling_dispersion import RollingDispersionCalculator


class TestDispersionConfidenceLogic:
    """Verify confidence-multiplier direction matches the documented rule."""

    def test_high_dispersion_gives_low_multiplier(self):
        """High dispersion (disagreement) must give a LOWER multiplier than low dispersion."""
        calc = RollingDispersionCalculator(window=50, min_samples=10)

        # Warmup history with stable, low-dispersion signals
        for _ in range(30):
            calc.update(p_rb=0.5, p_ml=0.5, p_hyb=0.5)

        state_low = calc.update(p_rb=0.5, p_ml=0.5, p_hyb=0.5)
        state_high = calc.update(p_rb=0.2, p_ml=0.8, p_hyb=0.5)

        assert state_high.confidence_multiplier <= state_low.confidence_multiplier, (
            f"High-dispersion mult {state_high.confidence_multiplier} "
            f"should be <= low-dispersion mult {state_low.confidence_multiplier}"
        )

    def test_multiplier_monotonically_decreases_with_dispersion(self):
        """As dispersion grows, multiplier should not grow."""
        calc = RollingDispersionCalculator(window=50, min_samples=10)

        for _ in range(30):
            calc.update(p_rb=0.5, p_ml=0.5, p_hyb=0.5)

        dispersion_inputs = [
            (0.50, 0.50, 0.50),  # zero dispersion
            (0.40, 0.60, 0.50),  # small
            (0.30, 0.70, 0.50),  # larger
            (0.20, 0.80, 0.50),  # largest
        ]

        multipliers = [
            calc.update(p_rb=p_rb, p_ml=p_ml, p_hyb=p_hyb).confidence_multiplier
            for p_rb, p_ml, p_hyb in dispersion_inputs
        ]

        for i in range(len(multipliers) - 1):
            assert multipliers[i] >= multipliers[i + 1], (
                f"Multiplier should not increase with dispersion: {multipliers}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
