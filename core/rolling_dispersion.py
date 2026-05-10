"""
Rolling Dispersion Calculator for Real-Time Position Sizing.

Tracks historical dispersion between P_rb, P_ml, P_hyb over rolling window.
CORRECTED LOGIC: Low dispersion (agreement) → high multiplier (more position).
High dispersion (chaos) → low multiplier (less position).

Integration: QuantAggregator._suggest_position_size()
"""
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class DispersionState:
    """Current dispersion metrics."""
    current_std: float           # Standard deviation of [P_rb, P_ml, P_hyb]
    rolling_mean_std: float      # Mean std over window
    rolling_percentile: float    # Current std percentile vs history (0-1)
    confidence_multiplier: float # Position size multiplier (0.3 - 1.5)
    n_samples: int               # Samples in rolling window


class RollingDispersionCalculator:
    """
    Calculates rolling dispersion metrics for position sizing.
    
    Logic:
    - Track std(P_rb, P_ml, P_hyb) over rolling window
    - Compare current dispersion to historical distribution
    - CORRECTED: Low dispersion (agreement) → high multiplier (more position)
    - High dispersion (chaos) → low multiplier (less position)
    
    Usage:
        calc = RollingDispersionCalculator(window=100)
        
        # On each bar:
        state = calc.update(p_rb=0.7, p_ml=0.65, p_hyb=0.72)
        position_multiplier = state.confidence_multiplier
    """
    
    def __init__(
        self,
        window: int = 100,
        min_samples: int = 20,
        low_dispersion_threshold: float = 0.05,
        high_dispersion_threshold: float = 0.20,
        min_multiplier: float = 0.3,
        max_multiplier: float = 1.5,
    ):
        """
        Initialize calculator.
        
        Args:
            window: Rolling window size (bars)
            min_samples: Minimum samples before calculating percentile
            low_dispersion_threshold: Below this = high confidence
            high_dispersion_threshold: Above this = low confidence
            min_multiplier: Minimum position multiplier
            max_multiplier: Maximum position multiplier
        """
        self.window = window
        self.min_samples = min_samples
        self.low_threshold = low_dispersion_threshold
        self.high_threshold = high_dispersion_threshold
        self.min_mult = min_multiplier
        self.max_mult = max_multiplier
        
        # Rolling history of std values
        self._std_history: deque = deque(maxlen=window)
        
        # Cache for efficiency
        self._last_state: Optional[DispersionState] = None
        
        logger.info(
            f"RollingDispersionCalculator initialized: "
            f"window={window}, thresholds=[{low_dispersion_threshold}, {high_dispersion_threshold}]"
        )
    
    def update(
        self,
        p_rb: Optional[float] = None,
        p_ml: Optional[float] = None,
        p_hyb: Optional[float] = None,
    ) -> DispersionState:
        """
        Update with new signals and return current dispersion state.
        
        Args:
            p_rb: Rule-based probability (0-1)
            p_ml: ML probability (0-1)
            p_hyb: Hybrid probability (0-1)
            
        Returns:
            DispersionState with current metrics
        """
        # Collect valid signals
        signals = [s for s in [p_rb, p_ml, p_hyb] if s is not None]
        
        # Calculate current dispersion
        if len(signals) >= 2:
            current_std = float(np.std(signals))
        elif len(signals) == 1:
            current_std = 0.0  # Single signal = no dispersion
        else:
            # No signals - return neutral state
            return DispersionState(
                current_std=0.0,
                rolling_mean_std=0.0,
                rolling_percentile=0.5,
                confidence_multiplier=1.0,
                n_samples=0,
            )
        
        # Add to history
        self._std_history.append(current_std)
        n_samples = len(self._std_history)
        
        # Calculate rolling statistics
        history_array = np.array(self._std_history)
        rolling_mean_std = float(np.mean(history_array))
        
        # Calculate percentile (where current std falls in history)
        if n_samples >= self.min_samples:
            rolling_percentile = float(np.mean(history_array <= current_std))
        else:
            rolling_percentile = 0.5  # Neutral until enough samples
        
        # Calculate confidence multiplier
        confidence_multiplier = self._calculate_multiplier(
            current_std, rolling_percentile, n_samples
        )
        
        state = DispersionState(
            current_std=current_std,
            rolling_mean_std=rolling_mean_std,
            rolling_percentile=rolling_percentile,
            confidence_multiplier=confidence_multiplier,
            n_samples=n_samples,
        )
        
        self._last_state = state
        return state
    
    def _calculate_multiplier(
        self,
        current_std: float,
        percentile: float,
        n_samples: int,
    ) -> float:
        """
        Calculate position multiplier based on dispersion.
        
        CORRECTED LOGIC:
        - Low dispersion (percentile→0, agreement) → high multiplier (max_mult = 1.5)
        - High dispersion (percentile→1, chaos) → low multiplier (min_mult = 0.3)
        
        Rationale: Agreement between models = high confidence = larger position
                   Disagreement = uncertainty = smaller position
        """
        # Before enough samples: use simple threshold logic
        if n_samples < self.min_samples:
            if current_std <= self.low_threshold:
                return self.max_mult  # Low dispersion → large position (agreement)
            elif current_std >= self.high_threshold:
                return self.min_mult  # High dispersion → small position (chaos)
            else:
                # Linear interpolation (inverted: low std → high mult)
                ratio = (current_std - self.low_threshold) / (
                    self.high_threshold - self.low_threshold
                )
                # Invert: as ratio increases, multiplier decreases
                return self.max_mult - ratio * (self.max_mult - self.min_mult)
        
        # After warmup: use percentile (CORRECTED)
        # percentile=0 (low dispersion, agreement) → max_mult
        # percentile=1 (high dispersion, chaos) → min_mult
        # Formula: multiplier = max_mult - (percentile * (max_mult - min_mult))
        return self.max_mult - percentile * (self.max_mult - self.min_mult)
    
    def get_state(self) -> Optional[DispersionState]:
        """Get last computed state without updating."""
        return self._last_state
    
    def reset(self) -> None:
        """Reset rolling history."""
        self._std_history.clear()
        self._last_state = None
        logger.info("RollingDispersionCalculator reset")
    
    def get_statistics(self) -> dict:
        """Get summary statistics of rolling window."""
        if len(self._std_history) == 0:
            return {"n_samples": 0}
        
        history = np.array(self._std_history)
        return {
            "n_samples": len(history),
            "mean_std": float(np.mean(history)),
            "median_std": float(np.median(history)),
            "min_std": float(np.min(history)),
            "max_std": float(np.max(history)),
            "std_of_std": float(np.std(history)),
        }


def integrate_dispersion_with_position_sizing(
    base_position: float,
    dispersion_state: DispersionState,
    agreement: Optional[float] = None,
) -> Tuple[float, str]:
    """
    Combine dispersion metrics with base position sizing.
    
    Args:
        base_position: Base position size (e.g., from Kelly or fixed)
        dispersion_state: Current dispersion metrics
        agreement: Optional agreement score (1 - max_spread)
        
    Returns:
        (adjusted_position, justification)
    """
    # Apply dispersion multiplier (corrected logic: agreement → high multiplier)
    adjusted_mult = dispersion_state.confidence_multiplier
    adjusted = base_position * adjusted_mult
    
    # Optional: Further adjust by agreement if provided
    if agreement is not None:
        # agreement 0.5 → 0.7x, agreement 1.0 → 1.0x
        agreement_mult = 0.7 + 0.3 * agreement
        adjusted *= agreement_mult
    
    # Clamp to reasonable bounds
    adjusted = float(np.clip(adjusted, 0.1, 2.0))
    
    # Build justification
    justification = (
        f"Dispersion: std={dispersion_state.current_std:.3f} "
        f"(p{dispersion_state.rolling_percentile*100:.0f}), "
        f"mult={dispersion_state.confidence_multiplier:.2f}"
    )
    
    return adjusted, justification


# Convenience function for quick integration
def create_dispersion_calculator(
    window: int = 100,
    conservative: bool = False,
) -> RollingDispersionCalculator:
    """
    Factory function to create calculator with presets.
    
    Args:
        window: Rolling window size
        conservative: If True, use tighter multiplier range
    """
    if conservative:
        return RollingDispersionCalculator(
            window=window,
            min_multiplier=0.5,
            max_multiplier=1.2,
        )
    else:
        return RollingDispersionCalculator(
            window=window,
            min_multiplier=0.3,
            max_multiplier=1.5,
        )