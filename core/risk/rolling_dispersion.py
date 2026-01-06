import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Union, List, Optional

@dataclass
class DispersionState:
    current_std: float
    rolling_mean_std: float
    rolling_percentile: float
    confidence_multiplier: float
    n_samples: int

class RollingDispersionCalculator:
    """
    Calculates market regime stability based on model disagreement (dispersion).
    Fixed: Handles Zero Variance Trap correctly.
    """
    def __init__(self, window: int = 20, min_mult: float = 0.5, max_mult: float = 1.0):
        self.window = window
        self.min_mult = min_mult
        self.max_mult = max_mult
        self.std_history: List[float] = []
        self._last_state: Optional[DispersionState] = None

    def reset(self):
        self.std_history = []
        self._last_state = None

    def update(self, predictions: Union[Dict[str, float], pd.Series]) -> DispersionState:
        # 1. Extract values
        if isinstance(predictions, pd.Series):
            values = predictions.values
        elif isinstance(predictions, dict):
            values = np.array(list(predictions.values()))
        else:
            raise ValueError(f"Unsupported input type: {type(predictions)}")
            
        # 2. Calculate current dispersion
        if len(values) < 2:
            current_std = 0.0
        else:
            current_std = float(np.std(values))
            
        # 3. Update History
        self.std_history.append(current_std)
        if len(self.std_history) > self.window:
            self.std_history.pop(0)
            
        # 4. Calculate Relative Dispersion (Percentile)
        if len(self.std_history) < 2:
            percentile = 0.0 
        elif current_std < 1e-9:
            # FIX: Perfect Agreement (Zero Variance) -> Always 0th percentile (Best case)
            percentile = 0.0
        else:
            # Rank current_std in history
            sorted_hist = sorted(self.std_history)
            # Use strict less than to handle ties better in consistent chaos
            # or keep <= but handle 0 separately (which we did above)
            rank = sum(1 for x in sorted_hist if x <= current_std)
            percentile = rank / len(sorted_hist)
            
        # 5. Calculate Multiplier
        # Low Percentile (0.0) -> Max Mult (1.0)
        # High Percentile (1.0) -> Min Mult (0.5)
        multiplier = self.max_mult - (percentile * (self.max_mult - self.min_mult))
        
        # 6. Create State
        state = DispersionState(
            current_std=current_std,
            rolling_mean_std=float(np.mean(self.std_history)),
            rolling_percentile=percentile,
            confidence_multiplier=multiplier,
            n_samples=len(self.std_history)
        )
        self._last_state = state
        return state

    def get_state(self) -> Optional[DispersionState]:
        return self._last_state
