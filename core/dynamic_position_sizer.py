"""
Dynamic Position Sizing Module.
Combines multiple signals into intelligent position sizing.

Inputs:
- P_hyb: Model probability
- Agreement: Strategy consensus
- Dispersion: Historical disagreement
- Regime: Market volatility state
- Drawdown: Current portfolio drawdown

Output:
- position_size: 0.0 to max_position (default 1.5)
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging
import yaml

from core.regime_detector import MarketRegime, RegimeState, REGIME_POSITION_MULTIPLIER
from core.rolling_dispersion import DispersionState

logger = logging.getLogger(__name__)


@dataclass
class PositionSizingInput:
    """All inputs for position sizing decision."""
    p_hyb: float                          # Hybrid probability [0, 1]
    agreement: float                       # Strategy agreement [0, 1]
    dispersion_state: Optional[DispersionState] = None
    regime_state: Optional[RegimeState] = None
    current_drawdown: float = 0.0         # Current DD as negative pct
    max_drawdown_limit: float = -0.30     # Stop trading at -30%


@dataclass
class PositionSizingOutput:
    """Position sizing decision with explanation."""
    position_size: float
    components: dict
    reasoning: str


class DynamicPositionSizer:
    """
    Calculates position size based on multiple factors.
    
    Formula:
    position = base * conviction * dispersion_mult * regime_mult * dd_mult
    
    Where:
    - base: 0.25 (minimum position)
    - conviction: f(|P_hyb - 0.5| * 2, agreement)
    - dispersion_mult: from dispersion state
    - regime_mult: from regime state
    - dd_mult: reduces as drawdown increases
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        base_position: float = 0.25,
        max_position: float = 1.5,
        conviction_weight: float = 0.5,
        agreement_weight: float = 0.5,
    ):
        if config_path:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
            self.max_position = self.config['trading']['risk_management']['max_position']
        else:
            # Honor the constructor parameter; config_path is the override path.
            self.max_position = max_position

        self.base_position = base_position
        self.conviction_weight = conviction_weight
        self.agreement_weight = agreement_weight
    
    def _conviction_from_probability(self, p_hyb: float) -> float:
        """
        Higher conviction when probability far from 0.5.
        P=0.5 → conviction=0
        P=0.0 or P=1.0 → conviction=1
        """
        return abs(p_hyb - 0.5) * 2
    
    def _drawdown_multiplier(self, dd: float, limit: float) -> float:
        """
        Reduce position as drawdown increases.
        DD=0% → mult=1.0
        DD=limit/2 → mult=0.5
        DD>=limit → mult=0.0 (stop trading)
        """
        if dd <= limit:
            return 0.0
        
        # Linear reduction from 0 to limit
        ratio = dd / limit  # 0 to 1 as DD approaches limit
        return max(0.0, 1.0 - ratio)
    
    def calculate(self, inputs: PositionSizingInput) -> PositionSizingOutput:
        """
        Calculate dynamic position size.
        
        Returns:
            PositionSizingOutput with size and reasoning
        """
        components = {}
        
        # 1. Base position
        components['base'] = self.base_position
        
        # 2. Conviction from probability
        conviction = self._conviction_from_probability(inputs.p_hyb)
        components['conviction'] = conviction
        
        # 3. Agreement contribution
        agreement = inputs.agreement if inputs.agreement is not None else 0.5
        components['agreement'] = agreement
        
        # 4. Combined signal strength
        signal_strength = (
            self.conviction_weight * conviction +
            self.agreement_weight * agreement
        )
        components['signal_strength'] = signal_strength
        
        # 5. Dispersion multiplier (inverted: high dispersion → high multiplier)
        if inputs.dispersion_state is not None:
            # Invert: high dispersion → high multiplier (more position)
            # low dispersion → low multiplier (less position)
            inverted_mult = 2.0 - inputs.dispersion_state.confidence_multiplier
            disp_mult = np.clip(inverted_mult, 0.5, 1.5)
        else:
            disp_mult = 1.0
        components['dispersion_mult'] = disp_mult
        
        # 6. Regime multiplier
        if inputs.regime_state is not None:
            regime_mult = REGIME_POSITION_MULTIPLIER.get(
                inputs.regime_state.regime, 1.0
            )
        else:
            regime_mult = 1.0
        components['regime_mult'] = regime_mult
        
        # 7. Drawdown multiplier
        dd_mult = self._drawdown_multiplier(
            inputs.current_drawdown,
            inputs.max_drawdown_limit
        )
        components['dd_mult'] = dd_mult
        
        # Final calculation
        position = (
            self.base_position +
            (self.max_position - self.base_position) * signal_strength
        ) * disp_mult * regime_mult * dd_mult
        
        position = np.clip(position, 0.0, self.max_position)
        components['final'] = position
        
        # Generate reasoning
        reasoning = self._generate_reasoning(inputs, components)
        
        return PositionSizingOutput(
            position_size=position,
            components=components,
            reasoning=reasoning,
        )
    
    def _generate_reasoning(
        self, 
        inputs: PositionSizingInput, 
        components: dict
    ) -> str:
        """Generate human-readable reasoning."""
        parts = []
        
        # Signal strength
        if components['signal_strength'] > 0.7:
            parts.append("Strong signal")
        elif components['signal_strength'] < 0.3:
            parts.append("Weak signal")
        
        # Regime
        if inputs.regime_state:
            regime = inputs.regime_state.regime
            if regime == MarketRegime.CRISIS:
                parts.append("CRISIS mode: minimal exposure")
            elif regime == MarketRegime.HIGH_VOL:
                parts.append("High volatility: reduced size")
            elif regime == MarketRegime.LOW_VOL:
                parts.append("Low volatility: increased size")
        
        # Drawdown
        if inputs.current_drawdown < -0.15:
            parts.append(f"DD={inputs.current_drawdown:.1%}: reducing risk")
        
        # Dispersion (inverted logic)
        if components['dispersion_mult'] < 0.7:
            parts.append("Low dispersion: reduced size")
        elif components['dispersion_mult'] > 1.2:
            parts.append("High dispersion: increased size")
        
        return "; ".join(parts) if parts else "Normal conditions"