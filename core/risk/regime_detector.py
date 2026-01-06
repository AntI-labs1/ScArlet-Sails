"""
Market Regime Detector.
Classifies market state: LOW_VOL, NORMAL, HIGH_VOL, CRISIS.

Used by:
- Position sizing (reduce in HIGH_VOL/CRISIS)
- Threshold adjustment (stricter in LOW_VOL)
- Risk management
"""
import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class MarketRegime(Enum):
    LOW_VOL = "low_vol"
    NORMAL = "normal"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"


@dataclass
class RegimeState:
    regime: MarketRegime
    volatility: float
    volatility_percentile: float
    trend_strength: float
    confidence: float


class RegimeDetector:
    """
    Detects market regime based on volatility and trend.
    
    Uses rolling ATR percentile to classify:
    - LOW_VOL: ATR < 25th percentile
    - NORMAL: 25th <= ATR < 75th percentile
    - HIGH_VOL: 75th <= ATR < 95th percentile
    - CRISIS: ATR >= 95th percentile
    """
    
    def __init__(
        self,
        atr_period: int = 14,
        lookback: int = 100,
        low_vol_pct: float = 25.0,
        high_vol_pct: float = 75.0,
        crisis_pct: float = 95.0,
    ):
        self.atr_period = atr_period
        self.lookback = lookback
        self.low_vol_pct = low_vol_pct
        self.high_vol_pct = high_vol_pct
        self.crisis_pct = crisis_pct
        
        self._atr_history: list = []
    
    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate current ATR."""
        if len(df) < self.atr_period + 1:
            return 0.0
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr = np.maximum(
            high[-self.atr_period:] - low[-self.atr_period:],
            np.maximum(
                np.abs(high[-self.atr_period:] - np.roll(close, 1)[-self.atr_period:]),
                np.abs(low[-self.atr_period:] - np.roll(close, 1)[-self.atr_period:])
            )
        )
        
        atr = np.mean(tr)
        return atr / close[-1] * 100  # Normalize as percentage
    
    def _calculate_trend_strength(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate trend strength using price vs SMA."""
        if len(df) < period:
            return 0.0
        
        close = df['close'].values
        sma = np.mean(close[-period:])
        
        # Normalized distance from SMA
        trend = (close[-1] - sma) / sma * 100
        return np.clip(trend, -10, 10) / 10  # Normalize to [-1, 1]
    
    def detect(self, df: pd.DataFrame) -> RegimeState:
        """
        Detect current market regime.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            RegimeState with classification
        """
        if len(df) < self.atr_period + 1:
            return RegimeState(
                regime=MarketRegime.NORMAL,
                volatility=0.0,
                volatility_percentile=50.0,
                trend_strength=0.0,
                confidence=0.0,
            )
        
        # Calculate current ATR
        atr = self._calculate_atr(df)
        
        # Update history
        self._atr_history.append(atr)
        if len(self._atr_history) > self.lookback:
            self._atr_history = self._atr_history[-self.lookback:]
        
        # Calculate percentile
        if len(self._atr_history) < 20:
            percentile = 50.0
            confidence = len(self._atr_history) / 20
        else:
            percentile = (np.sum(np.array(self._atr_history) <= atr) / 
                         len(self._atr_history) * 100)
            confidence = 1.0
        
        # Classify regime
        if percentile >= self.crisis_pct:
            regime = MarketRegime.CRISIS
        elif percentile >= self.high_vol_pct:
            regime = MarketRegime.HIGH_VOL
        elif percentile < self.low_vol_pct:
            regime = MarketRegime.LOW_VOL
        else:
            regime = MarketRegime.NORMAL
        
        trend_strength = self._calculate_trend_strength(df)
        
        return RegimeState(
            regime=regime,
            volatility=atr,
            volatility_percentile=percentile,
            trend_strength=trend_strength,
            confidence=confidence,
        )
    
    def reset(self):
        """Reset history for new backtest run."""
        self._atr_history = []


# Regime-based adjustments
REGIME_POSITION_MULTIPLIER = {
    MarketRegime.LOW_VOL: 1.2,    # Can take larger positions
    MarketRegime.NORMAL: 1.0,     # Baseline
    MarketRegime.HIGH_VOL: 0.7,   # Reduce exposure
    MarketRegime.CRISIS: 0.3,     # Minimal exposure
}

REGIME_THRESHOLD_ADJUSTMENT = {
    MarketRegime.LOW_VOL: 0.05,   # Stricter (fewer trades)
    MarketRegime.NORMAL: 0.0,     # Baseline 0.70
    MarketRegime.HIGH_VOL: -0.05, # Looser (more opportunities)
    MarketRegime.CRISIS: 0.10,    # Very strict (only high-confidence)
}