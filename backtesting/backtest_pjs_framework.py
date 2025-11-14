#!/usr/bin/env python3
"""
P_j(S) Framework - Backtest Engine
==================================

Реализует формулу:
P_j(S) = ML(market_state, portfolio_state, risk, regime, history)
       × ∏I_k                          (фильтры)
       × opportunity(S)                (выгодность)
       - costs(S)                      (издержки)
       - risk_penalty(S)              (штраф за риск)
       + γ·E[V_future]                (будущая ценность - позже для RL)

Архитектура:
1. ML component (XGBoost) - выдает base probability
2. Filters (∏I_k) - crisis, regime, correlation, liquidity
3. OpportunityScorer - дополнительная выгодность
4. CostCalculator - динамические издержки
5. RiskPenaltyCalculator - штраф за риск

ВАЖНО: Каждый компонент может быть включен/выключен для тестирования!
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable
import logging
from pathlib import Path
from enum import Enum

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class ComponentStatus(Enum):
    """Status of each component"""
    DISABLED = 0      # Component not used
    ENABLED = 1       # Component used
    DEBUG = 2         # Component used with detailed logging


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Trade:
    """Single trade record with full P_j(S) breakdown"""
    bar_idx: int
    entry_price: float
    entry_time: str

    exit_price: float
    exit_time: str
    exit_reason: str  # 'tp', 'sl', 'time'

    # P_j(S) components (for analysis)
    ml_score: float            # ML(state)
    filter_product: float      # ∏I_k
    opportunity_score: float   # opportunity(S)
    costs: float              # costs(S)
    risk_penalty: float       # risk_penalty(S)
    total_pjs: float          # Final P_j(S)

    # Results
    pnl: float                # Absolute P&L
    pnl_pct: float           # Percentage P&L

    def __repr__(self):
        return (f"Trade(idx={self.bar_idx}, pnl={self.pnl_pct:+.2f}%, "
                f"P_j(S)={self.total_pjs:.4f}, reason={self.exit_reason})")


@dataclass
class BacktestConfig:
    """Configuration for P_j(S) backtest"""
    # Core parameters
    initial_capital: float = 100000
    position_size_pct: float = 0.95

    # TP/SL parameters
    take_profit: float = 0.02        # 2%
    stop_loss: float = 0.01          # 1%
    max_hold_bars: int = 288         # 3 days for 15m

    # Costs
    commission: float = 0.001        # 0.1% per side
    slippage: float = 0.0005         # 0.05%

    # Cooldown between trades
    cooldown_bars: int = 10

    # Component status
    ml_enabled: bool = True
    filters_enabled: bool = False         # ∏I_k
    opportunity_enabled: bool = False
    cost_enabled: bool = True
    risk_penalty_enabled: bool = False

    # ML threshold
    ml_threshold: float = 0.5


@dataclass
class PjSComponents:
    """Holds computed P_j(S) components for a single signal"""
    ml_score: float = 0.0
    filter_product: float = 1.0      # Product starts at 1
    opportunity_score: float = 0.0
    costs: float = 0.0
    risk_penalty: float = 0.0
    final_pjs: float = 0.0


# ============================================================================
# STRATEGY INTERFACES
# ============================================================================

class BaseStrategy:
    """Base class for any strategy"""
    def generate_signals(self, ohlcv: pd.DataFrame) -> np.ndarray:
        """
        Generate buy signals

        Returns:
            array of 0/1 (no trade / buy signal)
        """
        raise NotImplementedError


class RuleBasedStrategy(BaseStrategy):
    """Simple Rule-Based strategy (RSI < 30)"""

    def __init__(self, rsi_threshold: float = 30, period: int = 14):
        self.rsi_threshold = rsi_threshold
        self.period = period

    def _calculate_rsi(self, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator"""
        deltas = np.diff(close)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(close)
        rsi[:period] = 100. - 100. / (1. + rs)

        for i in range(period, len(close)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi[i] = 100. - 100. / (1. + rs)

        return rsi

    def generate_signals(self, ohlcv: pd.DataFrame) -> np.ndarray:
        """Generate RSI < threshold signals"""
        rsi = self._calculate_rsi(ohlcv['close'].values)
        signals = (rsi < self.rsi_threshold).astype(int)
        return signals


# ============================================================================
# COMPONENT CALCULATORS
# ============================================================================

class OpportunityScorerComponent:
    """
    Calculates opportunity(S) - выгодность текущей ситуации

    Factors:
    - Volume (need sufficient volume)
    - Volatility (good opportunities in certain volatility ranges)
    - Pattern strength (for technical analysis)
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def score(self, ohlcv: pd.DataFrame, idx: int, base_score: float = 1.0) -> float:
        """
        Calculate opportunity score for bar at idx

        Returns: 0.0 to 2.0+ (can amplify good opportunities)
        """
        if not self.enabled:
            return 1.0

        if idx < 20:
            return 1.0

        # Simple: check volume (higher volume = more opportunity)
        recent_vol = ohlcv['volume'].iloc[max(0, idx-20):idx].mean()
        current_vol = ohlcv.iloc[idx]['volume']

        if current_vol < recent_vol * 0.5:
            return 0.5  # Low volume penalty
        elif current_vol > recent_vol * 1.5:
            return 1.2  # High volume bonus
        else:
            return 1.0


class CostCalculatorComponent:
    """
    Calculates costs(S) - динамические издержки

    Factors:
    - Commission (fixed per trade)
    - Slippage (depends on order size and liquidity)
    - Market impact (for large orders)
    """

    def __init__(
        self,
        commission: float = 0.001,  # 0.1% per side
        slippage: float = 0.0005,   # 0.05% base
        enabled: bool = True
    ):
        self.commission = commission
        self.slippage = slippage
        self.enabled = enabled

    def calculate(self,
                  volume: float,
                  typical_volume: float,
                  position_size_pct: float = 0.95) -> float:
        """
        Calculate costs as % of entry price

        Returns: costs in percentage form (e.g., 0.002 for 0.2%)
        """
        if not self.enabled:
            return 0.0

        # Base costs
        costs = self.commission * 2  # Entry + Exit
        costs += self.slippage

        # Slippage adjustment based on volume
        if typical_volume > 0:
            volume_ratio = volume / typical_volume
            if volume_ratio < 0.5:
                costs += 0.0005  # Extra slippage for low volume

        return costs


class RiskPenaltyComponent:
    """
    Calculates risk_penalty(S) - штраф за рискованные ситуации

    Factors:
    - High volatility (ATR-based)
    - OOD signals (z-score > 3σ)
    - Portfolio concentration
    - MDD limits
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def penalty(self,
                volatility_pct: float,  # ATR as % of price
                ml_score: float,        # If too low, increase penalty
                max_atr_pct: float = 0.05) -> float:  # 5% ATR threshold
        """
        Calculate risk penalty (0.0 to 1.0)

        Returns: penalty to subtract from P_j(S)
        """
        if not self.enabled:
            return 0.0

        penalty = 0.0

        # High volatility penalty
        if volatility_pct > max_atr_pct:
            penalty += 0.1 * (volatility_pct / max_atr_pct)
            penalty = min(penalty, 0.5)  # Cap at 0.5

        # Low confidence penalty
        if ml_score < 0.55:
            penalty += (0.55 - ml_score) * 0.2

        return min(penalty, 1.0)


class FilterComponent:
    """
    Implements ∏I_k - произведение всех фильтров

    Filters:
    - Crisis detection (0 if crisis)
    - Regime filter (adjust based on regime)
    - Correlation filter (avoid correlated positions)
    - Liquidity filter (need minimum liquidity)
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def product(self,
                crisis: bool = False,
                regime: str = 'sideways',  # bull, sideways, bear
                correlation: float = 0.3,  # 0.0-1.0
                volume_ok: bool = True) -> float:
        """
        Calculate filter product ∏I_k

        Each filter is 0.0-1.0, result is their product
        """
        if not self.enabled:
            return 1.0

        product = 1.0

        # Crisis filter - if crisis, skip signal (0)
        if crisis:
            return 0.0

        # Regime filter - adjust based on regime
        regime_mult = {
            'bull': 1.0,
            'sideways': 0.8,
            'bear': 0.5
        }.get(regime, 0.8)
        product *= regime_mult

        # Correlation filter - higher correlation = lower score
        if correlation > 0.7:
            product *= 0.5
        elif correlation > 0.5:
            product *= 0.7

        # Liquidity filter
        if not volume_ok:
            product *= 0.3

        return product


# ============================================================================
# MAIN BACKTEST ENGINE
# ============================================================================

class PjSBacktestEngine:
    """
    Main backtest engine implementing P_j(S) formula

    Allows step-by-step integration of components
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.signals_breakdown = []  # For analysis

        # Initialize components
        self.opportunity_scorer = OpportunityScorerComponent(
            enabled=config.opportunity_enabled
        )
        self.cost_calc = CostCalculatorComponent(
            commission=config.commission,
            slippage=config.slippage,
            enabled=config.cost_enabled
        )
        self.risk_penalty = RiskPenaltyComponent(
            enabled=config.risk_penalty_enabled
        )
        self.filters = FilterComponent(
            enabled=config.filters_enabled
        )

    def calculate_pjs(self,
                      ml_score: float,
                      ohlcv: pd.DataFrame,
                      idx: int,
                      volume: float) -> Tuple[float, PjSComponents]:
        """
        Calculate P_j(S) for a signal at given bar

        P_j(S) = ML(state)
               × ∏I_k
               × opportunity(S)
               - costs(S)
               - risk_penalty(S)

        Returns:
            (final_pjs, components_breakdown)
        """
        components = PjSComponents()

        # 1. Check if signal exists at all
        if ml_score <= 0:
            components.ml_score = 0.0
            components.final_pjs = 0.0
            return 0.0, components

        # 2. If ML filtering is enabled, apply threshold
        if self.config.ml_enabled:
            if ml_score < self.config.ml_threshold:
                components.ml_score = 0.0
                components.final_pjs = 0.0
                return 0.0, components

        # Signal passed all checks
        components.ml_score = ml_score

        # 2. Filters (∏I_k)
        components.filter_product = self.filters.product(
            crisis=False,  # TODO: get from crisis detector
            regime='sideways',  # TODO: get from regime detector
            correlation=0.3,  # TODO: get from portfolio
            volume_ok=volume > 0
        )

        if components.filter_product == 0.0:
            components.final_pjs = 0.0
            return 0.0, components

        # 3. Opportunity Score
        components.opportunity_score = self.opportunity_scorer.score(
            ohlcv, idx, components.ml_score
        )

        # 4. Costs
        recent_vol = ohlcv['volume'].iloc[max(0, idx-20):idx].mean() if idx >= 20 else volume
        components.costs = self.cost_calc.calculate(
            volume=volume,
            typical_volume=recent_vol,
            position_size_pct=self.config.position_size_pct
        )

        # 5. Risk Penalty
        atr_pct = 0.02  # TODO: calculate real ATR
        components.risk_penalty = self.risk_penalty.penalty(
            volatility_pct=atr_pct,
            ml_score=ml_score
        )

        # Calculate final P_j(S)
        pjs = (components.ml_score
               * components.filter_product
               * components.opportunity_score
               - components.costs
               - components.risk_penalty)

        # Ensure non-negative (can't be worse than 0)
        pjs = max(0.0, pjs)
        components.final_pjs = pjs

        return pjs, components

    def run(self,
            ohlcv: pd.DataFrame,
            raw_signals: np.ndarray,
            ml_scores: Optional[np.ndarray] = None) -> Dict:
        """
        Run backtest with P_j(S) framework

        Args:
            ohlcv: DataFrame with OHLCV data
            raw_signals: Binary signals (0/1)
            ml_scores: ML probability scores (0-1)

        Returns:
            Dictionary with metrics
        """
        if ml_scores is None:
            ml_scores = raw_signals.astype(float) * 0.7

        capital = self.config.initial_capital
        position = None
        last_exit_bar = -self.config.cooldown_bars

        self.trades = []
        self.equity_curve = [capital]

        logger.info(f"Starting backtest: {len(ohlcv)} bars, "
                   f"ML_enabled={self.config.ml_enabled}, "
                   f"Filters_enabled={self.config.filters_enabled}, "
                   f"Costs_enabled={self.config.cost_enabled}")

        for i in range(len(ohlcv)):
            current_bar = ohlcv.iloc[i]

            # Update equity curve
            current_equity = capital
            if position is not None:
                mtm = position['shares'] * current_bar['close']
                current_equity += mtm
            self.equity_curve.append(current_equity)

            # Check exit conditions
            if position is not None:
                bars_held = i - position['entry_bar']
                pnl_pct = (current_bar['close'] - position['entry_price']) / position['entry_price']

                exit_reason = None
                if current_bar['high'] >= position['tp_level']:
                    exit_reason = 'tp'
                    exit_price = position['tp_level']
                elif current_bar['low'] <= position['sl_level']:
                    exit_reason = 'sl'
                    exit_price = position['sl_level']
                elif bars_held >= self.config.max_hold_bars:
                    exit_reason = 'time'
                    exit_price = current_bar['close']

                if exit_reason:
                    # Close position
                    pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
                    pnl = capital * pnl_pct * self.config.position_size_pct
                    capital += pnl

                    trade = Trade(
                        bar_idx=i,
                        entry_price=position['entry_price'],
                        entry_time=str(position['entry_time']),
                        exit_price=exit_price,
                        exit_time=str(current_bar.get('time', '')),
                        exit_reason=exit_reason,
                        ml_score=position['ml_score'],
                        filter_product=position['filter_product'],
                        opportunity_score=position['opportunity_score'],
                        costs=position['costs'],
                        risk_penalty=position['risk_penalty'],
                        total_pjs=position['total_pjs'],
                        pnl=pnl,
                        pnl_pct=pnl_pct
                    )
                    self.trades.append(trade)
                    position = None
                    last_exit_bar = i  # CRITICAL: Reset cooldown timer

            # Check entry conditions
            if position is None and raw_signals[i] > 0:
                bars_since_exit = i - last_exit_bar
                if bars_since_exit >= self.config.cooldown_bars:
                    # Calculate P_j(S)
                    pjs, components = self.calculate_pjs(
                        ml_score=ml_scores[i],
                        ohlcv=ohlcv,
                        idx=i,
                        volume=current_bar['volume']
                    )

                    # Enter if P_j(S) > 0
                    if pjs > 0:
                        entry_price = current_bar['close']
                        position = {
                            'entry_bar': i,
                            'entry_price': entry_price,
                            'entry_time': current_bar.get('time', ''),
                            'shares': capital * self.config.position_size_pct / entry_price,
                            'tp_level': entry_price * (1 + self.config.take_profit),
                            'sl_level': entry_price * (1 - self.config.stop_loss),
                            'ml_score': components.ml_score,
                            'filter_product': components.filter_product,
                            'opportunity_score': components.opportunity_score,
                            'costs': components.costs,
                            'risk_penalty': components.risk_penalty,
                            'total_pjs': pjs
                        }

        # Close any open position at end
        if position is not None:
            exit_price = ohlcv.iloc[-1]['close']
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
            pnl = capital * pnl_pct * self.config.position_size_pct
            capital += pnl

            trade = Trade(
                bar_idx=len(ohlcv) - 1,
                entry_price=position['entry_price'],
                entry_time=str(position['entry_time']),
                exit_price=exit_price,
                exit_time='<end>',
                exit_reason='time',
                ml_score=position['ml_score'],
                filter_product=position['filter_product'],
                opportunity_score=position['opportunity_score'],
                costs=position['costs'],
                risk_penalty=position['risk_penalty'],
                total_pjs=position['total_pjs'],
                pnl=pnl,
                pnl_pct=pnl_pct
            )
            self.trades.append(trade)
            position = None
            last_exit_bar = len(ohlcv) - 1  # Update cooldown timer even at end

        # Calculate metrics
        if len(self.trades) > 0:
            wins = sum(1 for t in self.trades if t.pnl > 0)
            losses = sum(1 for t in self.trades if t.pnl < 0)
            wr = wins / len(self.trades) * 100

            total_pnl = sum(t.pnl for t in self.trades)
            avg_win = np.mean([t.pnl for t in self.trades if t.pnl > 0]) if wins > 0 else 0
            avg_loss = np.mean([t.pnl for t in self.trades if t.pnl < 0]) if losses > 0 else 0

            pf = (wins * avg_win) / (losses * abs(avg_loss)) if losses > 0 and avg_loss != 0 else 0
        else:
            wr = 0
            pf = 0
            total_pnl = 0

        # Final results
        results = {
            'initial_capital': self.config.initial_capital,
            'final_capital': capital,
            'total_pnl': capital - self.config.initial_capital,
            'total_pnl_pct': (capital - self.config.initial_capital) / self.config.initial_capital * 100,
            'trades': len(self.trades),
            'wins': wins if len(self.trades) > 0 else 0,
            'losses': losses if len(self.trades) > 0 else 0,
            'win_rate': wr,
            'profit_factor': pf,
            'equity_curve': self.equity_curve,
            'trades_detail': self.trades
        }

        logger.info(f"Backtest complete: {len(self.trades)} trades, "
                   f"WR={wr:.1f}%, PF={pf:.2f}")

        return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    # This is just a framework definition
    # Usage will be in separate backtest scripts
    print("P_j(S) Backtest Framework loaded successfully")
    print("Components: ML, Filters, OpportunityScorer, CostCalculator, RiskPenalty")
    print("Use PjSBacktestEngine with BacktestConfig to run backtests")
