# DEPRECATED 2026-05: используйте backtesting/vbt_engine.py (см. backtesting/MIGRATION_NOTES.md).
"""
Day 10: Full Integration Backtest.

Tests all components together and measures individual contributions.
This is the CORRECT path — not shortcuts, not overengineering.

Configurations tested:
1. Baseline: Static position (1.0), no filters
2. + Threshold: Only trade when P_ml > 0.70
3. + Regime: Adjust position by market regime
4. + Dispersion: Adjust by strategy agreement
5. + Q-Learner: Use adaptive α, β weights
6. + OOD: Reduce position for anomalous inputs
7. FULL: All components together
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ood_detector import OODDetector
from core.regime_detector import RegimeDetector, MarketRegime, REGIME_POSITION_MULTIPLIER
from core.dynamic_position_sizer import DynamicPositionSizer, PositionSizingInput
from core.rolling_dispersion import RollingDispersionCalculator
from strategies.hybrid_q_learner import HybridQLearner, MarketState


@dataclass
class BacktestConfig:
    """Configuration for backtest run."""
    name: str
    use_threshold: bool = False
    use_regime: bool = False
    use_dispersion: bool = False
    use_q_learner: bool = False
    use_ood: bool = False
    threshold: float = 0.70


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    name: str
    total_return: float
    sharpe: float
    max_dd: float
    calmar: float
    win_rate: float
    n_trades: int
    avg_position: float
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'total_return': round(self.total_return, 2),
            'sharpe': round(self.sharpe, 3),
            'max_dd': round(self.max_dd, 2),
            'calmar': round(self.calmar, 3),
            'win_rate': round(self.win_rate, 2),
            'n_trades': self.n_trades,
            'avg_position': round(self.avg_position, 3),
        }


class IntegrationBacktester:
    """
    Full integration backtester.
    
    Tests all components individually and together to measure
    the contribution of each to overall performance.
    """
    
    def __init__(self):
        # Load data
        self.pred = pd.read_parquet('models/xgboost_v3_btc_15m_predictions.parquet')
        self.feat = pd.read_parquet('data/features/BTC_USDT_15m_features.parquet')
        
        # Align
        n = len(self.pred)
        self.feat = self.feat.iloc[-n:].reset_index(drop=True)
        self.pred = self.pred.reset_index(drop=True)
        
        # Load feature names
        with open('models/xgboost_v3_btc_15m_metadata.json') as f:
            meta = json.load(f)
        self.feature_names = meta['feature_names']
        
        # Initialize components
        self.ood_detector = OODDetector()
        ood_path = Path('models/ood_detector_btc_15m.json')
        if ood_path.exists():
            self.ood_detector.load(str(ood_path))
        
        self.regime_detector = RegimeDetector()
        self.dispersion_calc = RollingDispersionCalculator(window=100)
        self.position_sizer = DynamicPositionSizer()
        
        self.q_learner = HybridQLearner()
        q_path = Path('models/hybrid_q_learner_btc.json')
        if q_path.exists():
            self.q_learner.load(str(q_path))
        
        # Data arrays
        self.p_ml = self.pred['y_pred'].values
        self.returns = self.pred['returns'].values
        
        # Simulate P_rb
        np.random.seed(42)
        self.p_rb = np.clip(self.p_ml + np.random.randn(len(self.p_ml)) * 0.15, 0, 1)
        
        print(f"Data loaded: {len(self.pred):,} bars")
    
    def _calculate_metrics(self, returns: np.ndarray, positions: np.ndarray, name: str) -> BacktestResult:
        """Calculate performance metrics."""
        strategy_returns = returns * positions
        active = positions > 0
        n_trades = int(active.sum())
        
        if n_trades == 0:
            return BacktestResult(name, 0, 0, 0, 0, 0, 0, 0)
        
        # Total return
        total_return = ((1 + strategy_returns).prod() - 1) * 100
        
        # Sharpe
        std = strategy_returns.std()
        sharpe = strategy_returns.mean() / std * np.sqrt(252 * 24 * 4) if std > 0 else 0
        
        # Max DD
        cum = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cum)
        dd = (cum - running_max) / running_max
        max_dd = dd.min() * 100
        
        # Calmar
        calmar = total_return / abs(max_dd) if max_dd != 0 else 0
        
        # Win rate
        wins = ((strategy_returns > 0) & active).sum()
        losses = ((strategy_returns < 0) & active).sum()
        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        
        # Avg position
        avg_pos = positions[active].mean()
        
        return BacktestResult(
            name=name,
            total_return=total_return,
            sharpe=sharpe,
            max_dd=max_dd,
            calmar=calmar,
            win_rate=win_rate,
            n_trades=n_trades,
            avg_position=avg_pos,
        )
    
    def run_config(self, config: BacktestConfig) -> BacktestResult:
        """Run backtest with specific configuration."""
        n = len(self.pred)
        positions = np.zeros(n)
        
        # Reset stateful components
        self.regime_detector.reset()
        self.dispersion_calc = RollingDispersionCalculator(window=100)
        
        for i in range(50, n):  # Skip warmup
            p_ml = self.p_ml[i]
            p_rb = self.p_rb[i]
            
            # Threshold filter
            if config.use_threshold and p_ml < config.threshold:
                continue
            
            # Base position
            position = 1.0
            
            # Q-Learner weights
            if config.use_q_learner:
                state = self.q_learner.get_state(self.returns[:i])
                action = self.q_learner.select_action(state, training=False)
                alpha, beta = self.q_learner.get_weights(action)
                p_hyb = alpha * p_rb + beta * p_ml
            else:
                p_hyb = 0.5 * p_rb + 0.5 * p_ml
            
            # Regime adjustment
            if config.use_regime:
                if 'open' in self.feat.columns:
                    window_start = max(0, i - 100)
                    ohlcv = self.feat.iloc[window_start:i+1][['open', 'high', 'low', 'close', 'volume']]
                    if len(ohlcv) >= 20:
                        regime_state = self.regime_detector.detect(ohlcv)
                        position *= REGIME_POSITION_MULTIPLIER.get(regime_state.regime, 1.0)
            
            # Dispersion adjustment (logic already inverted inside calculator)
            if config.use_dispersion:
                disp_state = self.dispersion_calc.update(p_rb, p_ml, p_hyb)
                if disp_state is not None:
                    position *= disp_state.confidence_multiplier
            
            # OOD adjustment
            if config.use_ood and self.ood_detector._fitted:
                features = self.feat.iloc[i][self.feature_names].values
                if not np.any(np.isnan(features)):
                    ood_state = self.ood_detector.detect(features)
                    position *= ood_state.confidence_multiplier
            
            # Clamp position
            position = np.clip(position, 0.0, 1.5)
            positions[i] = position
        
        return self._calculate_metrics(self.returns, positions, config.name)
    
    def run_all_configs(self) -> List[BacktestResult]:
        """Run all configurations and return results."""
        configs = [
            BacktestConfig("1. Baseline (static)", False, False, False, False, False),
            BacktestConfig("2. + Threshold 0.70", True, False, False, False, False),
            BacktestConfig("3. + Regime", True, True, False, False, False),
            BacktestConfig("4. + Dispersion", True, True, True, False, False),
            BacktestConfig("5. + Q-Learner", True, True, True, True, False),
            BacktestConfig("6. + OOD", True, True, True, True, True),
            BacktestConfig("FULL STACK", True, True, True, True, True),
        ]
        
        results = []
        for config in configs:
            print(f"Running: {config.name}...")
            result = self.run_config(config)
            results.append(result)
            print(f"  Sharpe: {result.sharpe:.2f}, Calmar: {result.calmar:.2f}, DD: {result.max_dd:.1f}%")
        
        return results


def main():
    print("=" * 70)
    print("DAY 10: FULL INTEGRATION BACKTEST")
    print("Testing all components together — the CORRECT path")
    print("=" * 70)
    
    backtester = IntegrationBacktester()
    results = backtester.run_all_configs()
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS: COMPONENT CONTRIBUTION MATRIX")
    print("=" * 70)
    
    print("\n{:<25} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Configuration", "Return%", "Sharpe", "MaxDD%", "Calmar", "Trades"
    ))
    print("-" * 75)
    
    for r in results:
        print("{:<25} {:>10.1f} {:>10.2f} {:>10.1f} {:>10.2f} {:>10,}".format(
            r.name[:25], r.total_return, r.sharpe, r.max_dd, r.calmar, r.n_trades
        ))
    
    # Analysis
    print("\n" + "=" * 70)
    print("COMPONENT CONTRIBUTION ANALYSIS")
    print("=" * 70)
    
    baseline = results[0]
    full = results[-1]
    
    print(f"\nBaseline → Full Stack:")
    print(f"  Return:  {baseline.total_return:.1f}% → {full.total_return:.1f}%")
    print(f"  Sharpe:  {baseline.sharpe:.2f} → {full.sharpe:.2f}")
    print(f"  Max DD:  {baseline.max_dd:.1f}% → {full.max_dd:.1f}%")
    print(f"  Calmar:  {baseline.calmar:.2f} → {full.calmar:.2f}")
    
    # Incremental contribution
    print("\n{:<20} {:>15} {:>15}".format("Component", "Δ Sharpe", "Δ Calmar"))
    print("-" * 50)
    
    for i in range(1, len(results)):
        prev = results[i-1]
        curr = results[i]
        d_sharpe = curr.sharpe - prev.sharpe
        d_calmar = curr.calmar - prev.calmar
        
        sign_s = "+" if d_sharpe >= 0 else ""
        sign_c = "+" if d_calmar >= 0 else ""
        
        component = curr.name.split("+")[-1].strip() if "+" in curr.name else curr.name
        print(f"{component:<20} {sign_s}{d_sharpe:>14.3f} {sign_c}{d_calmar:>14.3f}")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': [r.to_dict() for r in results],
        'analysis': {
            'baseline_sharpe': baseline.sharpe,
            'full_sharpe': full.sharpe,
            'sharpe_improvement': full.sharpe - baseline.sharpe,
            'baseline_calmar': baseline.calmar,
            'full_calmar': full.calmar,
            'calmar_improvement': full.calmar - baseline.calmar,
        }
    }
    
    with open('analysis/full_integration_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results saved: analysis/full_integration_results.json")
    
    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    if full.calmar > baseline.calmar * 1.5:
        print("✅ Integration SUCCESSFUL: Components work together synergistically")
    elif full.calmar > baseline.calmar:
        print("⚠️ Integration PARTIAL: Some improvement, but not multiplicative")
    else:
        print("❌ Integration PROBLEM: Full stack worse than baseline — investigate")


if __name__ == "__main__":
    main()


