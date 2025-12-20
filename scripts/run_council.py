#!/usr/bin/env python3
"""
ScArlet-Sails Council Runner

Main entry point for the Council decision support system.
Human-in-the-loop architecture: Council recommends, Human decides.

Usage:
    python scripts/run_council.py BTC --demo          # Demo mode with synthetic data
    python scripts/run_council.py BTC --tf 4h         # Real data, 4h timeframe
    python scripts/run_council.py BTC --verbose       # Verbose output

Philosophy:
    "Council — не автопилот, а 'вторая голова'. Человек несёт риск и ответственность."
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Council imports
from council.contracts import (
    CouncilContext,
    CouncilRecommendation,
    MarketSnapshot,
    PositionState,
    RiskConstraints,
    QuantSignals,
    RAGContext,
    AgentOpinion,
    ActionType,
    SeverityLevel,
    Regime,
)
from council.quant_aggregator import QuantAggregator, detect_regime
from council.pattern_detector import RuleBasedPatternDetector

# Interface imports
from interface.cli import (
    display_recommendation,
    get_human_decision,
    DecisionLogger,
    run_decision_cli,
)

# Core imports
try:
    from core.regime_detector import RegimeDetector, MarketRegime
    from core.rolling_dispersion import RollingDispersionCalculator
    from core.ood_detector import OODDetector
    from core.dynamic_position_sizer import DynamicPositionSizer, PositionSizingInput
    HAS_CORE = True
except ImportError as e:
    logging.warning(f"Core modules not available: {e}")
    HAS_CORE = False

# Strategy imports
try:
    from strategies.rule_based_v2 import RuleBasedStrategy
    from strategies.xgboost_ml_v3 import XGBoostMLStrategyV3
    HAS_STRATEGIES = True
except ImportError as e:
    logging.warning(f"Strategy modules not available: {e}")
    HAS_STRATEGIES = False

# RAG imports
try:
    from rag.hybrid_retriever import HybridRetriever
    HAS_RAG = True
except ImportError as e:
    logging.warning(f"RAG modules not available: {e}")
    HAS_RAG = False

logger = logging.getLogger(__name__)


# =============================================================================
# COUNCIL SESSION
# =============================================================================

class CouncilSession:
    """
    Orchestrates a single Council decision session.
    
    Flow:
    1. Load market data (real or demo)
    2. Build CouncilContext (multi-TF, signals, risk factors)
    3. Generate CouncilRecommendation
    4. Present to human
    5. Get human decision
    6. Log everything
    """
    
    def __init__(
        self,
        coin: str = "BTC",
        timeframe: str = "4h",
        verbose: bool = False,
    ):
        self.coin = coin
        self.timeframe = timeframe
        self.verbose = verbose
        
        # Components (lazy initialization)
        self._aggregator: Optional[QuantAggregator] = None
        self._pattern_detector: Optional[RuleBasedPatternDetector] = None
        self._regime_detector = None
        self._dispersion_calc = None
        self._ood_detector = None
        self._position_sizer = None
        self._rag_retriever = None
        self._decision_logger: Optional[DecisionLogger] = None
        
        # Strategies
        self._rb_strategy = None
        self._ml_strategy = None
        
        # Cached data
        self._features: Optional[pd.DataFrame] = None
        self._ohlcv: Optional[pd.DataFrame] = None
    
    def initialize(self) -> bool:
        """Initialize all components."""
        logger.info("Initializing Council session...")
        
        try:
            # Pattern detector (always available)
            self._pattern_detector = RuleBasedPatternDetector()
            
            # Decision logger
            self._decision_logger = DecisionLogger()
            
            # Core components
            if HAS_CORE:
                self._regime_detector = RegimeDetector()
                self._dispersion_calc = RollingDispersionCalculator(window=100)
                self._ood_detector = OODDetector()
                self._position_sizer = DynamicPositionSizer()
                
                # Load OOD detector if available
                ood_path = PROJECT_ROOT / f"models/ood_detector_{self.coin.lower()}_{self.timeframe}.json"
                if ood_path.exists():
                    self._ood_detector.load(str(ood_path))
            
            # Strategies
            if HAS_STRATEGIES:
                self._rb_strategy = RuleBasedStrategy()
                
                model_path = PROJECT_ROOT / f"models/xgboost_v3_{self.coin.lower()}_{self.timeframe}.json"
                if model_path.exists():
                    self._ml_strategy = XGBoostMLStrategyV3(str(model_path))
                    logger.info(f"Loaded XGBoost model: {model_path.name}")
            
            # RAG
            if HAS_RAG:
                self._rag_retriever = HybridRetriever()
            
            # Aggregator
            self._aggregator = QuantAggregator()
            if self._rb_strategy:
                self._aggregator.register_strategy('rule_based', self._rb_strategy)
            if self._ml_strategy:
                self._aggregator.register_strategy('xgboost_ml', self._ml_strategy)
            
            logger.info("Council session initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Council session: {e}")
            return False
    
    def load_data(self, demo: bool = False) -> bool:
        """Load market data."""
        if demo:
            return self._load_demo_data()
        else:
            return self._load_real_data()
    
    def _load_demo_data(self) -> bool:
        """Generate synthetic demo data."""
        logger.info("Generating demo data...")
        
        np.random.seed(42)
        n = 200
        
        # Generate OHLCV
        dates = pd.date_range(end=datetime.now(), periods=n, freq='4h')
        close = 43000 + np.cumsum(np.random.randn(n) * 200)
        
        self._ohlcv = pd.DataFrame({
            'open': close + np.random.randn(n) * 50,
            'high': close + np.abs(np.random.randn(n)) * 150,
            'low': close - np.abs(np.random.randn(n)) * 150,
            'close': close,
            'volume': np.random.exponential(1e9, n),
        }, index=dates)
        
        # Generate features (simplified)
        self._features = self._ohlcv.copy()
        self._features['RSI_14'] = 30 + np.random.rand(n) * 40  # 30-70
        self._features['ATR_pct'] = 0.015 + np.random.rand(n) * 0.02
        self._features['volume_ratio'] = 0.8 + np.random.rand(n) * 0.4
        
        logger.info(f"Demo data generated: {n} bars")
        return True
    
    def _load_real_data(self) -> bool:
        """Load real feature data."""
        features_path = PROJECT_ROOT / f"data/features/{self.coin}_USDT_{self.timeframe}_features.parquet"
        
        if not features_path.exists():
            logger.error(f"Features file not found: {features_path}")
            logger.info("Use --demo flag for synthetic data")
            return False
        
        try:
            self._features = pd.read_parquet(features_path)
            logger.info(f"Loaded {len(self._features)} rows from {features_path.name}")
            
            # Extract OHLCV
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            if all(c in self._features.columns for c in ohlcv_cols):
                self._ohlcv = self._features[ohlcv_cols].copy()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load features: {e}")
            return False
    
    def build_context(self) -> CouncilContext:
        """Build CouncilContext from current data."""
        if self._features is None or len(self._features) == 0:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Get latest data
        latest = self._features.iloc[-1]
        latest_ohlcv = self._ohlcv.iloc[-100:] if self._ohlcv is not None else None
        
        # 1. Market snapshot
        market = self._build_market_snapshot(latest, latest_ohlcv)
        
        # 2. Position state (assume flat for now)
        position = PositionState(
            current_action=ActionType.HOLD,
            size=0.0,
            entry_price=None,
            unrealized_pnl_pct=0.0,
        )
        
        # 3. Risk constraints
        constraints = RiskConstraints(
            max_position_size_pct=10.0,
            max_risk_per_trade_pct=0.5,
            max_leverage=1.0,
            daily_loss_remaining_pct=3.0,
            weekly_loss_remaining_pct=7.0,
        )
        
        # 4. Quant signals
        quant_signals = self._get_quant_signals()
        
        # 5. RAG context
        rag_context = self._get_rag_context()
        
        # Build context
        context = CouncilContext(
            market=market,
            position=position,
            constraints=constraints,
            quant_signals=quant_signals,
            rag=rag_context,
            request_id=f"{self.coin}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.utcnow(),
        )
        
        return context
    
    def _build_market_snapshot(
        self,
        latest: pd.Series,
        ohlcv: Optional[pd.DataFrame],
    ) -> MarketSnapshot:
        """Build MarketSnapshot from latest data."""
        
        # Extract values with fallbacks
        def get_val(keys, default=0.0):
            for k in keys if isinstance(keys, list) else [keys]:
                if k in latest.index and pd.notna(latest[k]):
                    return float(latest[k])
            return default
        
        # Detect regime
        regime = Regime.NORMAL
        if ohlcv is not None and self._regime_detector:
            try:
                regime_state = self._regime_detector.detect(ohlcv)
                regime = Regime(regime_state.regime.value.lower())
            except Exception:
                pass
        
        return MarketSnapshot(
            symbol=f"{self.coin}_USDT",
            timeframe=self.timeframe,
            timestamp=datetime.utcnow(),
            current_price=get_val(['close', 'Close'], 43000.0),
            spread_pct=0.01,
            volume_24h=get_val(['volume', 'Volume'], 1e9),
            rsi=get_val(['RSI_14', 'rsi_14', 'RSI'], 50.0),
            price_to_ema9_pct=get_val(['price_to_ema9_pct', 'ema9_dist'], 0.0),
            price_to_ema21_pct=get_val(['price_to_ema21_pct', 'ema21_dist'], 0.0),
            price_to_sma50_pct=get_val(['price_to_sma50_pct', 'sma50_dist'], 0.0),
            atr_pct=get_val(['ATR_pct', 'ATRr_14', 'atr_pct'], 0.02),
            bb_width_pct=get_val(['bb_width_pct', 'BBWidth'], 0.04),
            volume_ratio=get_val(['volume_ratio', 'vol_ratio'], 1.0),
            regime=regime,
            opportunity_score=get_val(['opportunity_score'], 0.5),
            risk_penalty=get_val(['risk_penalty'], 0.0),
        )
    
    def _get_quant_signals(self) -> QuantSignals:
        """Get signals from quant strategies."""
        p_rb = None
        p_ml = None
        p_hyb = None
        
        if self._features is not None and len(self._features) > 50:
            # Rule-based signal
            if self._rb_strategy:
                try:
                    rb_result = self._rb_strategy.generate_signals(self._features.tail(100))
                    if 'P_rb' in rb_result.columns:
                        p_rb = float(rb_result['P_rb'].iloc[-1])
                    elif 'signal' in rb_result.columns:
                        p_rb = float(rb_result['signal'].iloc[-1])
                except Exception as e:
                    logger.warning(f"Rule-based signal failed: {e}")
            
            # ML signal
            if self._ml_strategy:
                try:
                    feature_cols = [c for c in self._features.columns 
                                   if c in self._ml_strategy.feature_names]
                    if len(feature_cols) >= 50:
                        X = self._features[feature_cols].tail(1)
                        p_ml = float(self._ml_strategy.predict_single(X))
                except Exception as e:
                    logger.warning(f"ML signal failed: {e}")
            
            # Hybrid (simple average if both available)
            if p_rb is not None and p_ml is not None:
                p_hyb = 0.5 * p_rb + 0.5 * p_ml
        
        # Fallback to demo values
        if p_rb is None:
            p_rb = 0.55 + np.random.rand() * 0.2  # 0.55-0.75
        if p_ml is None:
            p_ml = 0.50 + np.random.rand() * 0.25  # 0.50-0.75
        if p_hyb is None:
            p_hyb = 0.5 * p_rb + 0.5 * p_ml
        
        signals = QuantSignals(p_rb=p_rb, p_ml=p_ml, p_hyb=p_hyb)
        signals.compute_agreement()
        
        # Update dispersion
        if self._dispersion_calc:
            self._dispersion_calc.update(p_rb=p_rb, p_ml=p_ml, p_hyb=p_hyb)
        
        return signals
    
    def _get_rag_context(self) -> RAGContext:
        """Get RAG context for pattern detection."""
        similar_patterns: List[Dict[str, Any]] = []
        
        if self._rag_retriever and self._features is not None:
            try:
                current_state = {
                    'symbol': self.coin,
                    'timeframe': self.timeframe,
                    'direction': 'long',
                }
                
                results = self._rag_retriever.retrieve(current_state, top_k=3)
                
                for r in results:
                    similar_patterns.append({
                        'pattern_id': getattr(r, 'pattern_id', 'unknown'),
                        'outcome': getattr(r, 'outcome', 'unknown'),
                        'pnl_pct': getattr(r, 'pnl_pct', 0.0),
                        'similarity': getattr(r, 'similarity', 0.0),
                    })
                    
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
        
        # Demo similar patterns if empty
        if not similar_patterns:
            similar_patterns = [
                {'pattern_id': 'ma50_bounce_001', 'outcome': 'win', 'pnl_pct': 3.2, 'similarity': 0.85},
                {'pattern_id': 'ma50_bounce_002', 'outcome': 'win', 'pnl_pct': 2.1, 'similarity': 0.78},
                {'pattern_id': 'oversold_reversal_001', 'outcome': 'loss', 'pnl_pct': -1.5, 'similarity': 0.72},
            ]
        
        return RAGContext(
            similar_patterns=similar_patterns,
            recent_trades=[],
            relevant_lessons=[],
        )
    
    def generate_recommendation(self, context: CouncilContext) -> CouncilRecommendation:
        """Generate Council recommendation from context."""
        
        # Get pattern detector opinion
        opinion = self._pattern_detector.detect(context)
        
        # Calculate position size
        position_size = self._calculate_position_size(context, opinion)
        
        # Calculate risk level
        risk_level = self._assess_risk_level(context, opinion)
        
        # Build rationale
        rationale = self._build_rationale(context, opinion)
        
        # Build dissent (contrarian view)
        dissent = self._build_dissent(context, opinion)
        
        # Check constraints
        violated = self._check_constraints(context, position_size)
        
        return CouncilRecommendation(
            final_action=opinion.proposed_action,
            final_position_size_pct=position_size,
            aggregate_confidence=opinion.confidence,
            risk_level=risk_level,
            rationale=rationale,
            sl_pct=opinion.suggested_sl_pct or 4.0,
            tp_pct=opinion.suggested_tp_pct or 8.0,
            dissent_summary=dissent,
            violated_constraints=violated if violated else None,
            request_id=context.request_id,
        )
    
    def _calculate_position_size(
        self,
        context: CouncilContext,
        opinion: AgentOpinion,
    ) -> float:
        """Calculate position size with all risk factors."""
        base_size = opinion.position_size_pct
        
        if self._position_sizer and HAS_CORE:
            try:
                # Get regime state
                regime_state = None
                if self._regime_detector and self._ohlcv is not None:
                    regime_state = self._regime_detector.detect(self._ohlcv.tail(100))
                
                # Get dispersion state
                disp_state = None
                if self._dispersion_calc:
                    disp_state = self._dispersion_calc.get_state()
                
                inputs = PositionSizingInput(
                    p_hyb=context.quant_signals.p_hyb or 0.5,
                    agreement=context.quant_signals.agreement or 0.5,
                    regime_state=regime_state,
                    dispersion_state=disp_state,
                    current_drawdown=0.0,  # TODO: track actual drawdown
                )
                
                output = self._position_sizer.calculate(inputs)
                base_size = output.position_size
                
            except Exception as e:
                logger.warning(f"Position sizing failed: {e}")
        
        # Apply constraints
        max_size = context.constraints.max_position_size_pct
        return min(base_size, max_size)
    
    def _assess_risk_level(
        self,
        context: CouncilContext,
        opinion: AgentOpinion,
    ) -> SeverityLevel:
        """Assess overall risk level."""
        risk_score = 0
        
        # Low confidence
        if opinion.confidence < 0.5:
            risk_score += 2
        elif opinion.confidence < 0.7:
            risk_score += 1
        
        # Low agreement
        if context.quant_signals.agreement and context.quant_signals.agreement < 0.7:
            risk_score += 1
        
        # High volatility regime
        if context.market.regime in [Regime.HIGH_VOL, Regime.CRISIS]:
            risk_score += 2
        
        # Low remaining risk budget
        if context.constraints.daily_loss_remaining_pct < 1.0:
            risk_score += 2
        
        # Map to severity
        if risk_score >= 5:
            return SeverityLevel.CRITICAL
        elif risk_score >= 3:
            return SeverityLevel.HIGH
        elif risk_score >= 2:
            return SeverityLevel.MEDIUM
        elif risk_score >= 1:
            return SeverityLevel.LOW
        else:
            return SeverityLevel.NONE
    
    def _build_rationale(
        self,
        context: CouncilContext,
        opinion: AgentOpinion,
    ) -> str:
        """Build human-readable rationale."""
        parts = []
        
        # Pattern info
        if opinion.pattern_detected:
            parts.append(f"Pattern: {opinion.pattern_detected}")
        
        # Confidence
        parts.append(f"Confidence: {opinion.confidence:.0%}")
        
        # Quant signals
        qs = context.quant_signals
        if qs.p_rb is not None and qs.p_ml is not None:
            parts.append(f"P_rb={qs.p_rb:.2f}, P_ml={qs.p_ml:.2f}")
        
        # Agreement
        if qs.agreement is not None:
            parts.append(f"Agreement: {qs.agreement:.0%}")
        
        # Regime
        parts.append(f"Regime: {context.market.regime.value}")
        
        # RSI
        parts.append(f"RSI: {context.market.rsi:.1f}")
        
        return " | ".join(parts)
    
    def _build_dissent(
        self,
        context: CouncilContext,
        opinion: AgentOpinion,
    ) -> Optional[str]:
        """Build contrarian view."""
        concerns = []
        
        # Low agreement
        if context.quant_signals.agreement and context.quant_signals.agreement < 0.7:
            concerns.append(f"Low model agreement ({context.quant_signals.agreement:.0%})")
        
        # Extreme RSI for the direction
        if opinion.proposed_action == ActionType.LONG and context.market.rsi > 65:
            concerns.append(f"RSI already elevated ({context.market.rsi:.0f})")
        elif opinion.proposed_action == ActionType.SHORT and context.market.rsi < 35:
            concerns.append(f"RSI already depressed ({context.market.rsi:.0f})")
        
        # High volatility
        if context.market.regime in [Regime.HIGH_VOL, Regime.CRISIS]:
            concerns.append(f"Elevated volatility ({context.market.regime.value})")
        
        # Low volume
        if context.market.volume_ratio < 0.8:
            concerns.append(f"Below-average volume ({context.market.volume_ratio:.2f}x)")
        
        # Historical losses in similar patterns
        losses = [p for p in context.rag.similar_patterns if p.get('outcome') == 'loss']
        if len(losses) >= 2:
            concerns.append(f"{len(losses)} similar patterns resulted in losses")
        
        if concerns:
            return "; ".join(concerns)
        return None
    
    def _check_constraints(
        self,
        context: CouncilContext,
        position_size: float,
    ) -> List[str]:
        """Check for constraint violations."""
        violations = []
        
        if position_size > context.constraints.max_position_size_pct:
            violations.append(f"Position size {position_size:.1f}% exceeds max {context.constraints.max_position_size_pct:.1f}%")
        
        if context.constraints.daily_loss_remaining_pct <= 0:
            violations.append("Daily loss limit reached")
        
        if context.constraints.weekly_loss_remaining_pct <= 0:
            violations.append("Weekly loss limit reached")
        
        return violations
    
    def run(self, demo: bool = False) -> bool:
        """Run full Council session."""
        print()
        print("=" * 60)
        print(f"  COUNCIL SESSION: {self.coin}/{self.timeframe}")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # 1. Initialize
        if not self.initialize():
            print("\n[X] Failed to initialize Council session")
            return False
        
        # 2. Load data
        if not self.load_data(demo=demo):
            return False
        
        # 3. Build context
        print("\n[*] Building context...")
        try:
            context = self.build_context()
            if self.verbose:
                print(f"   Request ID: {context.request_id}")
                print(f"   Market: {context.market.symbol}")
                print(f"   Regime: {context.market.regime.value}")
        except Exception as e:
            print(f"\n[X] Failed to build context: {e}")
            return False
        
        # 4. Generate recommendation
        print("\n[*] Generating recommendation...")
        try:
            recommendation = self.generate_recommendation(context)
        except Exception as e:
            print(f"\n[X] Failed to generate recommendation: {e}")
            return False
        
        # 5. Present to human and get decision
        response, trade_id = run_decision_cli(
            recommendation=recommendation,
            context=context,
            log_to_rag=True,
        )
        
        # 6. Summary
        print("\n" + "=" * 60)
        print("  SESSION COMPLETE")
        print("=" * 60)
        
        if trade_id:
            print(f"\n  Trade ID: {trade_id}")
            print(f"  Decision: {response.decision.value.upper()}")
            print(f"\n  To update outcome after trade closes:")
            print(f"    python scripts/update_outcome.py {trade_id} --outcome win --pnl 2.5")
        
        return True


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="ScArlet-Sails Council - Decision Support System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_council.py BTC --demo          # Demo with synthetic data
    python scripts/run_council.py BTC --tf 4h         # Real data, 4h timeframe
    python scripts/run_council.py ETH --tf 1h -v      # ETH 1h, verbose
        """
    )
    
    parser.add_argument(
        'coin',
        type=str,
        help='Coin symbol (e.g., BTC, ETH, SOL)'
    )
    
    parser.add_argument(
        '--tf', '--timeframe',
        type=str,
        default='4h',
        choices=['15m', '1h', '4h', '1d'],
        help='Timeframe (default: 4h)'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Use synthetic demo data'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run session
    session = CouncilSession(
        coin=args.coin.upper(),
        timeframe=args.tf,
        verbose=args.verbose,
    )
    
    success = session.run(demo=args.demo)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()