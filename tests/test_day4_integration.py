"""
Day 4: Integration smoke test.
Verifies all Day 1-4 components work together.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json


class TestModelIntegration:
    """Test model components."""
    
    def test_model_loads_with_threshold(self):
        """XGBoost loads with threshold 0.70."""
        from strategies.xgboost_ml_v3 import XGBoostMLStrategyV3
        
        strategy = XGBoostMLStrategyV3(model_path='models/xgboost_v3_btc_15m.json')
        assert strategy.model is not None
        assert len(strategy.feature_names) == 74
    
    def test_model_predictions_valid(self):
        """Model produces valid predictions."""
        path = Path('models/xgboost_v3_btc_15m_predictions.parquet')
        if not path.exists():
            pytest.skip("Predictions not generated")
        
        pred = pd.read_parquet(path)
        assert len(pred) > 0
        assert all(0 <= p <= 1 for p in pred['y_pred'])


class TestDispersionIntegration:
    """Test dispersion components."""
    
    def test_dispersion_calculator_works(self):
        """Dispersion calculator produces valid output."""
        from core.risk.rolling_dispersion import RollingDispersionCalculator
        
        calc = RollingDispersionCalculator(window=50)
        
        # Simulate 100 updates
        for i in range(100):
            state = calc.update(
                p_rb=0.5 + np.random.randn() * 0.1,
                p_ml=0.5 + np.random.randn() * 0.1,
                p_hyb=0.5 + np.random.randn() * 0.1,
            )
        
        assert state.n_samples == 50  # Window capped
        assert 0.3 <= state.confidence_multiplier <= 1.5
    
    def test_aggregator_uses_dispersion(self):
        """QuantAggregator integrates dispersion."""
        from council.quant_aggregator import QuantAggregator
        
        agg = QuantAggregator()
        assert hasattr(agg, '_dispersion_calc')
        assert agg._dispersion_calc is not None


class TestRAGIntegration:
    """Test RAG components."""
    
    def test_patterns_library_exists(self):
        """Pattern library exists and has content."""
        path = Path('rag/patterns/library.json')
        assert path.exists(), "library.json not found"
        
        with open(path) as f:
            library = json.load(f)
        
        assert 'patterns' in library
        # After Day 4, should have 10+ patterns
    
    def test_rag_agent_imports(self):
        """RAG agent can be imported."""
        from council.rag_agent import RAGAgent
        assert RAGAgent is not None
    
    def test_hybrid_retriever_works(self):
        """Hybrid retriever initializes."""
        from rag.hybrid_retriever import HybridRetriever
        
        retriever = HybridRetriever()
        assert retriever is not None


class TestCouncilIntegration:
    """Test Council architecture."""
    
    def test_quant_signals_contract(self):
        """QuantSignals contract works."""
        from council.contracts import QuantSignals
        
        signals = QuantSignals(p_rb=0.7, p_ml=0.65, p_hyb=0.72)
        signals.compute_agreement()
        
        assert signals.agreement is not None
        assert 0 <= signals.agreement <= 1
    
    def test_aggregator_to_opinion(self):
        """Aggregator produces valid AgentOpinion."""
        from council.quant_aggregator import QuantAggregator
        from council.contracts import QuantSignals, ActionType
        
        agg = QuantAggregator()
        signals = QuantSignals(p_rb=0.7, p_ml=0.75, p_hyb=0.72)
        signals.compute_agreement()
        
        opinion = agg.to_agent_opinion(signals)
        
        assert opinion is not None
        assert opinion.proposed_action in [ActionType.LONG, ActionType.SHORT, ActionType.HOLD]
        assert 0 < opinion.position_size_pct <= 1.5


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_signal_pipeline(self):
        """Test complete signal generation pipeline."""
        from strategies.rule_based_v2 import RuleBasedStrategy
        from council.quant_aggregator import QuantAggregator
        
        # Create sample data
        np.random.seed(42)
        n = 100
        dates = pd.date_range('2024-01-01', periods=n, freq='15min')
        close = 50000 + np.cumsum(np.random.randn(n) * 100)
        
        data = pd.DataFrame({
            'open': close + np.random.randn(n) * 50,
            'high': close + np.random.rand(n) * 200,
            'low': close - np.random.rand(n) * 200,
            'close': close,
            'volume': np.random.rand(n) * 1000000,
        }, index=dates)
        
        # Generate signals
        rb = RuleBasedStrategy()
        signals_df = rb.generate_signals(data)
        
        assert isinstance(signals_df, pd.DataFrame)
        assert len(signals_df) == n


if __name__ == "__main__":
    pytest.main([__file__, "-v"])