"""
Day 5: Council End-to-End Test
Full pipeline: Market Data → Strategies → Aggregator → Council Opinion
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCouncilEndToEnd:
    """Full pipeline integration tests."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Generate realistic market data."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2024-01-01', periods=n, freq='15min')
        
        # Realistic price movement
        returns = np.random.randn(n) * 0.002
        close = 50000 * np.exp(np.cumsum(returns))
        
        high = close * (1 + np.abs(np.random.randn(n)) * 0.001)
        low = close * (1 - np.abs(np.random.randn(n)) * 0.001)
        open_ = close * (1 + np.random.randn(n) * 0.0005)
        volume = np.random.exponential(1000000, n)
        
        return pd.DataFrame({
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
        }, index=dates)
    
    @pytest.fixture
    def real_features(self):
        """Load slice of real features."""
        path = Path('data/features/BTC_USDT_15m_features.parquet')
        if path.exists():
            df = pd.read_parquet(path)
            return df.iloc[-500:].copy()
        return None
    
    def test_rule_based_generates_signals(self, sample_market_data):
        """RuleBasedStrategy generates valid signals."""
        from strategies.rule_based_v2 import RuleBasedStrategy
        
        rb = RuleBasedStrategy()
        result = rb.generate_signals(sample_market_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_market_data)
        assert 'signal' in result.columns or 'P_rb' in result.columns
    
    def test_xgboost_with_real_features(self, real_features):
        """XGBoost works with real features."""
        if real_features is None:
            pytest.skip("No real features available")
        
        from strategies.xgboost_ml_v3 import XGBoostMLStrategyV3
        
        model_path = 'models/xgboost_v3_btc_15m.json'
        if not Path(model_path).exists():
            pytest.skip("Model not found")
        
        strategy = XGBoostMLStrategyV3(model_path=model_path)
        
        feature_cols = [c for c in real_features.columns if c in strategy.feature_names]
        X = real_features[feature_cols].dropna()
        
        if len(X) > 0:
            proba = strategy.predict_proba(X)
            assert len(proba) == len(X)
            assert all(0 <= p <= 1 for p in proba)
    
    def test_aggregator_combines_signals(self, sample_market_data):
        """QuantAggregator combines multiple strategy signals."""
        from council.quant_aggregator import QuantAggregator
        from strategies.rule_based_v2 import RuleBasedStrategy
        
        agg = QuantAggregator()
        agg.register_strategy('rule_based', RuleBasedStrategy())
        
        signals = agg.aggregate(sample_market_data)
        
        assert signals is not None
        assert signals.p_rb is not None or signals.p_ml is not None
    
    def test_aggregator_produces_opinion(self, sample_market_data):
        """Aggregator produces valid AgentOpinion."""
        from council.quant_aggregator import QuantAggregator
        from council.contracts import ActionType
        from strategies.rule_based_v2 import RuleBasedStrategy
        
        agg = QuantAggregator()
        agg.register_strategy('rule_based', RuleBasedStrategy())
        
        signals = agg.aggregate(sample_market_data)
        opinion = agg.to_agent_opinion(signals)
        
        assert opinion is not None
        assert opinion.proposed_action in [ActionType.LONG, ActionType.SHORT, ActionType.HOLD]
        assert 0 < opinion.position_size_pct <= 1.5
        assert 0 <= opinion.confidence <= 1
    
    def test_dispersion_affects_position_size(self):
        """Dispersion calculator affects position sizing."""
        from council.quant_aggregator import QuantAggregator
        from council.contracts import QuantSignals
        
        agg = QuantAggregator()
        
        # High agreement signals
        signals_agree = QuantSignals(p_rb=0.75, p_ml=0.74, p_hyb=0.76)
        signals_agree.compute_agreement()
        
        # Low agreement signals
        signals_disagree = QuantSignals(p_rb=0.3, p_ml=0.7, p_hyb=0.5)
        signals_disagree.compute_agreement()
        
        # Update dispersion with multiple samples
        for _ in range(50):
            agg._dispersion_calc.update(p_rb=0.5, p_ml=0.5, p_hyb=0.5)
        
        pos_agree = agg._suggest_position_size(signals_agree)
        pos_disagree = agg._suggest_position_size(signals_disagree)
        
        # Higher agreement should allow larger position
        assert pos_agree > pos_disagree
    
    def test_full_pipeline_produces_opinions(self, real_features):
        """Full pipeline produces valid opinions."""
        if real_features is None:
            pytest.skip("No real features available")
        
        from council.quant_aggregator import QuantAggregator
        from strategies.rule_based_v2 import RuleBasedStrategy
        
        agg = QuantAggregator()
        agg.register_strategy('rule_based', RuleBasedStrategy())
        
        # Process multiple bars
        opinions = []
        for i in range(0, min(100, len(real_features)), 10):
            window = real_features.iloc[max(0, i-50):i+1]
            if len(window) < 20:
                continue
            
            try:
                signals = agg.aggregate(window)
                opinion = agg.to_agent_opinion(signals)
                opinions.append(opinion)
            except Exception:
                continue
        
        # Just verify we get opinions
        assert len(opinions) > 0
        
        # Verify opinions are valid
        for op in opinions:
            assert 0 < op.position_size_pct <= 1.5
            assert 0 <= op.confidence <= 1


class TestRAGIntegration:
    """RAG system integration tests."""
    
    def test_pattern_library_has_patterns(self):
        """Pattern library has sufficient patterns."""
        import json
        
        path = Path('rag/patterns/library.json')
        assert path.exists()
        
        with open(path) as f:
            library = json.load(f)
        
        patterns = library.get('patterns', [])
        assert len(patterns) >= 10, f"Only {len(patterns)} patterns, need 10+"
    
    def test_hybrid_retriever_initializes(self):
        """HybridRetriever can be initialized."""
        from rag.hybrid_retriever import HybridRetriever
        
        # Use default initialization
        retriever = HybridRetriever()
        assert retriever is not None


class TestSystemHealth:
    """Overall system health checks."""
    
    def test_all_imports_work(self):
        """All critical imports succeed."""
        from strategies.rule_based_v2 import RuleBasedStrategy
        from strategies.xgboost_ml_v3 import XGBoostMLStrategyV3
        from council.quant_aggregator import QuantAggregator
        from council.contracts import QuantSignals, AgentOpinion, ActionType
        from rag.rag_agent import RAGAgent
        from core.rolling_dispersion import RollingDispersionCalculator
        from core.sanitize_features import sanitize_for_model
        from rag.hybrid_retriever import HybridRetriever
        
        assert True
    
    def test_model_files_present(self):
        """Required model files exist."""
        required = [
            'models/xgboost_v3_btc_15m.json',
            'models/xgboost_v3_btc_15m_metadata.json',
            'data/features/BTC_USDT_15m_features.parquet',
            'rag/patterns/library.json',
        ]
        
        for path in required:
            assert Path(path).exists(), f"Missing: {path}"
    
    def test_test_suite_complete(self):
        """Test suite has good coverage."""
        test_files = list(Path('tests').glob('test_*.py'))
        assert len(test_files) >= 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])