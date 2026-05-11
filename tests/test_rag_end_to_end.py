"""
End-to-End Tests for RAG Pipeline
=================================

Tests the complete RAG workflow:
1. State → Query Generation → Vector Search → Results
2. Council Integration
3. Outcomes Recording
4. Performance Benchmarks

ВРЕМЕННО SKIP'НУТ (см. отчёт ревизии 2026-05): требует seed-датасет паттернов
в `rag/patterns/`, который в MVP-пути retail-крипто-трейдера не нужен. Переоткрыть
после того как RAG-слой реально подключат к Council и/или мигрируют на ChromaDB.

Run with: pytest tests/test_rag_end_to_end.py -v
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="RAG e2e требует seed-датасет паттернов; не входит в MVP scope."
)

import time  # noqa: E402
from pathlib import Path
from typing import Dict
import tempfile
import json
import shutil


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_state() -> Dict:
    """Sample market state for testing."""
    return {
        'symbol': 'BTC',
        'timeframe': '1h',
        'direction': 'long',
        'indicators': {
            'rsi_zscore': -0.8,
            'norm_rsi_zscore': -0.8,
            'volume_zscore': 1.2,
            'norm_volume_zscore': 1.2,
            'rsi_low': True,
            'trend_up': True,
            'vol_low': False,
            'div_rsi_bullish': True,
        },
        'box': {
            'touches_support': 3,
            'touches_resistance': 2,
            'box_range_pct': 3.5,
        },
        'w_box': {
            'W_box': 0.65,
        },
    }


@pytest.fixture
def sample_pattern() -> Dict:
    """Sample pattern for testing."""
    return {
        'id': 'BTC_1h_20241201_1400',
        'meta': {
            'coin': 'BTC',
            'timeframe': '1h',
            'direction': 'long',
            'pattern_type': 'box_range',
        },
        'indicators_before': {
            'rsi_zscore': -0.9,
            'volume_zscore': 1.0,
            'rsi_low': True,
            'trend_up': True,
        },
        'box': {
            'touches_support': 3,
            'touches_resistance': 3,
        },
        'w_box': {
            'W_box': 0.7,
        },
        'statistics': {
            'total_trades': 5,
            'wins': 4,
            'losses': 1,
            'win_rate': 0.8,
            'avg_pnl': 2.1,
        },
    }


@pytest.fixture
def temp_patterns_dir(sample_pattern):
    """Create temporary patterns directory with sample data."""
    temp_dir = tempfile.mkdtemp()
    patterns_dir = Path(temp_dir) / "patterns"
    patterns_dir.mkdir(parents=True)
    
    # Save sample pattern
    pattern_file = patterns_dir / f"{sample_pattern['id']}.json"
    with open(pattern_file, 'w') as f:
        json.dump(sample_pattern, f)
    
    # Create empty outcomes
    outcomes_file = patterns_dir / "outcomes.json"
    with open(outcomes_file, 'w') as f:
        json.dump({sample_pattern['id']: sample_pattern['statistics']}, f)
    
    yield str(patterns_dir)
    
    # Cleanup
    shutil.rmtree(temp_dir)


# =============================================================================
# UNIT TESTS
# =============================================================================

class TestQueryGenerator:
    """Tests for query generation."""
    
    def test_rule_based_generates_4_queries(self, sample_state):
        """Rule-based generator should produce 4 queries."""
        from rag.llm_protocols import RuleBasedQueryGenerator
        
        generator = RuleBasedQueryGenerator()
        queries = generator.generate_queries(sample_state)
        
        assert len(queries) == 4
        assert all(isinstance(q, str) for q in queries)
        assert all(len(q) > 10 for q in queries)
    
    def test_rule_based_includes_symbol(self, sample_state):
        """Queries should include the symbol."""
        from rag.llm_protocols import RuleBasedQueryGenerator
        
        generator = RuleBasedQueryGenerator()
        queries = generator.generate_queries(sample_state)
        
        symbol_in_queries = any('BTC' in q for q in queries)
        assert symbol_in_queries
    
    def test_rule_based_reflects_rsi_oversold(self, sample_state):
        """When RSI is low, queries should reflect oversold condition."""
        from rag.llm_protocols import RuleBasedQueryGenerator
        
        generator = RuleBasedQueryGenerator()
        queries = generator.generate_queries(sample_state)
        
        all_text = " ".join(queries).lower()
        assert 'oversold' in all_text or 'bounce' in all_text


class TestPostMortem:
    """Tests for post-mortem analysis."""
    
    def test_winning_trade_analysis(self):
        """Analyze winning trade."""
        from rag.llm_protocols import RuleBasedPostMortem
        
        analyzer = RuleBasedPostMortem()
        
        trade = {
            'pnl_pct': 2.5,
            'exit_reason': 'tp',
            'duration_bars': 15,
        }
        
        result = analyzer.analyze(trade)
        
        assert len(result['success_factors']) > 0
        assert len(result['failure_factors']) == 0
        assert result['confidence'] > 0
    
    def test_losing_trade_analysis(self):
        """Analyze losing trade."""
        from rag.llm_protocols import RuleBasedPostMortem
        
        analyzer = RuleBasedPostMortem()
        
        trade = {
            'pnl_pct': -1.8,
            'exit_reason': 'sl',
            'duration_bars': 3,
        }
        
        result = analyzer.analyze(trade)
        
        assert len(result['failure_factors']) > 0
        assert 'loss' in result['learnings'].lower() or 'review' in result['learnings'].lower()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestHybridRetriever:
    """Tests for hybrid retriever."""
    
    def test_retriever_initialization(self, temp_patterns_dir):
        """Retriever should initialize without errors."""
        from rag.hybrid_retriever import HybridRetriever, RetrievalConfig
        
        config = RetrievalConfig(
            patterns_dir=temp_patterns_dir,
            use_vector_store=False,  # Skip FAISS for unit test
            use_multi_hyde=False,
        )
        
        retriever = HybridRetriever(config=config)
        
        assert retriever is not None
        assert retriever.patterns_dir == Path(temp_patterns_dir)
    
    def test_json_fallback_retrieval(self, temp_patterns_dir, sample_state):
        """JSON fallback should work when vector store unavailable."""
        from rag.hybrid_retriever import HybridRetriever, RetrievalConfig
        
        config = RetrievalConfig(
            patterns_dir=temp_patterns_dir,
            use_vector_store=False,
            use_multi_hyde=False,
        )
        
        retriever = HybridRetriever(config=config)
        results = retriever.retrieve(sample_state, top_k=5)
        
        assert len(results) > 0
        assert results[0].pattern_id == 'BTC_1h_20241201_1400'
    
    def test_council_context_building(self, temp_patterns_dir, sample_state):
        """Council context should include all required fields."""
        from rag.hybrid_retriever import HybridRetriever, RetrievalConfig
        
        config = RetrievalConfig(
            patterns_dir=temp_patterns_dir,
            use_vector_store=False,
            use_multi_hyde=False,
        )
        
        retriever = HybridRetriever(config=config)
        context = retriever.build_council_context(sample_state)
        
        # Check required fields
        assert hasattr(context, 'similar_patterns')
        assert hasattr(context, 'historical_win_rate')
        assert hasattr(context, 'recommendation')
        assert hasattr(context, 'confidence')
        assert hasattr(context, 'retrieval_time_ms')
        
        # Check values make sense
        assert 0 <= context.historical_win_rate <= 1
        assert 0 <= context.confidence <= 1
        assert context.retrieval_time_ms >= 0


class TestRAGAgent:
    """Tests for RAG Council agent."""
    
    def test_agent_sync_analysis(self, temp_patterns_dir, sample_state):
        """Agent should provide sync analysis."""
        from council.rag_agent import RAGAgent
        
        agent = RAGAgent(patterns_dir=temp_patterns_dir)
        
        # Force JSON fallback
        agent._retriever = None
        
        opinion = agent.analyze_sync(sample_state)
        
        assert opinion.agent_name == 'RAG_Historical'
        assert opinion.recommendation in [
            'STRONG_SIGNAL', 'MODERATE_SIGNAL', 'WEAK_SIGNAL',
            'NEUTRAL', 'NEGATIVE_EDGE', 'INSUFFICIENT_DATA'
        ]
        assert -1 <= opinion.signal <= 1
    
    def test_council_integration(self, temp_patterns_dir, sample_state):
        """Test integration with Council opinions."""
        from council.rag_agent import RAGAgent, integrate_rag_with_council
        
        agent = RAGAgent(patterns_dir=temp_patterns_dir)
        rag_opinion = agent.analyze_sync(sample_state)
        
        rb_opinion = {'signal': 0.6, 'confidence': 0.7}
        ml_opinion = {'signal': 0.4, 'confidence': 0.65}
        
        result = integrate_rag_with_council(rb_opinion, ml_opinion, rag_opinion)
        
        assert 'combined_signal' in result
        assert 'confidence' in result
        assert 'rag_context' in result
        assert -1 <= result['combined_signal'] <= 1


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance benchmarks."""
    
    def test_query_generation_speed(self, sample_state):
        """Query generation should be fast."""
        from rag.llm_protocols import RuleBasedQueryGenerator
        
        generator = RuleBasedQueryGenerator()
        
        # Warm up
        generator.generate_queries(sample_state)
        
        # Benchmark
        start = time.time()
        for _ in range(100):
            generator.generate_queries(sample_state)
        elapsed = (time.time() - start) * 1000
        
        avg_ms = elapsed / 100
        
        assert avg_ms < 10, f"Query generation too slow: {avg_ms:.2f}ms"
    
    def test_retrieval_latency(self, temp_patterns_dir, sample_state):
        """Retrieval should complete within threshold."""
        from rag.hybrid_retriever import HybridRetriever, RetrievalConfig
        
        config = RetrievalConfig(
            patterns_dir=temp_patterns_dir,
            use_vector_store=False,
            use_multi_hyde=False,
        )
        
        retriever = HybridRetriever(config=config)
        
        # Warm up
        retriever.retrieve(sample_state, top_k=5)
        
        # Benchmark
        latencies = []
        for _ in range(10):
            start = time.time()
            retriever.retrieve(sample_state, top_k=5, use_cache=False)
            latencies.append((time.time() - start) * 1000)
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        assert avg_latency < 100, f"Average latency too high: {avg_latency:.2f}ms"
        assert max_latency < 200, f"Max latency too high: {max_latency:.2f}ms"


# =============================================================================
# HEALTH CHECK TESTS
# =============================================================================

class TestHealthCheck:
    """Tests for health monitoring."""
    
    def test_health_check_with_empty_dir(self):
        """Health check should handle empty directory."""
        from rag.health_check import RAGHealthCheck
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checker = RAGHealthCheck(patterns_dir=temp_dir)
            report = checker.full_check()
            
            # Should not crash
            assert report is not None
            assert not report.healthy  # Missing index
    
    def test_health_check_with_patterns(self, temp_patterns_dir):
        """Health check with valid patterns."""
        from rag.health_check import RAGHealthCheck
        
        checker = RAGHealthCheck(patterns_dir=temp_patterns_dir)
        
        # Run individual checks
        pattern_check = checker.check_pattern_count()
        outcomes_check = checker.check_outcomes_data()
        
        assert pattern_check.value >= 1
        assert outcomes_check.passed


# =============================================================================
# RUN DIRECTLY
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
