"""
RAG Agent for Council
=====================

Integrates RAG retrieval into Council decision making.
Provides historical context for P_rb, P_ml, P_hyb evaluation.

Usage:
    from council.rag_agent import RAGAgent
    
    rag_agent = RAGAgent()
    opinion = await rag_agent.analyze(current_state)
    
    # opinion contains:
    # - recommendation (STRONG_SIGNAL, etc)
    # - confidence
    # - similar_patterns
    # - historical stats
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import asyncio


@dataclass
class RAGOpinion:
    """RAG agent's opinion for Council."""
    agent_name: str = "RAG_Historical"
    recommendation: str = "NEUTRAL"
    confidence: float = 0.5
    signal: float = 0.0  # -1 to 1 scale for Council
    
    # Details
    similar_cases_count: int = 0
    historical_win_rate: float = 0.5
    historical_avg_pnl: float = 0.0
    sample_size: int = 0
    retrieval_time_ms: float = 0.0
    
    # Reasoning
    reasoning: str = ""
    top_patterns: List[Dict] = None
    
    def __post_init__(self):
        if self.top_patterns is None:
            self.top_patterns = []
    
    def to_dict(self) -> Dict:
        return {
            'agent': self.agent_name,
            'recommendation': self.recommendation,
            'confidence': self.confidence,
            'signal': self.signal,
            'similar_cases_count': self.similar_cases_count,
            'historical_win_rate': self.historical_win_rate,
            'historical_avg_pnl': self.historical_avg_pnl,
            'sample_size': self.sample_size,
            'retrieval_time_ms': self.retrieval_time_ms,
            'reasoning': self.reasoning,
            'top_patterns': self.top_patterns,
        }


class RAGAgent:
    """
    RAG agent for Council.
    
    Responsibilities:
    1. Retrieve similar historical patterns
    2. Calculate historical statistics
    3. Generate recommendation
    4. Provide reasoning for Council debate
    """
    
    def __init__(
        self,
        patterns_dir: str = "rag/patterns",
        llm_provider: Optional[Any] = None,
    ):
        """
        Initialize RAG agent.
        
        Args:
            patterns_dir: Directory with patterns
            llm_provider: Optional LLM for enhanced reasoning
        """
        self.patterns_dir = Path(patterns_dir)
        self.llm_provider = llm_provider
        self._retriever = None
    
    @property
    def retriever(self):
        """Lazy load retriever."""
        if self._retriever is None:
            from rag.hybrid_retriever import HybridRetriever, RetrievalConfig
            
            config = RetrievalConfig(
                patterns_dir=str(self.patterns_dir),
                use_vector_store=True,
                use_multi_hyde=True,
                top_k=5,
                min_similarity=0.3,
            )
            
            self._retriever = HybridRetriever(
                config=config,
                llm_provider=self.llm_provider,
            )
        return self._retriever
    
    async def analyze(self, state: Dict[str, Any]) -> RAGOpinion:
        """
        Analyze current state using historical patterns.
        
        Args:
            state: Current market state
            
        Returns:
            RAGOpinion for Council
        """
        # Run retrieval (sync, but wrapped for async interface)
        context = await asyncio.to_thread(
            self.retriever.build_council_context,
            state
        )
        
        # Convert recommendation to signal
        signal = self._recommendation_to_signal(
            context.recommendation,
            context.confidence
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(context, state)
        
        # Extract top patterns for Council
        top_patterns = [
            {
                'id': p.pattern_id,
                'similarity': p.similarity,
                'explanation': p.explanation,
            }
            for p in context.similar_patterns[:3]
        ]
        
        return RAGOpinion(
            recommendation=context.recommendation,
            confidence=context.confidence,
            signal=signal,
            similar_cases_count=len(context.similar_patterns),
            historical_win_rate=context.historical_win_rate,
            historical_avg_pnl=context.historical_avg_pnl,
            sample_size=context.sample_size,
            retrieval_time_ms=context.retrieval_time_ms,
            reasoning=reasoning,
            top_patterns=top_patterns,
        )
    
    def analyze_sync(self, state: Dict[str, Any]) -> RAGOpinion:
        """Synchronous version of analyze."""
        context = self.retriever.build_council_context(state)
        
        signal = self._recommendation_to_signal(
            context.recommendation,
            context.confidence
        )
        
        reasoning = self._generate_reasoning(context, state)
        
        top_patterns = [
            {
                'id': p.pattern_id,
                'similarity': p.similarity,
                'explanation': p.explanation,
            }
            for p in context.similar_patterns[:3]
        ]
        
        return RAGOpinion(
            recommendation=context.recommendation,
            confidence=context.confidence,
            signal=signal,
            similar_cases_count=len(context.similar_patterns),
            historical_win_rate=context.historical_win_rate,
            historical_avg_pnl=context.historical_avg_pnl,
            sample_size=context.sample_size,
            retrieval_time_ms=context.retrieval_time_ms,
            reasoning=reasoning,
            top_patterns=top_patterns,
        )
    
    def _recommendation_to_signal(
        self, 
        recommendation: str, 
        confidence: float
    ) -> float:
        """
        Convert recommendation to numeric signal.
        
        Returns:
            Float in range [-1, 1]
            - Positive = bullish bias
            - Negative = bearish bias
            - Near 0 = neutral
        """
        base_signals = {
            'STRONG_SIGNAL': 0.8,
            'MODERATE_SIGNAL': 0.5,
            'WEAK_SIGNAL': 0.2,
            'NEUTRAL': 0.0,
            'NEGATIVE_EDGE': -0.6,
            'INSUFFICIENT_DATA': 0.0,
        }
        
        base = base_signals.get(recommendation, 0.0)
        
        # Scale by confidence
        return base * confidence
    
    def _generate_reasoning(
        self, 
        context: Any, 
        state: Dict
    ) -> str:
        """Generate human-readable reasoning."""
        parts = []
        
        # Summary
        n = len(context.similar_patterns)
        if n == 0:
            return "No similar historical patterns found."
        
        parts.append(f"Found {n} similar historical setups.")
        
        # Statistics
        if context.sample_size > 0:
            parts.append(
                f"Based on {context.sample_size} past trades: "
                f"{context.historical_win_rate:.0%} win rate, "
                f"{context.historical_avg_pnl:+.1f}% avg PnL."
            )
        
        # Recommendation
        rec_text = {
            'STRONG_SIGNAL': "Historical data strongly supports this setup.",
            'MODERATE_SIGNAL': "Historical data moderately supports this setup.",
            'WEAK_SIGNAL': "Historical data shows mixed results.",
            'NEUTRAL': "Historical data is inconclusive.",
            'NEGATIVE_EDGE': "Historical data suggests avoiding this setup.",
            'INSUFFICIENT_DATA': "Not enough historical data for conclusion.",
        }
        parts.append(rec_text.get(context.recommendation, ""))
        
        # Top match
        if context.similar_patterns:
            top = context.similar_patterns[0]
            parts.append(f"Best match: {top.pattern_id} (similarity: {top.similarity:.2f})")
        
        return " ".join(parts)
    
    def get_stats(self) -> Dict:
        """Get agent statistics."""
        return {
            'agent': 'RAG_Historical',
            'retriever_stats': self.retriever.get_stats(),
        }


# =============================================================================
# INTEGRATION WITH COUNCIL AGGREGATOR
# =============================================================================

def integrate_rag_with_council(
    rb_opinion: Dict,
    ml_opinion: Dict,
    rag_opinion: RAGOpinion,
    weights: Dict[str, float] = None,
) -> Dict[str, Any]:
    """
    Integrate RAG opinion with Council decision.
    
    Args:
        rb_opinion: Rule-based strategy opinion
        ml_opinion: ML strategy opinion
        rag_opinion: RAG historical opinion
        weights: Optional custom weights
        
    Returns:
        Integrated decision with RAG context
    """
    if weights is None:
        weights = {
            'rb': 0.35,
            'ml': 0.35,
            'rag': 0.30,
        }
    
    # Get signals
    rb_signal = rb_opinion.get('signal', 0)
    ml_signal = ml_opinion.get('signal', 0)
    rag_signal = rag_opinion.signal
    
    # Weighted combination
    combined_signal = (
        weights['rb'] * rb_signal +
        weights['ml'] * ml_signal +
        weights['rag'] * rag_signal
    )
    
    # Confidence adjustment based on agreement
    signals = [rb_signal, ml_signal, rag_signal]
    agreement = 1 - (max(signals) - min(signals))  # Higher if signals agree
    
    # RAG can boost or reduce confidence
    confidence_modifier = 1.0
    if rag_opinion.recommendation == 'STRONG_SIGNAL':
        confidence_modifier = 1.2
    elif rag_opinion.recommendation == 'NEGATIVE_EDGE':
        confidence_modifier = 0.7
    elif rag_opinion.recommendation == 'INSUFFICIENT_DATA':
        confidence_modifier = 0.9
    
    base_confidence = (
        rb_opinion.get('confidence', 0.5) * weights['rb'] +
        ml_opinion.get('confidence', 0.5) * weights['ml'] +
        rag_opinion.confidence * weights['rag']
    )
    
    final_confidence = min(1.0, base_confidence * agreement * confidence_modifier)
    
    return {
        'combined_signal': combined_signal,
        'confidence': final_confidence,
        'weights_used': weights,
        'rag_context': {
            'recommendation': rag_opinion.recommendation,
            'historical_win_rate': rag_opinion.historical_win_rate,
            'sample_size': rag_opinion.sample_size,
            'reasoning': rag_opinion.reasoning,
        },
        'agreement_score': agreement,
    }
