"""
Hybrid Retriever for ScArlet-Sails RAG
======================================

Unified API that combines:
- VectorStore (FAISS/ChromaDB) for semantic search
- MultiHyDE for multi-perspective retrieval
- JSON fallback for backward compatibility
- LLM hooks for future enhancement

Usage:
    # Basic (no LLM)
    retriever = HybridRetriever()
    results = retriever.retrieve(current_state, top_k=5)
    
    # With LLM (Week 3+)
    retriever = HybridRetriever(llm_provider=OllamaProvider())
    results = retriever.retrieve(current_state, top_k=5)
    
    # For Council
    context = retriever.build_council_context(current_state)
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import time

from .llm_protocols import (
    BaseLLMProvider,
    RuleBasedQueryGenerator,
    RuleBasedPostMortem,
    RuleBasedExplainer,
    get_query_generator,
    get_post_mortem_analyzer,
    get_pattern_explainer,
)


@dataclass
class RetrievalConfig:
    """Configuration for retriever."""
    patterns_dir: str = "rag/patterns"
    use_vector_store: bool = True
    use_multi_hyde: bool = True
    top_k: int = 5
    min_similarity: float = 0.3
    min_w_box: float = 0.0
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300


@dataclass  
class RetrievalResult:
    """Single retrieval result."""
    pattern_id: str
    similarity: float
    pattern: Dict[str, Any]
    explanation: str = ""
    historical_performance: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            'pattern_id': self.pattern_id,
            'similarity': self.similarity,
            'pattern': self.pattern,
            'explanation': self.explanation,
            'historical_performance': self.historical_performance,
        }


@dataclass
class CouncilContext:
    """Context for Council decision making."""
    similar_patterns: List[RetrievalResult]
    historical_win_rate: float
    historical_avg_pnl: float
    recommendation: str
    confidence: float
    sample_size: int
    retrieval_time_ms: float
    
    def to_dict(self) -> Dict:
        return {
            'similar_patterns': [p.to_dict() for p in self.similar_patterns],
            'historical_win_rate': self.historical_win_rate,
            'historical_avg_pnl': self.historical_avg_pnl,
            'recommendation': self.recommendation,
            'confidence': self.confidence,
            'sample_size': self.sample_size,
            'retrieval_time_ms': self.retrieval_time_ms,
        }


class HybridRetriever:
    """
    Unified retriever combining vector search and multi-query.
    
    Architecture:
        State → Query Generator → Vector Search → Results
                    ↓
              [Rule-based] or [LLM]
    """
    
    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            config: Retrieval configuration
            llm_provider: Optional LLM for enhanced queries
        """
        self.config = config or RetrievalConfig()
        self.llm_provider = llm_provider
        
        # Initialize components
        self._vector_store = None
        self._multi_hyde = None
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        
        # Get generators (rule-based or LLM)
        self.query_generator = get_query_generator(llm_provider)
        self.post_mortem = get_post_mortem_analyzer(llm_provider)
        self.explainer = get_pattern_explainer(llm_provider)
        
        # Paths
        self.patterns_dir = Path(self.config.patterns_dir)
        self.outcomes_file = self.patterns_dir / "outcomes.json"
        
        # Load outcomes
        self._outcomes = self._load_outcomes()
    
    @property
    def vector_store(self):
        """Lazy load vector store."""
        if self._vector_store is None and self.config.use_vector_store:
            try:
                from .vector_store import PatternVectorStore
                self._vector_store = PatternVectorStore(
                    patterns_dir=str(self.patterns_dir),
                    min_w_box=self.config.min_w_box,
                )
            except Exception as e:
                print(f"⚠️ Vector store not available: {e}")
                print("   Falling back to JSON search")
        return self._vector_store
    
    @property
    def multi_hyde(self):
        """Lazy load multi-hyde retriever."""
        if self._multi_hyde is None and self.config.use_multi_hyde:
            if self.vector_store is not None:
                try:
                    from .multi_hyde import MultiHyDERetriever
                    self._multi_hyde = MultiHyDERetriever(
                        vector_store=self.vector_store,
                        llm_client=self.llm_provider,
                    )
                except Exception as e:
                    print(f"⚠️ Multi-HyDE not available: {e}")
        return self._multi_hyde
    
    def _load_outcomes(self) -> Dict:
        """Load outcomes statistics."""
        if self.outcomes_file.exists():
            try:
                with open(self.outcomes_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _get_cache_key(self, state: Dict) -> str:
        """Generate cache key from state."""
        key_parts = [
            state.get('symbol', ''),
            state.get('timeframe', ''),
            str(state.get('indicators', {}).get('rsi_zscore', '')),
        ]
        return "|".join(key_parts)
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is valid."""
        if not self.config.cache_enabled:
            return False
        if key not in self._cache_timestamps:
            return False
        age = time.time() - self._cache_timestamps[key]
        return age < self.config.cache_ttl_seconds
    
    # =========================================================================
    # MAIN RETRIEVAL METHODS
    # =========================================================================
    
    def retrieve(
        self,
        state: Dict[str, Any],
        top_k: Optional[int] = None,
        filters: Optional[Dict] = None,
        use_cache: bool = True,
    ) -> List[RetrievalResult]:
        """
        Main retrieval method.
        
        Args:
            state: Current market state
            top_k: Number of results (default from config)
            filters: Optional filters {symbol, timeframe, direction}
            use_cache: Use result cache
            
        Returns:
            List of RetrievalResult
        """
        top_k = top_k or self.config.top_k
        start_time = time.time()
        
        # Check cache
        cache_key = self._get_cache_key(state)
        if use_cache and self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        results = []
        
        # Strategy 1: Multi-HyDE with vector search
        if self.multi_hyde is not None:
            try:
                raw_results = self.multi_hyde.retrieve(
                    state=state,
                    top_k=top_k,
                    **self._build_filters(filters)
                )
                results = self._convert_results(raw_results, state)
            except Exception as e:
                print(f"⚠️ Multi-HyDE failed: {e}")
        
        # Strategy 2: Direct vector search
        if not results and self.vector_store is not None:
            try:
                raw_results = self.vector_store.search_by_state(
                    state=state,
                    top_k=top_k,
                    **self._build_filters(filters)
                )
                results = self._convert_results(raw_results, state)
            except Exception as e:
                print(f"⚠️ Vector search failed: {e}")
        
        # Strategy 3: JSON fallback
        if not results:
            results = self._json_fallback(state, top_k, filters)
        
        # Filter by similarity
        results = [r for r in results if r.similarity >= self.config.min_similarity]
        
        # Cache results
        if use_cache:
            self._cache[cache_key] = results
            self._cache_timestamps[cache_key] = time.time()
        
        return results[:top_k]
    
    def _convert_results(
        self, 
        raw_results: List[Dict], 
        state: Dict
    ) -> List[RetrievalResult]:
        """Convert raw search results to RetrievalResult objects."""
        results = []
        
        for r in raw_results:
            pattern_id = r.get('pattern_id', r.get('id', 'unknown'))
            similarity = r.get('similarity', r.get('score', 0))
            pattern = r.get('pattern', r)
            
            # Get historical performance
            perf = self._outcomes.get(pattern_id)
            if not perf and 'statistics' in pattern:
                perf = pattern['statistics']
            
            # Generate explanation
            explanation = self.explainer.explain(pattern, state, similarity)
            
            results.append(RetrievalResult(
                pattern_id=pattern_id,
                similarity=similarity,
                pattern=pattern,
                explanation=explanation,
                historical_performance=perf,
            ))
        
        return results
    
    def _json_fallback(
        self, 
        state: Dict, 
        top_k: int,
        filters: Optional[Dict]
    ) -> List[RetrievalResult]:
        """Fallback to JSON file search."""
        results = []
        
        # Load all JSON patterns
        pattern_files = list(self.patterns_dir.glob("*.json"))
        pattern_files = [f for f in pattern_files if f.name not in ['outcomes.json', 'library.json']]
        
        for pf in pattern_files[:top_k * 2]:  # Load extra for filtering
            try:
                with open(pf, 'r') as f:
                    pattern = json.load(f)
                
                # Apply filters
                if filters:
                    meta = pattern.get('meta', {})
                    if filters.get('symbol') and meta.get('coin') != filters['symbol']:
                        continue
                    if filters.get('timeframe') and meta.get('timeframe') != filters['timeframe']:
                        continue
                
                pattern_id = pattern.get('id', pf.stem)
                
                results.append(RetrievalResult(
                    pattern_id=pattern_id,
                    similarity=0.5,  # Unknown similarity for JSON
                    pattern=pattern,
                    explanation="JSON fallback - no similarity score",
                    historical_performance=self._outcomes.get(pattern_id),
                ))
                
            except Exception as e:
                continue
        
        return results[:top_k]
    
    def _build_filters(self, filters: Optional[Dict]) -> Dict:
        """Build filter dict for vector search."""
        if not filters:
            return {}
        
        result = {}
        if 'symbol' in filters:
            result['filter_coin'] = filters['symbol']
        if 'timeframe' in filters:
            result['filter_timeframe'] = filters['timeframe']
        if 'direction' in filters:
            result['filter_direction'] = filters['direction']
        
        return result
    
    # =========================================================================
    # COUNCIL INTEGRATION
    # =========================================================================
    
    def build_council_context(
        self,
        state: Dict[str, Any],
        top_k: Optional[int] = None,
    ) -> CouncilContext:
        """
        Build context for Council decision making.
        
        Args:
            state: Current market state
            top_k: Number of patterns to retrieve
            
        Returns:
            CouncilContext with aggregated statistics
        """
        start_time = time.time()
        
        # Retrieve similar patterns
        results = self.retrieve(state, top_k=top_k)
        
        # Calculate aggregate statistics
        patterns_with_stats = [
            r for r in results
            if r.historical_performance is not None
        ]
        
        if patterns_with_stats:
            # Weighted average by similarity
            total_weight = sum(r.similarity for r in patterns_with_stats)
            
            if total_weight > 0:
                win_rate = sum(
                    r.similarity * r.historical_performance.get('win_rate', 0.5)
                    for r in patterns_with_stats
                ) / total_weight
                
                avg_pnl = sum(
                    r.similarity * r.historical_performance.get('avg_pnl', 0)
                    for r in patterns_with_stats
                ) / total_weight
            else:
                win_rate = 0.5
                avg_pnl = 0.0
            
            sample_size = sum(
                r.historical_performance.get('total_trades', 0)
                for r in patterns_with_stats
            )
        else:
            win_rate = 0.5
            avg_pnl = 0.0
            sample_size = 0
        
        # Generate recommendation
        recommendation, confidence = self._generate_recommendation(
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            sample_size=sample_size,
            num_patterns=len(results),
        )
        
        retrieval_time = (time.time() - start_time) * 1000
        
        return CouncilContext(
            similar_patterns=results,
            historical_win_rate=win_rate,
            historical_avg_pnl=avg_pnl,
            recommendation=recommendation,
            confidence=confidence,
            sample_size=sample_size,
            retrieval_time_ms=retrieval_time,
        )
    
    def _generate_recommendation(
        self,
        win_rate: float,
        avg_pnl: float,
        sample_size: int,
        num_patterns: int,
    ) -> tuple:
        """Generate recommendation and confidence."""
        
        # Not enough data
        if sample_size < 3:
            return 'INSUFFICIENT_DATA', 0.3
        
        # Strong signal
        if win_rate >= 0.7 and avg_pnl > 1.5:
            return 'STRONG_SIGNAL', 0.85
        
        # Moderate signal
        if win_rate >= 0.6 and avg_pnl > 0.5:
            return 'MODERATE_SIGNAL', 0.65
        
        # Weak signal
        if win_rate >= 0.5:
            return 'WEAK_SIGNAL', 0.45
        
        # Negative edge
        if win_rate < 0.4:
            return 'NEGATIVE_EDGE', 0.7
        
        return 'NEUTRAL', 0.5
    
    # =========================================================================
    # INDEX MANAGEMENT
    # =========================================================================
    
    def rebuild_index(self, verbose: bool = True) -> int:
        """Rebuild vector index from scratch."""
        if self.vector_store is None:
            print("⚠️ Vector store not available")
            return 0
        return self.vector_store.build_from_directory(verbose=verbose)
    
    def update_index(self, verbose: bool = True) -> int:
        """Update index with new patterns."""
        if self.vector_store is None:
            print("⚠️ Vector store not available")
            return 0
        return self.vector_store.update_index(verbose=verbose)
    
    def clear_cache(self):
        """Clear retrieval cache."""
        self._cache.clear()
        self._cache_timestamps.clear()
    
    def get_stats(self) -> Dict:
        """Get retriever statistics."""
        stats = {
            'config': {
                'use_vector_store': self.config.use_vector_store,
                'use_multi_hyde': self.config.use_multi_hyde,
                'min_similarity': self.config.min_similarity,
            },
            'cache': {
                'enabled': self.config.cache_enabled,
                'size': len(self._cache),
            },
            'outcomes': {
                'total_patterns_with_outcomes': len(self._outcomes),
            }
        }
        
        if self.vector_store is not None:
            try:
                stats['vector_store'] = self.vector_store.get_stats()
            except:
                pass
        
        return stats
