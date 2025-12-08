"""
Multi-HyDE Retrieval for Trading Patterns

Instead of simple query → search, we:
1. Generate MULTIPLE hypothetical ideal patterns
2. Embed each hypothesis
3. Search for EACH
4. Aggregate and rerank results

Based on: "Multi-HyDE: Multi-Hypothetical Document Embeddings" (2024)
Result: +11% accuracy on financial benchmarks
"""

from typing import List, Dict, Optional
from pathlib import Path

from .vector_store import PatternVectorStore


class MultiHyDERetriever:
    """
    Multi-Hypothesis Document Embeddings for pattern retrieval.
    
    Key insight: A single query might match patterns from DIFFERENT angles.
    Example: "BTC oversold" could match:
    - Technical: RSI < 30 patterns
    - Historical: Previous bottoms
    - Volume: Capitulation patterns
    
    We generate hypotheses for each angle, search separately,
    then merge results.
    """
    
    # Perspectives to consider
    PERSPECTIVES = [
        'technical',    # Indicators (RSI, MACD, etc.)
        'structure',    # Box quality, touches
        'context',      # Session, volatility regime
        'historical',   # Similar past outcomes
    ]
    
    def __init__(
        self,
        vector_store: PatternVectorStore,
        llm_client: Optional[object] = None,  # For hypothesis generation
    ):
        """
        Initialize Multi-HyDE retriever.
        
        Args:
            vector_store: PatternVectorStore instance
            llm_client: Optional LLM for hypothesis generation
                        If None, uses rule-based generation
        """
        self.store = vector_store
        self.llm = llm_client
    
    def generate_hypotheses(self, state: dict) -> List[str]:
        """
        Generate multiple hypothetical pattern descriptions.
        
        Each hypothesis represents ONE perspective on the current state.
        
        Args:
            state: Current market state
            
        Returns:
            List of hypothesis texts
        """
        if self.llm is not None:
            return self._generate_with_llm(state)
        else:
            return self._generate_rule_based(state)
    
    def _generate_rule_based(self, state: dict) -> List[str]:
        """
        Rule-based hypothesis generation (no LLM needed).
        
        Creates hypotheses from different perspectives.
        """
        hypotheses = []
        
        # Extract state info
        indicators = state.get('indicators', state)
        symbol = state.get('symbol', state.get('coin', 'BTC'))
        timeframe = state.get('timeframe', '1h')
        direction = state.get('direction', 'long')
        
        # === HYPOTHESIS 1: Technical Perspective ===
        tech_parts = [f"{symbol} {timeframe} {direction} setup"]
        
        # RSI
        rsi_z = indicators.get('rsi_zscore', indicators.get('norm_rsi_zscore', 0)) or 0
        if rsi_z < -1 or indicators.get('rsi_low') or indicators.get('regime_rsi_low'):
            tech_parts.append("RSI oversold zone")
            tech_parts.append("potential reversal")
        elif rsi_z > 1 or indicators.get('rsi_high') or indicators.get('regime_rsi_high'):
            tech_parts.append("RSI overbought zone")
            tech_parts.append("potential reversal")
        else:
            tech_parts.append("RSI neutral")
        
        # Volume
        vol_z = indicators.get('volume_zscore', indicators.get('norm_volume_zscore', 0)) or 0
        if vol_z > 1:
            tech_parts.append("high volume breakout")
        elif vol_z < -0.5:
            tech_parts.append("low volume")
        
        # Divergence
        if indicators.get('div_rsi_bullish'):
            tech_parts.append("bullish divergence confirmed")
        elif indicators.get('div_rsi_bearish'):
            tech_parts.append("bearish divergence warning")
        
        hypotheses.append(" | ".join(tech_parts))
        
        # === HYPOTHESIS 2: Structure Perspective ===
        struct_parts = [f"{symbol} box range pattern"]
        
        box = state.get('box', {})
        touches = (box.get('touches_support', 0) or 0) + (box.get('touches_resistance', 0) or 0)
        
        if touches >= 6:
            struct_parts.append("strong structure multiple touches")
            struct_parts.append("high probability breakout")
        elif touches >= 4:
            struct_parts.append("moderate structure")
        else:
            struct_parts.append("developing pattern")
        
        w_box = state.get('w_box', {}).get('W_box', 0) or 0
        if w_box > 0.6:
            struct_parts.append("high quality setup")
        elif w_box > 0.3:
            struct_parts.append("medium quality")
        
        hypotheses.append(" | ".join(struct_parts))
        
        # === HYPOTHESIS 3: Context Perspective ===
        context_parts = [f"{symbol} market context"]
        
        # Trend
        if indicators.get('trend_up') or indicators.get('regime_trend_up'):
            context_parts.append("uptrend environment")
            context_parts.append("continuation likely")
        elif indicators.get('trend_down') or indicators.get('regime_trend_down'):
            context_parts.append("downtrend environment")
        else:
            context_parts.append("ranging market")
        
        # Volatility
        if indicators.get('vol_low') or indicators.get('regime_vol_low'):
            context_parts.append("low volatility compression")
            context_parts.append("breakout imminent")
        elif indicators.get('vol_high') or indicators.get('regime_vol_high'):
            context_parts.append("high volatility")
        
        # Session
        if indicators.get('time_asian') or indicators.get('session_asian'):
            context_parts.append("Asian session")
        elif indicators.get('time_european') or indicators.get('session_european'):
            context_parts.append("European session high liquidity")
        elif indicators.get('time_american') or indicators.get('session_american'):
            context_parts.append("American session")
        
        hypotheses.append(" | ".join(context_parts))
        
        # === HYPOTHESIS 4: Outcome-focused (if outcomes available) ===
        outcome_parts = [f"{symbol} {direction}"]
        
        if direction == 'long':
            outcome_parts.append("successful long entries")
            outcome_parts.append("profitable bounce patterns")
        else:
            outcome_parts.append("successful short entries")
            outcome_parts.append("profitable breakdown patterns")
        
        hypotheses.append(" | ".join(outcome_parts))
        
        return hypotheses
    
    def _generate_with_llm(self, state: dict) -> List[str]:
        """
        LLM-based hypothesis generation.
        
        Generates richer, more contextual hypotheses.
        """
        # Build prompt
        prompt = f"""Given this trading state:
Symbol: {state.get('symbol', 'BTC')}
Timeframe: {state.get('timeframe', '1h')}
Direction: {state.get('direction', 'long')}
RSI: {state.get('indicators', {}).get('rsi_zscore', 'N/A')}
Volume: {state.get('indicators', {}).get('volume_zscore', 'N/A')}
Trend: {'up' if state.get('indicators', {}).get('trend_up') else 'down' if state.get('indicators', {}).get('trend_down') else 'range'}

Generate 4 DIFFERENT descriptions of ideal matching patterns from perspectives:
1. Technical indicators focus
2. Box/structure quality focus
3. Market context focus
4. Historical outcome focus

Each description should be one line, descriptive text (not JSON).
"""
        
        # Call LLM
        response = self.llm.generate(prompt)
        
        # Parse response into list
        hypotheses = [line.strip() for line in response.strip().split('\n') if line.strip()]
        
        # Ensure we have at least 4
        while len(hypotheses) < 4:
            hypotheses.append(self.store.state_to_text(state))
        
        return hypotheses[:4]
    
    def retrieve(
        self,
        state: dict,
        top_k: int = 5,
        per_hypothesis_k: int = 10,
        **filters
    ) -> List[Dict]:
        """
        Multi-HyDE retrieval: search from multiple perspectives.
        
        Args:
            state: Current market state
            top_k: Final number of results
            per_hypothesis_k: Results per hypothesis (before merging)
            **filters: Passed to vector_store.search()
            
        Returns:
            Merged and ranked results
        """
        # Step 1: Generate hypotheses
        hypotheses = self.generate_hypotheses(state)
        
        # Step 2: Search for each hypothesis
        all_results = []
        seen_ids = set()
        
        for hypo in hypotheses:
            results = self.store.search(
                query_text=hypo,
                top_k=per_hypothesis_k,
                **filters
            )
            
            for r in results:
                pid = r['pattern_id']
                if pid not in seen_ids:
                    r['matched_hypothesis'] = hypo
                    all_results.append(r)
                    seen_ids.add(pid)
        
        # Step 3: Aggregate scores
        # If pattern matched multiple hypotheses, it's more relevant
        id_to_results = {}
        for r in all_results:
            pid = r['pattern_id']
            if pid in id_to_results:
                # Boost score for multiple matches
                id_to_results[pid]['similarity'] += r['similarity'] * 0.5
                id_to_results[pid]['hypothesis_matches'] += 1
            else:
                r['hypothesis_matches'] = 1
                id_to_results[pid] = r
        
        # Step 4: Sort by combined score
        final_results = sorted(
            id_to_results.values(),
            key=lambda x: x['similarity'],
            reverse=True
        )
        
        return final_results[:top_k]
    
    def retrieve_with_explanation(
        self,
        state: dict,
        top_k: int = 5,
        **filters
    ) -> Dict:
        """
        Retrieve with detailed explanation of why patterns matched.
        
        Returns:
            {
                'results': List[Dict],
                'hypotheses_used': List[str],
                'match_summary': str
            }
        """
        hypotheses = self.generate_hypotheses(state)
        results = self.retrieve(state, top_k=top_k, **filters)
        
        # Build explanation
        if not results:
            summary = "No similar patterns found in database."
        else:
            multi_match = [r for r in results if r.get('hypothesis_matches', 1) > 1]
            
            summary = f"Found {len(results)} similar patterns. "
            if multi_match:
                summary += f"{len(multi_match)} matched multiple perspectives (high confidence). "
            
            top = results[0]
            summary += f"Best match: {top['pattern_id']} (similarity: {top['similarity']:.2f})"
        
        return {
            'results': results,
            'hypotheses_used': hypotheses,
            'match_summary': summary,
        }
