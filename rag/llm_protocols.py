"""
LLM Protocol Interfaces for ScArlet-Sails RAG
=============================================

Defines interfaces for LLM integration.
All RAG components accept these protocols, allowing:
- Rule-based implementation (default, no LLM)
- Ollama integration
- Custom local LLM
- Future cloud LLM (if needed)

Usage:
    # Without LLM (Week 1-2)
    retriever = HybridRetriever()  # Uses RuleBasedQueryGenerator
    
    # With LLM (Week 3+)
    from rag.llm_providers import OllamaProvider
    retriever = HybridRetriever(llm_provider=OllamaProvider())
"""

from typing import Protocol, List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


# =============================================================================
# PROTOCOL DEFINITIONS (Interfaces)
# =============================================================================

class QueryGenerator(Protocol):
    """
    Generates diverse queries for Multi-HyDE retrieval.
    
    Rule-based: Uses templates and state analysis
    LLM-based: Generates creative hypotheses
    """
    
    def generate_queries(self, state: Dict[str, Any]) -> List[str]:
        """
        Generate multiple search queries from current state.
        
        Args:
            state: Current market state with indicators
            
        Returns:
            List of 3-5 diverse query strings
        """
        ...


class PostMortemAnalyzer(Protocol):
    """
    Analyzes closed trades to extract learnings.
    
    Rule-based: Pattern matching on outcomes
    LLM-based: Reasoning about why trade worked/failed
    """
    
    def analyze(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze completed trade.
        
        Args:
            trade: {
                'pattern_id': str,
                'entry_price': float,
                'exit_price': float,
                'pnl_pct': float,
                'duration_bars': int,
                'exit_reason': str
            }
            
        Returns:
            {
                'success_factors': List[str],
                'failure_factors': List[str],
                'learnings': str,
                'confidence': float,
                'should_update_pattern': bool
            }
        """
        ...


class PatternExplainer(Protocol):
    """
    Explains why a pattern matches current state.
    
    Rule-based: Lists matching features
    LLM-based: Natural language explanation
    """
    
    def explain(
        self, 
        pattern: Dict[str, Any], 
        state: Dict[str, Any],
        similarity: float
    ) -> str:
        """
        Explain pattern-state match.
        
        Args:
            pattern: Retrieved pattern
            state: Current market state
            similarity: Cosine similarity score
            
        Returns:
            Human-readable explanation
        """
        ...


# =============================================================================
# RULE-BASED IMPLEMENTATIONS (Default, No LLM)
# =============================================================================

class RuleBasedQueryGenerator:
    """
    Generates queries using rules and templates.
    No LLM required. Works offline.
    """
    
    def generate_queries(self, state: Dict[str, Any]) -> List[str]:
        """Generate 4 perspective queries from state."""
        queries = []
        
        # Extract state info
        symbol = state.get('symbol', state.get('coin', 'BTC'))
        timeframe = state.get('timeframe', '1h')
        indicators = state.get('indicators', state)
        
        # === PERSPECTIVE 1: Technical Indicators ===
        tech_parts = [f"{symbol} {timeframe}"]
        
        # RSI
        rsi_z = indicators.get('rsi_zscore', indicators.get('norm_rsi_zscore', 0)) or 0
        if rsi_z < -1 or indicators.get('rsi_low') or indicators.get('regime_rsi_low'):
            tech_parts.append("RSI oversold")
            tech_parts.append("potential bounce")
        elif rsi_z > 1 or indicators.get('rsi_high') or indicators.get('regime_rsi_high'):
            tech_parts.append("RSI overbought")
            tech_parts.append("potential rejection")
        else:
            tech_parts.append("RSI neutral")
        
        # Divergence
        if indicators.get('div_rsi_bullish'):
            tech_parts.append("bullish divergence")
        elif indicators.get('div_rsi_bearish'):
            tech_parts.append("bearish divergence")
        
        queries.append(" | ".join(tech_parts))
        
        # === PERSPECTIVE 2: Volume Analysis ===
        vol_parts = [f"{symbol} volume analysis"]
        
        vol_z = indicators.get('volume_zscore', indicators.get('norm_volume_zscore', 0)) or 0
        if vol_z > 1.5:
            vol_parts.append("high volume breakout")
            vol_parts.append("institutional activity")
        elif vol_z > 0.5:
            vol_parts.append("above average volume")
        elif vol_z < -0.5:
            vol_parts.append("low volume")
            vol_parts.append("accumulation phase")
        else:
            vol_parts.append("normal volume")
        
        queries.append(" | ".join(vol_parts))
        
        # === PERSPECTIVE 3: Trend Context ===
        trend_parts = [f"{symbol} trend"]
        
        if indicators.get('trend_up') or indicators.get('regime_trend_up'):
            trend_parts.append("uptrend")
            trend_parts.append("continuation setup")
        elif indicators.get('trend_down') or indicators.get('regime_trend_down'):
            trend_parts.append("downtrend")
            trend_parts.append("reversal potential")
        else:
            trend_parts.append("ranging")
            trend_parts.append("breakout watch")
        
        # Volatility
        if indicators.get('vol_low') or indicators.get('regime_vol_low'):
            trend_parts.append("low volatility compression")
        elif indicators.get('vol_high') or indicators.get('regime_vol_high'):
            trend_parts.append("high volatility")
        
        queries.append(" | ".join(trend_parts))
        
        # === PERSPECTIVE 4: Box Pattern Quality ===
        box = state.get('box', {})
        w_box = state.get('w_box', {})
        
        box_parts = [f"{symbol} box range pattern"]
        
        touches = (box.get('touches_support', 0) or 0) + (box.get('touches_resistance', 0) or 0)
        if touches >= 6:
            box_parts.append("strong structure")
            box_parts.append("multiple touches")
        elif touches >= 4:
            box_parts.append("moderate structure")
        else:
            box_parts.append("developing pattern")
        
        w_score = w_box.get('W_box', 0) or 0
        if w_score > 0.6:
            box_parts.append("high quality setup")
        elif w_score > 0.3:
            box_parts.append("medium quality")
        
        queries.append(" | ".join(box_parts))
        
        return queries


class RuleBasedPostMortem:
    """
    Analyzes trades using rule-based logic.
    No LLM required.
    """
    
    def analyze(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trade outcome with rules."""
        pnl = trade.get('pnl_pct', 0)
        exit_reason = trade.get('exit_reason', 'unknown')
        duration = trade.get('duration_bars', 0)
        
        success_factors = []
        failure_factors = []
        
        # Analyze by outcome
        if pnl > 0:
            # Winning trade
            if exit_reason == 'tp':
                success_factors.append("Take profit reached")
            if duration < 10:
                success_factors.append("Quick execution")
            elif duration > 50:
                success_factors.append("Patient holding")
            
            if pnl > 2:
                success_factors.append("Strong move in favor")
        else:
            # Losing trade
            if exit_reason == 'sl':
                failure_factors.append("Stop loss triggered")
            if exit_reason == 'timeout':
                failure_factors.append("Position timed out")
            if duration < 3:
                failure_factors.append("Stopped out quickly - bad entry")
            
            if pnl < -1.5:
                failure_factors.append("Large adverse move")
        
        # Generate learning
        if pnl > 1:
            learning = "Pattern worked well. Consider similar setups."
        elif pnl > 0:
            learning = "Marginal win. Review entry timing."
        elif pnl > -0.5:
            learning = "Small loss. Normal variance."
        else:
            learning = "Significant loss. Review pattern quality filters."
        
        return {
            'success_factors': success_factors,
            'failure_factors': failure_factors,
            'learnings': learning,
            'confidence': 0.6,  # Rule-based = moderate confidence
            'should_update_pattern': abs(pnl) > 1.0  # Update if significant
        }


class RuleBasedExplainer:
    """
    Explains pattern matches using rules.
    No LLM required.
    """
    
    def explain(
        self, 
        pattern: Dict[str, Any], 
        state: Dict[str, Any],
        similarity: float
    ) -> str:
        """Generate rule-based explanation."""
        parts = []
        
        # Similarity
        if similarity > 0.8:
            parts.append("Very similar setup")
        elif similarity > 0.6:
            parts.append("Moderately similar")
        else:
            parts.append("Somewhat similar")
        
        # Pattern info
        pattern_meta = pattern.get('meta', pattern.get('pattern', {})).get('meta', {})
        coin = pattern_meta.get('coin', 'Unknown')
        tf = pattern_meta.get('timeframe', 'Unknown')
        direction = pattern_meta.get('direction', 'long')
        
        parts.append(f"to {coin} {tf} {direction}")
        
        # Historical performance
        perf = pattern.get('historical_performance', pattern.get('statistics', {}))
        if perf:
            win_rate = perf.get('win_rate', 0)
            avg_pnl = perf.get('avg_pnl', 0)
            total = perf.get('total_trades', 0)
            
            if total > 0:
                parts.append(f"Historical: {win_rate:.0%} win rate, {avg_pnl:+.1f}% avg ({total} trades)")
        
        # W_box quality
        w_box = pattern.get('w_box', {}).get('W_box', 0)
        if w_box > 0.6:
            parts.append("High quality pattern")
        elif w_box > 0.3:
            parts.append("Medium quality")
        
        return ". ".join(parts) + "."


# =============================================================================
# LLM PROVIDER BASE CLASS
# =============================================================================

@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    model: str = "qwen2.5:7b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 512
    timeout: int = 30


class BaseLLMProvider(ABC):
    """
    Base class for LLM providers.
    Implement this for Ollama, vLLM, or custom LLM.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        ...
    
    def generate_queries(self, state: Dict[str, Any]) -> List[str]:
        """Generate queries using LLM."""
        prompt = self._build_query_prompt(state)
        response = self.generate(prompt)
        return self._parse_queries(response)
    
    def analyze_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trade using LLM."""
        prompt = self._build_analysis_prompt(trade)
        response = self.generate(prompt)
        return self._parse_analysis(response)
    
    def explain_match(
        self, 
        pattern: Dict[str, Any], 
        state: Dict[str, Any],
        similarity: float
    ) -> str:
        """Explain match using LLM."""
        prompt = self._build_explanation_prompt(pattern, state, similarity)
        return self.generate(prompt)
    
    def _build_query_prompt(self, state: Dict[str, Any]) -> str:
        """Build prompt for query generation."""
        return f"""Generate 4 different search queries for finding similar trading patterns.

Current market state:
- Symbol: {state.get('symbol', 'BTC')}
- Timeframe: {state.get('timeframe', '1h')}
- RSI: {state.get('indicators', {}).get('rsi_zscore', 'N/A')}
- Volume: {state.get('indicators', {}).get('volume_zscore', 'N/A')}
- Trend: {'up' if state.get('indicators', {}).get('trend_up') else 'down' if state.get('indicators', {}).get('trend_down') else 'range'}

Generate 4 DIFFERENT perspectives:
1. Technical indicators focus
2. Volume and momentum focus
3. Trend and context focus
4. Pattern structure focus

Output exactly 4 lines, one query per line. No numbering, no explanations."""
    
    def _build_analysis_prompt(self, trade: Dict[str, Any]) -> str:
        """Build prompt for trade analysis."""
        return f"""Analyze this closed trade:

Entry: ${trade.get('entry_price', 0):.2f}
Exit: ${trade.get('exit_price', 0):.2f}
PnL: {trade.get('pnl_pct', 0):.2f}%
Duration: {trade.get('duration_bars', 0)} bars
Exit reason: {trade.get('exit_reason', 'unknown')}

Provide brief analysis:
1. Success factors (if profitable)
2. Failure factors (if losing)
3. Key learning (one sentence)

Be concise. Maximum 100 words."""
    
    def _build_explanation_prompt(
        self, 
        pattern: Dict[str, Any], 
        state: Dict[str, Any],
        similarity: float
    ) -> str:
        """Build prompt for match explanation."""
        return f"""Explain why this pattern matches the current setup.

Similarity score: {similarity:.2f}

Pattern: {pattern.get('id', 'Unknown')}
Current state: {state.get('symbol', 'BTC')} {state.get('timeframe', '1h')}

One sentence explanation. Be specific about what matches."""
    
    def _parse_queries(self, response: str) -> List[str]:
        """Parse LLM response into query list."""
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        # Remove any numbering
        queries = []
        for line in lines[:4]:
            # Remove "1.", "1)", "- " etc
            clean = line.lstrip('0123456789.-) ')
            if clean:
                queries.append(clean)
        return queries if queries else ["BTC trading pattern"]  # Fallback
    
    def _parse_analysis(self, response: str) -> Dict[str, Any]:
        """Parse LLM analysis response."""
        # Simple parsing - LLM output varies
        return {
            'success_factors': [],
            'failure_factors': [],
            'learnings': response[:500],  # Truncate if too long
            'confidence': 0.8,  # LLM = higher confidence
            'should_update_pattern': True
        }


# =============================================================================
# OLLAMA PROVIDER (Ready for Week 3+)
# =============================================================================

class OllamaProvider(BaseLLMProvider):
    """
    Ollama LLM provider.
    
    Setup:
        1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh
        2. Pull model: ollama pull qwen2.5:7b
        3. Use: OllamaProvider(LLMConfig(model="qwen2.5:7b"))
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        self._client = None
    
    @property
    def client(self):
        """Lazy load Ollama client."""
        if self._client is None:
            try:
                from langchain_community.llms import Ollama
                self._client = Ollama(
                    model=self.config.model,
                    base_url=self.config.base_url,
                    temperature=self.config.temperature,
                )
            except ImportError:
                raise ImportError(
                    "Ollama requires: pip install langchain-community\n"
                    "And Ollama server: https://ollama.com/download"
                )
        return self._client
    
    def generate(self, prompt: str) -> str:
        """Generate using Ollama."""
        try:
            return self.client.invoke(prompt)
        except Exception as e:
            print(f"⚠️ Ollama error: {e}")
            return ""


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_query_generator(llm_provider: Optional[BaseLLMProvider] = None) -> QueryGenerator:
    """
    Get query generator based on LLM availability.
    
    Args:
        llm_provider: Optional LLM provider
        
    Returns:
        QueryGenerator (rule-based or LLM-based)
    """
    if llm_provider is not None:
        return llm_provider
    return RuleBasedQueryGenerator()


def get_post_mortem_analyzer(llm_provider: Optional[BaseLLMProvider] = None) -> PostMortemAnalyzer:
    """Get post-mortem analyzer."""
    if llm_provider is not None:
        return llm_provider
    return RuleBasedPostMortem()


def get_pattern_explainer(llm_provider: Optional[BaseLLMProvider] = None) -> PatternExplainer:
    """Get pattern explainer."""
    if llm_provider is not None:
        return llm_provider
    return RuleBasedExplainer()
