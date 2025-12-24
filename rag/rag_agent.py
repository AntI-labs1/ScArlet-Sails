"""
ScArlet-Sails RAG Agent

RAG (Retrieval-Augmented Generation) как полноценный Agent.
Ищет похожие паттерны и формирует мнение на основе истории.

Philosophy:
    "Паттерны — это память системы"
    "Прошлое информирует будущее, но не диктует его"
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Добавляем путь к проекту
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from council.protocols import (
    ActionType,
    AgentConfig,
    AgentMetadata,
    AgentOpinion,
    BaseAgent,
    HealthCheckResult,
    HealthStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PatternMatch:
    """Результат поиска похожего паттерна."""
    pattern_id: str
    name: str
    similarity: float
    direction: str
    outcome: str
    pnl_pct: float
    description: str
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    lessons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_id': self.pattern_id,
            'name': self.name,
            'similarity': self.similarity,
            'direction': self.direction,
            'outcome': self.outcome,
            'pnl_pct': self.pnl_pct,
            'description': self.description,
            'category': self.category,
            'tags': self.tags,
            'lessons': self.lessons,
        }


@dataclass
class RAGAnalysisResult:
    """Результат RAG анализа."""
    matches: List[PatternMatch]
    win_rate: float
    avg_pnl: float
    dominant_direction: str
    confidence: float
    reasoning: str
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# RAG AGENT
# =============================================================================

class RAGAgent(BaseAgent):
    """
    RAG Agent — ищет похожие паттерны и формирует мнение.
    
    Реализует Agent protocol:
    - analyze(context) → AgentOpinion
    - health_check() → HealthCheckResult
    - get_metadata() → AgentMetadata
    
    Usage:
        agent = RAGAgent(top_k=5, min_similarity=0.6)
        opinion = agent.analyze(council_context)
    """
    
    name = "RAGAgent"
    version = "2.0.0"
    weight = 0.30
    description = "Pattern-based analysis using historical trades"
    
    def __init__(
        self,
        top_k: int = 5,
        min_similarity: float = 0.5,
        include_outcomes: bool = True,
        patterns_dir: Optional[str] = None,
        config: Optional[AgentConfig] = None,
    ):
        super().__init__(config)
        
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.include_outcomes = include_outcomes
        self.patterns_dir = Path(patterns_dir) if patterns_dir else PROJECT_ROOT / "rag" / "patterns"
        
        # Lazy load retriever
        self._retriever = None
        self._patterns_count = 0
        self._last_index_update: Optional[datetime] = None
    
    def _ensure_retriever(self) -> bool:
        """Убедиться, что retriever загружен."""
        if self._retriever is not None:
            return True
        
        try:
            from rag.hybrid_retriever import HybridRetriever
            self._retriever = HybridRetriever()
            
            # Получить статистику
            stats = self._retriever.get_stats()
            self._patterns_count = stats.get('vector_store', {}).get('total_patterns', 0)
            
            # Auto-rebuild если пусто
            if self._patterns_count == 0:
                logger.warning("RAG index empty, attempting rebuild...")
                if hasattr(self._retriever, 'rebuild_index'):
                    self._patterns_count = self._retriever.rebuild_index(verbose=False)
                    logger.info(f"Rebuilt RAG index: {self._patterns_count} patterns")
            
            self._last_index_update = datetime.now()
            return True
            
        except ImportError as e:
            logger.error(f"Cannot import HybridRetriever: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            return False
    
    def _do_analyze(self, context: Any) -> AgentOpinion:
        """
        Анализ контекста через RAG.
        
        1. Извлекаем текущее состояние рынка
        2. Ищем похожие паттерны
        3. Анализируем outcomes
        4. Формируем мнение
        """
        # Проверяем retriever
        if not self._ensure_retriever():
            return self._fallback_opinion("RAG retriever not available")
        
        # Извлекаем данные из контекста
        market_state = self._extract_market_state(context)
        
        # Ищем похожие паттерны
        matches = self._retrieve_patterns(market_state)
        
        # Анализируем результаты
        analysis = self._analyze_matches(matches, market_state)
        
        # Формируем мнение
        return self._build_opinion(analysis, matches)
    
    def _extract_market_state(self, context: Any) -> Dict[str, Any]:
        """Извлечь состояние рынка из контекста."""
        state = {}
        
        # Из CouncilContext
        if hasattr(context, 'market'):
            market = context.market
            state['symbol'] = getattr(market, 'symbol', 'BTC_USDT')
            state['timeframe'] = getattr(market, 'timeframe', '4h')
            state['rsi'] = getattr(market, 'rsi', 50)
            state['regime'] = getattr(market, 'regime', 'normal')
            if hasattr(market.regime, 'value'):
                state['regime'] = market.regime.value
            state['atr_pct'] = getattr(market, 'atr_pct', 0.02)
            state['volume_ratio'] = getattr(market, 'volume_ratio', 1.0)
        
        # Из quant_signals
        if hasattr(context, 'quant_signals'):
            qs = context.quant_signals
            state['p_rb'] = getattr(qs, 'p_rb', None)
            state['p_ml'] = getattr(qs, 'p_ml', None)
            state['agreement'] = getattr(qs, 'agreement', None)
        
        # Определяем направление на основе сигналов
        p_hyb = None
        if hasattr(context, 'quant_signals') and hasattr(context.quant_signals, 'p_hyb'):
            p_hyb = context.quant_signals.p_hyb
        
        if p_hyb is not None and not np.isnan(p_hyb):
            if p_hyb > 0.6:
                state['direction'] = 'long'
            elif p_hyb < 0.4:
                state['direction'] = 'short'
            else:
                state['direction'] = 'neutral'
        else:
            state['direction'] = 'neutral'
        
        return state
    
    def _retrieve_patterns(self, market_state: Dict[str, Any]) -> List[PatternMatch]:
        """Найти похожие паттерны."""
        matches = []
        
        if self._retriever is None:
            return matches
        
        try:
            # Формируем запрос
            query = self._build_query(market_state)
            
            # Ищем
            results = self._retriever.retrieve(market_state, top_k=self.top_k)
            
            # Конвертируем результаты
            for r in results:
                similarity = getattr(r, 'similarity', getattr(r, 'score', 0.5))
                
                if similarity < self.min_similarity:
                    continue
                
                match = PatternMatch(
                    pattern_id=getattr(r, 'pattern_id', getattr(r, 'id', 'unknown')),
                    name=getattr(r, 'name', 'Unknown Pattern'),
                    similarity=similarity,
                    direction=getattr(r, 'direction', 'neutral'),
                    outcome=getattr(r, 'outcome', 'unknown'),
                    pnl_pct=getattr(r, 'pnl_pct', 0.0),
                    description=getattr(r, 'description', ''),
                    category=getattr(r, 'category', None),
                    tags=getattr(r, 'tags', []),
                    lessons=getattr(r, 'lessons', []),
                )
                matches.append(match)
                
        except Exception as e:
            logger.warning(f"Pattern retrieval failed: {e}")
        
        return matches
    
    def _build_query(self, market_state: Dict[str, Any]) -> str:
        """Построить текстовый запрос для поиска."""
        parts = []
        
        direction = market_state.get('direction', 'neutral')
        parts.append(f"{direction} setup")
        
        rsi = market_state.get('rsi', 50)
        if rsi < 30:
            parts.append("RSI oversold")
        elif rsi > 70:
            parts.append("RSI overbought")
        
        regime = market_state.get('regime', 'normal')
        if regime != 'normal':
            parts.append(f"{regime} volatility")
        
        return " ".join(parts)
    
    def _analyze_matches(
        self,
        matches: List[PatternMatch],
        market_state: Dict[str, Any],
    ) -> RAGAnalysisResult:
        """Анализировать найденные паттерны."""
        warnings = []
        
        if not matches:
            return RAGAnalysisResult(
                matches=[],
                win_rate=0.5,
                avg_pnl=0.0,
                dominant_direction='neutral',
                confidence=0.3,
                reasoning="No similar patterns found",
                warnings=["No historical data available"],
            )
        
        # Считаем статистику
        wins = [m for m in matches if m.outcome == 'win']
        losses = [m for m in matches if m.outcome == 'loss']
        
        win_rate = len(wins) / len(matches) if matches else 0.5
        
        # Средний PnL (взвешенный по similarity)
        total_weight = sum(m.similarity for m in matches)
        if total_weight > 0:
            avg_pnl = sum(m.pnl_pct * m.similarity for m in matches) / total_weight
        else:
            avg_pnl = 0.0
        
        # Доминирующее направление
        directions = {}
        for m in matches:
            d = m.direction
            directions[d] = directions.get(d, 0) + m.similarity
        
        dominant_direction = max(directions, key=directions.get) if directions else 'neutral'
        
        # Confidence на основе:
        # 1. Количества паттернов
        # 2. Win rate
        # 3. Similarity scores
        n_patterns = len(matches)
        avg_similarity = np.mean([m.similarity for m in matches])
        
        confidence = (
            0.3 * min(n_patterns / 5, 1.0) +  # больше паттернов = лучше
            0.4 * win_rate +                    # win rate
            0.3 * avg_similarity                # качество matches
        )
        
        # Предупреждения
        if n_patterns < 3:
            warnings.append(f"Low pattern count ({n_patterns})")
        
        if avg_similarity < 0.6:
            warnings.append(f"Low similarity scores ({avg_similarity:.2f})")
        
        # Reasoning
        reasoning_parts = [
            f"Found {n_patterns} similar patterns",
            f"Win rate: {win_rate:.0%}",
            f"Avg PnL: {avg_pnl:+.1f}%",
        ]
        
        if wins:
            top_win = max(wins, key=lambda m: m.pnl_pct)
            reasoning_parts.append(f"Best: {top_win.name} (+{top_win.pnl_pct:.1f}%)")
        
        if losses:
            worst_loss = min(losses, key=lambda m: m.pnl_pct)
            reasoning_parts.append(f"Worst: {worst_loss.name} ({worst_loss.pnl_pct:.1f}%)")
        
        return RAGAnalysisResult(
            matches=matches,
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            dominant_direction=dominant_direction,
            confidence=confidence,
            reasoning=" | ".join(reasoning_parts),
            warnings=warnings,
        )
    
    def _build_opinion(
        self,
        analysis: RAGAnalysisResult,
        matches: List[PatternMatch],
    ) -> AgentOpinion:
        """Построить мнение на основе анализа."""
        
        # Определяем действие
        if analysis.confidence < 0.4:
            action = ActionType.HOLD
        elif analysis.dominant_direction == 'long' and analysis.win_rate > 0.5:
            action = ActionType.LONG
        elif analysis.dominant_direction == 'short' and analysis.win_rate > 0.5:
            action = ActionType.SHORT
        else:
            action = ActionType.HOLD
        
        # Position size на основе confidence и win_rate
        if action == ActionType.HOLD:
            position_size = 0.0
        else:
            position_size = min(10.0, 5.0 * analysis.confidence * analysis.win_rate)
        
        # Stop-loss и take-profit из паттернов
        sl_pct = 4.0  # default
        tp_pct = 8.0  # default
        
        if matches:
            # Анализируем исторические SL/TP
            losses_pcts = [abs(m.pnl_pct) for m in matches if m.outcome == 'loss']
            wins_pcts = [m.pnl_pct for m in matches if m.outcome == 'win']
            
            if losses_pcts:
                sl_pct = np.percentile(losses_pcts, 75)  # 75-й перцентиль потерь
            if wins_pcts:
                tp_pct = np.percentile(wins_pcts, 50)  # медиана выигрышей
        
        # Собираем lessons
        all_lessons = []
        for m in matches[:3]:  # top 3
            all_lessons.extend(m.lessons)
        
        return AgentOpinion(
            agent_name=self.name,
            proposed_action=action,
            confidence=analysis.confidence,
            reasoning=analysis.reasoning,
            position_size_pct=position_size,
            suggested_sl_pct=sl_pct,
            suggested_tp_pct=tp_pct,
            supporting_data={
                'matches': [m.to_dict() for m in matches],
                'win_rate': analysis.win_rate,
                'avg_pnl': analysis.avg_pnl,
                'dominant_direction': analysis.dominant_direction,
                'lessons': all_lessons[:5],  # top 5 lessons
            },
            warnings=analysis.warnings,
        )
    
    def _fallback_opinion(self, reason: str) -> AgentOpinion:
        """Fallback мнение когда RAG недоступен."""
        return AgentOpinion(
            agent_name=self.name,
            proposed_action=ActionType.HOLD,
            confidence=0.0,
            reasoning=f"RAG unavailable: {reason}",
            position_size_pct=0.0,
            warnings=[reason],
        )
    
    def health_check(self) -> HealthCheckResult:
        """Проверка здоровья RAG агента."""
        if not self._ensure_retriever():
            return HealthCheckResult(
                status=HealthStatus.CRITICAL,
                message="RAG retriever not available",
            )
        
        if self._patterns_count == 0:
            return HealthCheckResult(
                status=HealthStatus.DEGRADED,
                message="No patterns indexed",
                details={'patterns_count': 0},
            )
        
        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message=f"{self._patterns_count} patterns indexed",
            details={
                'patterns_count': self._patterns_count,
                'last_index_update': self._last_index_update.isoformat() if self._last_index_update else None,
                'min_similarity': self.min_similarity,
                'top_k': self.top_k,
            },
        )
    
    def get_metadata(self) -> AgentMetadata:
        """Метаданные RAG агента."""
        return AgentMetadata(
            name=self.name,
            version=self.version,
            description=self.description,
            tags=["rag", "patterns", "historical", "memory"],
            capabilities=[
                "pattern_retrieval",
                "similarity_search",
                "outcome_analysis",
                "lesson_extraction",
            ],
            dependencies=["faiss", "sentence-transformers"],
        )
    
    def rebuild_index(self, verbose: bool = True) -> int:
        """Пересобрать индекс паттернов."""
        if not self._ensure_retriever():
            return 0
        
        if hasattr(self._retriever, 'rebuild_index'):
            count = self._retriever.rebuild_index(verbose=verbose)
            self._patterns_count = count
            self._last_index_update = datetime.now()
            return count
        
        return 0
    
    def get_patterns_count(self) -> int:
        """Получить количество паттернов."""
        self._ensure_retriever()
        return self._patterns_count


# =============================================================================
# FACTORY
# =============================================================================

def create_rag_agent(config: Optional[Dict[str, Any]] = None) -> RAGAgent:
    """Фабрика для создания RAG агента."""
    config = config or {}
    
    return RAGAgent(
        top_k=config.get('top_k', 5),
        min_similarity=config.get('min_similarity', 0.5),
        include_outcomes=config.get('include_outcomes', True),
        patterns_dir=config.get('patterns_dir'),
    )