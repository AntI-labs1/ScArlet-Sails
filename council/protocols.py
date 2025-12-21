"""
ScArlet-Sails Agent Protocols

Единый контракт для ВСЕХ агентов системы.
Quant, RAG, LLM — все реализуют один интерфейс.

Philosophy:
    "Контракты > реализация"
    "Каждый агент — независимая единица с известным интерфейсом"
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


# =============================================================================
# ENUMS
# =============================================================================

class ActionType(Enum):
    """Возможные действия агента."""
    LONG = "long"
    SHORT = "short"
    HOLD = "hold"
    CLOSE = "close"


class HealthStatus(Enum):
    """Статус здоровья агента."""
    HEALTHY = "healthy"           # Всё работает
    DEGRADED = "degraded"         # Работает с ограничениями
    UNHEALTHY = "unhealthy"       # Не работает, но recoverable
    CRITICAL = "critical"         # Требует вмешательства
    UNKNOWN = "unknown"           # Не удалось определить


class ConfidenceLevel(Enum):
    """Уровень уверенности."""
    VERY_LOW = "very_low"      # < 0.3
    LOW = "low"                # 0.3 - 0.5
    MEDIUM = "medium"          # 0.5 - 0.7
    HIGH = "high"              # 0.7 - 0.85
    VERY_HIGH = "very_high"    # > 0.85


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AgentMetadata:
    """Метаданные агента для UI и мониторинга."""
    name: str
    version: str
    description: str
    author: str = "ScArlet-Sails Team"
    tags: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'tags': self.tags,
            'capabilities': self.capabilities,
            'dependencies': self.dependencies,
        }


@dataclass
class HealthCheckResult:
    """Результат проверки здоровья агента."""
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    latency_ms: Optional[float] = None
    
    @property
    def is_operational(self) -> bool:
        """Может ли агент работать?"""
        return self.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status.value,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'latency_ms': self.latency_ms,
            'is_operational': self.is_operational,
        }


@dataclass
class AgentOpinion:
    """
    Мнение агента — стандартный выход для ВСЕХ агентов.
    
    Независимо от того, Quant это, RAG или LLM —
    все возвращают AgentOpinion.
    """
    agent_name: str
    proposed_action: ActionType
    confidence: float  # 0.0 - 1.0
    reasoning: str
    
    # Опциональные поля
    position_size_pct: float = 5.0  # Рекомендуемый размер позиции
    suggested_sl_pct: Optional[float] = None  # Stop-loss %
    suggested_tp_pct: Optional[float] = None  # Take-profit %
    
    # Дополнительные данные (паттерны, сигналы и т.д.)
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    # Временные метки
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time_ms: Optional[float] = None
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Категория уверенности."""
        if self.confidence < 0.3:
            return ConfidenceLevel.VERY_LOW
        elif self.confidence < 0.5:
            return ConfidenceLevel.LOW
        elif self.confidence < 0.7:
            return ConfidenceLevel.MEDIUM
        elif self.confidence < 0.85:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH
    
    @property
    def is_actionable(self) -> bool:
        """Стоит ли действовать по этому мнению?"""
        return (
            self.proposed_action != ActionType.HOLD and
            self.confidence >= 0.5
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_name': self.agent_name,
            'proposed_action': self.proposed_action.value,
            'confidence': self.confidence,
            'confidence_level': self.confidence_level.value,
            'reasoning': self.reasoning,
            'position_size_pct': self.position_size_pct,
            'suggested_sl_pct': self.suggested_sl_pct,
            'suggested_tp_pct': self.suggested_tp_pct,
            'supporting_data': self.supporting_data,
            'warnings': self.warnings,
            'timestamp': self.timestamp.isoformat(),
            'processing_time_ms': self.processing_time_ms,
            'is_actionable': self.is_actionable,
        }


@dataclass
class AgentConfig:
    """Конфигурация агента."""
    name: str
    enabled: bool = True
    weight: float = 0.33  # Вес при агрегации (0.0 - 1.0)
    priority: int = 0  # Порядок вызова (больше = раньше)
    timeout_ms: int = 5000  # Таймаут вызова
    fallback_action: ActionType = ActionType.HOLD
    
    # Специфичные настройки агента
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'enabled': self.enabled,
            'weight': self.weight,
            'priority': self.priority,
            'timeout_ms': self.timeout_ms,
            'fallback_action': self.fallback_action.value,
            'custom_config': self.custom_config,
        }


# =============================================================================
# AGENT PROTOCOL
# =============================================================================

@runtime_checkable
class Agent(Protocol):
    """
    Единый протокол для ВСЕХ агентов.
    
    QuantAgent, RAGAgent, LLMAgent — все реализуют этот интерфейс.
    Council работает с Agent, не зная конкретной реализации.
    
    Usage:
        class MyCustomAgent:
            name = "MyAgent"
            version = "1.0.0"
            weight = 0.3
            
            def analyze(self, context) -> AgentOpinion:
                # Твоя логика
                pass
            
            def health_check(self) -> HealthCheckResult:
                return HealthCheckResult(HealthStatus.HEALTHY, "OK")
    """
    
    # Обязательные атрибуты
    name: str
    version: str
    weight: float
    
    def analyze(self, context: Any) -> AgentOpinion:
        """
        Анализ контекста и возврат мнения.
        
        Args:
            context: CouncilContext с рыночными данными
            
        Returns:
            AgentOpinion с предложенным действием
        """
        ...
    
    def health_check(self) -> HealthCheckResult:
        """
        Проверка готовности агента.
        
        Returns:
            HealthCheckResult с текущим статусом
        """
        ...
    
    def get_metadata(self) -> AgentMetadata:
        """
        Метаданные агента для UI и мониторинга.
        
        Returns:
            AgentMetadata с описанием агента
        """
        ...


# =============================================================================
# BASE AGENT (Abstract Implementation)
# =============================================================================

class BaseAgent(ABC):
    """
    Базовый класс для агентов.
    
    Реализует общую логику, оставляя analyze() абстрактным.
    
    Usage:
        class MyAgent(BaseAgent):
            name = "MyAgent"
            version = "1.0.0"
            
            def _do_analyze(self, context) -> AgentOpinion:
                # Твоя логика
                pass
    """
    
    # Должны быть переопределены в наследниках
    name: str = "BaseAgent"
    version: str = "0.0.0"
    weight: float = 0.33
    description: str = "Base agent implementation"
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig(name=self.name)
        self._last_health_check: Optional[HealthCheckResult] = None
        self._call_count: int = 0
        self._error_count: int = 0
    
    def analyze(self, context: Any) -> AgentOpinion:
        """
        Обёртка над _do_analyze с логированием и обработкой ошибок.
        """
        import time
        start = time.time()
        
        try:
            self._call_count += 1
            opinion = self._do_analyze(context)
            opinion.processing_time_ms = (time.time() - start) * 1000
            return opinion
            
        except Exception as e:
            self._error_count += 1
            # Fallback opinion
            return AgentOpinion(
                agent_name=self.name,
                proposed_action=self.config.fallback_action,
                confidence=0.0,
                reasoning=f"Agent error: {str(e)}",
                warnings=[f"Agent {self.name} failed: {str(e)}"],
                processing_time_ms=(time.time() - start) * 1000,
            )
    
    @abstractmethod
    def _do_analyze(self, context: Any) -> AgentOpinion:
        """
        Реальная логика анализа.
        Должна быть реализована в наследниках.
        """
        pass
    
    def health_check(self) -> HealthCheckResult:
        """
        Базовая проверка здоровья.
        Может быть переопределена в наследниках.
        """
        error_rate = self._error_count / max(self._call_count, 1)
        
        if error_rate > 0.5:
            status = HealthStatus.CRITICAL
            message = f"High error rate: {error_rate:.1%}"
        elif error_rate > 0.2:
            status = HealthStatus.DEGRADED
            message = f"Elevated error rate: {error_rate:.1%}"
        else:
            status = HealthStatus.HEALTHY
            message = "Agent operational"
        
        result = HealthCheckResult(
            status=status,
            message=message,
            details={
                'call_count': self._call_count,
                'error_count': self._error_count,
                'error_rate': error_rate,
            }
        )
        
        self._last_health_check = result
        return result
    
    def get_metadata(self) -> AgentMetadata:
        """Метаданные агента."""
        return AgentMetadata(
            name=self.name,
            version=self.version,
            description=self.description,
            capabilities=[],
            dependencies=[],
        )
    
    def reset_stats(self) -> None:
        """Сброс статистики."""
        self._call_count = 0
        self._error_count = 0
    
    def __repr__(self) -> str:
        return f"{self.name}(v{self.version}, weight={self.weight})"


# =============================================================================
# TYPE ALIASES
# =============================================================================

# Для удобства типизации
AgentList = List[Agent]
OpinionList = List[AgentOpinion]