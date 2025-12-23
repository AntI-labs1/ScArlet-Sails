"""
ScArlet-Sails Configuration Schema

Pydantic схемы для валидации конфигурации.
Человек пишет YAML/JSON, система валидирует.

Philosophy:
    "Декларативная конфигурация > императивный код"
    "Ошибки конфигурации ловятся ДО запуска"
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# ENUMS
# =============================================================================

class UIType(str, Enum):
    TERMINAL = "terminal"
    WEB = "web"
    API = "api"


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# =============================================================================
# AGENT CONFIG
# =============================================================================

class AgentConfigSchema(BaseModel):
    """Конфигурация одного агента."""
    
    name: str = Field(..., description="Уникальное имя агента")
    module: str = Field(..., description="Python модуль (e.g., 'council.quant_agent')")
    class_name: str = Field(..., alias="class", description="Имя класса в модуле")
    
    enabled: bool = Field(True, description="Включен ли агент")
    weight: float = Field(0.33, ge=0.0, le=1.0, description="Вес при агрегации")
    priority: int = Field(0, description="Приоритет вызова (больше = раньше)")
    timeout_ms: int = Field(5000, ge=100, le=60000, description="Таймаут в мс")
    
    # Специфичные настройки агента
    config: Dict[str, Any] = Field(default_factory=dict, description="Настройки агента")
    
    class Config:
        populate_by_name = True


# =============================================================================
# RAG CONFIG
# =============================================================================

class RAGConfigSchema(BaseModel):
    """Конфигурация RAG системы."""
    
    patterns_dir: str = Field("rag/patterns", description="Директория с паттернами")
    index_dir: str = Field("rag/faiss_index", description="Директория FAISS индекса")
    
    embedding_model: str = Field(
        "all-MiniLM-L6-v2",
        description="Модель для эмбеддингов"
    )
    
    top_k: int = Field(5, ge=1, le=20, description="Сколько паттернов возвращать")
    min_similarity: float = Field(0.5, ge=0.0, le=1.0, description="Минимальное сходство")
    
    auto_rebuild: bool = Field(True, description="Автоматически пересобирать индекс")
    
    @field_validator('patterns_dir', 'index_dir')
    @classmethod
    def validate_path(cls, v: str) -> str:
        # Просто валидация формата, не существования
        if not v:
            raise ValueError("Path cannot be empty")
        return v


# =============================================================================
# LLM CONFIG
# =============================================================================

class LLMConfigSchema(BaseModel):
    """Конфигурация LLM агента."""
    
    provider: str = Field("custom", description="Провайдер LLM (custom/openai/anthropic)")
    model: str = Field("", description="Название модели")
    
    endpoint: Optional[str] = Field(None, description="URL эндпоинта для custom LLM")
    api_key_env: Optional[str] = Field(None, description="Имя переменной окружения с API key")
    
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(1000, ge=100, le=8000)
    timeout_s: int = Field(30, ge=5, le=120)
    
    # Промпты
    system_prompt_file: Optional[str] = Field(None, description="Файл с system prompt")
    
    # Fallback
    fallback_enabled: bool = Field(True, description="Использовать fallback при ошибке")
    fallback_action: str = Field("hold", description="Действие при ошибке")


# =============================================================================
# UI CONFIG
# =============================================================================

class TerminalUIConfig(BaseModel):
    """Конфигурация терминального UI."""
    
    colors: bool = Field(True, description="Использовать цвета")
    unicode: bool = Field(True, description="Использовать Unicode символы")
    table_style: str = Field("rounded", description="Стиль таблиц (rounded/square/minimal)")
    show_charts: bool = Field(False, description="Показывать ASCII графики")


class WebUIConfig(BaseModel):
    """Конфигурация веб UI."""
    
    host: str = Field("127.0.0.1")
    port: int = Field(8050, ge=1024, le=65535)
    debug: bool = Field(False)
    auto_open: bool = Field(True, description="Открыть браузер автоматически")


class UIConfigSchema(BaseModel):
    """Конфигурация UI."""
    
    default: UIType = Field(UIType.TERMINAL)
    terminal: TerminalUIConfig = Field(default_factory=TerminalUIConfig)
    web: WebUIConfig = Field(default_factory=WebUIConfig)


# =============================================================================
# LEARNING CONFIG
# =============================================================================

class LearningConfigSchema(BaseModel):
    """Конфигурация learning loop."""
    
    enabled: bool = Field(True, description="Включен ли learning loop")
    
    decisions_dir: str = Field("rag/decisions", description="Директория для решений")
    
    auto_update_weights: bool = Field(
        False,
        description="Автоматически обновлять веса агентов"
    )
    
    min_decisions_for_update: int = Field(
        20,
        ge=5,
        description="Минимум решений для обновления весов"
    )
    
    report_schedule: Optional[str] = Field(
        None,
        description="Cron выражение для отчётов (e.g., '0 9 * * 1' = каждый понедельник 9:00)"
    )


# =============================================================================
# RISK CONFIG
# =============================================================================

class RiskConfigSchema(BaseModel):
    """Конфигурация риск-менеджмента."""
    
    max_position_pct: float = Field(10.0, ge=0.1, le=100.0)
    max_daily_loss_pct: float = Field(3.0, ge=0.1, le=50.0)
    max_weekly_loss_pct: float = Field(7.0, ge=0.1, le=100.0)
    
    default_sl_pct: float = Field(4.0, ge=0.5, le=20.0)
    default_tp_pct: float = Field(8.0, ge=1.0, le=50.0)
    
    min_confidence: float = Field(0.5, ge=0.0, le=1.0)
    min_agreement: float = Field(0.6, ge=0.0, le=1.0)


# =============================================================================
# MAIN CONFIG
# =============================================================================

class CouncilConfig(BaseModel):
    """
    Главная конфигурация Council.
    
    Example YAML:
        council:
          name: "ScArlet-Sails Council"
          version: "1.0.0"
          
        agents:
          - name: QuantAgent
            module: council.quant_agent
            class: QuantAgent
            weight: 0.3
            
          - name: RAGAgent
            module: rag.rag_agent
            class: RAGAgent
            weight: 0.3
            
          - name: LLMAgent
            module: council.llm_agent
            class: LLMAgent
            weight: 0.4
            
        rag:
          patterns_dir: rag/patterns
          top_k: 5
          
        llm:
          provider: custom
          endpoint: http://localhost:8000/predict
          
        ui:
          default: terminal
          
        risk:
          max_position_pct: 10.0
          max_daily_loss_pct: 3.0
    """
    
    # Метаданные
    name: str = Field("ScArlet-Sails Council", description="Название системы")
    version: str = Field("1.0.0")
    description: str = Field("")
    
    # Агенты
    agents: List[AgentConfigSchema] = Field(default_factory=list)
    
    # Подсистемы
    rag: RAGConfigSchema = Field(default_factory=RAGConfigSchema)
    llm: LLMConfigSchema = Field(default_factory=LLMConfigSchema)
    ui: UIConfigSchema = Field(default_factory=UIConfigSchema)
    learning: LearningConfigSchema = Field(default_factory=LearningConfigSchema)
    risk: RiskConfigSchema = Field(default_factory=RiskConfigSchema)
    
    # Логирование
    log_level: LogLevel = Field(LogLevel.INFO)
    log_file: Optional[str] = Field(None)
    
    @model_validator(mode='after')
    def validate_agent_weights(self) -> 'CouncilConfig':
        """Проверить, что веса агентов в сумме ≈ 1."""
        enabled_agents = [a for a in self.agents if a.enabled]
        if enabled_agents:
            total_weight = sum(a.weight for a in enabled_agents)
            if not (0.9 <= total_weight <= 1.1):
                # Warning, not error — можно нормализовать позже
                pass
        return self
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'CouncilConfig':
        """Загрузить конфиг из YAML файла."""
        import yaml
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> 'CouncilConfig':
        """Загрузить конфиг из JSON файла."""
        import json
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(**data)
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Сохранить конфиг в YAML файл."""
        import yaml
        
        # model_dump с mode='json' конвертирует Enum в строки
        data = self.model_dump(mode='json')
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    def to_json(self, path: Union[str, Path]) -> None:
        """Сохранить конфиг в JSON файл."""
        import json
        
        with open(path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)
    
    def get_agent_config(self, name: str) -> Optional[AgentConfigSchema]:
        """Получить конфиг агента по имени."""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None


# =============================================================================
# CONFIG LOADER
# =============================================================================

class ConfigLoader:
    """
    Загрузчик конфигурации с поддержкой:
    - Дефолтных значений
    - Переменных окружения
    - Нескольких файлов (merge)
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(".")
        self._config: Optional[CouncilConfig] = None
    
    def load(
        self,
        config_file: str = "config/council.yaml",
        env_prefix: str = "COUNCIL_",
    ) -> CouncilConfig:
        """
        Загрузить конфигурацию.
        
        Приоритет:
        1. Env variables (COUNCIL_*)
        2. Config file
        3. Defaults
        """
        import os
        
        config_path = self.base_dir / config_file
        
        # Загрузка из файла или дефолт
        if config_path.exists():
            if config_path.suffix in ['.yaml', '.yml']:
                self._config = CouncilConfig.from_yaml(config_path)
            else:
                self._config = CouncilConfig.from_json(config_path)
        else:
            self._config = CouncilConfig()
        
        # Переопределение из env
        self._apply_env_overrides(env_prefix)
        
        return self._config
    
    def _apply_env_overrides(self, prefix: str) -> None:
        """Применить переопределения из env переменных."""
        import os
        
        if self._config is None:
            return
        
        # Примеры:
        # COUNCIL_LOG_LEVEL=debug → config.log_level = "debug"
        # COUNCIL_RISK_MAX_POSITION_PCT=5.0 → config.risk.max_position_pct = 5.0
        
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            
            # Удаляем префикс и разбиваем на части
            path = key[len(prefix):].lower().split('_')
            
            # Применяем к конфигу (упрощённая версия)
            # TODO: полная реализация для вложенных путей
    
    @property
    def config(self) -> CouncilConfig:
        if self._config is None:
            self._config = self.load()
        return self._config


# =============================================================================
# DEFAULT CONFIG GENERATOR
# =============================================================================

def generate_default_config(output_path: Path) -> None:
    """Сгенерировать дефолтный конфиг."""
    config = CouncilConfig(
        name="ScArlet-Sails Council",
        version="1.0.0",
        description="Quantitative trading decision support system",
        agents=[
            AgentConfigSchema(
                name="QuantAgent",
                module="council.quant_agent",
                class_name="QuantAgent",
                weight=0.3,
                priority=10,
            ),
            AgentConfigSchema(
                name="RAGAgent",
                module="rag.rag_agent",
                class_name="RAGAgent",
                weight=0.3,
                priority=5,
            ),
            AgentConfigSchema(
                name="LLMAgent",
                module="council.llm_agent",
                class_name="LLMAgent",
                weight=0.4,
                priority=0,
                enabled=False,  # По умолчанию выключен
            ),
        ],
    )
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix in ['.yaml', '.yml']:
        config.to_yaml(output_path)
    else:
        config.to_json(output_path)
    
    print(f"Generated default config: {output_path}")