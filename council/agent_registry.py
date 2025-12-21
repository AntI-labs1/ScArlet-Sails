"""
ScArlet-Sails Agent Registry

Динамическая регистрация и управление агентами.
Позволяет добавлять/удалять агентов без изменения кода.

Philosophy:
    "Plugin system > Hardcoded agents"
    "Человек собирает систему декларативно через конфиг"
"""

import importlib.util
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import yaml

from .protocols import (
    Agent,
    AgentConfig,
    AgentMetadata,
    AgentOpinion,
    BaseAgent,
    HealthCheckResult,
    HealthStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class AgentRegistryError(Exception):
    """Базовая ошибка реестра агентов."""
    pass


class AgentNotFoundError(AgentRegistryError):
    """Агент не найден."""
    pass


class AgentAlreadyExistsError(AgentRegistryError):
    """Агент уже зарегистрирован."""
    pass


class InvalidAgentError(AgentRegistryError):
    """Агент не соответствует протоколу."""
    pass


# =============================================================================
# AGENT ENTRY
# =============================================================================

@dataclass
class AgentEntry:
    """Запись об агенте в реестре."""
    name: str
    agent: Agent
    config: AgentConfig
    registered_at: datetime = field(default_factory=datetime.now)
    source: str = "manual"  # manual, config, plugin
    
    @property
    def is_enabled(self) -> bool:
        return self.config.enabled
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'version': self.agent.version,
            'weight': self.config.weight,
            'enabled': self.config.enabled,
            'priority': self.config.priority,
            'source': self.source,
            'registered_at': self.registered_at.isoformat(),
        }


# =============================================================================
# AGENT REGISTRY
# =============================================================================

class AgentRegistry:
    """
    Реестр агентов — центральное место для управления всеми агентами.
    
    Features:
    - Динамическая регистрация агентов
    - Загрузка из конфига (YAML/JSON)
    - Загрузка плагинов из директории
    - Health check всех агентов
    - Приоритезация и взвешивание
    
    Usage:
        registry = AgentRegistry()
        
        # Ручная регистрация
        registry.register(QuantAgent())
        registry.register(RAGAgent())
        
        # Из конфига
        registry.load_from_config("config/agents.yaml")
        
        # Получить активных агентов
        agents = registry.get_enabled_agents()
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self._agents: Dict[str, AgentEntry] = {}
        self._plugins_dir: Optional[Path] = None
        
        if config_path:
            self.load_from_config(config_path)
    
    # =========================================================================
    # REGISTRATION
    # =========================================================================
    
    def register(
        self,
        agent: Agent,
        config: Optional[AgentConfig] = None,
        source: str = "manual",
        overwrite: bool = False,
    ) -> None:
        """
        Зарегистрировать агента.
        
        Args:
            agent: Экземпляр агента (реализует Agent protocol)
            config: Конфигурация агента (опционально)
            source: Источник регистрации (manual/config/plugin)
            overwrite: Перезаписать если существует
            
        Raises:
            InvalidAgentError: Если агент не соответствует протоколу
            AgentAlreadyExistsError: Если агент уже существует
        """
        # Валидация протокола
        if not self._validate_agent(agent):
            raise InvalidAgentError(
                f"Agent {getattr(agent, 'name', 'unknown')} does not implement Agent protocol"
            )
        
        name = agent.name
        
        # Проверка на дубликат
        if name in self._agents and not overwrite:
            raise AgentAlreadyExistsError(f"Agent '{name}' already registered")
        
        # Создание конфига если не передан
        if config is None:
            config = AgentConfig(
                name=name,
                weight=getattr(agent, 'weight', 0.33),
            )
        
        # Регистрация
        entry = AgentEntry(
            name=name,
            agent=agent,
            config=config,
            source=source,
        )
        
        self._agents[name] = entry
        logger.info(f"Registered agent: {name} (source={source}, weight={config.weight})")
    
    def unregister(self, name: str) -> None:
        """Удалить агента из реестра."""
        if name not in self._agents:
            raise AgentNotFoundError(f"Agent '{name}' not found")
        
        del self._agents[name]
        logger.info(f"Unregistered agent: {name}")
    
    def _validate_agent(self, agent: Any) -> bool:
        """Проверить, реализует ли объект Agent protocol."""
        required_attrs = ['name', 'version', 'weight']
        required_methods = ['analyze', 'health_check', 'get_metadata']
        
        for attr in required_attrs:
            if not hasattr(agent, attr):
                logger.warning(f"Agent missing attribute: {attr}")
                return False
        
        for method in required_methods:
            if not hasattr(agent, method) or not callable(getattr(agent, method)):
                logger.warning(f"Agent missing method: {method}")
                return False
        
        return True
    
    # =========================================================================
    # RETRIEVAL
    # =========================================================================
    
    def get(self, name: str) -> Agent:
        """Получить агента по имени."""
        if name not in self._agents:
            raise AgentNotFoundError(f"Agent '{name}' not found")
        return self._agents[name].agent
    
    def get_entry(self, name: str) -> AgentEntry:
        """Получить запись агента."""
        if name not in self._agents:
            raise AgentNotFoundError(f"Agent '{name}' not found")
        return self._agents[name]
    
    def get_all(self) -> List[Agent]:
        """Получить всех агентов."""
        return [entry.agent for entry in self._agents.values()]
    
    def get_enabled(self) -> List[Agent]:
        """Получить только включенных агентов."""
        return [
            entry.agent 
            for entry in self._agents.values() 
            if entry.is_enabled
        ]
    
    def get_by_priority(self) -> List[Agent]:
        """Получить агентов отсортированных по приоритету."""
        sorted_entries = sorted(
            self._agents.values(),
            key=lambda e: e.config.priority,
            reverse=True
        )
        return [entry.agent for entry in sorted_entries if entry.is_enabled]
    
    def get_weights(self) -> Dict[str, float]:
        """Получить веса всех агентов."""
        return {
            entry.name: entry.config.weight
            for entry in self._agents.values()
            if entry.is_enabled
        }
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    def load_from_config(self, config_path: Path) -> int:
        """
        Загрузить агентов из конфига.
        
        Config format (YAML):
            agents:
              - name: QuantAgent
                module: council.quant_agent
                class: QuantAgent
                weight: 0.3
                enabled: true
                
              - name: RAGAgent
                module: rag.rag_agent
                class: RAGAgent
                weight: 0.3
                config:
                  top_k: 5
        
        Returns:
            Количество загруженных агентов
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return 0
        
        # Загрузка конфига
        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        agents_config = config.get('agents', [])
        loaded = 0
        
        for agent_cfg in agents_config:
            try:
                agent = self._load_agent_from_config(agent_cfg)
                if agent:
                    self.register(
                        agent,
                        config=AgentConfig(
                            name=agent_cfg['name'],
                            weight=agent_cfg.get('weight', 0.33),
                            enabled=agent_cfg.get('enabled', True),
                            priority=agent_cfg.get('priority', 0),
                            custom_config=agent_cfg.get('config', {}),
                        ),
                        source="config",
                        overwrite=True,
                    )
                    loaded += 1
            except Exception as e:
                logger.error(f"Failed to load agent {agent_cfg.get('name')}: {e}")
        
        logger.info(f"Loaded {loaded} agents from config")
        return loaded
    
    def _load_agent_from_config(self, agent_cfg: Dict) -> Optional[Agent]:
        """Загрузить агента из конфигурации."""
        module_name = agent_cfg.get('module')
        class_name = agent_cfg.get('class')
        
        if not module_name or not class_name:
            logger.warning(f"Agent config missing module or class: {agent_cfg}")
            return None
        
        try:
            module = importlib.import_module(module_name)
            agent_class = getattr(module, class_name)
            
            # Создание экземпляра
            custom_config = agent_cfg.get('config', {})
            return agent_class(**custom_config)
            
        except ImportError as e:
            logger.error(f"Cannot import module {module_name}: {e}")
            return None
        except AttributeError as e:
            logger.error(f"Class {class_name} not found in {module_name}: {e}")
            return None
    
    def load_plugins(self, plugins_dir: Path) -> int:
        """
        Загрузить агентов-плагинов из директории.
        
        Каждый плагин — файл .py с классом, наследующим BaseAgent.
        
        Args:
            plugins_dir: Директория с плагинами
            
        Returns:
            Количество загруженных плагинов
        """
        plugins_dir = Path(plugins_dir)
        self._plugins_dir = plugins_dir
        
        if not plugins_dir.exists():
            logger.warning(f"Plugins directory not found: {plugins_dir}")
            return 0
        
        loaded = 0
        
        for plugin_file in plugins_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue
                
            try:
                agent = self._load_plugin(plugin_file)
                if agent:
                    self.register(agent, source="plugin", overwrite=True)
                    loaded += 1
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_file}: {e}")
        
        logger.info(f"Loaded {loaded} plugins from {plugins_dir}")
        return loaded
    
    def _load_plugin(self, plugin_file: Path) -> Optional[Agent]:
        """Загрузить плагин из файла."""
        spec = importlib.util.spec_from_file_location(
            plugin_file.stem,
            plugin_file
        )
        
        if spec is None or spec.loader is None:
            return None
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Ищем класс, наследующий BaseAgent
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type) and
                issubclass(attr, BaseAgent) and
                attr is not BaseAgent
            ):
                return attr()
        
        return None
    
    # =========================================================================
    # HEALTH & STATUS
    # =========================================================================
    
    def health_check_all(self) -> Dict[str, HealthCheckResult]:
        """Проверить здоровье всех агентов."""
        results = {}
        
        for name, entry in self._agents.items():
            try:
                results[name] = entry.agent.health_check()
            except Exception as e:
                results[name] = HealthCheckResult(
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {e}",
                )
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Получить статус реестра."""
        health = self.health_check_all()
        
        return {
            'total_agents': len(self._agents),
            'enabled_agents': len(self.get_enabled()),
            'agents': {
                name: {
                    **entry.to_dict(),
                    'health': health[name].to_dict() if name in health else None,
                }
                for name, entry in self._agents.items()
            },
            'plugins_dir': str(self._plugins_dir) if self._plugins_dir else None,
        }
    
    # =========================================================================
    # UTILITY
    # =========================================================================
    
    def enable(self, name: str) -> None:
        """Включить агента."""
        if name not in self._agents:
            raise AgentNotFoundError(f"Agent '{name}' not found")
        self._agents[name].config.enabled = True
        logger.info(f"Enabled agent: {name}")
    
    def disable(self, name: str) -> None:
        """Выключить агента."""
        if name not in self._agents:
            raise AgentNotFoundError(f"Agent '{name}' not found")
        self._agents[name].config.enabled = False
        logger.info(f"Disabled agent: {name}")
    
    def set_weight(self, name: str, weight: float) -> None:
        """Установить вес агента."""
        if name not in self._agents:
            raise AgentNotFoundError(f"Agent '{name}' not found")
        if not 0 <= weight <= 1:
            raise ValueError("Weight must be between 0 and 1")
        self._agents[name].config.weight = weight
        logger.info(f"Set weight for {name}: {weight}")
    
    def normalize_weights(self) -> None:
        """Нормализовать веса так, чтобы сумма = 1."""
        enabled = [e for e in self._agents.values() if e.is_enabled]
        if not enabled:
            return
        
        total = sum(e.config.weight for e in enabled)
        if total == 0:
            return
        
        for entry in enabled:
            entry.config.weight /= total
    
    def clear(self) -> None:
        """Очистить реестр."""
        self._agents.clear()
        logger.info("Cleared agent registry")
    
    def __len__(self) -> int:
        return len(self._agents)
    
    def __contains__(self, name: str) -> bool:
        return name in self._agents
    
    def __iter__(self):
        return iter(self._agents.values())
    
    def __repr__(self) -> str:
        return f"AgentRegistry({len(self)} agents)"


# =============================================================================
# GLOBAL REGISTRY (Singleton pattern)
# =============================================================================

_global_registry: Optional[AgentRegistry] = None


def get_registry() -> AgentRegistry:
    """Получить глобальный реестр агентов."""
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentRegistry()
    return _global_registry


def register_agent(agent: Agent, **kwargs) -> None:
    """Удобная функция для регистрации агента."""
    get_registry().register(agent, **kwargs)