#!/usr/bin/env python3
"""
ScArlet-Sails Agent Registration CLI

Регистрация агентов через командную строку.
Добавление новых агентов без изменения кода.

Usage:
    python scripts/register_agent.py --name MyAgent --module my_agent --class MyAgentClass
    python scripts/register_agent.py --list
    python scripts/register_agent.py --test MyAgent
    python scripts/register_agent.py --generate-config
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from council.protocols import Agent, HealthStatus
from council.agent_registry import AgentRegistry, get_registry
from council.config_schema import CouncilConfig, AgentConfigSchema, generate_default_config

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_config() -> Optional[CouncilConfig]:
    """Загрузить конфигурацию."""
    # Try configs/ first (project standard), then config/
    for config_dir in ["configs", "config"]:
        config_path = PROJECT_ROOT / config_dir / "council.yaml"
        if config_path.exists():
            return CouncilConfig.from_yaml(config_path)
        
        # Try JSON
        config_path = PROJECT_ROOT / config_dir / "council.json"
        if config_path.exists():
            return CouncilConfig.from_json(config_path)
    
    return None


def list_agents() -> None:
    """Показать список зарегистрированных агентов."""
    config = load_config()
    
    print()
    print("=" * 60)
    print("  REGISTERED AGENTS")
    print("=" * 60)
    
    if config is None or not config.agents:
        print("\n  No agents configured.")
        print("  Run: python scripts/register_agent.py --generate-config")
        print()
        return
    
    print()
    print(f"  {'Name':<20} {'Weight':<10} {'Priority':<10} {'Enabled':<10}")
    print("-" * 60)
    
    for agent in config.agents:
        enabled = "✓" if agent.enabled else "✗"
        print(f"  {agent.name:<20} {agent.weight:<10.2f} {agent.priority:<10} {enabled:<10}")
    
    print()
    print(f"  Total: {len(config.agents)} agents")
    
    # Проверка весов
    enabled_agents = [a for a in config.agents if a.enabled]
    if enabled_agents:
        total_weight = sum(a.weight for a in enabled_agents)
        if not (0.95 <= total_weight <= 1.05):
            print(f"\n  ⚠ Warning: Total weight = {total_weight:.2f} (should be ~1.0)")
    
    print()


def test_agent(agent_name: str) -> bool:
    """Протестировать агента."""
    config = load_config()
    
    if config is None:
        print("Error: No configuration found")
        return False
    
    agent_config = config.get_agent_config(agent_name)
    
    if agent_config is None:
        print(f"Error: Agent '{agent_name}' not found in config")
        return False
    
    print()
    print(f"Testing agent: {agent_name}")
    print("-" * 40)
    
    try:
        # Импортируем модуль
        import importlib
        module = importlib.import_module(agent_config.module)
        agent_class = getattr(module, agent_config.class_name)
        
        # Создаём экземпляр
        agent = agent_class(**agent_config.config)
        
        print(f"  ✓ Module loaded: {agent_config.module}")
        print(f"  ✓ Class found: {agent_config.class_name}")
        print(f"  ✓ Instance created")
        
        # Проверяем протокол
        has_analyze = hasattr(agent, 'analyze') and callable(agent.analyze)
        has_health = hasattr(agent, 'health_check') and callable(agent.health_check)
        has_metadata = hasattr(agent, 'get_metadata') and callable(agent.get_metadata)
        
        print(f"  {'✓' if has_analyze else '✗'} analyze() method")
        print(f"  {'✓' if has_health else '✗'} health_check() method")
        print(f"  {'✓' if has_metadata else '✗'} get_metadata() method")
        
        if not (has_analyze and has_health and has_metadata):
            print("\n  ✗ Agent does not implement full protocol")
            return False
        
        # Health check
        health = agent.health_check()
        status_icon = "✓" if health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED] else "✗"
        print(f"\n  Health check: {status_icon} {health.status.value}")
        print(f"  Message: {health.message}")
        
        # Метаданные
        meta = agent.get_metadata()
        print(f"\n  Metadata:")
        print(f"    Name: {meta.name}")
        print(f"    Version: {meta.version}")
        print(f"    Description: {meta.description}")
        
        print("\n  ✓ Agent test PASSED")
        return True
        
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False
    except AttributeError as e:
        print(f"  ✗ Class not found: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def register_agent(
    name: str,
    module: str,
    class_name: str,
    weight: float = 0.33,
    priority: int = 0,
    enabled: bool = True,
) -> bool:
    """Зарегистрировать нового агента в конфиге."""
    # Use configs/ (project standard)
    config_path = PROJECT_ROOT / "configs" / "council.yaml"
    
    # Загружаем или создаём конфиг
    if config_path.exists():
        config = CouncilConfig.from_yaml(config_path)
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = CouncilConfig()
    
    # Проверяем дубликат
    if config.get_agent_config(name) is not None:
        print(f"Error: Agent '{name}' already exists")
        return False
    
    # Добавляем агента
    new_agent = AgentConfigSchema(
        name=name,
        module=module,
        class_name=class_name,
        weight=weight,
        priority=priority,
        enabled=enabled,
    )
    
    config.agents.append(new_agent)
    
    # Сохраняем
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config.to_yaml(config_path)
    
    print(f"✓ Registered agent: {name}")
    print(f"  Module: {module}")
    print(f"  Class: {class_name}")
    print(f"  Weight: {weight}")
    print(f"  Config saved: {config_path}")
    
    return True


def unregister_agent(name: str) -> bool:
    """Удалить агента из конфига."""
    config_path = PROJECT_ROOT / "configs" / "council.yaml"
    
    if not config_path.exists():
        print("Error: No configuration found")
        return False
    
    config = CouncilConfig.from_yaml(config_path)
    
    # Ищем агента
    agent_idx = None
    for i, agent in enumerate(config.agents):
        if agent.name == name:
            agent_idx = i
            break
    
    if agent_idx is None:
        print(f"Error: Agent '{name}' not found")
        return False
    
    # Удаляем
    del config.agents[agent_idx]
    config.to_yaml(config_path)
    
    print(f"✓ Unregistered agent: {name}")
    return True


def generate_config() -> None:
    """Сгенерировать дефолтный конфиг."""
    config_path = PROJECT_ROOT / "configs" / "council.yaml"
    
    if config_path.exists():
        print(f"Warning: Config already exists: {config_path}")
        response = input("Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print("Cancelled")
            return
    
    generate_default_config(config_path)
    print(f"\n✓ Generated default config: {config_path}")
    print("\nEdit this file to configure your agents.")


def health_check_all() -> None:
    """Проверить здоровье всех агентов."""
    config = load_config()
    
    if config is None or not config.agents:
        print("Error: No agents configured")
        return
    
    print()
    print("=" * 60)
    print("  AGENT HEALTH CHECK")
    print("=" * 60)
    print()
    
    for agent_cfg in config.agents:
        if not agent_cfg.enabled:
            print(f"  {agent_cfg.name:<20} DISABLED")
            continue
        
        try:
            import importlib
            module = importlib.import_module(agent_cfg.module)
            agent_class = getattr(module, agent_cfg.class_name)
            agent = agent_class(**agent_cfg.config)
            
            health = agent.health_check()
            
            if health.status == HealthStatus.HEALTHY:
                icon = "✓"
            elif health.status == HealthStatus.DEGRADED:
                icon = "⚠"
            else:
                icon = "✗"
            
            print(f"  {icon} {agent_cfg.name:<20} {health.status.value:<12} {health.message}")
            
        except Exception as e:
            print(f"  ✗ {agent_cfg.name:<20} ERROR        {str(e)[:30]}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="ScArlet-Sails Agent Registration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List all agents
    python scripts/register_agent.py --list
    
    # Register new agent
    python scripts/register_agent.py --name MyAgent --module agents.my_agent --class MyAgentClass
    
    # Test agent
    python scripts/register_agent.py --test QuantAgent
    
    # Generate default config
    python scripts/register_agent.py --generate-config
    
    # Health check all agents
    python scripts/register_agent.py --health
        """
    )
    
    parser.add_argument('--list', action='store_true', help='List registered agents')
    parser.add_argument('--test', type=str, metavar='NAME', help='Test agent by name')
    parser.add_argument('--health', action='store_true', help='Health check all agents')
    parser.add_argument('--generate-config', action='store_true', help='Generate default config')
    
    # Registration options
    parser.add_argument('--name', type=str, help='Agent name')
    parser.add_argument('--module', type=str, help='Python module path')
    parser.add_argument('--class', type=str, dest='class_name', help='Class name')
    parser.add_argument('--weight', type=float, default=0.33, help='Agent weight (0-1)')
    parser.add_argument('--priority', type=int, default=0, help='Agent priority')
    parser.add_argument('--disabled', action='store_true', help='Register as disabled')
    
    parser.add_argument('--unregister', type=str, metavar='NAME', help='Unregister agent')
    
    args = parser.parse_args()
    
    # Команды
    if args.list:
        list_agents()
        return 0
    
    if args.test:
        success = test_agent(args.test)
        return 0 if success else 1
    
    if args.health:
        health_check_all()
        return 0
    
    if args.generate_config:
        generate_config()
        return 0
    
    if args.unregister:
        success = unregister_agent(args.unregister)
        return 0 if success else 1
    
    # Регистрация нового агента
    if args.name and args.module and args.class_name:
        success = register_agent(
            name=args.name,
            module=args.module,
            class_name=args.class_name,
            weight=args.weight,
            priority=args.priority,
            enabled=not args.disabled,
        )
        return 0 if success else 1
    
    # Если ничего не указано
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
