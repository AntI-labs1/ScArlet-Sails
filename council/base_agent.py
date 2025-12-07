# council/base_agent.py
"""
Базовый класс для агентов Council (Совет).

Идея: каждый агент (Quant, LLM, Human) наследует BaseAgent и предоставляет:
- decide(): основной метод принятия решения
- explain(): объяснение своей позиции
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseAgent(ABC):
    """
    Абстрактный агент, участвующий в Council.
    """

    def __init__(self, name: str):
        self.name = name
        self.last_decision: Optional[Dict[str, Any]] = None

    @abstractmethod
    def decide(self, state: Dict[str, Any], context: str = "") -> Dict[str, Any]:
        """
        Принимает решение на основе состояния и контекста.

        :param state: снимок состояния (например, из CanonicalState)
        :param context: RAG-контекст (паттерны, уроки)
        :return: {
            "action": "long" | "short" | "hold",
            "confidence": float,
            "reasoning": str,
        }
        """
        pass

    @abstractmethod
    def explain(self) -> str:
        """
        Объясняет последнее решение.
        """
        pass

    def update_decision(self, decision: Dict[str, Any]) -> None:
        """
        Сохраняет решение для дальнейшей рефлексии.
        """
        self.last_decision = decision


class QuantAgent(BaseAgent):
    """
    Квант-агент: агрегирует сигналы из множества стратегий.
    """

    def __init__(self, name: str = "QuantAgent", strategies=None):
        super().__init__(name)
        self.strategies = strategies or []  # список стратегий (rule-based, ML)

    def decide(self, state: Dict[str, Any], context: str = "") -> Dict[str, Any]:
        """
        Агрегирует сигналы всех стратегий.
        """
        # TODO: реализовать логику агрегации
        decision = {
            "action": "hold",
            "confidence": 0.5,
            "reasoning": "QuantAgent: пока заглушка",
        }
        self.update_decision(decision)
        return decision

    def explain(self) -> str:
        if not self.last_decision:
            return "No decision yet."
        return f"QuantAgent decision: {self.last_decision['action']} (confidence={self.last_decision['confidence']})"


class LLMAgent(BaseAgent):
    """
    LLM-агент: использует RAG-контекст и LLM для принятия решений.
    """

    def __init__(self, name: str = "LLMAgent", llm_model=None):
        super().__init__(name)
        self.llm_model = llm_model

    def decide(self, state: Dict[str, Any], context: str = "") -> Dict[str, Any]:
        # TODO: подключить LLM (например, GPT-4, Claude)
        decision = {
            "action": "hold",
            "confidence": 0.6,
            "reasoning": "LLMAgent: пока заглушка, RAG-контекст не использован",
        }
        self.update_decision(decision)
        return decision

    def explain(self) -> str:
        if not self.last_decision:
            return "No decision yet."
        return f"LLMAgent decision: {self.last_decision['action']} ({self.last_decision['reasoning']})"


class HumanAgent(BaseAgent):
    """
    Human-in-the-Loop агент: человек принимает окончательное решение.
    """

    def __init__(self, name: str = "HumanAgent"):
        super().__init__(name)

    def decide(self, state: Dict[str, Any], context: str = "") -> Dict[str, Any]:
        # TODO: реализовать интерфейс для ввода решения
        decision = {
            "action": "hold",
            "confidence": 1.0,
            "reasoning": "HumanAgent: пока заглушка (ждем UI)",
        }
        self.update_decision(decision)
        return decision

    def explain(self) -> str:
        if not self.last_decision:
            return "No decision yet."
        return f"HumanAgent decision: {self.last_decision['action']} ({self.last_decision['reasoning']})"
