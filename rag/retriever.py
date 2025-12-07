# rag/retriever.py
"""
Модуль для извлечения контекста из RAG-базы:
- patterns/ (торговые паттерны)
- trades/ (исторические сделки)
- lessons/ (уроки из ошибок)

Идея: построение контекста для Council и стратегий.
"""

import json
from typing import Dict, List, Any
from pathlib import Path


class RAGRetriever:
    """
    Простой извлекатель контекста из JSON-файлов.
    Можно расширить векторным поиском (например, Faiss).
    """

    def __init__(self, rag_root: str = "./rag"):
        self.rag_root = Path(rag_root)
        self.patterns_path = self.rag_root / "patterns" / "library.json"
        self.trades_path = self.rag_root / "trades" / "trade_log.json"
        self.lessons_path = self.rag_root / "lessons" / "lessons.json"

    def load_json(self, path: Path) -> List[Dict[str, Any]]:
        """Загрузить JSON-файл с обработкой ошибок"""
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return []

    def retrieve_patterns(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Извлекает паттерны из patterns/library.json.
        Пока без семантического поиска — возвращает все или первые top_k.
        """
        patterns = self.load_json(self.patterns_path)
        return patterns[:top_k]

    def retrieve_recent_trades(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Извлекает последние сделки из trades/trade_log.json.
        """
        trades = self.load_json(self.trades_path)
        return trades[-top_k:]

    def retrieve_lessons(self, category: str = "all") -> List[Dict[str, Any]]:
        """
        Извлекает уроки из lessons/lessons.json.
        Фильтрует по category если указано.
        """
        lessons = self.load_json(self.lessons_path)
        if category == "all":
            return lessons
        return [l for l in lessons if l.get("category") == category]

    def build_context(self, top_patterns: int = 3, top_trades: int = 5) -> str:
        """
        Строит текстовый контекст для LLM (советчик).
        """
        patterns = self.retrieve_patterns(top_k=top_patterns)
        trades = self.retrieve_recent_trades(top_k=top_trades)
        lessons = self.retrieve_lessons(category="all")

        context_lines = ["=== RAG CONTEXT ==="]

        if patterns:
            context_lines.append("\n[Patterns]:")
            for p in patterns:
                context_lines.append(f"- {p.get('name', 'N/A')}: {p.get('description', '')}")

        if trades:
            context_lines.append("\n[Recent Trades]:")
            for t in trades:
                context_lines.append(f"- Trade {t.get('id', 'N/A')}: {t.get('outcome', 'N/A')}")

        if lessons:
            context_lines.append("\n[Lessons]:")
            for lesson in lessons:
                context_lines.append(f"- {lesson.get('title', 'N/A')}: {lesson.get('lesson', '')}")

        return "\n".join(context_lines)
