"""
Модуль для извлечения контекста из RAG-базы:
- patterns/ (торговые паттерны)
- trades/ (исторические сделки)
- lessons/ (уроки из ошибок)

Идея: построение контекста для Council и стратегий.
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path


class RAGRetriever:
    """
    Простой извлекатель контекста из JSON-файлов.
    Можно расширить векторным поиском (например, Faiss).
    """

    def __init__(self, rag_root: str = "./rag", use_multi_hyde: bool = False):
        self.rag_root = Path(rag_root)
        self.patterns_path = self.rag_root / "patterns" / "library.json"
        self.trades_path = self.rag_root / "trades" / "trade_log.json"
        self.lessons_path = self.rag_root / "lessons" / "lessons.json"
        self.use_multi_hyde = use_multi_hyde

    def load_json(self, path: Path) -> List[Dict[str, Any]]:
        """Загрузить JSON-файл с обработкой ошибок"""
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Handle both direct array and {"patterns": []} format
                if isinstance(data, dict) and "patterns" in data:
                    return data["patterns"]
                return data if isinstance(data, list) else []
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return []

    def retrieve_patterns(
        self, 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Извлекает паттерны из patterns/library.json.
        
        Args:
            top_k: Количество паттернов
            filters: Фильтры (symbol, timeframe, direction)
        
        Returns:
            List of pattern dicts
        """
        patterns = self.load_json(self.patterns_path)
        
        # Apply filters if provided
        if filters:
            filtered = []
            for p in patterns:
                match = True
                if 'symbol' in filters and p.get('meta', {}).get('coin') != filters['symbol']:
                    match = False
                if 'timeframe' in filters and p.get('meta', {}).get('timeframe') != filters['timeframe']:
                    match = False
                if 'direction' in filters and p.get('meta', {}).get('direction') != filters['direction']:
                    match = False
                if match:
                    filtered.append(p)
            patterns = filtered
        
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
                context_lines.append(f"- {p.get('id', 'N/A')}: W_box={p.get('w_box', {}).get('W_box', 'N/A')}")

        if trades:
            context_lines.append("\n[Recent Trades]:")
            for t in trades:
                context_lines.append(f"- Trade {t.get('id', 'N/A')}: {t.get('outcome', 'N/A')}")

        if lessons:
            context_lines.append("\n[Lessons]:")
            for lesson in lessons:
                context_lines.append(f"- {lesson.get('title', 'N/A')}: {lesson.get('lesson', '')}")

        return "\n".join(context_lines)
    
    def build_council_context(
        self,
        current_state: Dict[str, Any],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Построить расширенный контекст для Council.
        
        Args:
            current_state: Текущее состояние рынка
                {
                    'symbol': 'BTC',
                    'timeframe': '4h',
                    'direction': 'long',
                    'features': {...}
                }
            top_k: Количество похожих паттернов
        
        Returns:
            Dict с расширенным контекстом:
            {
                'similar_patterns': [...],
                'historical_win_rate': float,
                'avg_pnl': float,
                'recommendation': str,
                'confidence': float,
                'lessons': [...]
            }
        """
        # Get similar patterns
        filters = {
            'symbol': current_state.get('symbol'),
            'timeframe': current_state.get('timeframe'),
            'direction': current_state.get('direction')
        }
        
        patterns = self.retrieve_patterns(top_k=top_k, filters=filters)
        
        # Calculate statistics
        if patterns:
            total_patterns = len(patterns)
            profitable = sum(1 for p in patterns 
                           if p.get('future_path', {}).get('max_profit_pct', 0) > 0)
            win_rate = profitable / total_patterns if total_patterns > 0 else 0
            
            avg_pnl = sum(p.get('future_path', {}).get('max_profit_pct', 0) 
                         for p in patterns) / total_patterns if total_patterns > 0 else 0
            
            avg_w_box = sum(p.get('w_box', {}).get('W_box', 0) 
                           for p in patterns) / total_patterns if total_patterns > 0 else 0
        else:
            win_rate = 0.5  # Default neutral
            avg_pnl = 0
            avg_w_box = 0
        
        # Get relevant lessons
        lessons = self.retrieve_lessons(category="all")[:3]
        
        # Generate recommendation
        if win_rate > 0.65 and avg_w_box > 0.6:
            recommendation = f"Strong {current_state.get('direction', 'long')} setup"
            confidence = min(0.9, (win_rate + avg_w_box) / 2)
        elif win_rate > 0.5 and avg_w_box > 0.4:
            recommendation = f"Moderate {current_state.get('direction', 'long')} setup"
            confidence = (win_rate + avg_w_box) / 2
        else:
            recommendation = "Weak setup, consider waiting"
            confidence = max(0.3, (win_rate + avg_w_box) / 2)
        
        return {
            'similar_patterns': patterns,
            'historical_win_rate': round(win_rate, 2),
            'avg_pnl': round(avg_pnl, 2),
            'avg_w_box': round(avg_w_box, 2),
            'recommendation': recommendation,
            'confidence': round(confidence, 2),
            'lessons': lessons,
            'patterns_count': len(patterns)
        }
    
    # Backward compatibility aliases
    def retrieve_similar_patterns(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Backward compatibility with old API"""
        return self.retrieve_patterns(top_k=limit)
    
    def retrieve_historical_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Backward compatibility with old API"""
        return self.retrieve_recent_trades(top_k=limit)
    
    def get_lessons(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Backward compatibility with old API"""
        lessons = self.retrieve_lessons(category="all")
        return lessons[:limit]
