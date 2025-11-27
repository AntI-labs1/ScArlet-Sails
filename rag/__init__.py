"""
Scarlet Sails RAG Pattern Extractor
===================================

Автоматическое извлечение индикаторов для паттернов.

Использование из командной строки:
    python -m rag.cli BTC 1h "2024-11-26 14:00"
    
Использование в коде:
    from rag.extractor import PatternExtractor
    
    extractor = PatternExtractor("BTC", "1h")
    data = extractor.extract("2024-11-26 14:00")
    extractor.save(data)
"""

from .extractor import PatternExtractor
from .config import COINS, TIMEFRAMES, PATTERNS_DIR

__all__ = ['PatternExtractor', 'COINS', 'TIMEFRAMES', 'PATTERNS_DIR']
__version__ = '1.0.0'