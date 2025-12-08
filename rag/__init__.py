"""
ScArlet-Sails RAG Module

Pattern extraction, storage, and intelligent retrieval.

Components:
- extractor: Extract patterns from market data (Time Capsule v2.0)
- retriever: Pattern retrieval interface
- config: Configuration for coins, timeframes, features
"""

from .extractor import PatternExtractor
from .config import COINS, TIMEFRAMES, PATTERNS_DIR, KEY_FEATURES
from .retriever import RAGRetriever

__all__ = [
    # Main classes
    'PatternExtractor',
    'RAGRetriever',
    # Config
    'COINS',
    'TIMEFRAMES',
    'PATTERNS_DIR',
    'KEY_FEATURES',
]

__version__ = '1.0.0'
