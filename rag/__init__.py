"""
ScArlet-Sails RAG Module v2.0

Pattern extraction, storage, and intelligent retrieval.

Components:
- extractor: Extract patterns from market data
- vector_store: FAISS-based semantic search
- multi_hyde: Multi-hypothesis retrieval
- retriever: Unified retrieval interface
- updater: Outcome tracking and statistics
"""

from .extractor import PatternExtractor
from .config import COINS, TIMEFRAMES, PATTERNS_DIR
from .vector_store import PatternVectorStore
from .multi_hyde import MultiHyDERetriever
from .retriever import RAGRetriever
from .updater import PatternUpdater

__all__ = [
    # Main classes
    'PatternExtractor',
    'PatternVectorStore',
    'MultiHyDERetriever',
    'RAGRetriever',
    'PatternUpdater',
    # Config
    'COINS',
    'TIMEFRAMES',
    'PATTERNS_DIR',
]

__version__ = '2.0.0'
