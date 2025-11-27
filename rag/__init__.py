"""RAG Module - Trading Pattern Extraction System.

This module provides tools for extracting and analyzing trading patterns
from cryptocurrency market data.
"""

from rag.extractor import PatternExtractor
from rag.config import RAGConfig

__version__ = "0.1.0"
__all__ = ["PatternExtractor", "RAGConfig"]
