"""Configuration for RAG pattern extraction."""

from pathlib import Path
from typing import List


class RAGConfig:
    """Configuration for pattern extraction."""

    # Supported coins
    COINS: List[str] = [
        "ALGO", "AVAX", "BTC", "DOT", "ENA", "ETH", "HBAR",
        "LDO", "LINK", "LTC", "ONDO", "SOL", "SUI", "UNI"
    ]

    # Supported timeframes
    TIMEFRAMES: List[str] = ["15m", "1h", "4h", "1d"]

    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data" / "raw"
    PATTERNS_DIR = BASE_DIR / "rag" / "patterns"
    FEATURES_DIR = BASE_DIR / "data" / "features"

    # Pattern extraction settings
    MIN_PATTERN_LENGTH: int = 10  # Minimum candles for pattern
    MAX_PATTERN_LENGTH: int = 100  # Maximum candles for pattern
    SIMILARITY_THRESHOLD: float = 0.85  # Pattern similarity threshold

    # Feature settings
    FEATURE_WINDOW: int = 20  # Rolling window for features

    @classmethod
    def validate_coin(cls, coin: str) -> bool:
        """Validate if coin is supported."""
        return coin.upper() in cls.COINS

    @classmethod
    def validate_timeframe(cls, timeframe: str) -> bool:
        """Validate if timeframe is supported."""
        return timeframe.lower() in cls.TIMEFRAMES
