"""Pattern Extractor - Main class for trading pattern extraction."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from rag.config import RAGConfig


class PatternExtractor:
    """Extract and analyze trading patterns from market data."""

    def __init__(self, coin: str, timeframe: str, config: Optional[RAGConfig] = None):
        """Initialize pattern extractor.

        Args:
            coin: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            timeframe: Timeframe for analysis (e.g., '15m', '1h')
            config: Optional configuration object
        """
        self.coin = coin.upper()
        self.timeframe = timeframe.lower()
        self.config = config or RAGConfig()

        # Validate inputs
        if not self.config.validate_coin(self.coin):
            raise ValueError(f"Unsupported coin: {self.coin}")
        if not self.config.validate_timeframe(self.timeframe):
            raise ValueError(f"Unsupported timeframe: {self.timeframe}")

        self.data: Optional[pd.DataFrame] = None
        self.patterns: List[Dict] = []

    def load_data(self, data_path: Optional[Path] = None) -> pd.DataFrame:
        """Load OHLCV data for analysis.

        Args:
            data_path: Optional custom path to data file

        Returns:
            DataFrame with OHLCV data
        """
        if data_path is None:
            data_path = self.config.DATA_DIR / f"{self.coin}_{self.timeframe}.csv"

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.data = pd.read_csv(data_path)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.data = self.data.sort_values('timestamp').reset_index(drop=True)

        return self.data

    def extract_patterns(self, pattern_type: str = 'bullish') -> List[Dict]:
        """Extract trading patterns from loaded data.

        Args:
            pattern_type: Type of pattern ('bullish', 'bearish', 'consolidation')

        Returns:
            List of extracted patterns
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        patterns = []
        min_len = self.config.MIN_PATTERN_LENGTH
        max_len = self.config.MAX_PATTERN_LENGTH

        # Simple pattern detection (placeholder for more advanced logic)
        for i in range(len(self.data) - max_len):
            window = self.data.iloc[i:i+max_len]

            if self._is_pattern(window, pattern_type):
                pattern = {
                    'type': pattern_type,
                    'start_idx': i,
                    'end_idx': i + max_len,
                    'start_time': str(window.iloc[0]['timestamp']),
                    'end_time': str(window.iloc[-1]['timestamp']),
                    'length': len(window),
                    'return': self._calculate_return(window)
                }
                patterns.append(pattern)

        self.patterns = patterns
        return patterns

    def _is_pattern(self, window: pd.DataFrame, pattern_type: str) -> bool:
        """Check if window contains specified pattern type.

        Args:
            window: Data window to analyze
            pattern_type: Type of pattern to detect

        Returns:
            True if pattern is detected
        """
        # Placeholder logic - implement actual pattern detection
        price_change = (window['close'].iloc[-1] - window['close'].iloc[0]) / window['close'].iloc[0]

        if pattern_type == 'bullish':
            return price_change > 0.02  # 2% increase
        elif pattern_type == 'bearish':
            return price_change < -0.02  # 2% decrease
        elif pattern_type == 'consolidation':
            return abs(price_change) < 0.01  # <1% change

        return False

    def _calculate_return(self, window: pd.DataFrame) -> float:
        """Calculate return for a pattern window."""
        return float((window['close'].iloc[-1] - window['close'].iloc[0]) / window['close'].iloc[0])

    def save_patterns(self, filename: Optional[str] = None) -> Path:
        """Save extracted patterns to JSON file.

        Args:
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        if not self.patterns:
            raise ValueError("No patterns to save. Run extract_patterns() first.")

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.coin}_{self.timeframe}_{timestamp}.json"

        output_path = self.config.PATTERNS_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                'coin': self.coin,
                'timeframe': self.timeframe,
                'patterns_count': len(self.patterns),
                'extraction_date': datetime.now().isoformat(),
                'patterns': self.patterns
            }, f, indent=2)

        return output_path

    def get_statistics(self) -> Dict:
        """Get statistics about extracted patterns."""
        if not self.patterns:
            return {}

        returns = [p['return'] for p in self.patterns]
        return {
            'total_patterns': len(self.patterns),
            'avg_return': np.mean(returns),
            'median_return': np.median(returns),
            'std_return': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns)
        }
