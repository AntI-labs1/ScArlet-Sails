# core/canonical_state.py
"""
Каноническое состояние системы — единое хранилище данных, которое:
1. Получает raw-данные из data_loader
2. Вычисляет фичи через feature_engine_v2
3. Предоставляет модулям (strat, risk, council) единый интерфейс

Идея: все данные живут в одном месте, фичи пересчитываются 1 раз.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime


class CanonicalState:
    """
    Центральное состояние торговой системы.
    Объединяет сырые данные, фичи и метаданные.
    """

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.raw_data: Optional[pd.DataFrame] = None  # OHLCV
        self.features: Optional[pd.DataFrame] = None  # Признаки
        self.last_update: Optional[datetime] = None
        self.metadata: Dict[str, Any] = {}

    def update_raw(self, df: pd.DataFrame) -> None:
        """
        Загружает сырые данные (OHLCV).
        Ожидается, что df имеет столбцы: open, high, low, close, volume.
        """
        self.raw_data = df.copy()
        self.last_update = datetime.now()

    def compute_features(self, feature_engine) -> None:
        """
        Вычисляет фичи через переданный feature_engine.
        Результат сохраняется в self.features.
        """
        if self.raw_data is None or self.raw_data.empty:
            raise ValueError("CanonicalState: нет сырых данных для вычисления фичей")

        self.features = feature_engine.compute(self.raw_data)
        self.metadata["features_computed_at"] = datetime.now()

    def get_snapshot(self) -> Dict[str, Any]:
        """
        Возвращает полный снимок состояния:
        - raw_data
        - features
        - metadata
        """
        return {
            "symbol": self.symbol,
            "raw_data": self.raw_data,
            "features": self.features,
            "last_update": self.last_update,
            "metadata": self.metadata,
        }

    def get_latest_row(self) -> Optional[pd.Series]:
        """
        Возвращает последний (актуальный) ряд фичей.
        Используется стратегиями для принятия решений.
        """
        if self.features is None or self.features.empty:
            return None
        return self.features.iloc[-1]

    def set_metadata(self, key: str, value: Any) -> None:
        """Добавить метаданные"""
        self.metadata[key] = value

    def get_metadata(self, key: str, default=None) -> Any:
        """Получить метаданные"""
        return self.metadata.get(key, default)
