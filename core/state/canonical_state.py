"""CORE - CANONICAL STATE
Immutable data container for Strategy Inputs.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime

@dataclass
class CanonicalState:
    """
    Единый формат данных для всех стратегий (RB, ML, RL).
    Гарантирует, что данные прошли валидацию и санитаризацию.
    """
    # Основные данные
    features: pd.DataFrame          # 75 validated features
    raw_ohlcv: pd.DataFrame         # Исходные свечи (для визуализации/расчетов PnL)
    
    # Метаданные состояния
    timestamp: datetime             # Время последнего бара
    symbol: str                     # Тикер (BTC, ETH...)
    timeframe: str                  # 15m, 1h...
    version: str                    # Версия фич (из Registry)
    
    # Дополнительные контексты (заполняются пайплайном)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Флаги валидации
    is_valid: bool = False
    validation_errors: list = field(default_factory=list)

    def __post_init__(self):
        # Автоматическая проверка целостности при создании
        if len(self.features) != len(self.raw_ohlcv):
            self.validation_errors.append("Length mismatch: Features vs OHLCV")
        
        if self.features.empty:
            self.validation_errors.append("Empty features DataFrame")

    def get_latest_row(self) -> pd.Series:
        """Возвращает последнюю строку фич (для live inference)"""
        return self.features.iloc[-1]

    def get_feature_matrix(self) -> np.ndarray:
        """Возвращает numpy массив для XGBoost (без колонок, чисто данные)"""
        return self.features.values

    @property
    def shape(self):
        return self.features.shape
