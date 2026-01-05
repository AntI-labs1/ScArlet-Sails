"""CORE - FEATURE REGISTRY
The Single Source of Truth for Feature Definitions.
Version: v3.0 (75 Features)
"""
from typing import List, Dict

class FeatureRegistry:
    # Версия набора фич. При изменении логики расчета - инкрементировать.
    VERSION = "v3_75_prod"

    # Базовые колонки (5)
    OHLCV = ['open', 'high', 'low', 'close', 'volume']

    # ВНИМАНИЕ: Это заглушка. В реальной системе здесь должен быть полный список 
    # из 75 строк, соответствующий твоим Parquet файлам.
    # Я добавил механизм auto-discovery в Pipeline, но в идеале этот список 
    # должен быть жестко зафиксирован (hardcoded) после первого успешного запуска.
    
    # Пример категорий (для справки):
    # REGIME_FEATURES = ['regime_volatility', 'regime_trend', ...]
    # MOMENTUM_FEATURES = ['norm_rsi_14', 'norm_macd', ...]

    @classmethod
    def get_all_features(cls) -> List[str]:
        """
        Возвращает ожидаемый список фич.
        В production здесь должен быть return [...] с полным списком.
        """
        return [] 
        
    @classmethod
    def is_valid_count(cls, count: int) -> bool:
        """Проверка количества (74 фичи + target или 75 с timestamp)"""
        # Допускаем 74 (без таргета) или 75+ (с таргетом/метаданными)
        return count >= 74
