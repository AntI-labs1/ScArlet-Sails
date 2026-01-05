"""CORE - CANONICAL PIPELINE
Orchestrator for Data Loading, Validation, and Sanitization.
Replaces: feature_engine.py, feature_engine_v2.py
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional, List

from .feature_registry import FeatureRegistry
from .canonical_state import CanonicalState
from .sanitize_features import sanitize_for_model, validate_features
from .feature_loader import FeatureLoader

logger = logging.getLogger(__name__)

class CanonicalPipeline:
    """
    Конвейер, превращающий сырые данные/файлы в CanonicalState.
    """
    def __init__(self, data_dir: str = "data/features", strict_mode: bool = True):
        self.loader = FeatureLoader(data_dir=data_dir)
        self.strict_mode = strict_mode
        self.registry = FeatureRegistry()
        
    def load_state(self, coin: str, timeframe: str) -> CanonicalState:
        """
        Главная точка входа. 
        Загружает Parquet -> Валидирует -> Санитизирует -> Создает State.
        """
        logger.info(f"Pipeline: Loading {coin} {timeframe}...")

        # 1. Loading (using existing FeatureLoader)
        # Грузим все колонки
        df_raw = self.loader.load_features(coin, timeframe, validate=False)
        
        # Разделяем OHLCV и ML Features
        ohlcv_cols = self.registry.OHLCV
        feature_cols = [c for c in df_raw.columns if c not in ohlcv_cols and c != 'timestamp']
        
        # 2. Validation Check (Structure)
        if len(feature_cols) < 70: # Допускаем небольшой люфт, но 31 vs 75 отловим
            msg = f"CRITICAL: Feature count mismatch! Expected ~75, got {len(feature_cols)}."
            logger.error(msg)
            if self.strict_mode:
                raise ValueError(msg)

        # 3. Sanitization (Cleaning)
        # Используем твой мощный модуль sanitize_features
        logger.info(f"Pipeline: Sanitizing {len(feature_cols)} features...")
        df_clean = sanitize_for_model(
            df_raw, 
            feature_names=feature_cols,
            drop_inf=True,
            drop_nan=True, # Важно для XGBoost
            clip_extreme=True,
            verbose=True
        )

        # 4. Packaging
        # Получаем features only DataFrame
        features_df = df_clean[feature_cols].copy()
        raw_ohlcv = df_clean[ohlcv_cols].copy()
        
        state = CanonicalState(
            features=features_df,
            raw_ohlcv=raw_ohlcv,
            timestamp=df_clean.index[-1],
            symbol=coin,
            timeframe=timeframe,
            version=self.registry.VERSION,
            is_valid=True
        )
        
        logger.info(f"✅ State Built: {state.shape} | Version: {state.version}")
        return state

    def transform_live(self, raw_ohlcv: pd.DataFrame) -> CanonicalState:
        """
        Для LIVE режима (позже).
        Здесь должен быть вызов FeatureCalculator, который генерирует те же 75 фич.
        Пока ставим заглушку (NotImplemented), чтобы не использовать старые engine.
        """
        raise NotImplementedError(
            "Live feature calculation not yet implemented for v3 features. "
            "Use load_state() with pre-computed parquet for now."
        )
