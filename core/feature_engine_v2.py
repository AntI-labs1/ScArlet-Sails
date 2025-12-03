"""
CORE - FEATURE ENGINE v2
Multi-timeframe feature engineering matching xgboost_normalized_model.json
31 features across 4 timeframes: 15m, 1h, 4h, 1d
"""
import pandas as pd
import numpy as np
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)

class FeatureEngine:
    """
    Создаёт multi-timeframe признаки для торговой стратегии.
    
    Features (31 total):
    - 15m: 13 features (base timeframe)
    - 1h:  6 features (higher TF context)
    - 4h:  6 features (trend context)
    - 1d:  6 features (macro context)
    """
    
    def __init__(self, config):
        self.config = config
        self.normalize = config.get('features', {}).get('normalize', True)
        self.scaler = None
        
        # Feature list (matches model exactly)
        self.feature_names = [
            # 15m features (13)
            "15m_RSI_14",
            "15m_price_to_EMA9",
            "15m_price_to_EMA21",
            "15m_price_to_SMA50",
            "15m_BB_width_pct",
            "15m_ATR_pct",
            "15m_returns_5",
            "15m_returns_10",
            "15m_volume_ratio_5",
            "15m_volume_ratio_10",
            "15m_price_to_EMA9_dup",  # Kept for compatibility
            "15m_price_to_EMA21_dup",
            "15m_price_to_SMA50_dup",
            # 1h features (6)
            "1h_RSI_14",
            "1h_returns_5",
            "1h_price_to_EMA9",
            "1h_price_to_EMA21",
            "1h_price_to_SMA50",
            "1h_ATR_pct",
            # 4h features (6)
            "4h_RSI_14",
            "4h_returns_5",
            "4h_price_to_EMA9",
            "4h_price_to_EMA21",
            "4h_price_to_SMA50",
            "4h_ATR_pct",
            # 1d features (6)
            "1d_RSI_14",
            "1d_returns_5",
            "1d_price_to_EMA9",
            "1d_price_to_EMA21",
            "1d_price_to_SMA50",
            "1d_ATR_pct"
        ]
        
        logger.info(f"FeatureEngine v2 инициализирован. Features: {len(self.feature_names)}, Normalize: {self.normalize}")
    
    def calculate_features(self, df_15m, df_1h=None, df_4h=None, df_1d=None):
        """
        Рассчитывает multi-timeframe признаки.
        
        Args:
            df_15m: DataFrame с OHLCV для 15m (required)
            df_1h: DataFrame с OHLCV для 1h (optional)
            df_4h: DataFrame с OHLCV для 4h (optional)
            df_1d: DataFrame с OHLCV для 1d (optional)
        
        Returns:
            DataFrame с 31 признаком
        """
        
        # Base features from 15m
        features_15m = self._calculate_timeframe_features(df_15m, '15m', full=True)
        
        # Higher timeframe features (if available)
        if df_1h is not None:
            features_1h = self._calculate_timeframe_features(df_1h, '1h', full=False)
            # Resample to 15m frequency
            features_1h = self._resample_to_15m(features_1h, df_15m.index)
        else:
            # Fill with neutral values if not available
            features_1h = pd.DataFrame(index=df_15m.index)
            for col in [f for f in self.feature_names if f.startswith('1h_')]:
                features_1h[col] = 0.0
        
        if df_4h is not None:
            features_4h = self._calculate_timeframe_features(df_4h, '4h', full=False)
            features_4h = self._resample_to_15m(features_4h, df_15m.index)
        else:
            features_4h = pd.DataFrame(index=df_15m.index)
            for col in [f for f in self.feature_names if f.startswith('4h_')]:
                features_4h[col] = 0.0
        
        if df_1d is not None:
            features_1d = self._calculate_timeframe_features(df_1d, '1d', full=False)
            features_1d = self._resample_to_15m(features_1d, df_15m.index)
        else:
            features_1d = pd.DataFrame(index=df_15m.index)
            for col in [f for f in self.feature_names if f.startswith('1d_')]:
                features_1d[col] = 0.0
        
        # Combine all features
        features = pd.concat([features_15m, features_1h, features_4h, features_1d], axis=1)
        
        # Reorder to match model
        features = features[self.feature_names]
        
        # Fill NaN
        features = features.fillna(method='ffill').fillna(0)
        
        # Normalize if enabled
        if self.normalize:
            features = self._normalize_features(features)
        
        logger.info(f"Признаков создано: {len(features.columns)}")
        
        return features
    
    def _calculate_timeframe_features(self, df, tf_prefix, full=False):
        """
        Рассчитывает признаки для одного таймфрейма.
        
        Args:
            df: OHLCV DataFrame
            tf_prefix: Prefix для колонок ('15m', '1h', etc)
            full: Если True, создаёт все 13 признаков для 15m. Если False, только 6.
        """
        df = df.copy()
        features = pd.DataFrame(index=df.index)
        
        # RSI
        features[f'{tf_prefix}_RSI_14'] = self._calculate_rsi(df['close'], 14)
        
        # Returns
        features[f'{tf_prefix}_returns_5'] = df['close'].pct_change(5)
        
        # Price to EMAs/SMA
        ema9 = df['close'].ewm(span=9, adjust=False).mean()
        ema21 = df['close'].ewm(span=21, adjust=False).mean()
        sma50 = df['close'].rolling(50).mean()
        
        features[f'{tf_prefix}_price_to_EMA9'] = (df['close'] - ema9) / ema9
        features[f'{tf_prefix}_price_to_EMA21'] = (df['close'] - ema21) / ema21
        features[f'{tf_prefix}_price_to_SMA50'] = (df['close'] - sma50) / sma50
        
        # ATR
        features[f'{tf_prefix}_ATR_pct'] = self._calculate_atr(df) / df['close']
        
        if full:  # 15m gets extra features
            # Returns 10
            features[f'{tf_prefix}_returns_10'] = df['close'].pct_change(10)
            
            # Bollinger Bands width
            sma20 = df['close'].rolling(20).mean()
            std20 = df['close'].rolling(20).std()
            bb_width = (std20 * 2) / sma20
            features[f'{tf_prefix}_BB_width_pct'] = bb_width
            
            # Volume ratios
            vol_ma5 = df['volume'].rolling(5).mean()
            vol_ma10 = df['volume'].rolling(10).mean()
            features[f'{tf_prefix}_volume_ratio_5'] = df['volume'] / vol_ma5
            features[f'{tf_prefix}_volume_ratio_10'] = df['volume'] / vol_ma10
            
            # Duplicates (for compatibility)
            features[f'{tf_prefix}_price_to_EMA9_dup'] = features[f'{tf_prefix}_price_to_EMA9']
            features[f'{tf_prefix}_price_to_EMA21_dup'] = features[f'{tf_prefix}_price_to_EMA21']
            features[f'{tf_prefix}_price_to_SMA50_dup'] = features[f'{tf_prefix}_price_to_SMA50']
        
        # Shift all engineered features by 1 to ensure the model only sees fully
        # closed bars (protects against look-ahead). NaNs are handled upstream
        # via forward-fill/zero-fill during assembly.
        return features.shift(1)
    
    def _calculate_rsi(self, series, period=14):
        """Calculate RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df, period=14):
        """Calculate ATR"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        atr = true_range.rolling(period).mean()
        return atr
    
    def _resample_to_15m(self, df, target_index):
        """Resample higher timeframe to 15m frequency"""
        df_resampled = df.reindex(target_index, method='ffill')
        return df_resampled
    
    def _normalize_features(self, features):
        """Normalize features using StandardScaler"""
        if self.scaler is None:
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features)
            logger.info("Признаки нормализованы (новый scaler)")
        else:
            features_scaled = self.scaler.transform(features)
            logger.info("Признаки нормализованы (существующий scaler)")
        
        return pd.DataFrame(
            features_scaled,
            index=features.index,
            columns=features.columns
        )
    
    def save_scaler(self, path):
        """Save scaler to file"""
        if self.scaler is not None:
            joblib.dump(self.scaler, path)
            logger.info(f"  💾 Scaler saved: {path}")
    
    def load_scaler(self, path):
        """Load scaler from file"""
        self.scaler = joblib.load(path)
        logger.info(f"  📂 Scaler loaded: {path}")


class CanonicalMarketStateBuilder:
    """Constructs canonical market states S_t from feature-enriched dataframes."""

    def __init__(self, df: pd.DataFrame, regime_column: str = "regime"):
        self.df = df.sort_index()
        self.regime_column = regime_column

    def _window_slice(self, idx: int, lookback: int = 100) -> pd.DataFrame:
        start = max(0, idx - lookback + 1)
        return self.df.iloc[start: idx + 1]

    def build_for_index(self, idx: int) -> Dict:
        if idx < 0 or idx >= len(self.df):
            raise IndexError("Index out of bounds for market state builder")

        window = self._window_slice(idx)
        prices = window["close"]
        returns = prices.pct_change().dropna()

        atr_series = window.get("atr")
        atr_value = (
            float(atr_series.iloc[-1])
            if atr_series is not None and not atr_series.isna().iloc[-1]
            else float(prices.pct_change().abs().rolling(14).mean().iloc[-1]) if len(prices) > 14 else 0.0
        )

        volume = float(window["volume"].iloc[-1])
        volume_ma = float(window["volume"].rolling(14).mean().iloc[-1]) if len(window) >= 14 else volume

        regime = window[self.regime_column].iloc[-1] if self.regime_column in window.columns else "normal"
        spread_series = window.get("spread", pd.Series([0.0], index=window.index))
        spread_value = float(spread_series.iloc[-1])

        base_cols = [col for col in ["open", "high", "low", "close", "volume"] if col in window.columns]
        state_features = window.drop(columns=base_cols).iloc[-1].fillna(0).values

        return {
            "returns": returns.values if len(returns) > 0 else np.array([]),
            "return_shock": float(returns.iloc[-1]) if len(returns) > 0 else 0.0,
            "atr": atr_value,
            "spread": spread_value,
            "volume": volume,
            "volume_ma": volume_ma,
            "regime": regime,
            "portfolio_value": 1.0,
            "drawdown": 0.0,
            "state_features": state_features,
            "window_slice": window.copy(),
        }


@dataclass
class FeatureSpecV3:
    """Canonical feature specification for Model 2 (74 single-TF features).

    This class centralises the ordered list of feature names used at both
    training and inference, addressing the historical "feature order" bug.
    """

    feature_names: List[str]

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        target_column: str = "target",
        drop_columns: Sequence[str] = ("raw_ret", "fee_ret", "rapnl"),
    ) -> "FeatureSpecV3":
        """Build a spec from a feature dataframe (drops target/label helpers)."""

        excluded = set([target_column, *drop_columns])
        names = [c for c in df.columns if c not in excluded]
        return cls(feature_names=names)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def enforce(self, df: pd.DataFrame, raise_on_missing: bool = True) -> pd.DataFrame:
        """Reorder dataframe columns to match the spec, validating presence."""

        missing = [c for c in self.feature_names if c not in df.columns]
        if missing and raise_on_missing:
            raise ValueError(f"Missing required features: {missing}")

        extra = [c for c in df.columns if c not in self.feature_names]
        if extra:
            df = df.drop(columns=extra)

        return df[self.feature_names]

    def validate(self, columns: Sequence[str]) -> Dict[str, List[str]]:
        """Return diagnostic information about missing/extra columns."""

        missing = [c for c in self.feature_names if c not in columns]
        extra = [c for c in columns if c not in self.feature_names]
        reordered = [c for c in columns if c in self.feature_names and self.feature_names.index(c) != list(columns).index(c)]

        return {"missing": missing, "extra": extra, "reordered": reordered}

    def to_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"feature_names": self.feature_names}, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "FeatureSpecV3":
        data = json.loads(Path(path).read_text())
        return cls(feature_names=data["feature_names"])