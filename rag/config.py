"""
RAG Configuration
=================

Пути, монеты, таймфреймы и features.
"""

import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

# Базовая директория проекта
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "features"
PATTERNS_DIR = BASE_DIR / "rag" / "patterns"

# Создаём папку для паттернов
PATTERNS_DIR.mkdir(parents=True, exist_ok=True)

# 14 монет проекта
COINS = [
    "ALGO", "AVAX", "BTC", "DOT", "ENA", "ETH", "HBAR",
    "LDO", "LINK", "LTC", "ONDO", "SOL", "SUI", "UNI"
]

# 4 таймфрейма
TIMEFRAMES = ["15m", "1h", "4h", "1d"]

# Маппинг таймфреймов в минуты (для поиска ближайшего бара)
TF_MINUTES = {
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440
}

# Features которые извлекаем для таблицы Егора 1
# (Ключевые индикаторы согласно ТЗ паттерна Box Range)
KEY_FEATURES = {
    # OHLCV
    "price": ["open", "high", "low", "close", "volume"],
    
    # Normalized (z-scores)
    "normalized": [
        "norm_rsi_zscore",
        "norm_macd_zscore", 
        "norm_atr_zscore",
        "norm_bb_width_zscore",
        "norm_volume_zscore"
    ],
    
    # Regime flags
    "regime": [
        "regime_rsi_low",      # RSI < 30
        "regime_rsi_mid",      # 30 <= RSI <= 70
        "regime_rsi_high",     # RSI > 70
        "regime_trend_up",     # Тренд вверх
        "regime_trend_down",   # Тренд вниз
        "regime_vol_low",      # Низкая волатильность
        "regime_vol_high"      # Высокая волатильность
    ],
    
    # Divergences
    "divergence": [
        "div_rsi_bullish",
        "div_rsi_bearish"
    ],
    
    # Time
    "time": [
        "time_hour",
        "time_asian",
        "time_european", 
        "time_american"
    ]
}


def get_file_path(coin: str, tf: str) -> Path:
    """
    Получить путь к parquet файлу.
    
    Parameters
    ----------
    coin : str
        Тикер монеты (BTC, ETH, ...)
    tf : str
        Таймфрейм (15m, 1h, 4h, 1d)
        
    Returns
    -------
    Path
        Полный путь к файлу
        
    Raises
    ------
    FileNotFoundError
        Если файл не существует
    """
    coin = coin.upper()
    tf = tf.lower()
    
    if coin not in COINS:
        raise ValueError(f"Монета {coin} не поддерживается. Доступные: {COINS}")
    
    if tf not in TIMEFRAMES:
        raise ValueError(f"Таймфрейм {tf} не поддерживается. Доступные: {TIMEFRAMES}")
    
    filename = f"{coin}_USDT_{tf}_features.parquet"
    path = DATA_DIR / filename
    
    if not path.exists():
        raise FileNotFoundError(
            f"Файл не найден: {path}\n"
            f"Убедись что данные скачаны: git pull"
        )
    
    return path


@dataclass
class TimeCapsuleSnapshot:
    """Unified snapshot format for RAG Time Capsule."""

    timestamp: str
    symbol: str
    timeframe: str
    market_state_window: Dict[str, Any]
    P_rb: Optional[float] = None
    P_ml: Optional[float] = None
    P_hyb: Optional[float] = None
    regime: str = "unknown"
    pattern_type: str = "unspecified"
    human_label: Optional[str] = None
    human_confidence: Optional[float] = None
    reviewed_by: Optional[str] = None
    trade_pnl: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)