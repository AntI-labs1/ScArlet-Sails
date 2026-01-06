#!/usr/bin/env python3
"""
🛡️ CANONICAL DATA PIPELINE v3.0

Философия: "Железный Купол" - ТОЛЬКО проверенные данные проходят дальше.

Архитектура:
    Raw Data → Validation Layer → Canonical Format → System
                     ↓
                 REJECT ❌

Важно:
- Один источник истины (Single Source of Truth)
- Fail-fast principle
- Структурная валидация
- Временная согласованность
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

# Добавляем корневую директорию в путь
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


class DataStatus(Enum):
    """Status of data validation"""
    VALID = "valid"
    INVALID = "invalid"
    QUARANTINE = "quarantine"
    REJECTED = "rejected"


@dataclass
class CanonicalData:
    """Canonical format for all market data"""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    source: str
    validated: bool
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class ValidationError(Exception):
    """Custom exception for validation failures"""
    pass


class CanonicalPipeline:
    """
    🛡️ Canonical Data Pipeline
    
    Отвечает за:
    1. Валидацию входящих данных
    2. Нормализацию формата
    3. Отброс невалидных данных
    4. Логирование всех операций
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = self._setup_logger()
        self.config = self._load_config(config_path)
        self.stats = {
            "processed": 0,
            "valid": 0,
            "invalid": 0,
            "rejected": 0
        }
        
    def _setup_logger(self) -> logging.Logger:
        """Setup structured logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load validation rules"""
        default_config = {
            "min_price": 0.0001,
            "max_price": 1000000,
            "min_volume": 0,
            "required_fields": ["timestamp", "symbol", "price", "volume"],
            "allowed_sources": ["binance", "coinbase", "kraken", "internal"],
            "max_age_seconds": 300  # 5 minutes
        }
        
        if config_path and config_path.exists():
            with open(config_path) as f:
                return {**default_config, **json.load(f)}
        
        return default_config
    
    def validate_structure(self, data: Dict) -> bool:
        """
        ✅ Структурная валидация
        """
        required = self.config["required_fields"]
        
        for field in required:
            if field not in data:
                raise ValidationError(f"Missing required field: {field}")
        
        return True
    
    def validate_types(self, data: Dict) -> bool:
        """
        ✅ Типовая валидация
        """
        if not isinstance(data.get("price"), (int, float)):
            raise ValidationError(f"Invalid price type: {type(data.get('price'))}")
        
        if not isinstance(data.get("volume"), (int, float)):
            raise ValidationError(f"Invalid volume type: {type(data.get('volume'))}")
        
        if not isinstance(data.get("symbol"), str):
            raise ValidationError(f"Invalid symbol type: {type(data.get('symbol'))}")
        
        return True
    
    def validate_ranges(self, data: Dict) -> bool:
        """
        ✅ Валидация диапазонов
        """
        price = data.get("price")
        if not (self.config["min_price"] <= price <= self.config["max_price"]):
            raise ValidationError(f"Price {price} out of range")
        
        volume = data.get("volume")
        if volume < self.config["min_volume"]:
            raise ValidationError(f"Volume {volume} below minimum")
        
        return True
    
    def validate_timestamp(self, data: Dict) -> bool:
        """
        ✅ Временная валидация
        """
        ts = data.get("timestamp")
        
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        elif isinstance(ts, (int, float)):
            ts = datetime.fromtimestamp(ts)
        elif not isinstance(ts, datetime):
            raise ValidationError(f"Invalid timestamp type: {type(ts)}")
        
        age = (datetime.now() - ts).total_seconds()
        if abs(age) > self.config["max_age_seconds"]:
            raise ValidationError(f"Data too old/future: {age}s")
        
        data["timestamp"] = ts
        return True
    
    def validate(self, raw_data: Dict) -> CanonicalData:
        """
        🛡️ Полная валидация
        
        Fail-fast: при первой ошибке бросаем ValidationError
        """
        try:
            self.stats["processed"] += 1
            
            # Проверяем все уровни
            self.validate_structure(raw_data)
            self.validate_types(raw_data)
            self.validate_timestamp(raw_data)
            self.validate_ranges(raw_data)
            
            # Создаём канонический объект
            canonical = CanonicalData(
                timestamp=raw_data["timestamp"],
                symbol=raw_data["symbol"],
                price=float(raw_data["price"]),
                volume=float(raw_data["volume"]),
                source=raw_data.get("source", "unknown"),
                validated=True,
                metadata={
                    "original_data": raw_data,
                    "validation_time": datetime.now().isoformat()
                }
            )
            
            self.stats["valid"] += 1
            self.logger.info(f"✅ Valid: {canonical.symbol} @ {canonical.price}")
            
            return canonical
            
        except ValidationError as e:
            self.stats["invalid"] += 1
            self.logger.error(f"❌ Validation failed: {e}")
            self.logger.debug(f"Raw data: {raw_data}")
            raise
    
    def process_batch(self, data_batch: List[Dict]) -> List[CanonicalData]:
        """
        🔄 Пакетная обработка
        """
        valid_data = []
        
        for raw in data_batch:
            try:
                canonical = self.validate(raw)
                valid_data.append(canonical)
            except ValidationError:
                self.stats["rejected"] += 1
                continue
        
        self.logger.info(
            f"📊 Batch stats: {len(valid_data)}/{len(data_batch)} valid"
        )
        
        return valid_data
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        return {
            **self.stats,
            "success_rate": (
                self.stats["valid"] / self.stats["processed"] * 100
                if self.stats["processed"] > 0 else 0
            )
        }


def main():
    """
    Example usage
    """
    pipeline = CanonicalPipeline()
    
    # Пример данных
    test_data = [
        {
            "timestamp": datetime.now().isoformat(),
            "symbol": "BTC/USDT",
            "price": 45000.0,
            "volume": 1.5,
            "source": "binance"
        },
        {
            "timestamp": datetime.now().isoformat(),
            "symbol": "ETH/USDT",
            "price": 3000.0,
            "volume": 10.0,
            "source": "coinbase"
        },
        {
            # Невалидные данные - нет price
            "timestamp": datetime.now().isoformat(),
            "symbol": "SOL/USDT",
            "volume": 5.0,
        }
    ]
    
    print("🚀 Starting Canonical Pipeline...\n")
    
    valid_data = pipeline.process_batch(test_data)
    
    print(f"\n📊 Statistics:")
    stats = pipeline.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n✅ Valid data:")
    for data in valid_data:
        print(f"  {data.symbol}: ${data.price} (vol: {data.volume})")


if __name__ == "__main__":
    main()
