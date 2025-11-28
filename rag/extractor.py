"""
Pattern Extractor v2.0 "Time Capsule"
=====================================

Извлекает данные паттерна по timestamp + сохраняет снапшот сырых данных.

Ключевое улучшение: Снапшот (100 баров до + 50 после) позволяет:
- Пересчитывать метрики при изменении формул
- Тестировать разные TP/SL на одних данных
- Искать корреляции в истории

Пример использования:
    extractor = PatternExtractor("BTC", "1h")
    data = extractor.extract("2024-11-26 14:00")
    extractor.save(data)  # Сохраняет JSON + CSV снапшот
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

from .config import (
    get_file_path, 
    PATTERNS_DIR, 
    KEY_FEATURES,
    TF_MINUTES
)


class PatternExtractor:
    """
    Извлекает все метрики для паттерна Box Range.
    Сохраняет "Капсулу Времени" — снапшот сырых данных.
    
    Workflow:
    1. Егор 1 находит паттерн на TradingView
    2. Записывает время свечи пробития
    3. Extractor находит эту свечу + свечу ДО неё
    4. Извлекает все 74 features
    5. Сохраняет в JSON + CSV снапшот
    """
    
    def __init__(self, coin: str, timeframe: str):
        """
        Инициализация.
        
        Parameters
        ----------
        coin : str
            Тикер монеты (BTC, ENA, ...)
        timeframe : str
            Таймфрейм (15m, 1h, 4h, 1d)
        """
        self.coin = coin.upper()
        self.timeframe = timeframe.lower()
        self.file_path = get_file_path(self.coin, self.timeframe)
        
        print(f"📂 Загрузка {self.file_path.name}...")
        self.df = pd.read_parquet(self.file_path)
        
        # Убедиться что индекс datetime
        if not isinstance(self.df.index, pd.DatetimeIndex):
            if 'timestamp' in self.df.columns:
                self.df.set_index('timestamp', inplace=True)
            else:
                self.df.index = pd.to_datetime(self.df.index)
        
        # UTC timezone
        if self.df.index.tz is None:
            self.df.index = self.df.index.tz_localize('UTC')
        
        print(f"✅ Загружено {len(self.df):,} баров")
        print(f"   Период: {self.df.index[0]} — {self.df.index[-1]}")
    
    def _find_bar(self, time_str: str) -> Tuple[int, pd.Timestamp]:
        """
        Найти бар по времени.
        """
        try:
            target = pd.Timestamp(time_str)
            if target.tz is None:
                target = target.tz_localize('UTC')
        except Exception as e:
            raise ValueError(f"Неверный формат времени: {time_str}. Используй YYYY-MM-DD HH:MM")
        
        tolerance = timedelta(minutes=TF_MINUTES[self.timeframe] // 2)
        
        time_diff = abs(self.df.index - target)
        min_idx = time_diff.argmin()
        actual = self.df.index[min_idx]
        
        if abs(actual - target) > tolerance:
            raise ValueError(
                f"Бар не найден. Ближайший: {actual}\n"
                f"Запрошен: {target}\n"
                f"Разница: {abs(actual - target)}"
            )
        
        return min_idx, actual
    
    def _extract_features(self, idx: int) -> Dict[str, Any]:
        """
        Извлечь features для одного бара.
        """
        row = self.df.iloc[idx]
        features = {}
        
        for group_name, feature_list in KEY_FEATURES.items():
            for feature in feature_list:
                if feature in row.index:
                    value = row[feature]
                    if pd.isna(value) or np.isinf(value):
                        value = None
                    elif isinstance(value, (np.integer, np.floating)):
                        value = float(value)
                    features[feature] = value
        
        return features
    
    def _calculate_box_metrics(
        self, 
        breakout_idx: int, 
        lookback: int = 48
    ) -> Dict[str, Any]:
        """
        Рассчитать метрики Box Range.
        """
        start_idx = max(0, breakout_idx - lookback)
        box = self.df.iloc[start_idx:breakout_idx]
        
        if len(box) < 10:
            return {"error": "Недостаточно данных для box"}
        
        support = float(box['low'].min())
        resistance = float(box['high'].max())
        box_range = resistance - support
        box_range_pct = (box_range / support) * 100 if support > 0 else 0
        
        tol = 0.003
        touches_support = int(sum(
            (box['low'] <= support * (1 + tol)) & 
            (box['low'] >= support * (1 - tol))
        ))
        touches_resistance = int(sum(
            (box['high'] >= resistance * (1 - tol)) & 
            (box['high'] <= resistance * (1 + tol))
        ))
        
        tr = pd.concat([
            box['high'] - box['low'],
            abs(box['high'] - box['close'].shift(1)),
            abs(box['low'] - box['close'].shift(1))
        ], axis=1).max(axis=1)
        atr = float(tr.mean())
        
        return {
            "support": round(support, 6),
            "resistance": round(resistance, 6),
            "box_range_pct": round(box_range_pct, 2),
            "touches_support": touches_support,
            "touches_resistance": touches_resistance,
            "atr_box": round(atr, 6),
            "duration_bars": len(box)
        }
    
    def _extract_snapshot(
        self,
        breakout_idx: int,
        lookback: int = 100,
        forward: int = 50
    ) -> pd.DataFrame:
        """
        Вырезать снапшот сырых данных вокруг пробоя.
        
        Это "Капсула Времени" — замороженные данные для:
        - Пересчёта метрик при изменении формул
        - Тестирования разных TP/SL
        - Поиска корреляций
        
        Parameters
        ----------
        breakout_idx : int
            Индекс бара пробития
        lookback : int
            Баров ДО пробоя (история/контекст)
        forward : int
            Баров ПОСЛЕ пробоя (будущее/результат)
            
        Returns
        -------
        pd.DataFrame
            Снапшот с lookback + 1 + forward баров
        """
        start_idx = max(0, breakout_idx - lookback)
        end_idx = min(len(self.df), breakout_idx + forward + 1)
        
        snapshot = self.df.iloc[start_idx:end_idx].copy()
        
        # Добавить метку позиции относительно пробоя
        snapshot['bar_position'] = range(-(breakout_idx - start_idx), end_idx - breakout_idx)
        
        return snapshot
    
    def _calculate_future_path(
        self,
        snapshot: pd.DataFrame,
        entry_price: float
    ) -> Dict[str, Any]:
        """
        Рассчитать метрики будущего пути цены.
        
        Это позволяет тестировать разные TP/SL без перезаписи данных.
        """
        # Только бары после пробоя (bar_position > 0)
        future = snapshot[snapshot['bar_position'] > 0]
        
        if len(future) == 0:
            return {"error": "Нет данных о будущем"}
        
        # Максимумы и минимумы после входа
        max_high = float(future['high'].max())
        min_low = float(future['low'].min())
        
        # Максимальный profit и drawdown
        max_profit_pct = ((max_high - entry_price) / entry_price) * 100
        max_drawdown_pct = ((entry_price - min_low) / entry_price) * 100
        
        # Когда достигнуты (в барах после входа)
        bars_to_max = int(future['high'].idxmax().value) if len(future) > 0 else None
        bars_to_min = int(future['low'].idxmin().value) if len(future) > 0 else None
        
        # Симуляция разных TP/SL
        tp_levels = [1.0, 1.5, 2.0, 2.5, 3.0]  # %
        sl_levels = [0.5, 1.0, 1.5, 2.0]  # %
        
        simulations = {}
        for tp in tp_levels:
            for sl in sl_levels:
                tp_price = entry_price * (1 + tp/100)
                sl_price = entry_price * (1 - sl/100)
                
                result = "OPEN"  # Позиция не закрыта
                exit_bar = None
                
                for i, (_, bar) in enumerate(future.iterrows()):
                    if bar['high'] >= tp_price:
                        result = "TP"
                        exit_bar = i + 1
                        break
                    if bar['low'] <= sl_price:
                        result = "SL"
                        exit_bar = i + 1
                        break
                
                simulations[f"TP{tp}_SL{sl}"] = {
                    "result": result,
                    "exit_bar": exit_bar
                }
        
        return {
            "max_profit_pct": round(max_profit_pct, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "future_bars": len(future),
            "simulations": simulations
        }
    
    def extract(
        self,
        breakout_time: str,
        pattern_type: str = "box_range",
        direction: str = "long",
        lookback: int = 48,
        snapshot_lookback: int = 100,
        snapshot_forward: int = 50,
        notes: str = ""
    ) -> Dict:
        """
        ГЛАВНЫЙ МЕТОД — извлечь все данные паттерна.
        
        Parameters
        ----------
        breakout_time : str
            Время пробития 'YYYY-MM-DD HH:MM'
        pattern_type : str
            Тип паттерна (box_range, breakout, ...)
        direction : str
            Направление (long, short)
        lookback : int
            Баров назад для box metrics
        snapshot_lookback : int
            Баров назад для снапшота (история)
        snapshot_forward : int
            Баров вперёд для снапшота (будущее)
        notes : str
            Заметки
            
        Returns
        -------
        Dict
            Полные данные паттерна + снапшот
        """
        try:
            # 1. Найти бар пробития
            breakout_idx, breakout_actual = self._find_bar(breakout_time)
            
            # 2. Бар ДО пробития (для индикаторов) — защита от look-ahead
            if breakout_idx < 2:
                return {"error": "Слишком мало данных до пробития"}
            
            setup_idx = breakout_idx - 1
            setup_time = self.df.index[setup_idx]
            
            # 3. Извлечь features
            setup_features = self._extract_features(setup_idx)
            breakout_features = self._extract_features(breakout_idx)
            
            # 4. Box metrics
            box_metrics = self._calculate_box_metrics(breakout_idx, lookback)
            
            # 5. W_box компоненты
            w_box = self._calculate_w_box(setup_features, box_metrics, direction)
            
            # 6. Снапшот (Капсула Времени)
            snapshot = self._extract_snapshot(breakout_idx, snapshot_lookback, snapshot_forward)
            
            # 7. Future Path (для тестирования TP/SL)
            entry_price = breakout_features.get("close", 0)
            future_path = self._calculate_future_path(snapshot, entry_price) if entry_price > 0 else {}
            
            # 8. Формируем результат
            pattern_id = f"{self.coin}_{self.timeframe}_{breakout_actual.strftime('%Y%m%d_%H%M')}"
            
            result = {
                "id": pattern_id,
                "version": "2.0",  # Time Capsule версия
                "created_at": datetime.now().isoformat(),
                
                "meta": {
                    "coin": self.coin,
                    "timeframe": self.timeframe,
                    "pattern_type": pattern_type,
                    "direction": direction,
                    "notes": notes
                },
                
                "timing": {
                    "breakout_time_input": breakout_time,
                    "breakout_time_actual": str(breakout_actual),
                    "setup_time": str(setup_time)
                },
                
                "box": box_metrics,
                
                "breakout_bar": {
                    "open": breakout_features.get("open"),
                    "high": breakout_features.get("high"),
                    "low": breakout_features.get("low"),
                    "close": breakout_features.get("close"),
                    "volume": breakout_features.get("volume")
                },
                
                "setup_bar": {
                    "open": setup_features.get("open"),
                    "high": setup_features.get("high"),
                    "low": setup_features.get("low"),
                    "close": setup_features.get("close"),
                    "volume": setup_features.get("volume")
                },
                
                "indicators_before": {
                    "rsi_zscore": setup_features.get("norm_rsi_zscore"),
                    "macd_zscore": setup_features.get("norm_macd_zscore"),
                    "atr_zscore": setup_features.get("norm_atr_zscore"),
                    "bb_width_zscore": setup_features.get("norm_bb_width_zscore"),
                    "volume_zscore": setup_features.get("norm_volume_zscore"),
                    
                    "rsi_low": setup_features.get("regime_rsi_low"),
                    "rsi_mid": setup_features.get("regime_rsi_mid"),
                    "rsi_high": setup_features.get("regime_rsi_high"),
                    "trend_up": setup_features.get("regime_trend_up"),
                    "trend_down": setup_features.get("regime_trend_down"),
                    "vol_low": setup_features.get("regime_vol_low"),
                    "vol_high": setup_features.get("regime_vol_high"),
                    
                    "div_rsi_bullish": setup_features.get("div_rsi_bullish"),
                    "div_rsi_bearish": setup_features.get("div_rsi_bearish"),
                    
                    "session_hour": setup_features.get("time_hour"),
                    "session_asian": setup_features.get("time_asian"),
                    "session_european": setup_features.get("time_european"),
                    "session_american": setup_features.get("time_american")
                },
                
                "w_box": w_box,
                
                "future_path": future_path,
                
                "snapshot": {
                    "lookback_bars": snapshot_lookback,
                    "forward_bars": snapshot_forward,
                    "total_bars": len(snapshot),
                    "file": None  # Будет заполнено при сохранении
                },
                
                "_snapshot_df": snapshot,  # Временное хранение для save()
                
                "all_features_setup": setup_features
            }
            
            return result
            
        except ValueError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Неожиданная ошибка: {str(e)}"}
    
    def _calculate_w_box(
        self, 
        features: Dict, 
        box: Dict,
        direction: str
    ) -> Dict:
        """
        Рассчитать компоненты W_box.
        """
        result = {}
        
        rsi_z = features.get("norm_rsi_zscore")
        if rsi_z is not None:
            if -0.5 <= rsi_z <= 0.5:
                I_rsi = 1.0
            elif -1.0 <= rsi_z <= 1.0:
                I_rsi = 0.7
            elif -1.5 <= rsi_z <= 1.5:
                I_rsi = 0.3
            else:
                I_rsi = 0.0
            result["I_rsi"] = round(I_rsi, 2)
        
        atr_z = features.get("norm_atr_zscore")
        if atr_z is not None:
            if atr_z < -0.5:
                I_vol = 1.0
            elif atr_z < 0:
                I_vol = 0.8
            elif atr_z < 0.5:
                I_vol = 0.5
            else:
                I_vol = 0.0
            result["I_volatility"] = round(I_vol, 2)
        
        vol_z = features.get("norm_volume_zscore")
        if vol_z is not None:
            if vol_z > 1.0:
                I_volume = 1.0
            elif vol_z > 0.5:
                I_volume = 0.8
            elif vol_z > 0:
                I_volume = 0.5
            else:
                I_volume = 0.3
            result["I_volume"] = round(I_volume, 2)
        
        if "touches_support" in box and "touches_resistance" in box:
            ts = box["touches_support"]
            tr = box["touches_resistance"]
            if ts >= 3 and tr >= 3:
                I_touches = 1.0
            elif ts >= 2 and tr >= 2:
                I_touches = 0.7
            else:
                I_touches = 0.3
            result["I_touches"] = round(I_touches, 2)
        
        components = [result.get(k) for k in ["I_rsi", "I_volatility", "I_volume", "I_touches"]]
        components = [c for c in components if c is not None]
        
        if components:
            W_box = 1.0
            for c in components:
                W_box *= c
            result["W_box"] = round(W_box, 4)
        
        return result
    
    def save(self, data: Dict) -> Optional[Path]:
        """
        Сохранить паттерн в JSON + CSV снапшот.
        
        Returns
        -------
        Path или None
            Путь к JSON файлу или None при ошибке
        """
        if "error" in data:
            print(f"❌ Ошибка: {data['error']}")
            return None
        
        pattern_id = data['id']
        
        # 1. Сохранить снапшот CSV
        snapshot_df = data.pop('_snapshot_df', None)
        if snapshot_df is not None:
            snapshots_dir = PATTERNS_DIR / "snapshots"
            snapshots_dir.mkdir(exist_ok=True)
            
            snapshot_path = snapshots_dir / f"{pattern_id}.csv"
            snapshot_df.to_csv(snapshot_path)
            
            # Обновить ссылку в данных
            data['snapshot']['file'] = str(snapshot_path.name)
            print(f"📸 Снапшот: {snapshot_path}")
        
        # 2. Сохранить JSON
        json_path = PATTERNS_DIR / f"{pattern_id}.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ Паттерн: {json_path}")
        
        return json_path