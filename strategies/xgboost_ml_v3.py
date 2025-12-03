"""
XGBoost ML Strategy v3
======================

Model 2: XGBoost на 74 features (single timeframe).

Формула из документации:
P_ml(S) = σ(f_XGB(Φ(S))) · ∏ₖ Fₖ(S) - C_adaptive(S) - R_ood(S)

Изменения v3.1:
- ИСПРАВЛЕНО: DMatrix теперь создаётся с feature_names
- Работает с 74 features (не 31)
- Убран multi-timeframe (каждый файл = один таймфрейм)

Использование:
    from strategies.xgboost_ml_v3 import XGBoostMLStrategyV3
    
    strategy = XGBoostMLStrategyV3("models/xgboost_v3_btc_15m.json")
    result = strategy.generate_signal(features_df)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from typing import Dict, Optional, Union

from core.feature_engine_v2 import FeatureSpecV3


class XGBoostMLStrategyV3:
    """
    Model 2: XGBoost ML Strategy.
    
    Архитектура:
    - Input: 74 features (normalized indicators)
    - Model: XGBoost binary classifier
    - Output: probability [0, 1] → signal {0, 1}
    """
    
    EXPECTED_FEATURES = 74
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Инициализация стратегии.
        
        Parameters
        ----------
        model_path : str, optional
            Путь к обученной модели (.json)
        """
        self.model: Optional[xgb.Booster] = None
        self.model_path: Optional[str] = None
        self.feature_names: Optional[list] = None
        self.metadata: Dict = {}
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, path: str) -> None:
        """Загрузить обученную модель."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        self.model = xgb.Booster()
        self.model.load_model(str(path))
        self.model_path = str(path)
        
        # Получить feature names из модели
        self.feature_names = self.model.feature_names

        # Загрузить metadata если есть
        metadata_path = path.parent / (path.stem + '_metadata.json')
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                # Если в модели нет feature_names, взять из metadata
                if self.feature_names is None:
                    self.feature_names = self.metadata.get('feature_names')

        if not self.feature_names:
            raise ValueError("Feature names not found in model or metadata; cannot enforce FeatureSpecV3")

        self.feature_spec = FeatureSpecV3(feature_names=self.feature_names)
        
        print(f"✅ Model loaded: {path.name}")
        if self.feature_names:
            print(f"   Features: {len(self.feature_names)}")
    
    def predict_proba(self, features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Получить вероятности для всех samples.
        
        Parameters
        ----------
        features : pd.DataFrame or np.ndarray
            Features shape (n_samples, 74)
            
        Returns
        -------
        np.ndarray
            Вероятности shape (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if self.feature_names is None:
            raise ValueError("Feature names are not set; load a model with metadata")

        # Обработка DataFrame
        if isinstance(features, pd.DataFrame):
            if 'target' in features.columns:
                features = features.drop(columns=['target'])

            missing = [c for c in self.feature_names if c not in features.columns]
            if missing:
                raise ValueError(f"Missing required features: {missing}")

            features = features[self.feature_names]

            # Проверить количество features
            if features.shape[1] != self.EXPECTED_FEATURES:
                raise ValueError(
                    f"Expected {self.EXPECTED_FEATURES} features, "
                    f"got {features.shape[1]}"
                )

            # Создать DMatrix напрямую из DataFrame (сохраняет column names)
            dmatrix = xgb.DMatrix(features)

        # Обработка numpy array
        else:
            if features.shape[1] != self.EXPECTED_FEATURES:
                raise ValueError(
                    f"Expected {self.EXPECTED_FEATURES} features, "
                    f"got {features.shape[1]}"
                )

            # Создать DMatrix с feature_names
            dmatrix = xgb.DMatrix(features, feature_names=self.feature_names)
        
        return self.model.predict(dmatrix)
    
    def predict_single(self, features: Union[pd.Series, pd.DataFrame, np.ndarray]) -> float:
        """Получить вероятность для одного sample."""
        
        # Если Series — конвертировать в DataFrame (сохраняет index как column names)
        if isinstance(features, pd.Series):
            features = features.to_frame().T
        
        # Если numpy 1D — reshape и передать с feature_names
        elif isinstance(features, np.ndarray):
            if features.ndim == 1:
                features = features.reshape(1, -1)
            # Для numpy создаём DMatrix с feature_names в predict_proba
        
        # Если DataFrame с одной строкой — OK
        elif isinstance(features, pd.DataFrame):
            if len(features) != 1:
                features = features.iloc[-1:].copy()
        
        return float(self.predict_proba(features)[0])
    
    def generate_signal(
        self,
        features: Union[pd.DataFrame, pd.Series, np.ndarray],
        threshold: float = 0.5,
        crisis_level: float = 0.0,
        drawdown: float = 0.0,
        regime: str = "normal"
    ) -> Dict:
        """
        Генерировать торговый сигнал.
        
        P_ml(S) = σ(f_XGB(Φ(S))) · ∏ₖ Fₖ(S)
        
        Parameters
        ----------
        features : 74 features
        threshold : порог для сигнала
        crisis_level : уровень кризиса [0, 1]
        drawdown : текущий drawdown [0, 1]
        regime : рыночный режим
            
        Returns
        -------
        dict
            signal, probability, P_ml, filters_pass, filter_details
        """
        # 1. Получить вероятность
        if isinstance(features, pd.DataFrame):
            if len(features) == 1:
                proba = self.predict_single(features)
            else:
                proba = self.predict_single(features.iloc[-1:])
        elif isinstance(features, pd.Series):
            proba = self.predict_single(features)
        else:
            proba = self.predict_single(features)
        
        # 2. Режимные фильтры
        filter_crisis = crisis_level < 0.7
        filter_drawdown = drawdown < 0.15
        filter_regime = regime != "crisis"
        
        filters_pass = filter_crisis and filter_drawdown and filter_regime
        
        # 3. Финальный сигнал
        signal = 1 if (proba >= threshold and filters_pass) else 0
        P_ml = proba if filters_pass else 0.0
        
        return {
            "signal": signal,
            "probability": proba,
            "P_ml": P_ml,
            "threshold": threshold,
            "filters_pass": filters_pass,
            "filter_details": {
                "crisis_ok": filter_crisis,
                "drawdown_ok": filter_drawdown,
                "regime_ok": filter_regime
            }
        }
    
    def generate_signals_batch(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """Генерировать сигналы для всего DataFrame."""
        result = df.copy()

        # Получить features как DataFrame (сохраняет column names)
        feature_cols = [c for c in df.columns if c != 'target']
        features_df = df[feature_cols]

        # Предсказать (передаём DataFrame, не numpy)
        probabilities = self.predict_proba(features_df)

        risk_penalty = result.get("risk_penalty", pd.Series([0.0] * len(result), index=result.index))
        penalty_clipped = risk_penalty.clip(lower=0.0, upper=1.0)

        p_ml = probabilities * (1 - penalty_clipped)

        result['ml_proba'] = probabilities
        result['P_ml'] = p_ml
        result['ml_signal'] = (p_ml >= threshold).astype(int)

        return result

    def validate_feature_order(self, df: pd.DataFrame) -> Dict[str, list]:
        """Return a validation report for incoming features."""

        if self.feature_names is None:
            raise ValueError("Feature names are not set; load a model first")

        return FeatureSpecV3(feature_names=self.feature_names).validate(df.columns)
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        threshold: float = 0.5
    ) -> Dict:
        """Оценить качество модели."""
        from sklearn.metrics import (
            roc_auc_score, f1_score, precision_score, 
            recall_score, accuracy_score
        )
        
        y_proba = self.predict_proba(X)
        y_pred = (y_proba >= threshold).astype(int)
        
        return {
            "auc": roc_auc_score(y, y_proba),
            "f1": f1_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "accuracy": accuracy_score(y, y_pred),
            "threshold": threshold,
            "samples": len(y),
            "class_balance": float(y.mean())
        }
    
    def find_optimal_threshold(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict:
        """Найти оптимальный threshold по F1."""
        from sklearn.metrics import precision_recall_curve
        
        y_proba = self.predict_proba(X)
        precision, recall, thresholds = precision_recall_curve(y, y_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores[:-1])
        
        return {
            "optimal_threshold": float(thresholds[best_idx]),
            "best_f1": float(f1_scores[best_idx]),
            "precision_at_best": float(precision[best_idx]),
            "recall_at_best": float(recall[best_idx])
        }
    
    def __repr__(self) -> str:
        status = "loaded" if self.model else "not loaded"
        return f"XGBoostMLStrategyV3(model={status})"
