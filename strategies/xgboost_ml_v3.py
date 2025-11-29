"""
XGBoost ML Strategy v3
======================

Model 2: XGBoost на 74 features (single timeframe).

Формула из документации:
P_ml(S) = σ(f_XGB(Φ(S))) · ∏ₖ Fₖ(S) - C_adaptive(S) - R_ood(S)

Изменения относительно v2:
- Работает с 74 features (не 31)
- Убран multi-timeframe (каждый файл = один таймфрейм)
- Добавлен generate_signal() для интеграции

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


class XGBoostMLStrategyV3:
    """
    Model 2: XGBoost ML Strategy.
    
    Архитектура:
    - Input: 74 features (normalized indicators)
    - Model: XGBoost binary classifier
    - Output: probability [0, 1] → signal {0, 1}
    
    Компоненты P_ml(S):
    - σ(f_XGB(Φ(S))): XGBoost probability
    - ∏ₖ Fₖ(S): Режимные фильтры (crisis, drawdown)
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
        
        # Загрузить metadata если есть
        metadata_path = path.parent / (path.stem + '_metadata.json')
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                self.feature_names = self.metadata.get('feature_names')
        
        print(f"✅ Model loaded: {path.name}")
    
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
        
        if isinstance(features, pd.DataFrame):
            if 'target' in features.columns:
                features = features.drop(columns=['target'])
            features = features.values
        
        if features.shape[1] != self.EXPECTED_FEATURES:
            raise ValueError(
                f"Expected {self.EXPECTED_FEATURES} features, "
                f"got {features.shape[1]}"
            )
        
        dmatrix = xgb.DMatrix(features)
        return self.model.predict(dmatrix)
    
    def predict_single(self, features: Union[pd.Series, np.ndarray]) -> float:
        """Получить вероятность для одного sample."""
        if isinstance(features, pd.Series):
            features = features.values.reshape(1, -1)
        elif features.ndim == 1:
            features = features.reshape(1, -1)
        
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
                proba = self.predict_single(features.iloc[0])
            else:
                proba = self.predict_single(features.iloc[-1])
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
        
        feature_cols = [c for c in df.columns if c != 'target']
        features = df[feature_cols].values
        
        probabilities = self.predict_proba(features)
        
        result['ml_proba'] = probabilities
        result['ml_signal'] = (probabilities >= threshold).astype(int)
        
        return result
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        threshold: float = 0.5
    ) -> Dict:
        """Оценить качество модели."""
        from sklearn.metrics import (
            roc_auc_score, f1_score, precision_score, 
            recall_score, accuracy_score, confusion_matrix
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
