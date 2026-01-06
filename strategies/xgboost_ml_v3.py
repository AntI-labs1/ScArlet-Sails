"""
XGBoost ML Strategy v3
======================
Model 2: XGBoost на 74 features (single timeframe).

Изменения v3.2 (Canonical Pipeline Edition):
- УДАЛЕНО: Зависимость от feature_engine_v2 (deleted module)
- Работает только с CanonicalState/готовыми features
- DMatrix создаётся с feature_names из metadata
"""
from pathlib import Path
from typing import Dict, Optional, Union, Any
import numpy as np
import pandas as pd
import xgboost as xgb

from core.ood_detector import OODDetector


class XGBoostMLStrategyV3:
    """
    Model 2: XGBoost ML Strategy.
    
    Архитектура:
    - Input: 74-75 features (normalized indicators)
    - Model: XGBoost binary classifier
    - Output: probability [0, 1] → signal {0, 1}
    """
    
    EXPECTED_FEATURES = 74

    def __init__(self, model_path: Optional[str] = None):
        """Инициализация стратегии."""
        self.model: Optional[xgb.Booster] = None
        self.model_path: Optional[str] = None
        self.feature_names: Optional[list] = None
        self.metadata: Dict = {}
        
        # OOD Detector
        self._ood_detector = OODDetector()
        ood_path = Path('models/ood_detector_btc_15m.json')
        if ood_path.exists():
            self._ood_detector.load(str(ood_path))
        
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
        
        # Загрузить metadata
        for suffix in ['_metadata.json', '_meta.json']:
            metadata_path = path.parent / (path.stem + suffix)
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                break
        
        # Если в модели нет feature_names, взять из metadata
        if self.feature_names is None:
            self.feature_names = self.metadata.get('feature_names')
        
        print(f"✅ Model loaded: {path.name}")
        if self.feature_names:
            print(f"   Features: {len(self.feature_names)}")

    def predict_proba(self, features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Получить вероятности для всех samples."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if isinstance(features, pd.DataFrame):
            if 'target' in features.columns:
                features = features.drop(columns=['target'])
            
            if self.feature_names:
                features = features[self.feature_names]
            
            dmatrix = xgb.DMatrix(features, feature_names=list(features.columns))
        else:
            dmatrix = xgb.DMatrix(features, feature_names=self.feature_names)
        
        return self.model.predict(dmatrix)

    def predict_single(self, features: Union[pd.Series, pd.DataFrame, np.ndarray]) -> float:
        """Получить вероятность для одного sample."""
        if isinstance(features, pd.Series):
            features = features.to_frame().T
        elif isinstance(features, np.ndarray):
            if features.ndim == 1:
                features = features.reshape(1, -1)
        elif isinstance(features, pd.DataFrame):
            if len(features) != 1:
                features = features.iloc[-1:].copy()
        
        return float(self.predict_proba(features)[0])

    def generate_signal(
        self,
        features: Union[pd.DataFrame, pd.Series, np.ndarray],
        threshold: float = 0.70,
        crisis_level: float = 0.0,
        drawdown: float = 0.0,
        regime: str = 'normal',
    ) -> Dict:
        """Генерировать торговый сигнал."""
        if isinstance(features, pd.DataFrame):
            proba = self.predict_single(features.iloc[-1:] if len(features) > 1 else features)
        elif isinstance(features, pd.Series):
            proba = self.predict_single(features)
        else:
            proba = self.predict_single(features)
        
        confidence = proba
        
        if self._ood_detector._fitted:
            ood_state = self._ood_detector.detect(features)
            confidence *= ood_state.confidence_multiplier
        
        filter_crisis = crisis_level < 0.7
        filter_drawdown = drawdown < 0.15
        filter_regime = regime != 'crisis'
        filters_pass = filter_crisis and filter_drawdown and filter_regime
        
        signal = 1 if (proba >= threshold and filters_pass) else 0
        P_ml = confidence if filters_pass else 0.0
        
        return {
            'signal': signal,
            'probability': proba,
            'P_ml': P_ml,
            'threshold': threshold,
            'filters_pass': filters_pass,
            'filter_details': {
                'crisis_ok': filter_crisis,
                'drawdown_ok': filter_drawdown,
                'regime_ok': filter_regime,
            }
        }

    def generate_signals_batch(self, df: pd.DataFrame, threshold: float = 0.70) -> pd.DataFrame:
        """Генерировать сигналы для всего DataFrame."""
        result = df.copy()
        feature_cols = [c for c in df.columns if c != 'target']
        features_df = df[feature_cols]
        
        probabilities = self.predict_proba(features_df)
        result['ml_proba'] = probabilities
        result['ml_signal'] = (probabilities >= threshold).astype(int)
        
        return result

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        threshold: float = 0.70,
    ) -> Dict:
        """Оценить качество модели."""
        from sklearn.metrics import (
            roc_auc_score, f1_score, precision_score, 
            recall_score, accuracy_score,
        )
        
        y_proba = self.predict_proba(X)
        y_pred = (y_proba >= threshold).astype(int)
        
        return {
            'auc': roc_auc_score(y, y_proba),
            'f1': f1_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'accuracy': accuracy_score(y, y_pred),
            'threshold': threshold,
            'samples': len(y),
            'class_balance': float(y.mean()),
        }

    def __repr__(self) -> str:
        status = 'loaded' if self.model else 'not loaded'
        return f'XGBoostMLStrategyV3(model={status})'
