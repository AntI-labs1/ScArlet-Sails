"""
CORE - MODELS
Model loading and management
"""
import logging
from pathlib import Path
import numpy as np

from strategies.xgboost_ml_v3 import XGBoostMLStrategyV3

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages all ML models"""
    
    def __init__(self, config):
        self.config = config
        self.project_root = Path(__file__).parent.parent
        self.models = {}
        
    def load_xgboost(self):
        """Load XGBoost model and scaler"""
        try:
            import xgboost as xgb
            import joblib
            
            # Load model
            model_path = self.project_root / self.config['models']['xgboost']['model_path']
            if model_path.exists():
                model = xgb.XGBClassifier()
                model.load_model(str(model_path))
                self.models['xgboost'] = model
                logger.info(f"  ✅ XGBoost loaded: {model_path.name}")
            else:
                logger.error(f"  ❌ XGBoost not found: {model_path}")
                return False
            
            # Load scaler
            scaler_path = self.project_root / self.config['models']['xgboost']['scaler_path']
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                self.models['scaler'] = scaler
                logger.info(f"  ✅ Scaler loaded: {scaler_path.name}")
            else:
                logger.warning(f"  ⚠️  Scaler not found: {scaler_path}")
                self.models['scaler'] = None
            
            return True
            
        except ImportError as e:
            logger.error(f"  ❌ Missing dependency: {e}")
            return False
        except Exception as e:
            logger.error(f"  ❌ Error loading XGBoost: {e}")
            return False
    
    def load_regime_detector(self):
        """Load regime detection model"""
        try:
            # Import regime detector module
            import sys
            models_dir = self.project_root / "models"
            sys.path.insert(0, str(models_dir))
            
            from regime_detector import RegimeDetector
            
            self.models['regime_detector'] = RegimeDetector()
            logger.info("  ✅ Regime detector loaded")
            return True
            
        except ImportError:
            logger.warning("  ⚠️  Regime detector not found (will skip)")
            self.models['regime_detector'] = None
            return False
        except Exception as e:
            logger.error(f"  ❌ Error loading regime detector: {e}")
            self.models['regime_detector'] = None
            return False
    
    def load_crisis_classifier(self):
        """Load crisis classification model"""
        try:
            import sys
            models_dir = self.project_root / "models"
            sys.path.insert(0, str(models_dir))
            
            from crisis_classifier import CrisisClassifier
            
            self.models['crisis_classifier'] = CrisisClassifier()
            logger.info("  ✅ Crisis classifier loaded")
            return True
            
        except ImportError:
            logger.warning("  ⚠️  Crisis classifier not found (will skip)")
            self.models['crisis_classifier'] = None
            return False

    def load_xgboost_v3(self):
        """Load XGBoost v3 with FeatureSpec enforcement."""
        cfg = self.config.get('models', {}).get('xgboost_v3', {})
        model_path = self.project_root / cfg.get('model_path', '')

        if not model_path.exists():
            raise FileNotFoundError(f"XGBoost v3 model not found at {model_path}")

        strategy = XGBoostMLStrategyV3(str(model_path))
        self.models['xgboost_v3'] = strategy
        logger.info(f"  ✅ XGBoost v3 loaded: {model_path.name}")
        return True
    
    def load_all(self):
        """Load all available models"""
        logger.info("Loading models...")
        
        success = True
        
        # Required models
        if not self.load_xgboost():
            success = False

        # Optional: new XGBoost v3
        try:
            self.load_xgboost_v3()
        except Exception as exc:
            logger.warning(f"  ⚠️  XGBoost v3 not loaded: {exc}")
        
        # Optional models (don't fail if missing)
        self.load_regime_detector()
        self.load_crisis_classifier()
        
        logger.info(f"✅ Loaded {len([m for m in self.models.values() if m is not None])} models")
        
        return success
    
    def predict_xgboost(self, X):
        """Get XGBoost predictions"""
        if 'xgboost' not in self.models:
            logger.error("XGBoost model not loaded")
            return None
        
        model = self.models['xgboost']
        scaler = self.models.get('scaler')
        
        # Scale features
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        # Predict probabilities
        proba = model.predict_proba(X_scaled)[:, 1]
        
        # Apply threshold
        threshold = self.config['models']['xgboost']['threshold']
        signals = (proba >= threshold).astype(int)
        
        return signals
    
    def predict_regime(self, df):
        """Detect market regime"""
        regime_detector = self.models.get('regime_detector')
        
        if regime_detector is None:
            # Default to BULL if no detector
            return np.array(['BULL'] * len(df))
        
        try:
            regimes = regime_detector.predict(df)
            return regimes
        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return np.array(['BULL'] * len(df))
    
    def predict_crisis(self, df):
        """Detect crisis conditions"""
        crisis_classifier = self.models.get('crisis_classifier')
        
        if crisis_classifier is None:
            # Default to no crisis
            return np.zeros(len(df))
        
        try:
            crisis_proba = crisis_classifier.predict_proba(df)
            return crisis_proba
        except Exception as e:
            logger.error(f"Crisis detection failed: {e}")
            return np.zeros(len(df))

    def predict_xgboost_v3(self, df):
        """Predict using XGBoost v3 strategy with column validation."""

        strategy: XGBoostMLStrategyV3 = self.models.get('xgboost_v3')
        if strategy is None:
            raise ValueError("XGBoost v3 strategy not loaded")

        cfg = self.config.get('models', {}).get('xgboost_v3', {})
        threshold = cfg.get('threshold', 0.5)

        result = strategy.generate_signals_batch(df, threshold=threshold)
        return result
    
    def get_model_info(self):
        """Get info about loaded models"""
        info = {
            'xgboost': self.models.get('xgboost') is not None,
            'scaler': self.models.get('scaler') is not None,
            'regime_detector': self.models.get('regime_detector') is not None,
            'crisis_classifier': self.models.get('crisis_classifier') is not None,
        }
        return info