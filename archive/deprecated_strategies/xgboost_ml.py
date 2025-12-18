"""
STRATEGY - XGBOOST ML
Machine learning strategy using XGBoost
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

class XGBoostStrategy:
    """ML-based trading strategy"""
    
    def __init__(self, config, feature_engine, model_manager):
        self.config = config
        self.feature_engine = feature_engine
        self.model_manager = model_manager
        self.name = "XGBoost-ML"
        
        self.threshold = config['models']['xgboost']['threshold']
    
    def generate_signals(self, df):
        """Generate trading signals using XGBoost"""
        # Calculate features
        features_df = self.feature_engine.calculate_features(df)
        
        # Get first 31 features for model
        feature_cols = features_df.columns[:31]
        X = features_df[feature_cols].values
        
        # Get model predictions
        signals = self.model_manager.predict_xgboost(X)
        
        logger.debug(f"  Generated {signals.sum()} ML signals")
        
        return signals
    
    def generate_signals_with_pj_s(self, df, regime=None, crisis_level=None):
        """Generate signals with P_j(S) formula components"""
        # Calculate ALL features first
        features_df = self.feature_engine.calculate_features(df)
        
        # Get first 31 features for model
        feature_cols = features_df.columns[:31]
        X = features_df[feature_cols].values
        
        # Base ML signals and scores
        signals = self.model_manager.predict_xgboost(X)
        ml_scores = self.model_manager.predict_proba(X)[:, 1]
        
        # P_j(S) = ML_score * filters + opportunity - costs - risk
        pj_s_values = ml_scores.copy()
        
        # Apply regime filter
        if regime is not None:
            regime_adjustments = {
                'BULL': 1.0,
                'BEAR': 0.5,
                'SIDEWAYS': 0.75
            }
            for i, r in enumerate(regime):
                adj = regime_adjustments.get(r, 1.0)
                pj_s_values[i] *= adj
        
        # Apply crisis filter
        if crisis_level is not None:
            crisis_threshold = self.config['models']['crisis_classifier']['threshold']
            pj_s_values = pj_s_values * (crisis_level < crisis_threshold)
        
        # Subtract costs
        costs = self.config['formula']['costs']['commission'] + \
                self.config['formula']['costs']['slippage']
        pj_s_values -= costs
        
        # Risk penalty for high volatility (from FULL features_df)
        if 'volatility' in features_df.columns:
            volatility = features_df['volatility'].values
            vol_threshold = self.config['formula']['risk_penalty']['volatility_threshold']
            risk_penalty = np.where(volatility > vol_threshold, 0.1, 0)
            pj_s_values -= risk_penalty
        
        # Update signals based on P_j(S)
        signals = (pj_s_values >= self.threshold).astype(int)
        
        return signals, ml_scores, pj_s_values
    
    def get_parameters(self):
        """Get strategy parameters"""
        return {
            'name': self.name,
            'threshold': self.threshold
        }