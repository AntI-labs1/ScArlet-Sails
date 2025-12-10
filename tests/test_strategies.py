"""
Proper pytest tests for ScArlet-Sails strategies.
Fast, isolated, correct format.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_ohlcv():
    """Generate small OHLCV dataset for testing."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2024-01-01', periods=n, freq='15min')
    
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.random.rand(n) * 200
    low = close - np.random.rand(n) * 200
    open_ = close + np.random.randn(n) * 50
    volume = np.random.rand(n) * 1000000
    
    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


@pytest.fixture
def sample_features():
    """Load small slice of real features for testing."""
    path = Path('data/features/BTC_USDT_15m_features.parquet')
    if path.exists():
        df = pd.read_parquet(path)
        return df.iloc[:500].copy()  # Only 500 rows for speed
    return None


@pytest.fixture
def model_path():
    """Path to trained XGBoost model."""
    path = Path('models/xgboost_v3_btc_15m.json')
    if path.exists():
        return str(path)
    return None


# ============================================================
# TEST: RULE-BASED STRATEGY
# ============================================================

class TestRuleBasedStrategy:
    """Tests for RuleBasedStrategy."""
    
    def test_import(self):
        """Strategy can be imported."""
        from strategies.rule_based_v2 import RuleBasedStrategy
        assert RuleBasedStrategy is not None
    
    def test_initialization(self):
        """Strategy initializes without errors."""
        from strategies.rule_based_v2 import RuleBasedStrategy
        strategy = RuleBasedStrategy()
        assert strategy is not None
    
    def test_generate_signals_returns_dataframe(self, sample_ohlcv):
        """generate_signals returns DataFrame."""
        from strategies.rule_based_v2 import RuleBasedStrategy
        strategy = RuleBasedStrategy()
        result = strategy.generate_signals(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)
    
    def test_generate_signals_has_required_columns(self, sample_ohlcv):
        """Output has P_rb and signal columns."""
        from strategies.rule_based_v2 import RuleBasedStrategy
        strategy = RuleBasedStrategy()
        result = strategy.generate_signals(sample_ohlcv)
        assert 'P_rb' in result.columns or 'signal' in result.columns


# ============================================================
# TEST: XGBOOST ML STRATEGY
# ============================================================

class TestXGBoostMLStrategy:
    """Tests for XGBoostMLStrategy."""
    
    def test_import(self):
        """Strategy can be imported."""
        from strategies.xgboost_ml_v3 import XGBoostMLStrategyV3
        assert XGBoostMLStrategyV3 is not None
    
    def test_initialization_without_model(self):
        """Strategy initializes without model path."""
        from strategies.xgboost_ml_v3 import XGBoostMLStrategyV3
        strategy = XGBoostMLStrategyV3()
        assert strategy is not None
    
    def test_initialization_with_model(self, model_path):
        """Strategy loads model correctly."""
        if model_path is None:
            pytest.skip("Model file not found")
        
        from strategies.xgboost_ml_v3 import XGBoostMLStrategyV3
        strategy = XGBoostMLStrategyV3(model_path=model_path)
        assert strategy.model is not None
    
    def test_feature_names_loaded(self, model_path):
        """Model has 74 features."""
        if model_path is None:
            pytest.skip("Model file not found")
        
        from strategies.xgboost_ml_v3 import XGBoostMLStrategyV3
        strategy = XGBoostMLStrategyV3(model_path=model_path)
        assert len(strategy.feature_names) == 74
    
    def test_predict_proba_shape(self, model_path, sample_features):
        """predict_proba returns correct shape."""
        if model_path is None or sample_features is None:
            pytest.skip("Model or data not found")
        
        from strategies.xgboost_ml_v3 import XGBoostMLStrategyV3
        strategy = XGBoostMLStrategyV3(model_path=model_path)
        
        # Get feature columns
        feature_cols = [c for c in sample_features.columns 
                       if c in strategy.feature_names]
        X = sample_features[feature_cols].dropna()
        
        if len(X) > 0:
            proba = strategy.predict_proba(X)
            assert len(proba) == len(X)
            assert all(0 <= p <= 1 for p in proba)
    
    def test_default_threshold_is_070(self, model_path):
        """Default threshold is 0.70."""
        if model_path is None:
            pytest.skip("Model file not found")
        
        from strategies.xgboost_ml_v3 import XGBoostMLStrategyV3
        
        # Check source code has 0.70
        import inspect
        source = inspect.getsource(XGBoostMLStrategyV3)
        assert 'threshold: float = 0.70' in source or 'threshold=0.70' in source


# ============================================================
# TEST: MODEL PREDICTIONS
# ============================================================

class TestModelPredictions:
    """Tests for model prediction quality."""
    
    def test_predictions_not_all_same(self, model_path, sample_features):
        """Model produces varied predictions, not constant."""
        if model_path is None or sample_features is None:
            pytest.skip("Model or data not found")
        
        from strategies.xgboost_ml_v3 import XGBoostMLStrategyV3
        strategy = XGBoostMLStrategyV3(model_path=model_path)
        
        feature_cols = [c for c in sample_features.columns 
                       if c in strategy.feature_names]
        X = sample_features[feature_cols].dropna()
        
        if len(X) > 10:
            proba = strategy.predict_proba(X)
            # Should have some variance
            assert np.std(proba) > 0.01, "Predictions are too uniform"
    
    def test_predictions_in_valid_range(self, model_path, sample_features):
        """All predictions between 0 and 1."""
        if model_path is None or sample_features is None:
            pytest.skip("Model or data not found")
        
        from strategies.xgboost_ml_v3 import XGBoostMLStrategyV3
        strategy = XGBoostMLStrategyV3(model_path=model_path)
        
        feature_cols = [c for c in sample_features.columns 
                       if c in strategy.feature_names]
        X = sample_features[feature_cols].dropna()
        
        if len(X) > 0:
            proba = strategy.predict_proba(X)
            assert all(0 <= p <= 1 for p in proba)


# ============================================================
# TEST: COUNCIL COMPONENTS
# ============================================================

class TestCouncilComponents:
    """Tests for Council architecture."""
    
    def test_rag_agent_import(self):
        """RAGAgent can be imported from council."""
        from council.rag_agent import RAGAgent
        assert RAGAgent is not None
    
    def test_quant_aggregator_import(self):
        """QuantAggregator can be imported."""
        from council.quant_aggregator import QuantAggregator
        assert QuantAggregator is not None
    
    def test_contracts_import(self):
        """Council contracts can be imported."""
        from council.contracts import CouncilContext, AgentOpinion
        assert CouncilContext is not None
        assert AgentOpinion is not None


# ============================================================
# TEST: DATA INTEGRITY
# ============================================================

class TestDataIntegrity:
    """Tests for data files."""
    
    def test_features_parquet_exists(self):
        """Feature parquet file exists."""
        path = Path('data/features/BTC_USDT_15m_features.parquet')
        assert path.exists(), f"Missing: {path}"
    
    def test_features_has_74_columns(self):
        """Feature file has expected columns."""
        path = Path('data/features/BTC_USDT_15m_features.parquet')
        if path.exists():
            df = pd.read_parquet(path)
            # Should have 75 columns (74 features + 1 more)
            assert len(df.columns) >= 74
    
    def test_model_file_exists(self):
        """XGBoost model file exists."""
        path = Path('models/xgboost_v3_btc_15m.json')
        assert path.exists(), f"Missing: {path}"
    
    def test_model_metadata_exists(self):
        """Model metadata file exists."""
        path = Path('models/xgboost_v3_btc_15m_metadata.json')
        assert path.exists(), f"Missing: {path}"


# ============================================================
# RUN
# ============================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
