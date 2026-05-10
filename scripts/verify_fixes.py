import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
sys.path.append('.')

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VERIFY")

def test_timeframe_validation():
    print("\n1. Testing Timeframe Validation...")
    from core.feature_engine_v2 import FeatureEngine
    
    # Mock config
    config = {'features': {'normalize': True}}
    engine = FeatureEngine(config)
    
    # Create fake 1h data (interval 1 hour)
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    df_1h = pd.DataFrame({'close': np.random.random(100)}, index=dates)
    
    try:
        # Should FAIL because default base_timeframe is usually 15m
        engine.compute_features(df_1h)
        print("❌ FAILED: Engine accepted 1h data expecting 15m!")
    except ValueError as e:
        print(f"✅ PASSED: Caught mismatch error: {e}")
    except Exception as e:
        print(f"⚠️ UNEXPECTED ERROR: {e}")

def test_memory_detox():
    print("\n2. Testing Memory Detox...")
    from strategies.hybrid_q_learner import HybridQLearner
    import json
    
    learner = HybridQLearner()
    # Poison the weights
    learner.weights = np.array([[np.nan, 0.5], [np.inf, -0.5]])
    
    # Try to save
    test_path = "test_q_table.json"
    learner.save(test_path)
    
    # Verify file
    with open(test_path, 'r') as f:
        data = json.load(f)
        
    # Check if NaNs were replaced by 0.0
    w = data['weights']
    if w[0][0] == 0.0 and w[1][0] == 0.0:
        print("✅ PASSED: NaN/Inf sanitized to 0.0")
    else:
        print(f"❌ FAILED: Weights still corrupted: {w}")
        
    Path(test_path).unlink(missing_ok=True)

if __name__ == "__main__":
    test_timeframe_validation()
    test_memory_detox()