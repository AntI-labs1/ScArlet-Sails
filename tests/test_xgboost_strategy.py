"""
TEST XGBOOST ML STRATEGY
Isolated test to verify XGBoost strategy works

Author: STAR_ANT + Claude Sonnet 4.5
Date: November 23, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

sys.path.append('.')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from strategies.xgboost_ml_v3 import XGBoostMLStrategyV3

def main():
    """Test XGBoost strategy in isolation"""
    print("="*80)
    print("TESTING XGBOOST ML STRATEGY")
    print("="*80)
    print()
    
    # Load data
    logger.info("Loading data...")
    data_file = 'data/features/BTC_USDT_15m_features.parquet'
    
    if not Path(data_file).exists():
        logger.error(f"Data file not found: {data_file}")
        return
    
    df = pd.read_parquet(data_file)
    logger.info(f"Loaded: {len(df)} bars")
    
    # Use 2024 data
    test_df = df[df.index >= '2024-01-01']
    logger.info(f"Test period: {len(test_df)} bars")
    
    # Check model file
    print()
    print("CHECKING MODEL FILE:")
    print("-"*80)
    
    model_path = 'models/xgboost_trained_v2.json'
    if Path(model_path).exists():
        size = Path(model_path).stat().st_size / 1024 / 1024
        print(f"✅ Model file exists: {model_path} ({size:.2f} MB)")
    else:
        print(f"❌ Model file missing: {model_path}")
        return
    
    # Initialize strategy
    print()
    print("INITIALIZING STRATEGY:")
    print("-"*80)
    
    try:
        strategy = XGBoostMLStrategy(model_path=model_path)
        logger.info("✅ XGBoost strategy initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test signal generation
    print()
    print("TESTING SIGNAL GENERATION:")
    print("-"*80)
    
    try:
        # Test on last 1000 bars
        test_sample = test_df.iloc[-1000:]
        logger.info(f"Testing on {len(test_sample)} bars...")
        
        result = strategy.generate_signal(test_sample)
        
        if result:
            print(f"✅ Signal generated!")
            print(f"   Action: {result.get('action', 'N/A')}")
            print(f"   Confidence: {result.get('confidence', 'N/A')}")
            print(f"   ML Score: {result.get('ml_score', 'N/A')}")
            print(f"   Keys: {list(result.keys())}")
        else:
            print(f"⚠️ No signal generated")
        
    except Exception as e:
        logger.error(f"❌ Signal generation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test on multiple windows
    print()
    print("TESTING MULTIPLE WINDOWS:")
    print("-"*80)
    
    signals_count = 0
    buy_signals = 0
    sell_signals = 0
    hold_signals = 0
    
    # Test every 100 bars
    for i in range(100, min(len(test_df), 1000), 100):
        try:
            window = test_df.iloc[:i]
            result = strategy.generate_signal(window)
            
            if result:
                action = result.get('action', 'hold')
                signals_count += 1
                
                if action == 'buy':
                    buy_signals += 1
                elif action == 'sell':
                    sell_signals += 1
                else:
                    hold_signals += 1
        except Exception as e:
            logger.error(f"Error at bar {i}: {e}")
    
    print(f"Total signals: {signals_count}")
    print(f"  Buy: {buy_signals}")
    print(f"  Sell: {sell_signals}")
    print(f"  Hold: {hold_signals}")
    
    if signals_count == 0:
        print()
        print("⚠️ WARNING: No signals generated!")
    else:
        print()
        print("✅ Strategy is generating signals!")
    
    print()
    print("="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()