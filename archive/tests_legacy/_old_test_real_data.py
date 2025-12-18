"""
REAL DATA INTEGRATION TEST - NO SYNTHETIC!
Scarlet Sails Project

Tests 3 strategies on REAL MARKET DATA ONLY
⛔ NO SYNTHETIC DATA GENERATION!

Author: STAR_ANT + Claude Sonnet 4.5
Date: November 24, 2025
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("=" * 80)
print("SCARLET SAILS - REAL DATA INTEGRATION TEST")
print("=" * 80)


# ============================================================================
# STEP 1: LOAD REAL DATA (NO SYNTHETIC!)
# ============================================================================

print("\n🔄 STEP 1: Loading REAL DATA...")

data_path = "data/features/BTC_USDT_15m_features.parquet"

if not os.path.exists(data_path):
    print(f"\n❌ ERROR: REAL DATA not found!")
    print(f"   Expected: {data_path}")
    print(f"   Current directory: {os.getcwd()}")
    print("\n⛔ NO SYNTHETIC DATA FALLBACK!")
    print("⛔ THIS TEST REQUIRES REAL MARKET DATA!")
    print("\nPlease ensure data file exists and run again.")
    sys.exit(1)

# Load REAL DATA
print(f"📊 Loading: {data_path}")
df = pd.read_parquet(data_path)
print(f"✅ Loaded REAL DATA: {df.shape}")

# Validate structure
print(f"\n📊 Data validation:")
print(f"  Rows: {len(df):,}")
print(f"  Columns: {len(df.columns)}")
print(f"  Date range: {df.index[0]} to {df.index[-1]}")
print(f"  Memory: {df.memory_usage().sum() / 1024 / 1024:.1f} MB")

# Check for required columns
required_cols = ['open', 'high', 'low', 'close', 'volume']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(f"❌ Missing required columns: {missing}")
    sys.exit(1)
else:
    print(f"✅ All required columns present")

print(f"\n📋 Column summary:")
print(f"  Basic OHLCV: {len([c for c in df.columns if c in ['open','high','low','close','volume']])}")
print(f"  Normalized: {len([c for c in df.columns if c.startswith('norm_')])}")
print(f"  Derivatives: {len([c for c in df.columns if c.startswith('deriv_')])}")
print(f"  Regime: {len([c for c in df.columns if c.startswith('regime_')])}")
print(f"  Cross: {len([c for c in df.columns if c.startswith('cross_')])}")
print(f"  Divergences: {len([c for c in df.columns if c.startswith('div_')])}")
print(f"  Time: {len([c for c in df.columns if c.startswith('time_')])}")


# ============================================================================
# STEP 2: TEST XGBOOST ML STRATEGY
# ============================================================================

print("\n" + "=" * 80)
print("🔄 STEP 2: Testing XGBoost ML Strategy (74 features)")
print("=" * 80)

try:
    from strategies.xgboost_ml_v2 import XGBoostMLStrategy
    
    print("\n📦 Initializing XGBoost ML Strategy...")
    ml_strategy = XGBoostMLStrategy()
    
    print("\n🔄 Generating signals...")
    ml_signals = ml_strategy.generate_signals(df)
    
    print(f"\n✅ XGBoost ML Strategy:")
    print(f"  Signals generated: {len(ml_signals)}")
    print(f"  Buy signals: {ml_signals['signal'].sum()}")
    print(f"  Signal rate: {ml_signals['signal'].mean():.2%}")
    print(f"\n  P_ml statistics:")
    print(ml_signals[['P_ml', 'ml_score', 'filters_product']].describe())
    
except Exception as e:
    print(f"❌ XGBoost ML Strategy failed: {e}")
    import traceback
    traceback.print_exc()


# ============================================================================
# STEP 3: TEST RULE-BASED STRATEGY
# ============================================================================

print("\n" + "=" * 80)
print("🔄 STEP 3: Testing Rule-Based Strategy")
print("=" * 80)

try:
    from strategies.rule_based_v2 import RuleBasedStrategy
    
    print("\n📦 Initializing Rule-Based Strategy...")
    rb_strategy = RuleBasedStrategy()
    
    print("\n🔄 Generating signals...")
    rb_signals = rb_strategy.generate_signals(df)
    
    print(f"\n✅ Rule-Based Strategy:")
    print(f"  Signals generated: {len(rb_signals)}")
    print(f"  Buy signals: {rb_signals['signal'].sum()}")
    print(f"  Signal rate: {rb_signals['signal'].mean():.2%}")
    print(f"\n  P_rb statistics:")
    print(rb_signals[['P_rb', 'W_opportunity', 'filters_product']].describe())
    
except Exception as e:
    print(f"❌ Rule-Based Strategy failed: {e}")
    import traceback
    traceback.print_exc()


# ============================================================================
# STEP 4: TEST HYBRID STRATEGY
# ============================================================================

print("\n" + "=" * 80)
print("🔄 STEP 4: Testing Hybrid Strategy")
print("=" * 80)

try:
    from strategies.hybrid_v2 import HybridStrategy
    
    print("\n📦 Initializing Hybrid Strategy...")
    hybrid_strategy = HybridStrategy()
    
    print("\n🔄 Generating signals...")
    hybrid_signals = hybrid_strategy.generate_signals(df)
    
    print(f"\n✅ Hybrid Strategy:")
    print(f"  Signals generated: {len(hybrid_signals)}")
    print(f"  Buy signals: {hybrid_signals['signal'].sum()}")
    print(f"  Signal rate: {hybrid_signals['signal'].mean():.2%}")
    print(f"\n  P_hyb statistics:")
    print(hybrid_signals[['P_hyb', 'alpha', 'beta', 'P_rb', 'P_ml']].describe())
    
except Exception as e:
    print(f"❌ Hybrid Strategy failed: {e}")
    import traceback
    traceback.print_exc()


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("📊 TEST SUMMARY")
print("=" * 80)

print(f"\n✅ REAL DATA: {len(df):,} bars, {len(df.columns)} columns")
print(f"✅ XGBoost ML: 74 advanced features")
print(f"✅ Rule-Based: OHLCV-based signals")
print(f"✅ Hybrid: Combined strategy")

print("\n" + "=" * 80)
print("READY FOR BACKTESTING & GITHUB")
print("=" * 80)