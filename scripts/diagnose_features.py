"""
SCRIPT: DIAGNOSE FEATURES
Показывает, какие именно колонки доступны в CanonicalState,
чтобы мы могли правильно настроить RL Agent.
"""
import sys
import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.canonical_pipeline import CanonicalPipeline

def main():
    print("🔍 DIAGNOSING FEATURES...")
    
    pipeline = CanonicalPipeline(strict_mode=False)
    try:
        # Грузим BTC 4h (на нем мы учили агента)
        state = pipeline.load_state("BTC", "4h")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    cols = list(state.features.columns)
    print(f"\n✅ Total Features: {len(cols)}")
    
    print("\n--- VOLATILITY CANDIDATES ---")
    vol_cols = [c for c in cols if 'atr' in c.lower() or 'vol' in c.lower() or 'std' in c.lower()]
    for c in vol_cols:
        sample = state.features[c].iloc[-1]
        print(f"  {c}: {sample:.4f}")

    print("\n--- TREND CANDIDATES ---")
    trend_cols = [c for c in cols if 'trend' in c.lower() or 'sma' in c.lower() or 'ema' in c.lower() or 'regime' in c.lower()]
    for c in trend_cols:
        sample = state.features[c].iloc[-1]
        print(f"  {c}: {sample:.4f}")
        
    print("\n--- REGIME CANDIDATES ---")
    reg_cols = [c for c in cols if 'regime' in c.lower()]
    for c in reg_cols:
        sample = state.features[c].iloc[-1]
        print(f"  {c}: {sample:.4f}")

if __name__ == "__main__":
    main()
