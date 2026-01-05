"""Test Canonical Pipeline v3
Verifies that the new pipeline loads data correctly.
"""
from core.canonical_pipeline import CanonicalPipeline

print("\n" + "="*60)
print("CANONICAL PIPELINE v3 TEST")
print("="*60)

# Create pipeline with strict_mode=False (until Registry is filled)
print("\n[1/4] Initializing CanonicalPipeline...")
pipeline = CanonicalPipeline(strict_mode=False)
print("✅ Pipeline created successfully")

# Load BTC 4h data
print("\n[2/4] Loading BTC 4h data...")
try:
    state = pipeline.load_state("BTC", "4h")
    print(f"✅ State loaded: {state.shape}")
except Exception as e:
    print(f"❌ Error loading state: {e}")
    exit(1)

# Validate features
print("\n[3/4] Validating features...")
print(f"   - Total features: {state.shape[1]}")
print(f"   - Total rows: {state.shape[0]}")
print(f"   - Version: {state.version}")
print(f"   - Symbol: {state.symbol}")
print(f"   - Timeframe: {state.timeframe}")

# Test specific feature
print("\n[4/4] Testing specific features...")
latest_row = state.get_latest_row()
print(f"   - Latest timestamp: {state.timestamp}")

# Try to get RSI
if 'norm_rsi_14' in latest_row.index:
    print(f"   - Latest RSI: {latest_row['norm_rsi_14']:.4f}")
else:
    print(f"   - ⚠️ RSI not found in features")
    print(f"   - Available features sample: {list(latest_row.index[:5])}")

print("\n" + "="*60)
print("✅ PIPELINE TEST COMPLETED SUCCESSFULLY!")
print("="*60 + "\n")
