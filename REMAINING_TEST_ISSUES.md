# Remaining Test Issues (8 FAILED)

## Status: 172 PASSED, 8 FAILED (95.6% pass rate)

## 1. Data Loader Issues (5 tests) - FILE NOT FOUND

### Problem:
Missing data files: `data/raw/BTC_USDT_15m.parquet`, `ALGO_USDT_15m.parquet`, `AVAX_USDT_15m.parquet`

### Failed Tests:
- `test_load_btc_15m` - FileNotFoundError
- `test_load_with_date_filter` - FileNotFoundError
- `test_ohlc_relationships` - FileNotFoundError
- `test_no_negative_values` - FileNotFoundError
- `test_load_multiple_coins` - ValueError (all coins failed to load)

### Solution:
```bash
# Option 1: Create mock test fixtures
mkdir -p tests/fixtures/data
python -c "
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate mock BTC data
dates = pd.date_range(start='2024-01-01', periods=1000, freq='15min')
df = pd.DataFrame({
    'timestamp': dates,
    'open': np.random.uniform(40000, 45000, 1000),
    'high': np.random.uniform(45000, 50000, 1000),
    'low': np.random.uniform(35000, 40000, 1000),
    'close': np.random.uniform(40000, 45000, 1000),
    'volume': np.random.uniform(100, 1000, 1000)
})
df.to_parquet('tests/fixtures/data/BTC_USDT_15m.parquet')
df.to_parquet('tests/fixtures/data/ALGO_USDT_15m.parquet')
df.to_parquet('tests/fixtures/data/AVAX_USDT_15m.parquet')
print('Mock data created!')
"

# Option 2: Update tests to use mock data path
# Modify tests/test_data_loader.py to use fixtures
```

---

## 2. Dispersion Logic Issues (3 tests) - INVERTED LOGIC BUG

### Problem:
**CRITICAL:** The `confidence_multiplier` logic is INVERTED!
- Perfect agreement (low dispersion) gives LOW multiplier (0.3)
- High disagreement (high dispersion) gives HIGH multiplier (1.5)

**This is backwards!** High confidence (agreement) should give HIGH multiplier.

### Failed Tests:

#### a) `test_high_dispersion_gives_high_multiplier`
```
AssertionError: High disp mult 1.5 should be > low disp mult 1.5
assert 1.5 > 1.5  # Both are 1.5!
```
**Issue:** High and low dispersion produce same multiplier

#### b) `test_perfect_agreement_low_dispersion`
```
assert 0.3 == 1.5  # FAIL!
```
**Issue:** Perfect agreement (std ≈ 0) gives min_mult=0.3, expected max_mult=1.5

#### c) `test_high_disagreement_high_dispersion`
```
assert 1.5 < 1.0  # FAIL!
```
**Issue:** High disagreement (std=0.245) gives max_mult=1.5, expected <1.0

### Root Cause Analysis:
File: `core/rolling_dispersion.py`

Current (WRONG) logic:
```python
# Low dispersion -> Low multiplier (WRONG!)
# High dispersion -> High multiplier (WRONG!)
```

### Solution:
```python
# Fix in core/rolling_dispersion.py
# Around line 80-90 in _calculate_multiplier()

# BEFORE (WRONG):
# multiplier = self.min_mult + (percentile * (self.max_mult - self.min_mult))

# AFTER (CORRECT - INVERT):
multiplier = self.max_mult - (percentile * (self.max_mult - self.min_mult))

# This makes:
# - Low percentile (low dispersion, high agreement) -> HIGH multiplier ✓
# - High percentile (high dispersion, low agreement) -> LOW multiplier ✓
```

### Fix Command:
```bash
# 1. Edit core/rolling_dispersion.py
# Find the _calculate_multiplier method
# Invert the formula as shown above

# 2. Test the fix
pytest tests/test_dispersion_inverted.py tests/test_rolling_dispersion.py -v

# 3. Commit
git add core/rolling_dispersion.py
git commit -m "fix(dispersion): invert confidence_multiplier logic - high agreement = high multiplier"
git push
```

---

## Priority Actions:

1. **HIGH PRIORITY:** Fix dispersion logic inversion (affects trading confidence!)
2. **LOW PRIORITY:** Add mock test data fixtures (doesn't affect production)

## Files to Fix:
- [ ] `core/rolling_dispersion.py` - Invert multiplier calculation
- [ ] `tests/fixtures/` - Add mock data files (optional)
