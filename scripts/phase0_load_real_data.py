"""
PHASE 0: LOAD REAL DATA
========================

Load and prepare ALL data needed for comprehensive validation:
- BTC/USDT 1h 2018-2025 (already have)
- Mark major crash events (COVID, Luna, FTX)
- Validate data quality
- Prepare for component testing

Author: Scarlet Sails Team
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

print("="*80)
print("PHASE 0: LOAD & PREPARE REAL DATA")
print("="*80)

# ============================================================================
# LOAD EXISTING DATA
# ============================================================================
print("\n📂 Step 1: Load existing BTC data...")

data_path = Path('data/raw/BTC_USDT_1h_FULL.parquet')

if not data_path.exists():
    print(f"❌ ERROR: {data_path} not found!")
    print("\nExpected location:")
    print(f"   {data_path.absolute()}")
    print("\nPlease ensure data file exists.")
    exit(1)

df = pd.read_parquet(data_path)
print(f"✅ Loaded {len(df):,} bars")
print(f"   Period: {df.index[0]} to {df.index[-1]}")
print(f"   Days: {(df.index[-1] - df.index[0]).days}")
print(f"   Price range: ${df['close'].min():.0f} - ${df['close'].max():.0f}")

# ============================================================================
# MARK MAJOR CRASH EVENTS
# ============================================================================
print("\n📊 Step 2: Mark major crash events...")

# Define known crash events (UTC timestamps)
crash_events = {
    'COVID_CRASH': {
        'start': pd.Timestamp('2020-03-12 00:00:00', tz='UTC'),
        'end': pd.Timestamp('2020-03-13 23:00:00', tz='UTC'),
        'description': 'COVID-19 Black Thursday (-50% in 24h)',
        'severity': 'EXTREME',
    },
    'LUNA_COLLAPSE': {
        'start': pd.Timestamp('2022-05-09 00:00:00', tz='UTC'),
        'end': pd.Timestamp('2022-05-12 23:00:00', tz='UTC'),
        'description': 'Terra/Luna ecosystem collapse (-60%)',
        'severity': 'EXTREME',
    },
    'FTX_COLLAPSE': {
        'start': pd.Timestamp('2022-11-08 00:00:00', tz='UTC'),
        'end': pd.Timestamp('2022-11-10 23:00:00', tz='UTC'),
        'description': 'FTX bankruptcy (-25%)',
        'severity': 'HIGH',
    },
    'CHINA_BAN': {
        'start': pd.Timestamp('2021-05-19 00:00:00', tz='UTC'),
        'end': pd.Timestamp('2021-05-20 23:00:00', tz='UTC'),
        'description': 'China mining ban announcement (-30%)',
        'severity': 'HIGH',
    },
    'EVERGRANDE': {
        'start': pd.Timestamp('2021-09-20 00:00:00', tz='UTC'),
        'end': pd.Timestamp('2021-09-21 23:00:00', tz='UTC'),
        'description': 'Evergrande default fears (-15%)',
        'severity': 'MEDIUM',
    },
}

# Ensure df index is timezone-aware
if df.index.tz is None:
    df.index = df.index.tz_localize('UTC')

# Mark crash periods
df['is_crash'] = False
df['crash_event'] = None
df['crash_severity'] = None

for event_name, event_data in crash_events.items():
    mask = (df.index >= event_data['start']) & (df.index <= event_data['end'])
    df.loc[mask, 'is_crash'] = True
    df.loc[mask, 'crash_event'] = event_name
    df.loc[mask, 'crash_severity'] = event_data['severity']

    n_bars = mask.sum()
    if n_bars > 0:
        print(f"   ✅ {event_name}: {n_bars} bars marked")
        print(f"      {event_data['description']}")
    else:
        print(f"   ⚠️  {event_name}: NOT FOUND in data range")

crash_bars = df['is_crash'].sum()
crash_pct = crash_bars / len(df) * 100
print(f"\n   Total crash bars: {crash_bars} ({crash_pct:.1f}%)")

# ============================================================================
# DATA QUALITY CHECKS
# ============================================================================
print("\n🔍 Step 3: Data quality validation...")

# Check for missing values
missing = df.isnull().sum()
if missing.any():
    print("   ⚠️  Missing values detected:")
    for col, count in missing[missing > 0].items():
        print(f"      {col}: {count} ({count/len(df)*100:.1f}%)")
else:
    print("   ✅ No missing values")

# Check for zero/negative prices
zero_prices = (df['close'] <= 0).sum()
if zero_prices > 0:
    print(f"   ❌ {zero_prices} bars with zero/negative prices!")
else:
    print("   ✅ All prices positive")

# Check for extreme price jumps (>50% in 1 bar = likely error)
price_change = df['close'].pct_change().abs()
extreme_jumps = (price_change > 0.5).sum()
if extreme_jumps > 0:
    print(f"   ⚠️  {extreme_jumps} bars with >50% price jump")
    # Show examples
    extreme_idx = price_change[price_change > 0.5].index[:3]
    for idx in extreme_idx:
        prev_idx = df.index[df.index.get_loc(idx) - 1]
        prev_price = df.loc[prev_idx, 'close']
        curr_price = df.loc[idx, 'close']
        change = (curr_price - prev_price) / prev_price * 100
        print(f"      {idx}: ${prev_price:.0f} → ${curr_price:.0f} ({change:+.1f}%)")
else:
    print("   ✅ No extreme price jumps")

# Check for gaps in timeline (missing bars)
expected_freq = '1H'
expected_bars = (df.index[-1] - df.index[0]) / pd.Timedelta(expected_freq)
actual_bars = len(df)
missing_bars = int(expected_bars - actual_bars)

if missing_bars > 100:  # Allow small gaps
    print(f"   ⚠️  ~{missing_bars:,} bars missing from timeline")
    gap_pct = missing_bars / expected_bars * 100
    print(f"      Gap: {gap_pct:.1f}% of expected data")
else:
    print("   ✅ Timeline complete")

# ============================================================================
# CALCULATE BASIC FEATURES (for component tests)
# ============================================================================
print("\n📈 Step 4: Calculate basic features...")

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

# RSI
df['rsi'] = calculate_rsi(df['close'], 14)
print("   ✅ RSI calculated")

# ATR
df['atr'] = calculate_atr(df, 14)
print("   ✅ ATR calculated")

# Moving averages
df['ma20'] = df['close'].rolling(20).mean()
df['ma50'] = df['close'].rolling(50).mean()
df['ma200'] = df['close'].rolling(200).mean()
print("   ✅ MAs calculated (20, 50, 200)")

# Volatility
df['volatility'] = df['close'].pct_change().rolling(24).std()
print("   ✅ Volatility calculated")

# Volume
df['volume_ma'] = df['volume'].rolling(24).mean()
df['volume_ratio'] = df['volume'] / df['volume_ma']
print("   ✅ Volume features calculated")

# Drop NaN rows
initial_len = len(df)
df = df.dropna()
dropped = initial_len - len(df)
print(f"   Dropped {dropped} rows with NaN (after calculations)")

# ============================================================================
# SAVE PREPARED DATA
# ============================================================================
print("\n💾 Step 5: Save prepared data...")

output_path = Path('data/processed/btc_prepared_phase0.parquet')
output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_parquet(output_path)
print(f"   ✅ Saved to: {output_path}")
print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PHASE 0 COMPLETE - DATA SUMMARY")
print("="*80)

print(f"\n📊 Dataset:")
print(f"   Bars: {len(df):,}")
print(f"   Period: {df.index[0]} to {df.index[-1]}")
print(f"   Days: {(df.index[-1] - df.index[0]).days:,}")
print(f"   Price range: ${df['close'].min():.0f} - ${df['close'].max():.0f}")

print(f"\n💥 Crash Events:")
for event_name, event_data in crash_events.items():
    mask = df['crash_event'] == event_name
    if mask.sum() > 0:
        event_df = df[mask]
        start_price = event_df['close'].iloc[0]
        min_price = event_df['close'].min()
        drop = (min_price - start_price) / start_price * 100
        print(f"   {event_name}: {mask.sum()} bars, {drop:.1f}% drop")

print(f"\n📈 Features Available:")
features = ['rsi', 'atr', 'ma20', 'ma50', 'ma200', 'volatility', 'volume_ratio']
for feat in features:
    print(f"   ✅ {feat}")

print(f"\n✅ Ready for Phase 1 component testing!")
print("="*80)
