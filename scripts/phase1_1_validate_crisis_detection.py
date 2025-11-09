"""
PHASE 1.1: CRISIS DETECTION VALIDATION
=======================================

Test crisis detection component on REAL crashes:
- COVID crash (March 2020)
- Luna collapse (May 2022)
- FTX collapse (Nov 2022)

Metrics:
- Detection time (before/after crash start)
- False positives (normal volatility flagged as crisis)
- False negatives (missed crashes)
- Severity scoring accuracy

Expected performance:
- Detection within 1-2h of crash start
- <5% false positives on normal periods
- 0% false negatives (catch ALL crashes)

Author: Scarlet Sails Team
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from models.crisis_classifier import CrisisClassifier

print("="*80)
print("PHASE 1.1: CRISIS DETECTION VALIDATION")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📂 Loading prepared data...")

data_path = Path('data/processed/btc_prepared_phase0.parquet')
if not data_path.exists():
    print(f"❌ ERROR: {data_path} not found!")
    print("Run phase0_load_real_data.py first!")
    exit(1)

df = pd.read_parquet(data_path)
print(f"✅ Loaded {len(df):,} bars")

# ============================================================================
# INITIALIZE CRISIS DETECTOR
# ============================================================================
print("\n🔍 Initializing crisis detector...")

try:
    crisis_detector = CrisisClassifier()
    print("✅ CrisisClassifier loaded")
except Exception as e:
    print(f"❌ Failed to load CrisisClassifier: {e}")
    print("\nWill use simple rule-based detector for testing...")

    # Fallback: simple rule-based detector
    class SimpleCrisisDetector:
        def detect(self, df, bar_idx):
            """Simple crisis detection: volatility spike + volume surge"""
            if bar_idx < 24:
                return {'is_crisis': False, 'confidence': 0.0, 'severity': 'NONE'}

            # Recent volatility vs baseline
            recent_vol = df['close'].iloc[bar_idx-24:bar_idx].pct_change().std()
            baseline_vol = df['close'].iloc[bar_idx-168:bar_idx-24].pct_change().std()

            # Recent volume surge
            recent_volume = df['volume'].iloc[bar_idx-24:bar_idx].mean()
            baseline_volume = df['volume'].iloc[bar_idx-168:bar_idx-24].mean()

            # Price drop
            price_change_24h = (df['close'].iloc[bar_idx] - df['close'].iloc[bar_idx-24]) / df['close'].iloc[bar_idx-24]

            # Crisis if: high vol spike + volume surge + price drop
            vol_spike = recent_vol / baseline_vol if baseline_vol > 0 else 1.0
            vol_surge = recent_volume / baseline_volume if baseline_volume > 0 else 1.0

            is_crisis = (vol_spike > 3.0 and vol_surge > 2.0 and price_change_24h < -0.15)

            if is_crisis:
                if vol_spike > 5.0 and price_change_24h < -0.3:
                    severity = 'EXTREME'
                    confidence = 0.9
                elif vol_spike > 4.0 and price_change_24h < -0.2:
                    severity = 'HIGH'
                    confidence = 0.7
                else:
                    severity = 'MEDIUM'
                    confidence = 0.5
            else:
                severity = 'NONE'
                confidence = 0.0

            return {
                'is_crisis': is_crisis,
                'confidence': confidence,
                'severity': severity,
                'vol_spike': vol_spike,
                'vol_surge': vol_surge,
                'price_drop': price_change_24h
            }

    crisis_detector = SimpleCrisisDetector()
    print("✅ Using SimpleCrisisDetector (rule-based)")

# ============================================================================
# TEST 1: KNOWN CRASH EVENTS
# ============================================================================
print("\n" + "="*80)
print("TEST 1: VALIDATE ON KNOWN CRASH EVENTS")
print("="*80)

crash_events = {
    'COVID_CRASH': pd.Timestamp('2020-03-12 00:00:00', tz='UTC'),
    'LUNA_COLLAPSE': pd.Timestamp('2022-05-09 00:00:00', tz='UTC'),
    'FTX_COLLAPSE': pd.Timestamp('2022-11-08 00:00:00', tz='UTC'),
}

results = []

for event_name, event_start in crash_events.items():
    print(f"\n📊 Testing {event_name}...")
    print(f"   Known start: {event_start}")

    # Find event in data
    if event_start not in df.index:
        # Find closest timestamp
        closest_idx = (df.index - event_start).abs().argmin()
        event_start = df.index[closest_idx]
        print(f"   Adjusted to: {event_start}")

    event_bar = df.index.get_loc(event_start)

    # Test detection 24h before and after event start
    test_range = range(max(0, event_bar - 24), min(len(df), event_bar + 48))

    detections = []
    for bar in test_range:
        result = crisis_detector.detect(df, bar)

        if result['is_crisis']:
            timestamp = df.index[bar]
            hours_from_start = (timestamp - event_start).total_seconds() / 3600

            detections.append({
                'timestamp': timestamp,
                'hours_from_start': hours_from_start,
                'confidence': result['confidence'],
                'severity': result['severity'],
            })

    if detections:
        # Find first detection
        first_detection = min(detections, key=lambda x: x['hours_from_start'])

        print(f"   ✅ DETECTED!")
        print(f"      First detection: {first_detection['hours_from_start']:.1f}h from start")
        print(f"      Confidence: {first_detection['confidence']:.1%}")
        print(f"      Severity: {first_detection['severity']}")
        print(f"      Total detections in 72h window: {len(detections)}")

        results.append({
            'event': event_name,
            'detected': True,
            'detection_time_hours': first_detection['hours_from_start'],
            'confidence': first_detection['confidence'],
            'severity': first_detection['severity'],
        })
    else:
        print(f"   ❌ NOT DETECTED in 72h window!")
        results.append({
            'event': event_name,
            'detected': False,
            'detection_time_hours': np.nan,
            'confidence': 0.0,
            'severity': 'NONE',
        })

# ============================================================================
# TEST 2: FALSE POSITIVES (NORMAL PERIODS)
# ============================================================================
print("\n" + "="*80)
print("TEST 2: FALSE POSITIVE RATE (NORMAL PERIODS)")
print("="*80)

# Test on periods outside crashes (sample 1000 bars)
normal_bars = df[~df['is_crash']]

if len(normal_bars) > 1000:
    # Sample evenly across dataset
    sample_indices = np.linspace(200, len(normal_bars)-200, 1000, dtype=int)
    test_bars = [normal_bars.index.get_loc(normal_bars.index[i]) for i in sample_indices]
else:
    test_bars = range(200, len(normal_bars)-200)

print(f"\nTesting {len(test_bars):,} bars from normal periods...")

false_positives = 0
for bar in test_bars:
    result = crisis_detector.detect(df, bar)
    if result['is_crisis']:
        false_positives += 1

false_positive_rate = false_positives / len(test_bars)

print(f"\nFalse Positive Rate: {false_positive_rate:.1%}")
print(f"   {false_positives} / {len(test_bars)} normal bars flagged as crisis")

if false_positive_rate < 0.05:
    print("   ✅ PASS: <5% false positives")
elif false_positive_rate < 0.10:
    print("   ⚠️  WARNING: 5-10% false positives")
else:
    print("   ❌ FAIL: >10% false positives - detector too sensitive!")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PHASE 1.1 RESULTS SUMMARY")
print("="*80)

results_df = pd.DataFrame(results)

print("\n📊 Crash Detection Results:")
print(results_df.to_string(index=False))

# Calculate metrics
detected_count = results_df['detected'].sum()
total_crashes = len(results_df)

print(f"\n🎯 Overall Metrics:")
print(f"   Detection Rate: {detected_count}/{total_crashes} ({detected_count/total_crashes:.0%})")

if detected_count > 0:
    avg_detection_time = results_df[results_df['detected']]['detection_time_hours'].mean()
    print(f"   Avg Detection Time: {avg_detection_time:.1f}h from crash start")

    avg_confidence = results_df[results_df['detected']]['confidence'].mean()
    print(f"   Avg Confidence: {avg_confidence:.1%}")

print(f"   False Positive Rate: {false_positive_rate:.1%}")

# Pass/Fail criteria
print(f"\n✅ PASS/FAIL:")

passed = True

if detected_count == total_crashes:
    print(f"   ✅ All crashes detected (0% false negatives)")
else:
    print(f"   ❌ Missed {total_crashes - detected_count} crashes!")
    passed = False

if detected_count > 0:
    if avg_detection_time <= 2.0:
        print(f"   ✅ Detection within 2h average")
    elif avg_detection_time <= 4.0:
        print(f"   ⚠️  Detection 2-4h average (acceptable)")
    else:
        print(f"   ❌ Detection >4h average (too slow!)")
        passed = False

if false_positive_rate < 0.05:
    print(f"   ✅ False positive rate <5%")
elif false_positive_rate < 0.10:
    print(f"   ⚠️  False positive rate 5-10% (acceptable)")
else:
    print(f"   ❌ False positive rate >10% (too high!)")
    passed = False

print("\n" + "="*80)
if passed:
    print("✅ PHASE 1.1 PASSED - Crisis detection validated!")
else:
    print("❌ PHASE 1.1 FAILED - Crisis detection needs improvement")
print("="*80)
