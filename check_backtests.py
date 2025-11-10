"""
Check which backtest scripts can actually run
"""
import os
from pathlib import Path

print("=" * 80)
print("🔍 CHECKING WHICH BACKTESTS CAN RUN")
print("=" * 80)
print()

backtests = {
    'master_comprehensive_audit.py': 'data/raw/{ASSET}USDT_{TF}.parquet',
    'day11_forensic_analysis.py': 'data/processed/btc_prepared_phase0.parquet',
    'test_week1_improvements.py': 'data/raw/BTC_USDT_1h_FULL.parquet',
    'comprehensive_exit_test_REAL.py': '?',
}

print("📋 BACKTEST SCRIPTS vs DATA REQUIREMENTS:")
print("-" * 80)

for script, data_req in backtests.items():
    script_path = Path('scripts') / script
    exists = "✅" if script_path.exists() else "❌"
    
    print(f"{exists} {script}")
    print(f"   Needs: {data_req}")
    
    # Check if data exists
    if '{ASSET}' in data_req:
        # Check for BTC example
        data_file = Path(data_req.replace('{ASSET}', 'BTC').replace('{TF}', '1h'))
        data_exists = "✅" if data_file.exists() else "❌ MISSING"
        print(f"   Data: {data_exists}")
    else:
        data_file = Path(data_req)
        data_exists = "✅" if data_file.exists() else "❌ MISSING"
        print(f"   Data: {data_exists}")
    
    print()

print("=" * 80)
print("🎯 CONCLUSION:")
print("=" * 80)

# Check what data we actually have
raw_files = list(Path('data/raw').glob('*.parquet')) if Path('data/raw').exists() else []
processed_files = list(Path('data/processed').glob('*.parquet')) if Path('data/processed').exists() else []

print(f"Raw parquet files: {len(raw_files)}")
if raw_files:
    for f in raw_files[:5]:
        print(f"  • {f.name}")

print(f"\nProcessed parquet files: {len(processed_files)}")
if processed_files:
    for f in processed_files[:5]:
        print(f"  • {f.name}")

print()

if len(raw_files) == 0 and len(processed_files) == 0:
    print("❌ NO DATA AVAILABLE - Cannot run any backtest!")
    print()
    print("To create results, one of these must have been run BEFORE:")
    print("  1. Data was downloaded")
    print("  2. Backtests were run")
    print("  3. Results saved to reports/")
    print()
    print("Current results in reports/ are from PREVIOUS runs.")

