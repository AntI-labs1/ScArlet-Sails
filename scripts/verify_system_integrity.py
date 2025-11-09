"""
SYSTEM INTEGRITY VERIFICATION
==============================

Проверяет что вся система синхронизирована и готова к работе:
- Git status и commits
- Все необходимые файлы на месте
- Dependencies установлены
- Data files существуют и корректны
- Scripts могут запускаться

Author: Scarlet Sails Team
"""

import sys
from pathlib import Path
import subprocess

print("="*80)
print("SYSTEM INTEGRITY VERIFICATION")
print("="*80)

all_checks_passed = True

# ============================================================================
# CHECK 1: GIT STATUS
# ============================================================================
print("\n📋 CHECK 1: Git Repository Status")
print("-"*80)

try:
    # Check current branch
    result = subprocess.run(['git', 'branch', '--show-current'],
                          capture_output=True, text=True, check=True)
    current_branch = result.stdout.strip()
    print(f"✅ Current branch: {current_branch}")

    expected_branch = "claude/debug-naive-strategy-performance-011CUx4E9miJvu4k8Sn2gBkH"
    if current_branch == expected_branch:
        print(f"   ✅ Correct branch!")
    else:
        print(f"   ⚠️  Expected: {expected_branch}")
        print(f"   Consider: git checkout {expected_branch}")

    # Check git status
    result = subprocess.run(['git', 'status', '--porcelain'],
                          capture_output=True, text=True, check=True)
    if not result.stdout.strip():
        print("✅ Working tree clean (no uncommitted changes)")
    else:
        print("⚠️  Uncommitted changes detected:")
        print(result.stdout)

    # Check latest commits
    result = subprocess.run(['git', 'log', '--oneline', '-5'],
                          capture_output=True, text=True, check=True)
    print("\n📝 Latest 5 commits:")
    for line in result.stdout.strip().split('\n'):
        print(f"   {line}")

except Exception as e:
    print(f"❌ Git check failed: {e}")
    all_checks_passed = False

# ============================================================================
# CHECK 2: REQUIRED FILES
# ============================================================================
print("\n📋 CHECK 2: Required Files Existence")
print("-"*80)

required_files = {
    'Phase 0': [
        'scripts/phase0_load_real_data.py',
    ],
    'Phase 1': [
        'scripts/phase1_1_validate_crisis_detection.py',
        'scripts/phase1_2_validate_regime_detection.py',
        'scripts/phase1_3_validate_entry_signals.py',
        'scripts/phase1_5_validate_ml_models.py',
        'scripts/comprehensive_exit_test_REAL.py',
    ],
    'Phase 2-4': [
        'scripts/phase2_walk_forward_validation.py',
        'scripts/phase3_root_cause_analysis.py',
        'scripts/phase4_decision_matrix.py',
    ],
    'Master': [
        'scripts/run_comprehensive_audit.py',
    ],
    'Models': [
        'models/position_manager.py',
        'models/hybrid_position_manager.py',
        'models/regime_detector.py',
        'models/exit_strategy.py',
        'models/crisis_classifier.py',
        'models/decision_formula_v2.py',
        'models/xgboost_model.py',
    ],
    'Documentation': [
        'docs/COMPREHENSIVE_AUDIT_GUIDE.md',
    ],
}

missing_files = []

for category, files in required_files.items():
    print(f"\n{category}:")
    for filepath in files:
        path = Path(filepath)
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"   ✅ {filepath} ({size_kb:.1f} KB)")
        else:
            print(f"   ❌ {filepath} MISSING!")
            missing_files.append(filepath)
            all_checks_passed = False

if missing_files:
    print(f"\n❌ {len(missing_files)} file(s) missing!")
    print("   Run: git pull origin claude/debug-naive-strategy-performance-011CUx4E9miJvu4k8Sn2gBkH")
else:
    print(f"\n✅ All required files present!")

# ============================================================================
# CHECK 3: DATA FILES
# ============================================================================
print("\n📋 CHECK 3: Data Files")
print("-"*80)

data_files = {
    'Raw Data': 'data/raw/BTC_USDT_1h_FULL.parquet',
    'Processed Data': 'data/processed/btc_prepared_phase0.parquet',
}

for name, filepath in data_files.items():
    path = Path(filepath)
    if path.exists():
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"✅ {name}: {filepath}")
        print(f"   Size: {size_mb:.1f} MB")

        # Quick validation
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            print(f"   Bars: {len(df):,}")
            print(f"   Period: {df.index[0]} to {df.index[-1]}")

            # Expected values
            if 'FULL' in filepath:
                expected_bars = 71238
                if len(df) == expected_bars:
                    print(f"   ✅ Correct size ({expected_bars} bars)")
                else:
                    print(f"   ⚠️  Expected {expected_bars} bars, got {len(df)}")
            elif 'prepared' in filepath:
                expected_bars_min = 70000
                if len(df) >= expected_bars_min:
                    print(f"   ✅ Correct size (>={expected_bars_min} bars)")
                else:
                    print(f"   ⚠️  Expected >={expected_bars_min} bars, got {len(df)}")
                    all_checks_passed = False
        except Exception as e:
            print(f"   ❌ Failed to load: {e}")
            all_checks_passed = False
    else:
        print(f"❌ {name}: {filepath} NOT FOUND!")
        if 'FULL' in filepath:
            print(f"   This is YOUR local data file - ensure it exists")
            all_checks_passed = False
        elif 'prepared' in filepath:
            print(f"   Run: python scripts/phase0_load_real_data.py")
            all_checks_passed = False

# ============================================================================
# CHECK 4: PYTHON DEPENDENCIES
# ============================================================================
print("\n📋 CHECK 4: Python Dependencies")
print("-"*80)

required_packages = {
    'pandas': '2.0.0',
    'numpy': '1.20.0',
    'xgboost': '1.0.0',
    'sklearn': '1.0.0',
}

for package, min_version in required_packages.items():
    try:
        if package == 'sklearn':
            import sklearn
            module = sklearn
            package_name = 'scikit-learn'
        else:
            module = __import__(package)
            package_name = package

        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name}: {version}")
    except ImportError:
        print(f"❌ {package_name}: NOT INSTALLED!")
        print(f"   Install: pip install {package_name}")
        all_checks_passed = False

# ============================================================================
# CHECK 5: MODEL IMPORTS
# ============================================================================
print("\n📋 CHECK 5: Model Imports")
print("-"*80)

sys.path.insert(0, str(Path(__file__).parent.parent))

models_to_test = [
    ('models.position_manager', 'PositionManager'),
    ('models.hybrid_position_manager', 'HybridPositionManager'),
    ('models.regime_detector', 'SimpleRegimeDetector'),
    ('models.exit_strategy', 'AdaptiveStopLoss'),
    ('models.crisis_classifier', 'CrisisClassifier'),
    ('models.decision_formula_v2', 'DecisionFormulaV2'),
]

for module_name, class_name in models_to_test:
    try:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        print(f"✅ {module_name}.{class_name}")
    except Exception as e:
        print(f"❌ {module_name}.{class_name}: {e}")
        all_checks_passed = False

# ============================================================================
# CHECK 6: RESULTS VALIDATION
# ============================================================================
print("\n📋 CHECK 6: Results Validation (from Phase 2)")
print("-"*80)

expected_results = {
    'Total years tested': 9,
    'Hybrid total P&L': '+256%',
    'Hybrid annual return': '~28.4%',
    'Profitable years': '8/9 (88.9%)',
    'Best year': '2020 (+62.1%)',
    'Worst year': '2022 (-11.8%)',
}

print("Expected results from your Phase 2 run:")
for key, value in expected_results.items():
    print(f"   {key}: {value}")

print("\n⚠️  NOTE: These should match your Phase 2 output!")
print("   If numbers are different, data may be out of sync.")

# ============================================================================
# CHECK 7: DIRECTORY STRUCTURE
# ============================================================================
print("\n📋 CHECK 7: Directory Structure")
print("-"*80)

required_dirs = [
    'scripts',
    'models',
    'data/raw',
    'data/processed',
    'docs',
    'reports',
]

for dirname in required_dirs:
    path = Path(dirname)
    if path.exists() and path.is_dir():
        file_count = len(list(path.glob('*')))
        print(f"✅ {dirname}/ ({file_count} items)")
    else:
        print(f"❌ {dirname}/ NOT FOUND!")
        all_checks_passed = False

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)

if all_checks_passed:
    print("\n✅ ALL CHECKS PASSED!")
    print()
    print("Your system is fully synchronized and ready!")
    print()
    print("Next steps:")
    print("   1. Review Phase 2-4 results")
    print("   2. Discuss optimization strategy")
    print("   3. Implement Quick Win #1 (entry improvements)")
    print()
else:
    print("\n⚠️  SOME CHECKS FAILED!")
    print()
    print("Fix issues above, then re-run this script.")
    print()
    print("Common fixes:")
    print("   - git pull origin claude/debug-naive-strategy-performance-011CUx4E9miJvu4k8Sn2gBkH")
    print("   - python scripts/phase0_load_real_data.py")
    print("   - pip install -r requirements.txt")
    print()

print("="*80)
