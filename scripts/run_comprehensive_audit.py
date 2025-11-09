"""
COMPREHENSIVE SYSTEM AUDIT - MASTER SCRIPT
===========================================

Runs all 4 phases of system validation:

PHASE 0: Load REAL data
PHASE 1: Component validation (1.1-1.5)
PHASE 2: Walk-forward validation
PHASE 3: Root cause analysis
PHASE 4: Decision matrix

Usage:
    python scripts/run_comprehensive_audit.py [--phase PHASE_NUMBER]

    --phase: Run specific phase only (0, 1, 2, 3, 4)
             Default: run all phases

Author: Scarlet Sails Team
"""

import subprocess
import sys
from pathlib import Path
import argparse
from datetime import datetime

def run_script(script_name, description):
    """Run a Python script and capture results"""
    print("\n" + "="*100)
    print(f"RUNNING: {description}")
    print(f"Script: {script_name}")
    print("="*100 + "\n")

    script_path = Path(__file__).parent / script_name

    if not script_path.exists():
        print(f"❌ ERROR: {script_path} not found!")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            check=True
        )
        print(f"\n✅ {description} - COMPLETED")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} - FAILED")
        print(f"Exit code: {e.returncode}")
        return False

    except Exception as e:
        print(f"\n❌ {description} - ERROR: {e}")
        return False


def run_phase_0():
    """Phase 0: Load and prepare data"""
    return run_script('phase0_load_real_data.py', 'PHASE 0: Load REAL data')


def run_phase_1():
    """Phase 1: Component validation"""
    print("\n" + "="*100)
    print("PHASE 1: COMPONENT VALIDATION")
    print("="*100)

    results = {}

    # 1.1: Crisis detection
    results['1.1'] = run_script(
        'phase1_1_validate_crisis_detection.py',
        'PHASE 1.1: Crisis Detection Validation'
    )

    # 1.2: Regime detection (to be created)
    print("\n⚠️  PHASE 1.2: Regime Detection - NOT YET IMPLEMENTED")
    results['1.2'] = None

    # 1.3: Entry signals (to be created)
    print("\n⚠️  PHASE 1.3: Entry Signals - NOT YET IMPLEMENTED")
    results['1.3'] = None

    # 1.4: Exit strategies (already tested in comprehensive_exit_test_REAL.py)
    results['1.4'] = run_script(
        'comprehensive_exit_test_REAL.py',
        'PHASE 1.4: Exit Strategy Validation'
    )

    # 1.5: ML models (to be created)
    print("\n⚠️  PHASE 1.5: ML Models - NOT YET IMPLEMENTED")
    results['1.5'] = None

    # Summary
    print("\n" + "="*100)
    print("PHASE 1 SUMMARY")
    print("="*100)
    for phase, result in results.items():
        if result is None:
            status = "⚠️  NOT IMPLEMENTED"
        elif result:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        print(f"   Phase {phase}: {status}")

    return all(r for r in results.values() if r is not None)


def run_phase_2():
    """Phase 2: Walk-forward validation"""
    print("\n⚠️  PHASE 2: Walk-forward Validation - NOT YET IMPLEMENTED")
    return None


def run_phase_3():
    """Phase 3: Root cause analysis"""
    print("\n⚠️  PHASE 3: Root Cause Analysis - NOT YET IMPLEMENTED")
    return None


def run_phase_4():
    """Phase 4: Decision matrix"""
    print("\n⚠️  PHASE 4: Decision Matrix - NOT YET IMPLEMENTED")
    return None


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive system audit')
    parser.add_argument('--phase', type=int, choices=[0, 1, 2, 3, 4], help='Run specific phase only')
    args = parser.parse_args()

    start_time = datetime.now()

    print("="*100)
    print("COMPREHENSIVE SYSTEM AUDIT & VALIDATION FRAMEWORK")
    print("="*100)
    print(f"\nStart time: {start_time}")

    if args.phase is not None:
        print(f"Running Phase {args.phase} only...\n")

    results = {}

    # Run phases
    if args.phase is None or args.phase == 0:
        results['Phase 0'] = run_phase_0()

    if args.phase is None or args.phase == 1:
        results['Phase 1'] = run_phase_1()

    if args.phase is None or args.phase == 2:
        results['Phase 2'] = run_phase_2()

    if args.phase is None or args.phase == 3:
        results['Phase 3'] = run_phase_3()

    if args.phase is None or args.phase == 4:
        results['Phase 4'] = run_phase_4()

    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*100)
    print("COMPREHENSIVE AUDIT - FINAL SUMMARY")
    print("="*100)

    print(f"\n⏱️  Duration: {duration}")

    print(f"\n📊 Results:")
    for phase, result in results.items():
        if result is None:
            status = "⚠️  NOT IMPLEMENTED"
        elif result:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        print(f"   {phase}: {status}")

    # Overall verdict
    completed_phases = [r for r in results.values() if r is not None]
    if not completed_phases:
        print(f"\n⚠️  No phases completed yet")
    elif all(completed_phases):
        print(f"\n✅ ALL COMPLETED PHASES PASSED!")
    else:
        failed = sum(1 for r in completed_phases if not r)
        print(f"\n❌ {failed} PHASE(S) FAILED - needs attention")

    print("="*100)


if __name__ == '__main__':
    main()
