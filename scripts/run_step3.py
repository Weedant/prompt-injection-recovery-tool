# ============================================================================
# scripts/run_step3.py
# ============================================================================
"""
Run Step 3 (Sandbox) pipeline tasks.

Usage:
    python scripts/run_step3.py           # Evaluate pattern detectors (Phase A)
    python scripts/run_step3.py --train   # Train ML classifier (Phase B)
    python scripts/run_step3.py --test    # Run Step 3 unit tests
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 3 (Sandbox) pipeline tasks.")
    parser.add_argument("--train", action="store_true", help="Run ML training (Phase B)")
    parser.add_argument("--test", action="store_true", help="Run functionality tests")
    args = parser.parse_args()

    if args.test:
        print("\n-- Running Step 3 Tests --")
        run([sys.executable, "scripts/run_tests.py", "--step", "3"])
        
    elif args.train:
        print("\n-- Phase B: Training ML Classifier --")
        run([sys.executable, "scripts/build_behavior_classifier.py"])
        
    else:
        print("\n-- Phase A: Evaluating Pattern Detectors --")
        run([sys.executable, "scripts/evaluate_detectors.py"])
        
    print("\nStep 3 task complete.")

if __name__ == "__main__":
    main()
