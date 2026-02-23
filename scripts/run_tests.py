# ============================================================================
# scripts/run_tests.py
# ============================================================================
"""
Run the full test suite.

Usage:
    python scripts/run_tests.py              # all tests
    python scripts/run_tests.py --step 2    # only Step 2 tests
    python scripts/run_tests.py --step 3    # only Step 3 tests
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STEP_TEST_MAP = {
    "2": "tests/test_step2_prefilter.py",
    "3": "tests/test_step3_sandbox.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Recovery pipeline tests.")
    parser.add_argument("--step", type=str, choices=["2", "3"], help="Run tests for a specific step only.")
    args = parser.parse_args()

    if args.step:
        test_file = STEP_TEST_MAP.get(args.step)
        if not (ROOT / test_file).exists():
            print(f"Test file not found: {test_file}")
            sys.exit(1)
        targets = [test_file]
    else:
        targets = [f for f in STEP_TEST_MAP.values() if (ROOT / f).exists()]

    cmd = [sys.executable, "-m", "pytest"] + targets + ["-v"]
    result = subprocess.run(cmd, cwd=ROOT)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
