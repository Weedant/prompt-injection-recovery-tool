# ============================================================================
# scripts/build_behavior_classifier.py
# ============================================================================
"""
Phase B: Train sandbox behavior ML classifier.
Delegates to scripts/train_sandbox_classifier.py.

Usage (via run_step3.py):
    python scripts/run_step3.py --train

Or directly:
    python scripts/build_behavior_classifier.py
    python scripts/build_behavior_classifier.py --use-distilbert
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    # Pass all args through to the real trainer
    extra_args = sys.argv[1:]
    cmd = [sys.executable, str(ROOT / "scripts" / "train_sandbox_classifier.py")] + extra_args
    result = subprocess.run(cmd, cwd=ROOT)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
