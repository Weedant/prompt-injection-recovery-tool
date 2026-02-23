# ============================================================================
# scripts/run_step2.py
# ============================================================================
"""
Run the full Step 2 (Prefilter) pipeline end-to-end.

Steps:
    1. Prepare raw data → data/processed/filtered_data.csv
    2. Build SBERT embeddings → data/processed/filtered_embeddings.npy
    3. Train LogisticRegression → models/step2/filtered_clf.joblib

Usage:
    python scripts/run_step2.py
    python scripts/run_step2.py --skip-data   # skip data prep if CSV exists
    python scripts/run_step2.py --skip-embed  # skip embedding if .npy exists
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
    parser = argparse.ArgumentParser(description="Run Step 2 prefilter pipeline.")
    parser.add_argument("--skip-data",  action="store_true", help="Skip data preparation step.")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding step.")
    args = parser.parse_args()

    if not args.skip_data:
        print("\n-- Step 2a: Preparing data (using prefilter_merged.csv) --")
        run([sys.executable, "-m", "pipeline.step2_prefilter.data_prep", "--raw-path", str(ROOT / "data" / "raw" / "prefilter_merged.csv")])

    if not args.skip_embed:
        print("\n-- Step 2b: Building SBERT embeddings --")
        run([sys.executable, "-m", "pipeline.step2_prefilter.embedder", "--build", "--force", "--batch", "128"])

    print("\n-- Step 2c: Training classifier --")
    run([sys.executable, "-m", "pipeline.step2_prefilter.train"])

    print("\nStep 2 complete. Model saved to models/step2/")


if __name__ == "__main__":
    main()
