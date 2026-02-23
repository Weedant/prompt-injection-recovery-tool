# ============================================================================
# pipeline/step2_prefilter/data_prep.py
# ============================================================================
"""
Prepare and normalize raw prompt datasets into a single CSV for the prefilter.

- Preserves existing id/source/meta columns if present
- Deduplicates, shuffles, caps to MAX_ROWS
- Outputs: data/processed/filtered_data.csv

Usage:
    python -m pipeline.step2_prefilter.data_prep
    python -m pipeline.step2_prefilter.data_prep --raw-path data/raw/prefilter_merged.csv
    python -m pipeline.step2_prefilter.data_prep --merge   # auto-merges original + prefilter_merged.csv
    # or via scripts/run_step2.py --merge
"""
import argparse
from pathlib import Path
import pandas as pd
import uuid
import json
import sys

ROOT = Path(__file__).resolve().parents[2]

# --- CONFIG ---
RAW_CSV_PATH = ROOT / "data" / "raw" / "test.csv"
OUTPUT_PATH  = ROOT / "data" / "processed" / "filtered_data.csv"
SAMPLE_PATH  = ROOT / "data" / "processed" / "filtered_sample.csv"
DEFAULT_SOURCE = "huggingface"


def fatal(msg: str) -> None:
    print("ERROR:", msg)
    sys.exit(1)


def prepare(raw_path: Path = RAW_CSV_PATH, output_path: Path = OUTPUT_PATH) -> None:
    """
    Load raw CSV, clean, deduplicate, and save processed dataset.

    Args:
        raw_path:    Path to the raw input CSV (must have 'text' and 'label' columns).
        output_path: Destination path for the processed CSV.
    """
    if not raw_path.exists():
        fatal(f"Raw file not found: {raw_path.resolve()}")

    print(f"Loading raw data from {raw_path}...")
    df = pd.read_csv(raw_path, encoding="utf-8", low_memory=False)

    if "text" not in df.columns or "label" not in df.columns:
        fatal(f"Input CSV must contain 'text' and 'label' columns. Found: {list(df.columns)}")

    # Keep only recognised columns
    cols_to_keep = [c for c in ["id", "text", "label", "source", "meta"] if c in df.columns]
    cols_to_keep = ["text", "label"] + [c for c in cols_to_keep if c not in ("text", "label")]
    df = df[cols_to_keep]

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]

    try:
        df["label"] = df["label"].astype(int)
    except Exception as e:
        fatal(f"Could not convert label column to int: {e}")

    if not df["label"].isin([0, 1]).all():
        fatal("Label column must contain only 0 and 1 values")

    if "id" not in df.columns:
        df.insert(0, "id", [str(uuid.uuid4()) for _ in range(len(df))])
    else:
        df["id"] = df["id"].astype(str)

    if "source" not in df.columns:
        df["source"] = DEFAULT_SOURCE

    if "meta" not in df.columns:
        df["meta"] = "{}"
    else:
        def ensure_json(x):
            if pd.isna(x):
                return "{}"
            if isinstance(x, (dict, list)):
                return json.dumps(x)
            try:
                json.loads(x)
                return x
            except Exception:
                return json.dumps({"raw": str(x)})
        df["meta"] = df["meta"].apply(ensure_json)

    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate rows")

    df["text_norm"] = df["text"].str.lower()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    cols_order = [c for c in ["id", "text", "text_norm", "label", "source", "meta"] if c in df.columns]
    df = df[cols_order]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved processed dataset ({len(df)} rows) → {output_path}")

    sample_n = min(200, len(df))
    df.sample(sample_n, random_state=42).to_csv(SAMPLE_PATH, index=False, encoding="utf-8")
    print(f"Saved {sample_n}-row sample → {SAMPLE_PATH}")
    print("Label distribution:", df["label"].value_counts().to_dict())
    print("Done.")


def _merge_sources(original: Path, merged: Path, out: Path) -> Path:
    """Combine original raw CSV with prefilter_merged.csv and write to out."""
    dfs = []
    if original.exists():
        df = pd.read_csv(original, encoding="utf-8", low_memory=False)
        print(f"  Original: {len(df):,} rows")
        dfs.append(df[[c for c in ["text", "label", "source"] if c in df.columns]])
    if merged.exists():
        df2 = pd.read_csv(merged, encoding="utf-8", low_memory=False)
        print(f"  prefilter_merged.csv: {len(df2):,} rows")
        dfs.append(df2[[c for c in ["text", "label", "source"] if c in df2.columns]])
    if not dfs:
        fatal("No source files found for --merge")
    combined = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"  Combined (deduped): {len(combined):,} rows")
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False, encoding="utf-8")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare raw data for Step 2 prefilter.")
    parser.add_argument("--raw-path", type=Path, default=None,
                        help="Override input CSV path (must have 'text' and 'label' columns).")
    parser.add_argument("--merge", action="store_true",
                        help="Auto-merge data/raw/test.csv + data/raw/prefilter_merged.csv before processing.")
    args = parser.parse_args()

    raw_path = args.raw_path
    if args.merge:
        merged_out = ROOT / "data" / "raw" / "prefilter_combined.csv"
        raw_path = _merge_sources(
            original=ROOT / "data" / "raw" / "test.csv",
            merged=ROOT / "data" / "raw" / "prefilter_merged.csv",
            out=merged_out,
        )

    prepare(raw_path=raw_path or RAW_CSV_PATH)
