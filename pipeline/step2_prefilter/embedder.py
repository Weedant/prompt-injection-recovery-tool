# ============================================================================
# pipeline/step2_prefilter/embedder.py
# ============================================================================
"""
Build and cache SBERT embeddings for the prefilter dataset.

Usage:
    python -m pipeline.step2_prefilter.embedder --build
    python -m pipeline.step2_prefilter.embedder --build --force --model all-MiniLM-L6-v2
    python -m pipeline.step2_prefilter.embedder --info
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

ROOT          = Path(__file__).resolve().parents[2]
DATA_CSV      = ROOT / "data" / "processed" / "filtered_data.csv"
EMB_OUT       = ROOT / "data" / "processed" / "filtered_embeddings.npy"
IDS_OUT       = ROOT / "data" / "processed" / "filtered_ids.npy"
META_OUT      = ROOT / "data" / "processed" / "embedder_meta.json"
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def load_dataframe(path: Path = DATA_CSV) -> pd.DataFrame:
    """
    Load the processed CSV.

    Args:
        path: Path to filtered_data.csv.

    Returns:
        DataFrame with at least 'id' and 'text' columns.
    """
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found at: {path}")
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    if "id" not in df.columns or "text" not in df.columns:
        raise ValueError("Expected columns 'id' and 'text' in processed CSV.")
    return df


def embed_texts(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode a list of texts with SBERT.

    Args:
        texts:         List of raw text strings.
        model_name:    SentenceTransformers model identifier.
        batch_size:    Number of texts per encoding batch.
        show_progress: Show tqdm progress bar.

    Returns:
        2-D numpy array of shape (len(texts), embedding_dim).
    """
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SBERT model on {device.upper()}...")
    model = SentenceTransformer(model_name, device=device)
    embeddings = []
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Embedding batches", unit="batch")
    for i in iterator:
        batch = texts[i : i + batch_size]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(emb)
    return np.vstack(embeddings)


def build_and_cache(
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    force: bool = False,
) -> Tuple[Path, Path, dict]:
    """
    Build SBERT embeddings and cache them to disk.

    Args:
        model_name: SentenceTransformers model identifier.
        batch_size: Encoding batch size.
        force:      Rebuild even if cached files already exist.

    Returns:
        Tuple of (embeddings_path, ids_path, metadata_dict).
    """
    df   = load_dataframe()
    ids  = df["id"].astype(str).tolist()
    texts = df["text"].astype(str).tolist()

    if EMB_OUT.exists() and IDS_OUT.exists() and not force:
        print(f"Embeddings already exist at {EMB_OUT}. Use --force to rebuild.")
        meta = json.loads(META_OUT.read_text(encoding="utf-8")) if META_OUT.exists() else {}
        return EMB_OUT, IDS_OUT, meta

    print(f"Building embeddings with model={model_name}, batch_size={batch_size}")
    t0 = time.time()
    embeddings = embed_texts(texts, model_name, batch_size)
    duration = time.time() - t0
    print(f"Built embeddings for {len(texts)} texts in {duration:.1f}s")

    EMB_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMB_OUT, embeddings)
    np.save(IDS_OUT, np.array(ids, dtype=object))

    meta = {
        "model":      model_name,
        "batch_size": batch_size,
        "rows":       len(texts),
        "dim":        int(embeddings.shape[1]),
        "time_s":     duration,
    }
    META_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved embeddings → {EMB_OUT}")
    print(f"Saved ids        → {IDS_OUT}")
    print(f"Saved meta       → {META_OUT}")
    return EMB_OUT, IDS_OUT, meta


def info() -> None:
    """Print current cache status."""
    print("Embedder info:")
    print(f"  Processed CSV : {DATA_CSV}")
    print(f"  Embeddings    : {EMB_OUT} (exists={EMB_OUT.exists()})")
    print(f"  IDs           : {IDS_OUT} (exists={IDS_OUT.exists()})")
    if META_OUT.exists():
        meta = json.loads(META_OUT.read_text(encoding="utf-8"))
        print("  Meta:", json.dumps(meta, indent=4))
    else:
        print("  Meta: none")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Build SBERT embeddings for Step 2 prefilter.")
    parser.add_argument("--build", action="store_true", help="Build and cache embeddings.")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if cached.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="SentenceTransformers model name.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for encoding.")
    parser.add_argument("--info",  action="store_true", help="Show cache info and exit.")
    args = parser.parse_args()

    if args.info:
        info()
        return
    if args.build:
        build_and_cache(model_name=args.model, batch_size=args.batch, force=args.force)
    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
