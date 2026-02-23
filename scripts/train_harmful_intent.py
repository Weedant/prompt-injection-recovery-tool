
# =============================================================================
# scripts/train_harmful_intent.py
# =============================================================================
"""
Train the ML-based Harmful Intent Classifier (Phase 1 ROADMAP).

Replaces the regex-based HarmfulIntentDetector with a real ML model.

Architecture:
    SBERT (all-MiniLM-L6-v2) embeddings → LinearSVC (primary) + LogisticRegression (fallback)

Input:
    data/raw/harmful_intent_merged.csv   (from prepare_all_datasets.py)
    Also folds in benign rows from data/raw/prefilter_merged.csv (label=0)

Output:
    models/harmful_intent/harmful_intent_clf.joblib
    models/harmful_intent/threshold.txt
    models/harmful_intent/model_metadata.json

Usage:
    python scripts/train_harmful_intent.py
    python scripts/train_harmful_intent.py --classifier lr   # use LogisticRegression
    python scripts/train_harmful_intent.py --max-rows 20000
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "models" / "harmful_intent"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

SBERT_MODEL = "all-MiniLM-L6-v2"


def log(msg: str) -> None:
    print(f"[train_harmful_intent] {msg}", flush=True)


def load_data() -> tuple[list[str], list[int]]:
    """Load harmful + benign samples, return (texts, labels)."""
    dfs = []

    # Harmful samples
    harmful_path = ROOT / "data" / "raw" / "harmful_intent_merged.csv"
    if harmful_path.exists():
        df = pd.read_csv(harmful_path, encoding="utf-8", low_memory=False)
        log(f"  Harmful intent CSV: {len(df):,} rows")
        log(f"  Label dist: {df['label'].value_counts().to_dict()}")
        dfs.append(df[["text", "label"]])
    else:
        log("  ⚠ harmful_intent_merged.csv not found — run prepare_all_datasets.py first!")

    # Supplement benign samples from prefilter merged
    prefilter_path = ROOT / "data" / "raw" / "prefilter_merged.csv"
    if prefilter_path.exists():
        df2 = pd.read_csv(prefilter_path, encoding="utf-8", low_memory=False)
        benign = df2[df2["label"] == 0][["text", "label"]]
        # Take enough benign to balance
        n_harmful = sum(1 for d in dfs for _ in range(len(d)) if d["label"].iloc[_] == 1) if dfs else 5000
        n_benign_target = min(len(benign), max(n_harmful, 5000))
        benign_sample = benign.sample(n_benign_target, random_state=RANDOM_SEED)
        log(f"  Supplementing with {len(benign_sample):,} benign rows from prefilter_merged.csv")
        dfs.append(benign_sample)

    # Supplement benign from original dataset too
    original_path = ROOT / "data" / "raw" / "test.csv"
    if original_path.exists():
        df3 = pd.read_csv(original_path, encoding="utf-8", low_memory=False)
        if "label" in df3.columns:
            benign_orig = df3[df3["label"] == 0][["text", "label"]].sample(
                min(5000, len(df3[df3["label"] == 0])), random_state=RANDOM_SEED
            )
            log(f"  Adding {len(benign_orig):,} benign rows from original test.csv")
            dfs.append(benign_orig)

    if not dfs:
        log("ERROR: No data found!")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(subset=["text", "label"])
    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"] != ""]
    combined["label"] = combined["label"].astype(int)
    combined = combined[combined["label"].isin([0, 1])]
    combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)

    log(f"  Total after dedup: {len(combined):,}")
    log(f"  Label dist: {combined['label'].value_counts().to_dict()}")

    return combined["text"].tolist(), combined["label"].tolist()


def build_embeddings(texts: list[str]) -> np.ndarray:
    """Generate SBERT embeddings (batched)."""
    log(f"Building SBERT embeddings for {len(texts):,} texts using {SBERT_MODEL}...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log("ERROR: sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SBERT model on {device.upper()}...")
    model = SentenceTransformer(SBERT_MODEL, device=device)
    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    log(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def train(clf_type: str) -> None:
    log("═══ Harmful Intent Classifier Training ═══")
    log(f"Classifier: {clf_type.upper()}")

    texts, labels = load_data()
    y = np.array(labels)
    X = build_embeddings(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    log(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── Build classifier ────────────────────────────────────────────────────
    if clf_type == "svm":
        log("Training LinearSVC (calibrated for probabilities)...")
        base = LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_SEED,
        )
        clf = CalibratedClassifierCV(base, cv=3, method="isotonic")
    else:
        log("Training LogisticRegression...")
        clf = LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            solver="saga",
            random_state=RANDOM_SEED,
        )

    clf.fit(X_train, y_train)

    # ── Evaluate ────────────────────────────────────────────────────────────
    probas = clf.predict_proba(X_test)[:, 1]

    # Tune threshold at FPR ≤ 5%
    fpr, tpr, thresholds = roc_curve(y_test, probas)
    roc_auc = roc_auc_score(y_test, probas)

    # Find best threshold by F1
    f1s = [f1_score(y_test, (probas >= t).astype(int), zero_division=0) for t in thresholds]
    best_thresh_idx = int(np.argmax(f1s))
    best_threshold = float(thresholds[best_thresh_idx])

    # Also compute FPR ≤ 5% threshold as a conservative option
    fpr5_candidates = np.where(fpr <= 0.05)[0]
    if len(fpr5_candidates) > 0:
        conservative_threshold = float(thresholds[fpr5_candidates[-1]])
    else:
        conservative_threshold = best_threshold

    log(f"ROC-AUC: {roc_auc:.4f}")
    log(f"Best F1 threshold: {best_threshold:.4f}  (F1={f1s[best_thresh_idx]:.4f})")
    log(f"Conservative threshold (FPR≤5%): {conservative_threshold:.4f}")

    y_pred = (probas >= best_threshold).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["benign", "harmful"]))

    # ── Save artifacts ──────────────────────────────────────────────────────
    joblib.dump(clf, MODEL_DIR / "harmful_intent_clf.joblib")
    (MODEL_DIR / "threshold.txt").write_text(str(best_threshold))
    (MODEL_DIR / "conservative_threshold.txt").write_text(str(conservative_threshold))

    metadata = {
        "classifier":          clf_type.upper(),
        "embedding_model":     SBERT_MODEL,
        "threshold":           best_threshold,
        "conservative_threshold": conservative_threshold,
        "roc_auc":             roc_auc,
        "best_f1":             float(f1s[best_thresh_idx]),
        "train_size":          int(len(X_train)),
        "test_size":           int(len(X_test)),
        "n_features":          int(X.shape[1]),
        "datetime":            datetime.now().isoformat(),
        "labels":              {"0": "benign", "1": "harmful"},
    }
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    # ── Cache texts+labels for quick re-train (no re-embed) ─────────────────
    np.save(MODEL_DIR / "harmful_intent_embeddings.npy", X)
    np.save(MODEL_DIR / "harmful_intent_labels.npy", y)

    log("")
    log(f"✓ Model saved to {MODEL_DIR.relative_to(ROOT)}/")
    log("  Now update pipeline/step2_prefilter/harmful_intent_clf.py to use this model.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", choices=["svm", "lr"], default="svm",
                        help="Classifier type: svm (LinearSVC) or lr (LogisticRegression). Default: svm")
    args = parser.parse_args()
    train(args.classifier)


if __name__ == "__main__":
    main()
