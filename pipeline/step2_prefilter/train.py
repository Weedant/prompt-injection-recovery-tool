# ============================================================================
# pipeline/step2_prefilter/train.py
# ============================================================================
"""
Train the LogisticRegression classifier on SBERT embeddings.

Requires:
    data/processed/filtered_embeddings.npy  (built by embedder.py)
    data/processed/filtered_ids.npy
    data/processed/embedder_meta.json
    data/processed/filtered_data.csv

Outputs:
    models/step2/filtered_clf.joblib
    models/step2/threshold.txt
    models/step2/model_metadata.json

Usage:
    python -m pipeline.step2_prefilter.train
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

ROOT      = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "step2"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from shared.logger import logger

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def train() -> None:
    """
    Load embeddings, train LogisticRegression, tune threshold at FPR ≤ 5%,
    and save model artifacts to models/step2/.
    """
    emb_path  = ROOT / "data" / "processed" / "filtered_embeddings.npy"
    ids_path  = ROOT / "data" / "processed" / "filtered_ids.npy"
    meta_path = ROOT / "data" / "processed" / "embedder_meta.json"

    missing = [p for p in [emb_path, ids_path, meta_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Run embedder first! Missing: {missing}")

    X    = np.load(emb_path)
    ids  = np.load(ids_path, allow_pickle=True)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    df = pd.read_csv(ROOT / "data" / "processed" / "filtered_data.csv")
    df = df.set_index("id").loc[ids].reset_index()
    y  = df["label"].values

    logger.info(f"Loaded {len(X)} embeddings | dim={X.shape[1]} | model={meta['model']}")

    train_idx, test_idx = train_test_split(
        range(len(X)), test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        solver="saga",
    )
    clf.fit(X_train, y_train)
    probas = clf.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, probas)
    threshold = float(thresholds[np.where(fpr <= 0.05)[0][-1]])
    logger.info(f"Threshold (FPR ≤ 5%): {threshold:.4f}")

    joblib.dump(clf, MODEL_DIR / "filtered_clf.joblib")
    (MODEL_DIR / "threshold.txt").write_text(str(threshold))

    metadata = {
        "classifier":      "LogisticRegression",
        "embedding_model": meta["model"],
        "threshold":       threshold,
        "roc_auc":         float(roc_auc_score(y_test, probas)),
        "train_size":      int(len(train_idx)),
        "test_size":       int(len(test_idx)),
        "datetime":        datetime.now().isoformat(),
    }
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    logger.info(f"Model saved to {MODEL_DIR}. ROC-AUC: {metadata['roc_auc']:.4f}")
    print(classification_report(y_test, (probas >= threshold).astype(int)))


if __name__ == "__main__":
    train()
