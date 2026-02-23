
# =============================================================================
# scripts/train_sandbox_classifier.py
# =============================================================================
"""
Train Step 3 Sandbox Behavior Classifier.

Predicts whether an LLM response indicates a compromised sandbox:
  label=1 → LLM complied with harmful/jailbreak request
  label=0 → LLM refused or responded safely

Architecture choices:
  PRIMARY:  SBERT embeddings of [prompt + " [SEP] " + response] → LinearSVC
            (fast, runs offline, no GPU needed)
  OPTIONAL: Fine-tuned DistilBERT on (prompt, response) pairs  [--use-distilbert]
            (better accuracy, needs ~2GB RAM, no GPU required for inference)

Input:
    data/raw/sandbox_behavior.csv   (from prepare_all_datasets.py)

Output:
    models/sandbox/sandbox_clf.joblib          (SBERT+SVC)
    models/sandbox/threshold.txt
    models/sandbox/model_metadata.json
    models/sandbox/distilbert/                 (if --use-distilbert)

Usage:
    python scripts/train_sandbox_classifier.py
    python scripts/train_sandbox_classifier.py --use-distilbert
    python scripts/train_sandbox_classifier.py --max-rows 20000
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "models" / "sandbox"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

SBERT_MODEL = "all-MiniLM-L6-v2"
MAX_SEQ_LEN = 256   # truncate prompt+response combined


def log(msg: str) -> None:
    print(f"[train_sandbox] {msg}", flush=True)


def load_data() -> tuple[list[str], list[str], list[int]]:
    """Load (prompt, response, label) from sandbox_behavior.csv."""
    path = ROOT / "data" / "raw" / "sandbox_behavior.csv"
    if not path.exists():
        log("ERROR: sandbox_behavior.csv not found!")
        log("  Run: python scripts/prepare_all_datasets.py")
        sys.exit(1)

    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    log(f"Loaded {len(df):,} rows | Label dist: {df['label'].value_counts().to_dict()}")

    df = df.dropna(subset=["prompt", "response", "label"])
    df["prompt"]   = df["prompt"].astype(str).str.strip()
    df["response"] = df["response"].astype(str).str.strip()
    df = df[df["prompt"] != ""]
    df = df[df["response"] != ""]
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])]
    df = df.drop_duplicates(subset=["prompt", "response"]).reset_index(drop=True)

    return df["prompt"].tolist(), df["response"].tolist(), df["label"].tolist()


def build_sbert_embeddings(prompts: list[str], responses: list[str]) -> np.ndarray:
    """
    Concatenate prompt + [SEP] + response, embed with SBERT.
    Uses a 'checkpoint' system to allow stopping/resuming for cooling.
    """
    CACHE_DIR = ROOT / "data" / "processed" / "sandbox_chunks"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    CHUNK_SIZE = 50_000
    n_rows = len(prompts)
    n_chunks = (n_rows // CHUNK_SIZE) + (1 if n_rows % CHUNK_SIZE != 0 else 0)
    
    all_embeddings = []
    
    log(f"Building SBERT embeddings for {n_rows:,} pairs in {n_chunks} chunks...")
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log("ERROR: pip install sentence-transformers")
        sys.exit(1)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None # Lazy load only if needed
    
    for i in range(n_chunks):
        chunk_file = CACHE_DIR / f"chunk_{i}.npy"
        
        if chunk_file.exists():
            log(f"  Chunk {i+1}/{n_chunks}: Loading from cache...")
            all_embeddings.append(np.load(chunk_file))
            continue
            
        # If we reach here, we need to process this chunk
        if model is None:
            log(f"Loading SBERT model on {device.upper()}...")
            model = SentenceTransformer(SBERT_MODEL, device=device)
            
        start_idx = i * CHUNK_SIZE
        end_idx = min((i + 1) * CHUNK_SIZE, n_rows)
        
        log(f"  Chunk {i+1}/{n_chunks}: Processing indices {start_idx:,} to {end_idx:,}...")
        
        CHAR_LIMIT = 400
        batch_texts = [
            f"{p[:CHAR_LIMIT]} [SEP] {r[:CHAR_LIMIT]}"
            for p, r in zip(prompts[start_idx:end_idx], responses[start_idx:end_idx])
        ]
        
        chunk_emb = model.encode(
            batch_texts,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        np.save(chunk_file, chunk_emb)
        all_embeddings.append(chunk_emb)
        log(f"  Chunk {i+1}/{n_chunks}: Saved to disk. (Total progress: {end_idx/n_rows*100:.1f}%)")
        
        # Optional: Force memory cleanup
        if device == "cuda":
            torch.cuda.empty_cache()

    embeddings = np.vstack(all_embeddings)
    log(f"Final embeddings shape: {embeddings.shape}")
    return embeddings


def train_sbert_svm(prompts: list[str], responses: list[str], labels: list[int]) -> None:
    """Train LinearSVC on SBERT embeddings."""
    log("═══ Sandbox Classifier — SBERT + LinearSVC ═══")

    X = build_sbert_embeddings(prompts, responses)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    log(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    log("Training LinearSVC (calibrated)...")
    base = LinearSVC(C=1.0, class_weight="balanced", max_iter=3000, random_state=RANDOM_SEED)
    clf = CalibratedClassifierCV(base, cv=3, method="isotonic")
    clf.fit(X_train, y_train)

    probas = clf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, probas)
    fpr, tpr, thresholds = roc_curve(y_test, probas)
    f1s = [f1_score(y_test, (probas >= t).astype(int), zero_division=0) for t in thresholds]
    best_thresh = float(thresholds[int(np.argmax(f1s))])
    best_f1 = float(max(f1s))

    log(f"ROC-AUC: {roc_auc:.4f} | Best F1: {best_f1:.4f} at threshold={best_thresh:.4f}")
    y_pred = (probas >= best_thresh).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["safe/refused", "compromised"]))

    joblib.dump(clf, MODEL_DIR / "sandbox_clf.joblib")
    (MODEL_DIR / "threshold.txt").write_text(str(best_thresh))

    metadata = {
        "classifier":       "LinearSVC_calibrated",
        "embedding_model":  SBERT_MODEL,
        "threshold":        best_thresh,
        "roc_auc":          roc_auc,
        "best_f1":          best_f1,
        "train_size":       int(len(X_train)),
        "test_size":        int(len(X_test)),
        "n_features":       int(X.shape[1]),
        "input_format":     "prompt [SEP] response",
        "datetime":         datetime.now().isoformat(),
        "labels":           {"0": "safe/refused", "1": "compromised"},
    }
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    log(f"✓ Model saved to {MODEL_DIR.relative_to(ROOT)}/")


def train_distilbert(prompts: list[str], responses: list[str], labels: list[int]) -> None:
    """
    Fine-tune DistilBERT on (prompt, response) pairs.
    Uses HuggingFace Transformers — needs: pip install transformers torch datasets
    """
    log("═══ Sandbox Classifier — DistilBERT Fine-tuning ═══")
    try:
        from transformers import (
            DistilBertTokenizerFast,
            DistilBertForSequenceClassification,
            TrainingArguments,
            Trainer,
        )
        import torch
        from datasets import Dataset as HFDataset
    except ImportError:
        log("ERROR: pip install transformers torch datasets")
        sys.exit(1)

    db_dir = MODEL_DIR / "distilbert"
    db_dir.mkdir(exist_ok=True)

    MODEL_NAME = "distilbert-base-uncased"
    tokenizer  = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    CHAR_LIMIT = 256
    texts = [f"{p[:CHAR_LIMIT]} [SEP] {r[:CHAR_LIMIT]}" for p, r in zip(prompts, responses)]

    train_texts, test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=RANDOM_SEED
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_SEQ_LEN)

    train_ds = HFDataset.from_dict({"text": train_texts, "label": y_train}).map(tokenize, batched=True)
    test_ds  = HFDataset.from_dict({"text": test_texts,  "label": y_test }).map(tokenize, batched=True)
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=str(db_dir),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=str(db_dir / "logs"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )
    trainer.train()
    trainer.save_model(str(db_dir))
    tokenizer.save_pretrained(str(db_dir))
    log(f"✓ DistilBERT model saved to {db_dir.relative_to(ROOT)}/")
    log("  To use: load with transformers pipeline('text-classification', model=str(db_dir))")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-distilbert", action="store_true",
                        help="Fine-tune DistilBERT instead of SBERT+SVC (needs transformers + torch)")
    args = parser.parse_args()

    prompts, responses, labels = load_data()

    if args.use_distilbert:
        train_distilbert(prompts, responses, labels)
    else:
        train_sbert_svm(prompts, responses, labels)


if __name__ == "__main__":
    main()
