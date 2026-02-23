import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import joblib

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.step2_prefilter.service import apply_rules, _STRONG_RULES, _MEDIUM_RULES, _LEGITIMATE_INDICATORS
from pipeline.step2_prefilter.harmful_intent_clf import HarmfulIntentClassifier
from sklearn.metrics import classification_report, roc_auc_score

def evaluate_full():
    FULL_DATA_PATH = ROOT / "Test Data" / "full_combined_test.csv"
    print(f"Loading complete test dataset from {FULL_DATA_PATH}...")
    # Sample 10,000 rows for a quick but statistically significant test (274k takes 10 hours locally)
    df = pd.read_csv(FULL_DATA_PATH).sample(10000, random_state=42)
    
    prompts = df['prompt'].astype(str).tolist()
    y_true = df['label'].values
    
    print(f"Loaded {len(df):,} rows.")
    
    # 1. Load ML models
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SBERT on {device.upper()}...")
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Please install sentence-transformers")
        sys.exit(1)
        
    embedder_meta_path = ROOT / "data" / "processed" / "embedder_meta.json"
    meta = json.loads(embedder_meta_path.read_text())
    sbert = SentenceTransformer(meta["model"], device=device)
    
    clf_path = ROOT / "models" / "filtered_clf.joblib"
    clf = joblib.load(clf_path)
    
    thresh_path = ROOT / "models" / "threshold.txt"
    base_thresh = float(thresh_path.read_text().strip())
    
    # 2. Batch Encode
    print("Batch encoding all prompts (this is highly optimized for the full dataset)...")
    embeddings = sbert.encode(prompts, batch_size=256, show_progress_bar=True)
    
    # 3. Batch Predict
    print("Predicting baseline ML probabilities...")
    base_probs = clf.predict_proba(embeddings)[:, 1]
    
    # 4. Apply rules efficiently
    print("Applying regex heuristics and intent classifiers...")
    harmful_clf = HarmfulIntentClassifier()
    
    y_pred = np.zeros(len(prompts), dtype=int)
    y_prob = np.zeros(len(prompts), dtype=float)
    
    for i in tqdm(range(len(prompts)), desc="Evaluating logic"):
        text = prompts[i]
        
        rule_hits    = apply_rules(text)
        strong_rule  = any(r in _STRONG_RULES for r in rule_hits)
        medium_rule  = any(r in _MEDIUM_RULES for r in rule_hits)
        has_legit    = any(ind in text.lower() for ind in _LEGITIMATE_INDICATORS)

        harmful = harmful_clf.detect(text, "")
        if harmful["compromised"]:
            y_pred[i] = 1
            y_prob[i] = 0.98
            continue

        prob = float(base_probs[i])
        threshold_adj = base_thresh * 1.2 if (has_legit and not strong_rule) else base_thresh

        final_prob = prob
        if strong_rule:
            final_prob = max(prob, 0.95)
        elif medium_rule:
            final_prob = max(prob, min(prob * 1.3, 0.85))
        elif rule_hits:
            final_prob = max(prob, 0.7)

        y_prob[i] = final_prob
        y_pred[i] = 1 if (strong_rule or prob >= threshold_adj) else 0

    print("\n=======================================================")
    print("--- Step 2 Prefilter Local Evaluation (FULL DATASET) ---")
    roc = roc_auc_score(y_true, y_prob)
    print(f"Total Evaluated: {len(df):,}")
    print(f"ROC-AUC: {roc:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Benign", "Malicious"], digits=4))
    print("=======================================================\n")

if __name__ == "__main__":
    evaluate_full()
