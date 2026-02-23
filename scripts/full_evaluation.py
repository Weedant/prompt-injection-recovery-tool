
# =============================================================================
# scripts/full_evaluation.py
# =============================================================================
"""
Final Evaluation Script: Calculates metrics across all three defense stages.
Run this after training finishes to see the performance of the new models.

Metrics:
- Accuracy: Overall correctness
- Recall (Sensitivity): % of actual attacks we caught
- Precision: % of flagged inputs that were actually attacks
- F1-Score: Balance of Precision and Recall
"""

import pandas as pd
import numpy as np
import sys
import time
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from pipeline.step2_prefilter.service import is_suspicious
from pipeline.step2_prefilter.harmful_intent_clf import HarmfulIntentClassifier
from pipeline.step3_sandbox.behavior_detectors import BehaviorAnalyzer

def log(msg: str):
    print(f"[eval] {msg}")

def evaluate_prefilter(limit=5000):
    log("--- Evaluating Step 2: Prefilter ---")
    path = ROOT / "data" / "raw" / "prefilter_merged.csv"
    if not path.exists():
        log("Error: prefilter_merged.csv not found.")
        return
    
    df = pd.read_csv(path).sample(min(limit, 20000), random_state=42)
    y_true = df["label"].tolist()
    y_pred = []
    scores = []
    
    start = time.time()
    for text in df["text"]:
        res = is_suspicious(str(text))
        y_pred.append(1 if res["suspicious"] else 0)
        scores.append(res["score"])
    
    elapsed = time.time() - start
    print(classification_report(y_true, y_pred, target_names=["Benign", "Injection"]))
    print(f"ROC-AUC: {roc_auc_score(y_true, scores):.4f}")
    print(f"Latency: {elapsed/len(df)*1000:.2f}ms per query")
    print()

def evaluate_harmful_intent(limit=2000):
    log("--- Evaluating Harmful Intent Classifier ---")
    path = ROOT / "data" / "raw" / "harmful_intent_merged.csv"
    if not path.exists():
        log("Error: harmful_intent_merged.csv not found.")
        return
        
    df = pd.read_csv(path).sample(min(limit, 5000), random_state=42)
    clf = HarmfulIntentClassifier()
    
    y_true = df["label"].tolist()
    y_pred = []
    probs = []
    
    for text in df["text"]:
        res = clf.detect(str(text), "")
        y_pred.append(1 if res["compromised"] else 0)
        probs.append(res["confidence"])
        
    print(classification_report(y_true, y_pred, target_names=["Benign", "Harmful"]))
    print()

def evaluate_sandbox(limit=2000):
    log("--- Evaluating Step 3: Sandbox Behavior ---")
    path = ROOT / "data" / "raw" / "sandbox_behavior.csv"
    if not path.exists():
        log("Error: sandbox_behavior.csv not found.")
        return
        
    df = pd.read_csv(path).sample(min(limit, 10000), random_state=42)
    analyzer = BehaviorAnalyzer()
    
    y_true = df["label"].tolist() # 1 = LLM complied, 0 = LLM refused
    y_pred = []
    
    for _, row in df.iterrows():
        res = analyzer.analyze(str(row["prompt"]), str(row["response"]))
        y_pred.append(1 if res["compromised"] else 0)
        
    print(classification_report(y_true, y_pred, target_names=["Refusal", "Compliance/Compromise"]))
    print("\nNote: Step 3 behavior metrics reflect the ability to distinguish whether the LLM actually followed a harmful prompt.")
    print()

def main():
    print("="*60)
    print("RECOVERY DEFENSE PIPELINE - FINAL PERFORMANCE REPORT")
    print("="*60)
    
    evaluate_prefilter()
    evaluate_harmful_intent()
    evaluate_sandbox()
    
    print("="*60)
    print("EVOLUTION SUMMARY")
    print("-" * 60)
    print("Old Baseline: Regex-heavy, limited data diversity (~65k rows).")
    print("New Model: ML-first, high data diversity (358k rows), GPU-trained.")
    print("Expectation: Broad coverage, higher recall on novel attacks, lower false positives.")
    print("="*60)

if __name__ == "__main__":
    main()
