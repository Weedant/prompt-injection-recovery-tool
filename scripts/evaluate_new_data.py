import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
from sklearn.metrics import classification_report, roc_auc_score

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.step2_prefilter.service import is_suspicious
from pipeline.step3_sandbox.sandbox_llm import SandboxLLM
from pipeline.step3_sandbox.behavior_detectors import BehaviorAnalyzer

# Paths
FULL_DATA_PATH = ROOT / "Test Data" / "full_combined_test.csv"
SAMPLE_DATA_PATH = ROOT / "Test Data" / "sampled_test_for_api.csv"

def evaluate_step2(limit=10000):
    """
    Evaluates Step 2 on a subset of the massive dataset (e.g. 10k rows).
    This doesn't use the API, but validating 274k rows takes a few minutes,
    so we sample it to get a quick metric for now.
    """
    print(f"\nEvaluating Step 2 (Prefilter) locally on {limit:,} samples...")
    print("This runs entirely locally without touching the Groq API.")
    df = pd.read_csv(FULL_DATA_PATH).sample(limit, random_state=42)
    
    y_true = df['label'].values
    y_pred, y_prob = [], []
    
    for prompt in tqdm(df['prompt'].astype(str), desc="Step 2 (Local ML)"):
        # is_suspicious returns a dict
        res = is_suspicious(prompt)
        y_pred.append(int(res["suspicious"]))
        y_prob.append(res["score"])
        
    print("\n--- Step 2 Prefilter Local Evaluation ---")
    roc = roc_auc_score(y_true, y_prob)
    print(f"ROC-AUC: {roc:.3f}")
    print(classification_report(y_true, y_pred, target_names=["Benign", "Malicious"]))


def evaluate_step3():
    """
    Evaluates Step 3 on the 100-row presentation-safe dataset.
    Has an explicit sleep timer to protect the Groq API limits.
    """
    print("\nEvaluating Step 3 (Sandbox) on the **100-row presentation-safe** dataset...")
    print("Enforcing strict 2.5 second delays between API calls to prevent Free Tier rate limiting!")
    
    df = pd.read_csv(SAMPLE_DATA_PATH)
    y_true = df['label'].values
    y_pred = []
    
    sandbox = SandboxLLM()
    analyzer = BehaviorAnalyzer()
    
    for prompt in tqdm(df['prompt'].astype(str), desc="Step 3 (Groq API)"):
        try:
            sandbox_result = sandbox.query(prompt)
            if "error" in sandbox_result:
                y_pred.append(0)
                continue
                
            llm_output = sandbox_result.get("output", "")
            behavior_result = analyzer.analyze(prompt, llm_output)
            
            y_pred.append(int(behavior_result["compromised"]))
            time.sleep(2.5) 
        except Exception as e:
            print(f"API Error skipped: {str(e)}")
            y_pred.append(0) # Default to false on error 

    print("\n--- Step 3 Sandbox Groq Evaluation (100 Samples) ---")
    print(classification_report(y_true, y_pred, target_names=["Benign", "Malicious"]))
    print("\n✅ Sandbox evaluation complete. API Limits protected for tomorrow's presentation!")

if __name__ == "__main__":
    evaluate_step2(limit=500)  # Evaluates local models fast
    evaluate_step3()           # Throttles API requests safely
