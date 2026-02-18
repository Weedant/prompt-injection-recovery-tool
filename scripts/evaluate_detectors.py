# ============================================================================
# scripts/evaluate_detectors.py
# ============================================================================
"""
Phase A: Evaluate pattern-based detectors on the ReNeLLM dataset (offline).
Usage: python scripts/evaluate_detectors.py
"""
import sys
import logging
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from pipeline.step3_sandbox.behavior_detectors import BehaviorAnalyzer
from scripts.load_renellm_json import load_renellm
from shared.logger import logger

def evaluate():
    logger.info("Loading ReNeLLM dataset...")
    df = load_renellm()
    
    if df.empty:
        logger.error("Dataset empty or not found. Cannot evaluate.")
        sys.exit(1)
        
    logger.info(f"Loaded {len(df)} records. Evaluating detectors...")
    
    analyzer = BehaviorAnalyzer()
    
    y_true = []
    y_pred = []
    
    # Track individual detector performance
    detector_hits = {
        "InstructionFollowing": 0,
        "RoleSwitch": 0,
        "DataExfiltration": 0
    }
    
    detected_count = 0
    total_jailbreaks = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing"):
        prompt = row["prompt"]
        response = row["response"]
        label = row["label"] # 1 = successful jailbreak
        
        # Ground truth: Did the attack succeed? (1=yes)
        is_successful_attack = (label == 1)
        
        # Only evaluate on successful jailbreaks to measure Recall (True Positive Rate)
        # Assuming dataset is mostly successful jailbreaks (ReNeLLM generates successful ones)
        if is_successful_attack:
            total_jailbreaks += 1
            y_true.append(True)
            
            # Behavior analysis on the response
            result = analyzer.analyze(prompt, response)
            is_detected = result["compromised"]
            y_pred.append(is_detected)
            
            if is_detected:
                detected_count += 1
                detections = result["detections"]
                if detections["InstructionFollowing"]["compromised"]:
                    detector_hits["InstructionFollowing"] += 1
                if detections["RoleSwitch"]["compromised"]:
                    detector_hits["RoleSwitch"] += 1
                if detections["DataExfiltration"]["compromised"]:
                    detector_hits["DataExfiltration"] += 1
        
            
    # Metrics
    # Recall = TP / (TP + FN)
    recall = detected_count / total_jailbreaks if total_jailbreaks > 0 else 0
    
    print("\n" + "="*60)
    print("DETECTOR EVALUATION REPORT (Phase A)")
    print("="*60)
    print(f"Total Jailbreaks Evaluated: {total_jailbreaks}")
    print(f"Correctly Detected:         {detected_count}")
    print(f"Recall (Sensitivity):       {recall:.2%}")
    print("-" * 60)
    print("Breakdown by Detector:")
    for name, hits in detector_hits.items():
        hit_rate = hits / total_jailbreaks if total_jailbreaks > 0 else 0
        print(f"  - {name}: {hits} ({hit_rate:.2%})")
    print("="*60)
    
    # Target validation
    if recall > 0.80:
        print("\n[PASS] Target Met: Recall > 80%")
    else:
        print(f"\n[FAIL] Target Missed: Recall {recall:.2%} < 80%")
        print("\nNOTE: ReNeLLM dataset has all label=1 (125,494 successful jailbreaks).")
        print("      Responses are actual harmful Claude 2 completions, not refusals.")
        print("      Low recall means the regex detectors don't match the response style.")
        print("      Consider: (1) expanding regex patterns, or (2) using an LLM-as-judge approach.")

if __name__ == "__main__":
    evaluate()
