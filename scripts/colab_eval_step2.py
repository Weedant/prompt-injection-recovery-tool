# ---
# RECOVERY DEFENSE PIPELINE - MASSIVE EVALUATION SCRIPT
# ---
# 1. Open Google Colab (colab.research.google.com)
# 2. Select: Runtime -> Change runtime type -> GPU (T4) -> Save
# 3. Paste this code into a cell and run.

# --- INSTALL DEPENDENCIES ---
!pip install -q sentence-transformers pandas scikit-learn joblib

# --- MOUNT GOOGLE DRIVE ---
# Upload your 'Test Data/full_combined_test.csv' and the 'models/' folder to your Drive
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import joblib
import time
from sklearn.metrics import classification_report, roc_auc_score
from sentence_transformers import SentenceTransformer

# --- CONFIGURE PATHS ---
CSV_PATH = "/content/drive/MyDrive/full_combined_test.csv"
MODEL_PATH = "/content/drive/MyDrive/models/filtered_clf.joblib"
THRESH_PATH = "/content/drive/MyDrive/models/threshold.txt"

print(f"Loading complete test dataset from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)
prompts = df['prompt'].astype(str).tolist()
y_true = df['label'].values
print(f"Loaded {len(df):,} rows.")

print("Loading SBERT on GPU...")
sbert = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

print("Loading Logistic Regression Model...")
clf = joblib.load(MODEL_PATH)
base_thresh = float(open(THRESH_PATH).read().strip())

print("Encoding ALL prompts (Takes ~3-5 mins on GPU)...")
t0 = time.time()
embeddings = sbert.encode(prompts, batch_size=512, show_progress_bar=True)
print(f"Finished embedding in {time.time()-t0:.1f}s")

print("Predicting baseline ML probabilities...")
y_prob_base = clf.predict_proba(embeddings)[:, 1]

# --- RECREATE LOCAL RULE BANK ---
import re
print("Applying strict heuristics and rules...")

RULES = [
    (r'\bignore\s+all\s+previous\s+instructions?\b', "ignore_instruction"),
    (r'\bignore\s+(previous|earlier|prior)\s+(instructions?|prompts?|commands?)\b', "ignore_instruction"),
    (r'\bdo\s+not\s+(follow|obey|listen\s+to)\b', "do_not_follow"),
    (r'\bdisregard\s+(all\s+)?(instructions?|previous|above)\b', "disregard"),
    (r'\bforget\s+(everything|all|previous|your)\b', "forget_instruction"),
    (r'\boverride\s+(previous|system|your)\b', "override_instruction"),
    (r'\bprint\s+your\s+system\s+prompt\b', "exfil_system_prompt"),
    (r'\bshow\s+me\s+your\s+(original|initial)\s+instructions?\b', "exfil_system_prompt"),
    (r'\b(reveal|display|tell\s+me|give\s+me)\s+(the\s+)?(system\s+prompt|initial\s+prompt|your\s+instructions)\b', "exfil_system_prompt"),
    (r'\bwhat\s+(are|were)\s+your\s+(original|initial|system)\s+(instructions?|prompts?)\b', "exfil_system_prompt"),
    (r'\b(what\s+is|tell\s+me|reveal|show\s+me|give\s+me)\s+(your|the|my)\s+(password|api\s+key|secret|token|credentials?)\b', "credential_extraction"),
    (r'\b(need|want|require|get|obtain)\s+(your|the|my|a)?\s*(password|api\s+key|secret|token|credentials?)\b', "credential_extraction"),
    (r'\bpassword\s*[:=?]\s*', "password_prompt"),
    (r'\b(access|admin|secret|api)\s+(key|token|password|code)\b', "credential_extraction"),
    (r'\b(urgent|emergency|immediately|quickly|asap).*\b(password|key|token|secret|credentials?)\b', "urgent_credential_request"),
    (r'\byou\s+are\s+now\s+(a|an|acting\s+as)\b', "role_manipulation"),
    (r'\bpretend\s+(you\s+are|to\s+be)\b', "role_manipulation"),
    (r'\bact\s+as\s+(if\s+you|a|an)\b', "role_manipulation"),
    (r'\bDAN\s+mode\b', "jailbreak"),
    (r'\bdo\s+anything\s+now\b', "jailbreak"),
    (r'<\|?(?:im_start|im_end|endoftext)\|?>', "prompt_injection_token"),
    (r'\[INST\]|\[/INST\]', "prompt_injection_token"),
    (r'<system>|</system>', "prompt_injection_token"),
]

_STRONG_RULES = {
    "ignore_instruction", "do_not_follow", "disregard", "forget_instruction",
    "exfil_system_prompt", "credential_extraction", "password_prompt",
    "jailbreak", "prompt_injection_token", "urgent_credential_request",
    "override_instruction",
}
_MEDIUM_RULES = {"role_manipulation", "jailbreak_pretext"}

_LEGITIMATE_INDICATORS = [
    "data security", "cybersecurity research", "academic study",
    "ethical hacking", "penetration testing", "security audit",
]

def apply_rules(text: str) -> list[str]:
    return [name for pattern, name in RULES if re.search(pattern, text, re.IGNORECASE)]

y_pred = np.zeros(len(prompts), dtype=int)
y_prob = np.zeros(len(prompts), dtype=float)

for i in range(len(prompts)):
    text = prompts[i]
    rule_hits    = apply_rules(text)
    strong_rule  = any(r in _STRONG_RULES for r in rule_hits)
    medium_rule  = any(r in _MEDIUM_RULES for r in rule_hits)
    has_legit    = any(ind in text.lower() for ind in _LEGITIMATE_INDICATORS)

    prob = float(y_prob_base[i])
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
print("--- Step 2 Prefilter + Rules Evaluation (FULL 274k+ SET) ---")
roc = roc_auc_score(y_true, y_prob)
print(f"Total Evaluated: {len(df):,}")
print(f"ROC-AUC: {roc:.4f}")
print(classification_report(y_true, y_pred, target_names=["Benign", "Malicious"], digits=4))
print("=======================================================\n")
