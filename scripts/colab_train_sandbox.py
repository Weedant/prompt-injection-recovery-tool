# ---
# RECOVERY DEFENSE PIPELINE - STEP 3 CLOUD TRAINER
# ---
# 1. Open Google Colab (colab.research.google.com)
# 2. Select: Runtime -> Change runtime type -> GPU (T4) -> Save
# 3. Paste the code below into a cell and run.
# NOTE: Uncomment the "!pip" and "drive.mount" lines below when running in Google Colab.

# --- INSTALL DEPENDENCIES ---
!pip install -q sentence-transformers pandas scikit-learn joblib

# --- MOUNT GOOGLE DRIVE ---
# Preparing your CSV by uploading it to your Google Drive root or a folder
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import joblib
import time
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
# Change this path to where you uploaded your CSV
CSV_PATH = "/content/drive/MyDrive/sandbox_behavior.csv" 
MODEL_OUT = "/content/drive/MyDrive/sandbox_clf.joblib"
THRESH_OUT = "/content/drive/MyDrive/threshold.txt"

SBERT_MODEL = "all-MiniLM-L6-v2"
RANDOM_SEED = 42

def log(msg):
    print(f"[colab_train] {msg}")

# --- LOAD DATA ---
log(f"Loading data from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH).dropna(subset=["prompt", "response", "label"])
prompts = df["prompt"].astype(str).tolist()
responses = df["response"].astype(str).tolist()
labels = df["label"].astype(int).tolist()
log(f"Loaded {len(prompts):,} rows.")

# --- EMBEDDING ---
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"Loading SBERT on {device.upper()}...")
model = SentenceTransformer(SBERT_MODEL, device=device)

# Combine prompt+response for context
CHAR_LIMIT = 400
texts = [f"{p[:CHAR_LIMIT]} [SEP] {r[:CHAR_LIMIT]}" for p, r in zip(prompts, responses)]

log("Encoding embeddings (This will be fast on T4!)...")
t0 = time.time()
embeddings = model.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
log(f"Finished embedding in {time.time()-t0:.1f}s")

# --- TRAINING ---
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, stratify=labels, random_state=RANDOM_SEED)

log("Training Calibrated LinearSVC...")
base = LinearSVC(C=1.0, class_weight="balanced", max_iter=3000, random_state=RANDOM_SEED)
clf = CalibratedClassifierCV(base, cv=3, method="isotonic")
clf.fit(X_train, y_train)

# --- SAVE RESULT ---
joblib.dump(clf, MODEL_OUT)
# Simple static threshold for now (matching our local logic)
Path(THRESH_OUT).write_text("0.5") 

log(f"SUCCESS! Download these files from your Drive and put them in models/sandbox/:")
log(f"1. {MODEL_OUT}")
log(f"2. {THRESH_OUT}")
