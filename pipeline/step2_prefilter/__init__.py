# pipeline/step2_prefilter/__init__.py
"""
Step 2 — Prefilter (COMPLETE)

Flags suspicious inputs before they reach the production LLM using:
- SBERT embeddings (all-MiniLM-L6-v2) + LogisticRegression classifier
- 17 hand-crafted regex rules (instruction override, credential extraction, jailbreak tokens, etc.)
- Legitimate context detection to reduce false positives

Performance: 97% ROC-AUC, FPR ≤ 5%
"""
