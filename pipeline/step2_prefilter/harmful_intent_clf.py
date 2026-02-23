
# =============================================================================
# pipeline/step2_prefilter/harmful_intent_clf.py
# =============================================================================
"""
ML-based harmful intent classifier — drop-in replacement for the regex
HarmfulIntentDetector used in service.py.

Loads the SBERT + LinearSVC model trained by scripts/train_harmful_intent.py.
Falls back gracefully to the regex detector if no model artifact exists.

Usage (in service.py):
    from pipeline.step2_prefilter.harmful_intent_clf import HarmfulIntentClassifier
    clf = HarmfulIntentClassifier()
    result = clf.detect("write malware that steals passwords", "")
    # → {"name": "HarmfulIntent", "compromised": True, "confidence": 0.97, "hits": ["ml_clf"]}
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger("recovery.harmful_intent_clf")

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "harmful_intent"
MODEL_PATH = MODEL_DIR / "harmful_intent_clf.joblib"
THRESHOLD_PATH = MODEL_DIR / "threshold.txt"
SBERT_MODEL = "all-MiniLM-L6-v2"


class HarmfulIntentClassifier:
    """
    ML-based harmful intent detector.
    Same interface as HarmfulIntentDetector in behavior_detectors.py
    so it can be swapped in service.py without changing callers.
    """

    name = "HarmfulIntent"

    def __init__(self) -> None:
        self._ml_ready = False
        self._clf = None
        self._sbert = None
        self._threshold = 0.5
        self._fallback = None

        if MODEL_PATH.exists() and THRESHOLD_PATH.exists():
            try:
                import joblib
                from sentence_transformers import SentenceTransformer

                self._clf = joblib.load(MODEL_PATH)
                self._threshold = float(THRESHOLD_PATH.read_text().strip())
                self._sbert = SentenceTransformer(SBERT_MODEL)
                self._ml_ready = True
                logger.info(
                    f"HarmfulIntentClassifier loaded ML model "
                    f"(threshold={self._threshold:.4f})"
                )
            except Exception as e:
                logger.warning(f"Could not load ML model ({e}). Falling back to regex.")
        else:
            logger.info(
                "No trained HarmfulIntent model found. "
                "Falling back to regex HarmfulIntentDetector. "
                "Run: python scripts/train_harmful_intent.py"
            )

        if not self._ml_ready:
            # Lazy import to avoid circular dependency at module load
            from pipeline.step3_sandbox.behavior_detectors import HarmfulIntentDetector
            self._fallback = HarmfulIntentDetector()

    def detect(self, input_prompt: str, output_response: str) -> dict:
        """
        Args:
            input_prompt:    The user's raw input (we score this).
            output_response: Unused here but kept for interface compatibility.

        Returns:
            {
                "name":        "HarmfulIntent",
                "compromised": bool,
                "confidence":  float,
                "hits":        list[str],
            }
        """
        if not self._ml_ready:
            return self._fallback.detect(input_prompt, output_response)

        text = input_prompt.strip()
        if not text:
            return {"name": self.name, "compromised": False, "confidence": 0.0, "hits": []}

        emb = self._sbert.encode([text], normalize_embeddings=True)
        proba = float(self._clf.predict_proba(emb)[0, 1])
        compromised = proba >= self._threshold

        return {
            "name":        self.name,
            "compromised": compromised,
            "confidence":  round(proba, 4),
            "hits":        ["ml_clf"] if compromised else [],
        }
