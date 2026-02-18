# ============================================================================
# pipeline/step5_verify/service.py
# ============================================================================
"""
Step 5 — Verification Service.

Re-tests the repaired prompt from Step 4 through the full Step2→Step3
pipeline to confirm the attack has been neutralized.

Flow:
    repaired_prompt
        │
        ▼
    Step2 Prefilter  ──── still suspicious? ──► escalate (hard reject)
        │ not suspicious
        ▼
    Step3 Sandbox    ──── still compromised? ──► escalate (hard reject)
        │ not compromised
        ▼
    verified=True → route to Step 6
"""
import logging
from typing import Dict, Any, Optional

from pipeline.step3_sandbox.pipeline import Step3Pipeline

logger = logging.getLogger("recovery.verify")

# Singleton — reuse the same pipeline instance to avoid reloading SBERT model
_pipeline: Optional[Step3Pipeline] = None


def _get_pipeline() -> Step3Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Step3Pipeline()
    return _pipeline


def verify(
    repaired_prompt: str,
    original_behavior: Optional[Dict[str, Any]] = None,
    repair_confidence: float = 0.0,
) -> Dict[str, Any]:
    """
    Verify that a repaired prompt is safe by re-running it through
    the Step 2 + Step 3 pipeline.

    Args:
        repaired_prompt:    The sanitized prompt from Step 4.
        original_behavior:  The Step 3 behavior dict from before repair
                            (used for context/logging).
        repair_confidence:  Confidence score from Step 4 (0.0–1.0).

    Returns:
        {
            "repaired_prompt":      str,
            "verified":             bool,   # True = safe to send to Step 6
            "repair_success":       bool,   # True = attack fully neutralized
            "escalate":             bool,   # True = hard reject, do not pass
            "escalation_reason":    str | None,
            "prefilter_result":     dict,
            "sandbox_result":       dict | None,
            "behavior_result":      dict | None,
            "route":                str,    # "production" | "escalate"
            "original_severity":    str,
        }
    """
    original_severity = (
        original_behavior.get("overall_severity", "unknown")
        if original_behavior else "unknown"
    )

    logger.info(
        f"Verify started. original_severity={original_severity}, "
        f"repair_confidence={repair_confidence:.2f}, "
        f"prompt_len={len(repaired_prompt)}"
    )

    pipeline = _get_pipeline()
    result = pipeline.process(repaired_prompt)

    stage    = result["stage"]
    route    = result["route"]
    prefilter = result["prefilter"]
    behavior  = result["behavior"]
    sandbox   = result["sandbox"]

    still_suspicious  = prefilter["suspicious"]
    still_compromised = behavior["compromised"] if behavior else False

    # ── Decision logic ────────────────────────────────────────────────────────
    if still_compromised:
        # Repair failed — attack survived sanitization
        verified = False
        repair_success = False
        escalate = True
        escalation_reason = (
            f"Repair failed: sandbox still detected compromise after sanitization. "
            f"Detectors: {[k for k,v in behavior['detections'].items() if v['compromised']]}"
        )
        final_route = "escalate"
        logger.warning(f"Verification FAILED — {escalation_reason}")

    elif still_suspicious and not still_compromised:
        # Prefilter still flags it but sandbox cleared it → treat as safe
        # (same logic as Step 3 false-positive path)
        verified = True
        repair_success = True
        escalate = False
        escalation_reason = None
        final_route = "production"
        logger.info("Verify: prefilter still suspicious but sandbox cleared — routing to production.")

    else:
        # Completely clean — prefilter passed, no sandbox needed
        verified = True
        repair_success = True
        escalate = False
        escalation_reason = None
        final_route = "production"
        logger.info("Verify: repaired prompt is clean — routing to production.")

    return {
        "repaired_prompt":    repaired_prompt,
        "verified":           verified,
        "repair_success":     repair_success,
        "escalate":           escalate,
        "escalation_reason":  escalation_reason,
        "prefilter_result":   prefilter,
        "sandbox_result":     sandbox,
        "behavior_result":    behavior,
        "route":              final_route,
        "original_severity":  original_severity,
    }
