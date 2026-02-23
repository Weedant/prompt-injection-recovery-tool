# ============================================================================
# pipeline/step6_route/service.py
# ============================================================================
"""
Step 6 — Final Router.

The single public entry point for the entire Recovery pipeline.
Orchestrates Steps 2 → 3 → 4 → 5 and emits a final routing decision
with a full structured audit log.

Usage:
    from pipeline.step6_route.service import route
    result = route("some user input")
    print(result["final_route"])   # "production" | "reject"
    print(result["reason"])        # "clean" | "false_alarm" | "repaired" | "unrecoverable"
"""
import time
import logging
from typing import Dict, Any, Optional

from pipeline.step2_prefilter.service import is_suspicious
from pipeline.step3_sandbox.sandbox_llm import SandboxLLM
from pipeline.step3_sandbox.behavior_detectors import BehaviorAnalyzer
from pipeline.step4_repair.service import repair
from pipeline.step5_verify.service import verify

logger = logging.getLogger("recovery.route")

# Reason codes — human-readable labels for each terminal state
REASON_CLEAN         = "clean"          # Passed prefilter, never suspicious
REASON_FALSE_ALARM   = "false_alarm"    # Prefilter flagged, sandbox cleared
REASON_REPAIRED      = "repaired"       # Attack detected, repaired, verified safe
REASON_UNRECOVERABLE = "unrecoverable"  # Attack survived repair — hard reject


def route(
    user_input: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full 6-stage Recovery pipeline on a user input.

    Args:
        user_input: Raw prompt string from the user.
        api_key:    Optional Groq API key override (defaults to GROQ_API_KEY env var).

    Returns:
        {
            "final_route":   "production" | "reject",
            "reason":        str,          # one of the REASON_* constants
            "prompt_used":   str,          # the prompt that reached production
                                           # (original or repaired)
            "total_tokens":  int,          # total Groq tokens consumed
            "latency_ms":    float,        # end-to-end wall time in ms
            "stages":        dict,         # per-stage outputs for audit
            "audit_log":     dict,         # structured summary for logging/storage
        }
    """
    t_start = time.perf_counter()
    total_tokens = 0
    stages: Dict[str, Any] = {}

    logger.info(f"[Route] START — input_len={len(user_input)}")

    # ── Stage 2: Prefilter ────────────────────────────────────────────────────
    t2 = time.perf_counter()
    prefilter_result = is_suspicious(user_input)
    stages["step2_prefilter"] = {**prefilter_result, "latency_ms": _ms(t2)}

    if not prefilter_result["suspicious"]:
        # Fast path — clean input, no API calls needed
        return _build_result(
            final_route  = "production",
            reason       = REASON_CLEAN,
            prompt_used  = user_input,
            total_tokens = 0,
            t_start      = t_start,
            stages       = stages,
        )

    logger.info(
        f"[Route] Step2 flagged (score={prefilter_result['score']:.3f}). "
        f"Routing to sandbox."
    )

    # --- OPTIMIZATION: Confidence Gating ---
    # If the prefilter gave a score > 0.85 or triggered a strong rule/intent,
    # skip the sandbox entirely to save time & money, routing directly to Repair.
    if prefilter_result.get("skip_sandbox", False):
        logger.info(f"[Route] Prefilter confidence extremely high ({prefilter_result['score']:.3f}). Bypassing sandbox -> Repair.")
        behavior_result = {
            "compromised": True,
            "overall_severity": "high",
            "hits": prefilter_result.get("rule_hits", []),
            "detected_by": ["Prefilter_High_Confidence_Bypass"]
        }
        stages["step3_sandbox"] = {"status": "skipped_due_to_high_confidence", "latency_ms": 0}
    else:
        # ── Stage 3: Sandbox ──────────────────────────────────────────────────────
        t3 = time.perf_counter()
        sandbox = SandboxLLM(api_key=api_key)
        analyzer = BehaviorAnalyzer()
    
        sandbox_result = sandbox.query(user_input)
        llm_output = sandbox_result.get("output", "")
        tokens_s3 = sandbox_result.get("tokens", {}).get("total", 0)
        total_tokens += tokens_s3
    
        if "error" in sandbox_result:
            # Sandbox error → fail-safe reject
            stages["step3_sandbox"] = {**sandbox_result, "latency_ms": _ms(t3)}
            logger.error(f"[Route] Sandbox error: {sandbox_result['error']} — fail-safe reject.")
            return _build_result(
                final_route  = "reject",
                reason       = "sandbox_error",
                prompt_used  = None,
                total_tokens = total_tokens,
                t_start      = t_start,
                stages       = stages,
            )
    
        behavior_result = analyzer.analyze(user_input, llm_output)
        stages["step3_sandbox"] = {
            "sandbox":  sandbox_result,
            "behavior": behavior_result,
            "latency_ms": _ms(t3),
        }

    if not behavior_result["compromised"]:
        # Prefilter false positive — sandbox cleared it
        logger.info("[Route] Sandbox cleared — false alarm. Routing to production.")
        return _build_result(
            final_route  = "production",
            reason       = REASON_FALSE_ALARM,
            prompt_used  = user_input,
            total_tokens = total_tokens,
            t_start      = t_start,
            stages       = stages,
        )

    logger.info(
        f"[Route] Compromise confirmed (severity={behavior_result['overall_severity']}). "
        f"Routing to repair."
    )

    # ── Stage 4: Repair ───────────────────────────────────────────────────────
    t4 = time.perf_counter()
    repair_result = repair(
        original_prompt = user_input,
        behavior        = behavior_result,
        api_key         = api_key,
    )
    tokens_s4 = repair_result.get("tokens_used", 0)
    total_tokens += tokens_s4
    stages["step4_repair"] = {**repair_result, "latency_ms": _ms(t4)}

    if not repair_result["has_legitimate_intent"]:
        # Pure attack — no legitimate intent found, reject immediately
        logger.info("[Route] No legitimate intent found — hard reject.")
        return _build_result(
            final_route  = "reject",
            reason       = REASON_UNRECOVERABLE,
            prompt_used  = None,
            total_tokens = total_tokens,
            t_start      = t_start,
            stages       = stages,
        )

    # ── Stage 5: Verify ───────────────────────────────────────────────────────
    t5 = time.perf_counter()
    verify_result = verify(
        repaired_prompt   = repair_result["repaired_prompt"],
        original_behavior = behavior_result,
        repair_confidence = repair_result["repair_confidence"],
    )
    # Count any sandbox tokens used during re-verification
    sandbox_v = verify_result.get("sandbox_result") or {}
    tokens_s5 = sandbox_v.get("tokens", {}).get("total", 0) if sandbox_v else 0
    total_tokens += tokens_s5
    stages["step5_verify"] = {**verify_result, "latency_ms": _ms(t5)}

    if verify_result["verified"]:
        logger.info("[Route] Repair verified — routing repaired prompt to production.")
        return _build_result(
            final_route  = "production",
            reason       = REASON_REPAIRED,
            prompt_used  = repair_result["repaired_prompt"],
            total_tokens = total_tokens,
            t_start      = t_start,
            stages       = stages,
        )
    else:
        logger.warning(
            f"[Route] Repair failed verification — {verify_result['escalation_reason']}. "
            f"Hard reject."
        )
        return _build_result(
            final_route  = "reject",
            reason       = REASON_UNRECOVERABLE,
            prompt_used  = None,
            total_tokens = total_tokens,
            t_start      = t_start,
            stages       = stages,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ms(t_start: float) -> float:
    """Elapsed milliseconds since t_start."""
    return round((time.perf_counter() - t_start) * 1000, 2)


def _build_result(
    final_route:  str,
    reason:       str,
    prompt_used:  Optional[str],
    total_tokens: int,
    t_start:      float,
    stages:       Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the final result dict with audit log."""
    latency_ms = _ms(t_start)

    audit_log = {
        "final_route":   final_route,
        "reason":        reason,
        "total_tokens":  total_tokens,
        "latency_ms":    latency_ms,
        "stages_run":    list(stages.keys()),
        "prompt_used":   prompt_used[:120] + "..." if prompt_used and len(prompt_used) > 120 else prompt_used,
    }

    logger.info(
        f"[Route] END — route={final_route}, reason={reason}, "
        f"tokens={total_tokens}, latency={latency_ms}ms"
    )

    return {
        "final_route":  final_route,
        "reason":       reason,
        "prompt_used":  prompt_used,
        "total_tokens": total_tokens,
        "latency_ms":   latency_ms,
        "stages":       stages,
        "audit_log":    audit_log,
    }
