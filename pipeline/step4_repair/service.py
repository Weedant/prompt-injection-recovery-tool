# ============================================================================
# pipeline/step4_repair/service.py
# ============================================================================
"""
Step 4 Repair Service — orchestrates rule stripping + LLM rewriting.

Flow:
    flagged prompt
        │
        ▼
    rule_stripper.strip_injections()   ← fast, offline
        │
        ▼
    llm_rewriter.rewrite()             ← Groq API call
        │
        ▼
    RepairResult {
        repaired_prompt,
        changes_made,
        repair_confidence,
        has_legitimate_intent,
        route_hint: "verify" | "reject"
    }
"""
import logging
from typing import Dict, Any, Optional

from pipeline.step4_repair.rule_stripper import strip_injections
from pipeline.step4_repair.llm_rewriter import LLMRewriter

logger = logging.getLogger("recovery.repair")


def repair(
    original_prompt: str,
    behavior: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Repair a flagged prompt.

    Args:
        original_prompt: The raw suspicious prompt from Step 3.
        behavior: The behavior analysis dict from Step 3 (used to tailor repair).
        api_key: Optional Groq API key override.

    Returns:
        {
            "original_prompt":      str,
            "repaired_prompt":      str | None,
            "has_legitimate_intent": bool,
            "changes_made":         List[str],
            "repair_confidence":    float,   # 0.0–1.0
            "rule_strip":           dict,    # raw rule stripper output
            "llm_rewrite":          dict,    # raw LLM rewriter output
            "route_hint":           str,     # "verify" | "reject"
            "tokens_used":          int,
        }
    """
    severity = behavior.get("overall_severity", "low") if behavior else "low"
    logger.info(f"Repair started. severity={severity}, prompt_len={len(original_prompt)}")

    # ── Phase 1: Rule-based stripping ────────────────────────────────────────
    strip_result = strip_injections(original_prompt)
    stripped_text = strip_result["cleaned_text"]
    rules_fired = strip_result["rules_fired"]

    logger.info(
        f"Rule stripper: {strip_result['n_removed']} removal(s), "
        f"rules={rules_fired}, "
        f"{strip_result['original_length']} → {strip_result['cleaned_length']} chars"
    )

    # ── Phase 2: LLM rewriting ────────────────────────────────────────────────
    rewriter = LLMRewriter(api_key=api_key)
    # Feed the already-stripped text to the LLM for a cleaner rewrite
    llm_result = rewriter.rewrite(stripped_text or original_prompt)

    has_intent = llm_result["has_legitimate_intent"]
    rewritten = llm_result["rewritten"]
    tokens_used = llm_result.get("tokens", 0)

    # ── Assemble changes_made list ────────────────────────────────────────────
    changes_made = []
    if rules_fired:
        changes_made.extend([f"rule:{r}" for r in rules_fired])
    if has_intent and rewritten:
        changes_made.append("llm_rewrite")
    elif not has_intent:
        changes_made.append("no_legitimate_intent_found")

    # ── Repair confidence heuristic ───────────────────────────────────────────
    # Higher confidence when:
    #   - LLM confirmed legitimate intent exists
    #   - Rules fired (we know what was removed)
    #   - Severity was not critical (critical = likely pure attack)
    confidence = 0.0
    if has_intent:
        confidence += 0.5
    if rules_fired:
        confidence += 0.3
    if severity not in ("critical",):
        confidence += 0.2
    confidence = min(confidence, 1.0)

    # ── Route hint ────────────────────────────────────────────────────────────
    # "verify" → pass repaired prompt to Step 5 for re-testing
    # "reject" → no legitimate intent, block entirely
    route_hint = "verify" if has_intent else "reject"

    final_prompt = rewritten if has_intent else None

    logger.info(
        f"Repair complete: has_intent={has_intent}, confidence={confidence:.2f}, "
        f"route_hint={route_hint}, tokens={tokens_used}"
    )

    return {
        "original_prompt":       original_prompt,
        "repaired_prompt":       final_prompt,
        "has_legitimate_intent": has_intent,
        "changes_made":          changes_made,
        "repair_confidence":     confidence,
        "rule_strip":            strip_result,
        "llm_rewrite":           llm_result,
        "route_hint":            route_hint,
        "tokens_used":           tokens_used,
    }
