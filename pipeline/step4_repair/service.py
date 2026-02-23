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
from functools import lru_cache
import json

logger = logging.getLogger("recovery.repair")

@lru_cache(maxsize=2000)
def _cached_repair(original_prompt: str, behavior_str: str, api_key: Optional[str] = None) -> str:
    """Inner cached repair function. Strings are hashable."""
    behavior = json.loads(behavior_str) if behavior_str else None
    
    severity = behavior.get("overall_severity", "low") if behavior else "low"
    logger.info(f"Repair started (cache miss). severity={severity}, prompt_len={len(original_prompt)}")

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
    confidence = 0.0
    if has_intent: confidence += 0.5
    if rules_fired: confidence += 0.3
    if severity not in ("critical",): confidence += 0.2
    confidence = min(confidence, 1.0)

    route_hint = "verify" if has_intent else "reject"
    final_prompt = rewritten if has_intent else None

    logger.info(f"Repair complete: has_intent={has_intent}, confidence={confidence:.2f}, route_hint={route_hint}, tokens={tokens_used}")

    return json.dumps({
        "original_prompt":       original_prompt,
        "repaired_prompt":       final_prompt,
        "has_legitimate_intent": has_intent,
        "changes_made":          changes_made,
        "repair_confidence":     confidence,
        "rule_strip":            strip_result,
        "llm_rewrite":           llm_result,
        "route_hint":            route_hint,
        "tokens_used":           tokens_used,
    })


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
    behavior_str = json.dumps(behavior, sort_keys=True) if behavior else ""
    return json.loads(_cached_repair(original_prompt, behavior_str, api_key))
