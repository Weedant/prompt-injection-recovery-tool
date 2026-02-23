# ============================================================================
# pipeline/__init__.py
# ============================================================================
"""
Recovery Pipeline — 6-stage prompt injection defense system.

Stages:
    step1_baseline  — Measure baseline LLM vulnerability
    step2_prefilter — ML + rule-based suspicious input detection (COMPLETE)
    step3_sandbox   — Sandbox LLM + behavior analysis (IN PROGRESS)
    step4_repair    — Sanitize flagged inputs (PLANNED)
    step5_verify    — Re-test repaired inputs (PLANNED)
    step6_route     — Final production/reject routing (PLANNED)
"""
