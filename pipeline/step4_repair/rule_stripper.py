# ============================================================================
# pipeline/step4_repair/rule_stripper.py
# ============================================================================
"""
Rule-based stripping of known prompt injection patterns.
Fast, offline, no API calls. First pass before LLM rewriting.
"""
import re
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger("recovery.repair.rules")

# ── Injection pattern rules ───────────────────────────────────────────────────
# Each rule: (name, compiled_regex)
# Order matters — more destructive patterns first.

_RULES: List[Tuple[str, re.Pattern]] = [
    # Instruction override openers
    ("ignore_instructions",
     re.compile(
         # Stop at sentence end (. ! ?) or newline — don't eat the rest of the line
         r"ignore\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|rules?|prompts?|guidelines?)"
         r"[^.!?\n]*[.!?]?",
         re.IGNORECASE
     )),

    # Persona / role-switch triggers
    ("role_switch_dan",
     re.compile(
         r"(you\s+are\s+now|pretend\s+(to\s+be|you\s+are)|act\s+as|become|you\s+will\s+be)\s+"
         r"(dan|aim|jailbreak|unrestricted|uncensored|free|evil|god|root|an?\s+ai\s+with\s+no\s+restrictions)[^\n]*",
         re.IGNORECASE
     )),

    # "Do Anything Now" / DAN explicit
    ("dan_explicit",
     re.compile(
         r"\bDAN\b.*?(do\s+anything\s+now|no\s+restrictions?|no\s+limits?)[^\n]*",
         re.IGNORECASE | re.DOTALL
     )),

    # System / injection token wrappers
    ("injection_tokens",
     re.compile(
         r"(\[INST\]|\[\/INST\]|<\|system\|>|<\|user\|>|<\|assistant\|>|<system>|<\/system>"
         r"|<<SYS>>|<</SYS>>|\[SYS\]|\[\/SYS\])",
         re.IGNORECASE
     )),

    # "New instructions are:" style overrides
    ("new_instructions",
     re.compile(
         r"(your\s+new\s+instructions?\s+(are|is)|new\s+task\s*:)[^\n]*",
         re.IGNORECASE
     )),

    # "Stay in character" enforcement
    ("stay_in_character",
     re.compile(
         r"(stay\s+in\s+character|remain\s+in\s+character|keep\s+up\s+the\s+act)[^\n]*",
         re.IGNORECASE
     )),

    # "No restrictions / no rules / no limits" declarations
    ("no_restrictions",
     re.compile(
         r"(has?\s+no\s+(restrictions?|rules?|limits?|guidelines?|ethics?|morals?)"
         r"|without\s+(restrictions?|limits?|rules?))[^\n]*",
         re.IGNORECASE
     )),

    # Jailbreak game framing
    ("game_framing",
     re.compile(
         r"(let'?s?\s+play\s+a\s+(game|role[- ]?play)\s*[.,]?\s*)"
         r"(you\s+are|i\s+want\s+you\s+to\s+be|pretend)[^\n]*",
         re.IGNORECASE
     )),

    # "Reveal your system prompt / API keys / secrets"
    ("exfiltration_request",
     re.compile(
         r"(reveal|show|print|output|tell\s+me|give\s+me)\s+"
         r"(your\s+)?(system\s+prompt|api\s+key|secret|token|password|credentials?)[^\n]*",
         re.IGNORECASE
     )),
]


def strip_injections(text: str) -> Dict[str, Any]:
    """
    Apply all injection-stripping rules to the input text.

    Returns:
        {
            "cleaned_text": str,       # text after stripping
            "rules_fired": List[str],  # names of rules that matched
            "n_removed": int,          # number of pattern removals
            "original_length": int,
            "cleaned_length": int,
        }
    """
    cleaned = text
    rules_fired: List[str] = []
    n_removed = 0

    for name, pattern in _RULES:
        new_text, count = pattern.subn("", cleaned)
        if count > 0:
            rules_fired.append(name)
            n_removed += count
            cleaned = new_text

    # Collapse excessive whitespace / blank lines left by removals
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = cleaned.strip()

    logger.debug(f"Rule stripper: {n_removed} removal(s) via rules {rules_fired}")

    return {
        "cleaned_text": cleaned,
        "rules_fired": rules_fired,
        "n_removed": n_removed,
        "original_length": len(text),
        "cleaned_length": len(cleaned),
    }
