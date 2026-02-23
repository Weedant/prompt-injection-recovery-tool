# Step 4 — Repair Engine

## Status: Planned

## Purpose
Sanitize inputs that Step 3 (Sandbox) has confirmed as compromised/jailbreak attempts.

## Planned Approach
- Strip injection tokens (`[INST]`, `<system>`, etc.)
- Neutralize role-switch phrases ("you are now DAN" → removed)
- Rewrite instruction-override attempts into benign equivalents
- Preserve the user's legitimate underlying intent where possible

## Input
```json
{
  "original_input": "...",
  "behavior": {
    "compromised": true,
    "severity": "high",
    "detections": { ... }
  }
}
```

## Output
```json
{
  "repaired_input": "...",
  "changes_made": ["removed_role_switch", "stripped_injection_token"],
  "repair_confidence": 0.85
}
```

## Next Step
→ **Step 5: Verify** — re-test the repaired input to confirm the attack is neutralized
