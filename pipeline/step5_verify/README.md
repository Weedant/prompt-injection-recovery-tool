# Step 5 — Verification

## Status: Planned

## Purpose
Re-test the repaired input from Step 4 through the sandbox to confirm the attack has been
successfully neutralized before allowing it into production.

## Planned Approach
- Send repaired input through `Step3Pipeline` again
- If `behavior.compromised == False` → repair was successful → route to Step 6
- If `behavior.compromised == True` → repair failed → escalate (hard reject or human review)
- Track repair success rate as a quality metric

## Input
```json
{
  "repaired_input": "...",
  "original_behavior": { "severity": "high", ... }
}
```

## Output
```json
{
  "verified": true,
  "route": "production",
  "repair_success": true
}
```

## Next Step
→ **Step 6: Route** — final decision: production or reject
