# Step 6 — Final Routing

## Status: Planned

## Purpose
The final gate. Aggregates decisions from all previous steps and routes the input to either
production (safe to process) or rejection (block + log).

## Planned Routing Logic
```
Input
  ↓
[Step 2: Prefilter]
  ├── suspicious=False → PRODUCTION ✅
  └── suspicious=True
        ↓
      [Step 3: Sandbox]
        ├── compromised=False → PRODUCTION ✅ (false alarm)
        └── compromised=True
              ↓
            [Step 4: Repair]
              ↓
            [Step 5: Verify]
              ├── verified=True  → PRODUCTION ✅ (repaired)
              └── verified=False → REJECT ❌ (log + alert)
```

## Output
```json
{
  "final_route": "production" | "reject",
  "reason": "clean" | "false_alarm" | "repaired" | "unrecoverable",
  "audit_log": { ... }
}
```

## Metrics to Track
- Overall block rate
- False positive rate (legitimate inputs blocked)
- Repair success rate
- End-to-end latency per route
