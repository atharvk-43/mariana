# NOC Escalation Procedure

## Severity Levels

| Level | Definition | Response Time |
|-------|------------|---------------|
| P1 | Service outage, multiple sites affected | 5 minutes |
| P2 | Single site degraded, redundant path available | 15 minutes |
| P3 | Non-critical, no traffic impact | 60 minutes |
| P4 | Informational, cosmetic only | Next business day |

## Escalation Path

```
Level 1 NOC (Shift Engineer)
    → Level 2 NOC (Senior Engineer) — 15 min unresolved
        → Level 3 (SME / Engineering) — 30 min unresolved
            → Management — 60 min unresolved
```

## Handoff Procedure

1. Document current state in ticket
2. Share relevant show commands output
3. Share timeline of events
4. Escalate with recommendation (not just problem statement)

## Post-Incident

- Root cause analysis within 24 hours
- Update runbook with lessons learned
- Adjust alert thresholds if false positive
