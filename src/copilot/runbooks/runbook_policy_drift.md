# Configuration / Policy Drift

## Severity
Medium — causes routing instability, potential blackhole

## Symptoms
- Unexpected route advertisements
- VRF route count changes
- BGP updates spike without topology change
- CPU/memory pressure on PE routers

## Affected Roles
PE routers primarily; downstream CEs can receive leaked routes

## Troubleshooting Steps

### 1. Compare running vs. baseline config
```
show running-config | compare
show commit changes
```

### 2. Check BGP policy
```
show route policy
show bgp neighbors <peer> advertised-routes
show bgp neighbors <peer> received-routes
```

### 3. Verify VRF import/export
```
show vrf detail
show ip bgp vpnv4 vrf <vrf-name>
```

### 4. Check route-target consistency
```
show bgp vpnv4 unicast all rt <rt-value>
```

## Recovery Actions

| Condition | Action |
|-----------|--------|
| Unauthorized prefix | `withdraw route <prefix>` |
| RT mismatch | Correct RT on VRF definition |
| CPU saturation | Apply route-policy to filter leaks |

## Prevention
- Use `commit confirmed` for all changes
- Config backup every 6 hours
- RADIUS/TACACS+ for change authorization

## Escalation
If VRF corruption suspected, escalate to Security team for audit.
