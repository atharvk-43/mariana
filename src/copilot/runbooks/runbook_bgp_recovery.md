# BGP Session Recovery

## Severity
Critical — service-impacting

## Symptoms
- BGP prefixes withdrawn
- Route table incomplete
- Traffic blackholing
- `bgp_updates_per_min` spike > 50

## Affected Roles
PE routers only. P and CE routers have no BGP.

## Troubleshooting Steps

### 1. Verify BGP sessions
```
show bgp summary
show bgp neighbors
```

### 2. Check keepalive timers
```
show bgp neighbor <peer> timers
```
Default: 30s keepalive, 120s hold. If timers are mismatched, sessions flap.

### 3. Verify TCP connectivity
```
ping <peer-ip> source <local-ip>
telnet <peer-ip> 179
```
BGP runs over TCP/179. If telnet fails, check ACLs and reachability.

### 4. Check BGP table
```
show bgp table
show ip bgp
```
Look for missing prefixes or unexpected routes.

### 5. Reset session if needed
```
clear bgp <peer> soft out
clear bgp <peer> soft in
```
Soft reset is preferred — no traffic impact.

## Recovery Actions

| Condition | Action |
|-----------|--------|
| Hold timer expired | Check physical link, increase hold timer |
| Prefix limit hit | `neighbor <peer> maximum-prefix <limit> <threshold> restart <minutes>` |
| TCP reset | Check MTU, path MTU discovery |
| BGP configuration change | `commit confirmed 10` to roll back |

## Escalation
If unrecovered after 5 minutes, escalate to Level 2 NOC.
