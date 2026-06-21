# MPLS Tunnel / LSP Degradation

## Severity
High — may cause traffic loss on affected LSPs

## Symptoms
- Tunnel packet loss > 2%
- Tunnel latency spike
- LSP flap counters increasing
- Label table size changes on P routers

## Affected Roles
P routers and PE routers (LSP head-end)

## Troubleshooting Steps

### 1. Verify LDP sessions
```
show mpls ldp neighbor
show mpls ldp discovery
```
All core-facing LDP sessions must be up.

### 2. Check LSP status
```
show mpls forwarding-table
show mpls traffic-eng tunnels
```

### 3. Verify label stack
```
show mpls label table
show mpls forwarding-table <prefix>
```

### 4. Check MPLS OAM
```
ping mpls ipv4 <dest> <label>
traceroute mpls ipv4 <dest> <label>
```

## Recovery Actions

| Condition | Action |
|-----------|--------|
| LDP session down | Check IGP reachability between peers |
| Label table corruption | `clear mpls ldp neighbor *` to re-establish |
| LSP blackhole | Check PHP (Penultimate Hop Popping) |

## Escalation
If LSPs remain down after 10 minutes, escalate to Core team.
