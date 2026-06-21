# Link Congestion Management

## Severity
Warning → Critical as utilization approaches 90%

## Symptoms
- Interface utilization > 75%
- Queue depth > 40
- Packet loss > 1%
- Latency increase

## Affected Roles
CE access links and PE uplinks

## Troubleshooting Steps

### 1. Identify congested interface
```
show interface <iface>
show interface <iface> queue
```

### 2. Check top talkers
```
show ip traffic
show flow monitor cache
```

### 3. Verify QoS policy
```
show policy-map interface <iface>
show qos interface <iface>
```
Confirm marking and queuing are configured correctly.

### 4. Check bandwidth utilization
```
show interface <iface> bandwidth
show interface <iface> accounting
```

## Recovery Actions

| Condition | Action |
|-----------|--------|
| Utilization > 75% | Apply QoS, shape traffic |
| Utilization > 90% | Reroute traffic, add capacity |
| Queue depth > 100 | Increase buffer, check tail-drop |

## Prevention
- Monitor utilization trends with Prophet
- Pre-stage bandwidth upgrades when trend > 70%
- Deploy WRED on WAN interfaces

## Escalation
If congestion persists beyond 15 minutes, escalate to transport provider.
