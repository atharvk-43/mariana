# Interface Link Flapping

## Severity
Warning — degrades routing stability

## Symptoms
- Interface state toggling UP/DOWN
- `errors_in` increasing on interface
- OSPF adjacency flapping
- BGP session resets

## Affected Roles
Any interface on any node

## Troubleshooting Steps

### 1. Check interface counters
```
show interface <iface>
show interface <iface> errors
show interface <iface> crc
```

### 2. Verify physical layer
```
show interface <iface> transceiver
show interface <iface> statistics
show logging | include <iface>
```

### 3. Check OSPF impact
```
show ip ospf interface <iface>
show ip ospf neighbor
```

## Recovery Actions

| Condition | Action |
|-----------|--------|
| CRC errors increasing | Replace SFP/cable |
| Link state toggling | Set debounce timer: `link-debounce time 1000` |
| Speed/duplex mismatch | Hard-code: `speed 1000`, `duplex full` |

## Prevention
- Use optical monitoring for pre-failure detection
- Set errdisable recovery
- Enable loopback detection

## Escalation
If physical layer suspected, escalate to Field Engineering.
