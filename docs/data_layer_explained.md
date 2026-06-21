# Data Layer Explained

## What It Produces

A telemetry CSV where **one row = one node at one timestamp** (10 nodes × 21,600 ticks/day at 4s intervals ≈ 216,000 rows/day). Each row has 92 columns organized into tiers.

## The 4 Fault Scenarios (Problem Statement)

| Fault | What breaks | Precursor signal | Active signal |
|-------|------------|-----------------|---------------|
| **Congestion** | CE access link (wan0) + upstream PE | utilization rising, queue building | drops, high latency, packet loss |
| **BGP Flap** | PE router BGP sessions | BGP updates rising | sessions dropping, convergence storm |
| **MPLS Failure** | P-router core (eth2/eth3) | errors_in creeping up | link flapping, tunnels degrading |
| **Policy Drift** | PE router config change | OSPF SPF runs increasing | routes changing, CPU/memory spike |

## Why 3 Tiers of Fields?

```
Row (92 columns)
├── Identity (4)         → timestamp, node_id, node_type, site
├── Per-Interface (40)   → 8 fields × 5 possible ifaces (eth0-eth3, wan0)
│                           Only fields that faults actually touch:
│                           errors_in, drops_in, utilization_pct,
│                           queue_depth, latency_ms, jitter_ms,
│                           packet_loss_pct, link_state
├── Per-Tunnel (18)      → 3 fields × 6 MPLS LSPs
│                           loss_pct, jitter_ms, latency_ms
├── NUMERIC_COLS (26)    → 23 numeric aggregates + 3 non-numeric
│                           This is what ML models actually see
└── Labels (4)           → fault_type, fault_phase, is_anomaly, is_precursor
```

## The Trick: Aggregates → ML, Raw → Alerting

**NUMERIC_COLS (23)** = the only fields ML models touch. These are aggregates computed from per-interface data:

```python
bytes_in        = sum(eth0.bytes_in, eth1.bytes_in, ...)
utilization_pct = max(eth0.util, eth1.util, ...) × 100
latency_ms      = max(eth0.latency, eth1.latency, ...)
errors_in       = sum(eth0.errors, eth1.errors, ...)
queue_depth     = max(eth0.queue, eth1.queue, ...)
# ... plus node-level fields (BGP sessions, CPU, memory, etc.)
```

**Per-interface raw fields** are for the copilot/alerting to answer *"which interface?"* — e.g., the ensemble says "congestion" and `wan0_utilization_pct=95%` tells you it's the CE access link, not a core link.

## FRRouting Compatibility

| Field | FRR Source | How |
|-------|-----------|-----|
| `{iface}_errors_in` | `/proc/net/dev` | `cat /proc/net/dev \| grep eth0` |
| `{iface}_drops_in` | `/proc/net/dev` | same file, "drop" column |
| `{iface}_utilization_pct` | computed | `bytes/sec / link_speed` |
| `{iface}_queue_depth` | tc qdisc | `tc -s qdisc show dev eth0` |
| `{iface}_latency_ms` | ICMP probe | `ping -c 3 <neighbor>` or FRR BFD |
| `{iface}_link_state` | ip link | `ip link show eth0 \| grep state` |
| `cpu_load_pct` | `/proc/stat` | standard Linux |
| `bgp_sessions_active` | FRR vtysh | `vtysh -c "show bgp summary json"` |
| `mpls_lsp_count` | FRR vtysh | `vtysh -c "show mpls lsp json"` |

## Why Not 123?

The original schema had 13 fields per interface (including `bytes_in`, `bytes_out`, `packets_in`, `packets_out`, `drops_out`). None of those 5 are ever changed by a fault injector — they only exist as normal telemetry. The aggregates in NUMERIC_COLS already capture the same information (bytes_in = sum of all iface bytes). So the per-interface versions were redundant: they took up space, added no fault signal, and made integration harder for zero gain.

92 columns = every column earns its place.
