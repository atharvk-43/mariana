# Spec: Data Layer
## Files: `src/data/topology.py`, `src/data/network_gen.py`, `src/data/anomaly_injector.py`

---

## Overview

This layer is responsible for generating a realistic, correlated, time-series synthetic network telemetry dataset with **per-interface** and **per-tunnel** granularity. All downstream ML models and the API are fed from this layer.

Key advancement: Instead of a single set of interface metrics per node, each router tracks telemetry per physical interface (e.g., `eth0`, `eth1`, `wan0`) and per MPLS tunnel (e.g., `lsp-b1-dc1`). Node-level aggregates (`bytes_in`, `utilization_pct`) are computed from per-interface data for backward compatibility.

---

## File 1: `src/data/topology.py`

### Purpose
Define the static network graph: nodes, their roles, their interfaces, neighbors, link capacities, and tunnel definitions. This is the "truth table" for the simulated SD-WAN/MPLS environment.

### EDGES Format — Per-Interface Naming

Each edge now includes **both** source and destination interface names, enabling per-interface telemetry:

```python
# Format: (source_id, dest_id, capacity_mbps, src_interface, dst_interface)
EDGES = [
    ("CE-B1",  "PE-1",   100,    "wan0", "eth1"),     # CE-B1 wan0 → PE-1 eth1
    ("CE-B2",  "PE-1",   100,    "wan0", "eth2"),     # CE-B2 wan0 → PE-1 eth2
    ("CE-B3",  "PE-2",   100,    "wan0", "eth1"),     # CE-B3 wan0 → PE-2 eth1
    ("PE-1",   "P-1",    1000,   "eth0", "eth0"),     # PE-1 eth0 → P-1 eth0
    ("PE-2",   "P-1",    1000,   "eth0", "eth1"),     # PE-2 eth0 → P-1 eth1
    ("P-1",    "P-2",    10000,  "eth2", "eth0"),     # P-1 eth2 → P-2 eth0
    ("P-1",    "P-3",    10000,  "eth3", "eth0"),     # P-1 eth3 → P-3 eth0
    ("P-2",    "CE-DC1", 1000,   "eth1", "wan0"),     # P-2 eth1 → CE-DC1 wan0
    ("P-3",    "CE-DC2", 1000,   "eth1", "wan0"),     # P-3 eth1 → CE-DC2 wan0
    ("PE-1",   "PE-2",   1000,   "eth3", "eth3"),     # PE-1 eth3 → PE-2 eth3
]
```

### Per-Node Interface Map

| Node | Interfaces |
|------|------------|
| CE-B1 | wan0 (→PE-1, 100M) |
| CE-B2 | wan0 (→PE-1, 100M) |
| CE-B3 | wan0 (→PE-2, 100M) |
| PE-1 | eth1 (→CE-B1, 100M), eth2 (→CE-B2, 100M), eth0 (→P-1, 1G), eth3 (→PE-2, 1G) |
| PE-2 | eth1 (→CE-B3, 100M), eth0 (→P-1, 1G), eth3 (→PE-1, 1G) |
| P-1 | eth0 (→PE-1, 1G), eth1 (→PE-2, 1G), eth2 (→P-2, 10G), eth3 (→P-3, 10G) |
| P-2 | eth0 (→P-1, 10G), eth1 (→CE-DC1, 1G) |
| P-3 | eth0 (→P-1, 10G), eth1 (→CE-DC2, 1G) |
| CE-DC1 | wan0 (→P-2, 1G) |
| CE-DC2 | wan0 (→P-3, 1G) |

### New Functions

```python
def get_node_interfaces(node_id: str) -> list[tuple[str, str, int]]:
    """Return list of (interface_name, neighbor, capacity_mbps) for the given node."""

def get_node_tunnels(node_id: str) -> list[str]:
    """Return list of LSP tunnel names that this node participates in."""
```

### Notes
- `get_graph_as_dict()` now returns `src_interface` and `dst_interface` per edge.
- `get_adjacency_matrix()` and `get_edge_index_coo()` are unchanged (capacity-based).

---

## File 2: `src/data/network_gen.py`

### Purpose
Generate realistic, correlated synthetic network telemetry with per-interface and per-tunnel granularity. Each call to `next_tick()` produces one row per node, where each row contains per-interface fields (prefixed by interface name), per-tunnel fields (prefixed by tunnel name), node-level fields, and aggregate fields.

### Key Design Principles
- **Per-interface**: Each physical interface on every router tracks its own utilization, queue, errors, latency, jitter, loss, and link state.
- **Per-tunnel**: Each MPLS LSP (end-to-end path) tracks tunnel-level loss, jitter, latency, and IPSec rekey count.
- **Correlated**: Interface utilization drives queue depth, queue drives latency, latency drives jitter and loss — all within each interface.
- **Diurnal patterns**: Traffic follows a sine-wave pattern with peak at 14:00 UTC and trough at 04:00 UTC.
- **Aggregate backward compat**: Old field names (`bytes_in`, `utilization_pct`, `latency_ms`, etc.) are preserved as computed aggregates (sum or max across interfaces).

### New Dataclasses

```python
@dataclass
class InterfaceState:
    """Per-interface telemetry state."""
    name:              str          # e.g. "eth0", "wan0"
    neighbor:          str          # connected to which node
    capacity_mbps:     float
    b_util:            float        # baseline utilization
    b_latency:         float        # baseline latency (ms)
    b_jitter:          float        # baseline jitter (ms)
    b_loss:            float        # baseline packet loss (%)
    b_queue:           float        # baseline queue depth
    # Running state:
    util, queue, errors_in, drops_in, drops_out,
    latency_ms, jitter_ms, packet_loss_pct, link_state


@dataclass
class TunnelState:
    """Per-LSP tunnel health metrics."""
    name:              str          # e.g. "lsp-b1-dc1"
    path_nodes:        list[str]    # ordered node IDs along path
    b_loss, b_jitter, b_latency     # baselines
    # Running state:
    loss_pct, jitter_ms, latency_ms, uptime_sec, ipsec_rekeyed_last_hr
```

### Updated NodeState

```python
@dataclass
class NodeState:
    node_id:      str
    node_type:    str
    site:         str
    # Node-level running state:
    cpu, memory, bgp_active, bgp_prefixes, bgp_updates,
    bgp_withdrawals, ospf_spf, ldp_active, lsp_count,
    label_table_size, vpn_routes
    # Per-interface state:
    interfaces:   dict[str, InterfaceState]    # e.g. {"eth0": ..., "eth1": ...}
    # Per-tunnel state:
    tunnels:      dict[str, TunnelState]       # e.g. {"lsp-b1-dc1": ..., ...}
    # Fault labels:
    fault_type, fault_phase
```

### Telemetry Schema

The output row has 5 tiers — identity, per-interface, per-tunnel, NUMERIC_COLS aggregates, and labels.

**1. Identity (4):** `timestamp`, `node_id`, `node_type`, `site`

**2. Per-interface (5 ifaces × 8 = 40):** Only fields directly manipulated by fault injectors:
```
{eth0|eth1|eth2|eth3|wan0}_errors_in, {iface}_drops_in,
{iface}_utilization_pct, {iface}_queue_depth,
{iface}_latency_ms, {iface}_jitter_ms, {iface}_packet_loss_pct,
{iface}_link_state
```
(bytes_in/out/packets still computed internally for aggregate NUMERIC_COLS but not stored per-interface)

**3. Per-tunnel (6 tunnels × 3 = 18):**
```
{lsp-*}_loss_pct, {lsp-*}_jitter_ms, {lsp-*}_latency_ms
```

**4. NUMERIC_COLS (23) + 3 non-numeric aggregates:**
```python
# NUMERIC_COLS — primary ML input (23 numeric fields)
NUMERIC_COLS = [
    "bytes_in", "bytes_out", "packets_in", "packets_out",
    "errors_in", "drops_in", "drops_out", "utilization_pct",
    "bgp_sessions_active", "bgp_prefixes_received",
    "bgp_updates_per_min", "bgp_withdrawals_per_min",
    "ospf_spf_runs",
    "ldp_sessions_active", "mpls_lsp_count",
    "mpls_label_table_size", "vpn_routes_count",
    "cpu_load_pct", "memory_used_pct", "queue_depth",
    "latency_ms", "jitter_ms", "packet_loss_pct",
]
```
Non-numeric extras: `link_state` (worst across interfaces), `tunnel_packet_loss_pct` (max tunnel loss), `ipsec_rekeyed_last_hr` (total rekeys).

**5. Labels (4):** `fault_type`, `fault_phase`, `is_anomaly`, `is_precursor`

> The 23 NUMERIC_COLS are the primary ML input. Per-interface/per-tunnel fields enable granular alerting and feature engineering without affecting ML model interfaces.

### Baseline Values Per Node Type (unchanged)

See `network_gen.py` `BASELINES` dict. Per-interface baselines are derived by distributing node-type utilization across interfaces proportionally to capacity.

---

## File 3: `src/data/anomaly_injector.py`

### Purpose
Inject realistic fault patterns into the running `NetworkTelemetryGenerator`. Each fault now targets **specific interfaces** and **specific tunnels**, not just node-level metrics.

### Fault Patterns — Per-Interface Targeting

#### Fault 1: Progressive Congestion Buildup
```
Target: CE wan0 interface (access link)
Cascade: Upstream PE interface facing the congested CE

Precursor (0–20 min):
  - CE wan0.util: ramps from baseline to 80%
  - CE wan0.queue: +2/min
  - CE wan0.latency: rises gradually
  - PE facing-interface (e.g., PE-1 eth1): util rises proportionally

Active (20–35 min):
  - CE wan0.util: 90–100%, drops_in spikes (15–75)
  - CE wan0.latency: 4–8× baseline
  - CE wan0.packet_loss: 2–8%
  - PE facing-interface: full congestion effects

Recovery (35–45 min):
  - Exponential decay on all interface metrics
```

#### Fault 2: BGP Route Flap + Downstream Cascade
```
Target: PE router (node-level BGP) → all PE interfaces affected
Cascade: CE wan0 interfaces facing the flapping PE

Precursor (0–8 min):
  - bgp_updates rising (node-level)
  - All PE interfaces: latency +2ms, loss +0.2%

Active (8–13 min):
  - bgp_active drops, bgp_updates/convergence storm (node-level)
  - PE interfaces: latency spike (+30ms), loss spike (+5%)
  - Cascade CE interfaces: latency +20ms, loss +4%

Recovery (13–20 min):
  - BGP stabilizes, interface metrics decay
```

#### Fault 3: MPLS Underlay Failure + Tunnel Degradation
```
Target: P-router core interfaces (eth2, eth3) + all tunnels through this P
Cascade: All downstream nodes' interfaces

Precursor (0–25 min):
  - P-router eth2/eth3: errors_in rising
  - All tunnels through P: loss rising (0→2.5%)
  - P-router CPU elevated

Active (25–45 min):
  - P-router all interfaces: FLAPPING then DOWN
  - P-router LDP sessions drop, LSPs break
  - All tunnels: severe degradation (loss 8–20%)
  - Downstream node interfaces: loss + reroute delay penalty

Recovery (45–60 min):
  - P interfaces → UP, tunnels recover exponentially
```

#### Fault 4: Controller Misconfiguration → Policy Drift
```
Target: PE access interfaces (eth1/eth2) + all PE tunnels
Cascade: Neighboring node interfaces

Precursor (0–12 min):
  - PE CPU +12%, bgp_updates +8/min
  - PE access interfaces: latency +3ms, loss +0.3%
  - vpn_routes, label_table_size: +25%

Active (12–32 min):
  - PE CPU 80–98%, memory 80–97%
  - PE routes 5× baseline, label table 3× baseline
  - All PE interfaces: latency spike, loss, queue blowout
  - All PE tunnels: loss +4%, latency +20ms

Recovery (32–42 min):
  - Config rolled back, route table drains, metrics decay
```

### Classes

See `anomaly_injector.py`. Key methods per fault now use `_get_all_ifaces(s)` and `_get_ifaces_for_neighbor(s, neighbor)` helpers to target the correct interfaces.

---

## File 4: `src/training/generate_dataset.py`

### CSV Columns

The CSV now contains per-interface, per-tunnel, NUMERIC_COLS aggregates, and labels. Example header (abbreviated):

```
timestamp,node_id,node_type,site,
eth0_errors_in,eth0_drops_in,eth0_utilization_pct,...,eth0_link_state,
eth1_errors_in,...,eth1_link_state,
eth2_errors_in,...,eth2_link_state,
eth3_errors_in,...,eth3_link_state,
wan0_errors_in,...,wan0_link_state,
lsp-b1-dc1_loss_pct,lsp-b1-dc1_jitter_ms,lsp-b1-dc1_latency_ms,
...,lsp-b3-dc2_latency_ms,
bytes_in,bytes_out,...,packet_loss_pct,
link_state,tunnel_packet_loss_pct,ipsec_rekeyed_last_hr,
fault_type,fault_phase,is_anomaly,is_precursor
```

Total columns: 92 (identity 4 + per-interface 5×8 + per-tunnel 6×3 + NUMERIC_COLS 23 + 3 non-numeric aggregates + labels 4)

### Graph Snapshot Format (unchanged)

Node features remain 20-element aggregates from `NUMERIC_COLS[:20]` for GAT model compatibility. The `_state_to_feature_row()` function computes aggregates from per-interface data.

---

## Dependencies (unchanged)
```
numpy
pandas
dataclasses (stdlib)
pickle (stdlib)
```
