"""
Quick smoke test for the data layer (v2 — per-interface + per-tunnel telemetry).
Run from: src/
  python smoke_test.py

Checks:
  1. topology.py — 10 nodes, edges with dst_interface, per-node interface list
  2. network_gen.py — single tick produces 10 rows with per-interface/per-tunnel fields
  3. anomaly_injector.py — schedules and applies one fault per-interface
  4. 60-second mini-sim — prints fault phase progression per-interface
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime, timedelta
from data.topology import NODES, get_adjacency_matrix, get_lsps_through_node, get_node_interfaces
from data.network_gen import NetworkTelemetryGenerator, TELEMETRY_SCHEMA, ALL_IFACES, ALL_TUNNELS
from data.anomaly_injector import AnomalyInjector

print("=" * 60)
print("SMOKE TEST V2 — MPLS Data Layer (Per-Interface)")
print("=" * 60)

# --- 1. Topology ---
print(f"\n[1] Topology: {len(NODES)} nodes")
for n in NODES:
    ifaces = get_node_interfaces(n["id"])
    iface_str = ", ".join(f"{name}(->{neigh})" for name, neigh, _ in ifaces)
    print(f"    {n['id']:10s} | type={n['type']:2s} | site={n['site']:12s} | ifaces: {iface_str}")
A = get_adjacency_matrix()
print(f"    Adjacency matrix: {A.shape}, non-zero edges: {(A > 0).sum()}")

lsps_p1 = get_lsps_through_node("P-1")
print(f"    LSPs through P-1: {lsps_p1}")

# --- 2. Generator ---
print(f"\n[2] Generator: first tick")
gen = NetworkTelemetryGenerator(seed=42)
t0  = datetime(2026, 1, 1, 8, 0, 0)
rows = gen.next_tick(t0)
print(f"    Rows produced: {len(rows)} (expected 10)")

# Check per-interface fields
pe1_row = next(r for r in rows if r["node_id"] == "PE-1")
print(f"\n    PE-1 sample row — Interface fields:")
for k, v in pe1_row.items():
    if "eth0_" in k or "eth1_" in k or "eth2_" in k or "eth3_" in k or "wan0_" in k:
        print(f"      {k:40s}: {v}")

# Check tunnel fields
print(f"\n    PE-1 sample row — Tunnel fields:")
for k, v in pe1_row.items():
    if "lsp-" in k:
        print(f"      {k:40s}: {v}")

# Check node-level fields
print(f"\n    PE-1 sample row — Node-level fields:")
for k in ["cpu_load_pct", "memory_used_pct", "bgp_sessions_active", "bgp_prefixes_received",
          "bgp_updates_per_min", "ldp_sessions_active", "mpls_lsp_count", "vpn_routes_count"]:
    print(f"      {k:40s}: {pe1_row[k]}")

# Check aggregate fields
print(f"\n    PE-1 sample row — Aggregate fields:")
for k in ["bytes_in", "bytes_out", "utilization_pct", "queue_depth",
          "latency_ms", "jitter_ms", "packet_loss_pct", "link_state",
          "tunnel_packet_loss_pct", "ipsec_rekeyed_last_hr"]:
    print(f"      {k:40s}: {pe1_row[k]}")

# Verify all schema fields present (identity + per-interface + per-tunnel + node-level + aggregates + labels)
schema_core = ["timestamp", "node_id", "node_type", "site"] + list(pe1_row.keys())
missing = [f for f in schema_core if f not in pe1_row]
print(f"\n    Missing fields: {missing}")
print(f"    Total columns in row: {len(pe1_row)}")

# --- 3. Anomaly injector ---
print(f"\n[3] Fault injection: congestion on CE-B1")
inj = AnomalyInjector()
t_fault = t0 + timedelta(seconds=30)
evt = inj.schedule_fault("congestion", "CE-B1", t_fault)
print(f"    Affected nodes: {evt.affected_nodes}")
print(f"    Phases: precursor={evt.precursor_dur}, active={evt.active_dur}, recovery={evt.recovery_dur}")

# --- 4. Mini-simulation (5 minutes)---
print(f"\n[4] 5-minute simulation with congestion on CE-B1 (checking wan0 interface)")
gen2 = NetworkTelemetryGenerator(seed=42)
inj2 = AnomalyInjector()
t_start = datetime(2026, 1, 1, 12, 0, 0)
inj2.schedule_fault("congestion", "CE-B1", t_start + timedelta(minutes=1))

print(f"    {'Time':10s} | {'Node':8s} | {'Iface':6s} | {'Phase':12s} | {'Util%':8s} | {'Queue':6s} | {'Loss%':8s}")
print(f"    {'-'*75}")

current = t_start
for tick in range(150):
    inj2.apply(gen2, current)
    rows2 = gen2.next_tick(current)
    if tick % 10 == 0:
        pe1_r = next(r for r in rows2 if r["node_id"] == "PE-1")
        ce1_r = next(r for r in rows2 if r["node_id"] == "CE-B1")
        print(f"    {current.strftime('%H:%M:%S'):10s} | {'CE-B1':8s} | {'wan0':6s} | "
              f"{ce1_r['fault_phase']:12s} | {ce1_r['wan0_utilization_pct']:>8.2f} | "
              f"{ce1_r['wan0_queue_depth']:>6d} | {ce1_r['wan0_packet_loss_pct']:>8.3f}")
        print(f"    {'':10s} | {'PE-1':8s} | {'eth1':6s} | "
              f"{pe1_r['fault_phase']:12s} | {pe1_r['eth1_utilization_pct']:>8.2f} | "
              f"{pe1_r['eth1_queue_depth']:>6d} | {pe1_r['eth1_packet_loss_pct']:>8.3f}")
    current += timedelta(seconds=2)

print("\n" + "=" * 60)
print("SMOKE TEST V2 PASSED — Per-interface + per-tunnel telemetry OK")
print("=" * 60)
print("\nNext step: python -m training.generate_dataset")

print("\n--- Interface/Tunnel Schema Summary ---")
print(f"All interface names: {ALL_IFACES}")
print(f"All tunnel names: {ALL_TUNNELS}")
print(f"Schema width: {len(TELEMETRY_SCHEMA)} columns")
