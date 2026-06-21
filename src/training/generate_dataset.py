"""
Dataset Generator
=================
Generates training-quality synthetic telemetry for the MPLS NOC project.

Produces:
  telemetry_train.csv       — 3 days of data, ~5.5M rows/hour, ~1.3M total
  telemetry_val.csv         — 1 day of data (held out)
  graph_snapshots.pkl       — graph-level snapshots every 60 ticks for GAT training

Usage:
  python -m training.generate_dataset

Fault injection schedule:
  - 1 fault per ~8 hours, all 4 fault types rotated
  - ~6% anomaly rate in final dataset (precursor + active + recovery combined)
  - Each fault type appears at least twice in training set

Output rows per day: 10 nodes × 43200 ticks (2s intervals) = 432,000 rows/day
Train (3 days): ~1,296,000 rows — enough for LSTM seq_len=60 with ample sequences
Val   (1 day):  ~432,000 rows
"""

import os
import sys
import csv
import pickle
import argparse
import logging
import collections
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

# Make sure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data.topology import NODES, NODE_IDS, get_adjacency_matrix, NODE_INDEX
from data.network_gen import NetworkTelemetryGenerator, NUMERIC_COLS
from data.anomaly_injector import AnomalyInjector, FAULT_TYPES, ELIGIBLE_PRIMARY

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TICK_INTERVAL_SEC = 2
SNAPSHOT_EVERY_N  = 60    # save graph snapshot every 60 ticks (2 minutes)
TRAIN_DAYS        = 3
VAL_DAYS          = 1
START_DATE        = datetime(2026, 1, 1, 0, 0, 0)   # fixed for reproducibility
SEED              = 42
BASE_DIR          = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR        = os.path.join(BASE_DIR, "data")

# Fault injection schedule:
# One fault every ~8 hours of simulated time, all 4 types rotated
FAULT_CYCLE = ["congestion", "bgp_flap", "mpls_failure", "policy_drift"]

# Primary node for each fault type (fixed for reproducibility)
FAULT_PRIMARIES = {
    "congestion":   ["CE-B1", "CE-B2", "CE-B3"],
    "bgp_flap":     ["PE-1",  "PE-2"],
    "mpls_failure": ["P-1",   "P-2",   "P-3"],
    "policy_drift": ["PE-1",  "PE-2"],
}

# ---------------------------------------------------------------------------
# Schedule builder
# ---------------------------------------------------------------------------

def build_fault_schedule(start: datetime, end: datetime, seed: int) -> list[dict]:
    """
    Build a deterministic list of fault events for the simulation window.
    Returns list of dicts: {fault_type, primary_node, start_time}
    """
    rng   = np.random.default_rng(seed)
    faults = []
    cursor = start + timedelta(hours=4)   # first fault 4 hours in (warm-up period)
    idx    = 0

    while cursor < end - timedelta(hours=2):
        ft      = FAULT_CYCLE[idx % len(FAULT_CYCLE)]
        primaries = FAULT_PRIMARIES[ft]
        primary   = primaries[rng.integers(0, len(primaries))]

        # Jitter start time ±30 minutes
        jitter = timedelta(minutes=int(rng.integers(-30, 30)))
        faults.append({
            "fault_type":   ft,
            "primary_node": primary,
            "start_time":   cursor + jitter,
        })

        # Next fault in 7–9 hours
        gap_hours = 7 + rng.integers(0, 2)
        cursor   += timedelta(hours=int(gap_hours))
        idx      += 1

    log.info(f"Scheduled {len(faults)} faults over {(end - start).days} days")
    for f in faults:
        log.info(f"  {f['fault_type']:15s} on {f['primary_node']:8s} at {f['start_time'].strftime('%Y-%m-%d %H:%M')}")
    return faults


# ---------------------------------------------------------------------------
# Core simulation runner
# ---------------------------------------------------------------------------

def simulate(
    start: datetime,
    end:   datetime,
    generator: NetworkTelemetryGenerator,
    injector:  AnomalyInjector,
    output_csv: str,
    snapshot_every: int = SNAPSHOT_EVERY_N,
    flush_every: int = 50_000,
) -> tuple[list[dict], int, int, collections.Counter]:
    """
    Run the simulation from start to end.
    Writes CSV incrementally to avoid holding all rows in memory.
    Returns (graph_snapshots, total_rows, stats_tracker).
    stats_tracker = Counter of fault_type per row + anomaly count.
    """
    total_ticks  = int((end - start).total_seconds() / TICK_INTERVAL_SEC)
    snapshots    = []
    current_time = start
    buffer     = []
    header_written = False
    anomaly_count  = 0
    fault_counts   = collections.Counter()
    total_rows     = 0

    with open(output_csv, "w", newline="") as f:
        fieldnames = None

        for tick in tqdm(range(total_ticks), desc="Simulating", unit="tick", ncols=80, position=0, leave=True):
            injector.apply(generator, current_time)
            rows = generator.next_tick(current_time)
            buffer.extend(rows)

            if tick % snapshot_every == 0:
                snapshots.append(_build_snapshot(generator, current_time))

            if len(buffer) >= flush_every:
                if fieldnames is None:
                    fieldnames = list(buffer[0].keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    header_written = True
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerows(buffer)
                total_rows += len(buffer)
                for r in buffer:
                    if r["is_anomaly"]:
                        anomaly_count += 1
                    fault_counts[r["fault_type"]] += 1
                buffer.clear()

            current_time += timedelta(seconds=TICK_INTERVAL_SEC)

        # Flush remaining
        if buffer:
            if fieldnames is None:
                fieldnames = list(buffer[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(buffer)
            total_rows += len(buffer)
            for r in buffer:
                if r["is_anomaly"]:
                    anomaly_count += 1
                fault_counts[r["fault_type"]] += 1

    return snapshots, total_rows, anomaly_count, fault_counts


def _build_snapshot(generator: NetworkTelemetryGenerator, t: datetime) -> dict:
    """
    Build a graph snapshot for the GAT model.
    node_features: (10, 20) numpy array — first 20 NUMERIC_COLS
    node_labels: (10,) binary — 1 if anomalous, 0 if normal
    edge_index: COO from topology (static, same every snapshot)
    """
    from data.topology import get_edge_index_coo
    states = generator.get_all_states()

    node_features = np.zeros((len(NODE_IDS), 20), dtype=np.float32)
    node_labels   = np.zeros(len(NODE_IDS), dtype=np.int32)
    
    gat_cols = NUMERIC_COLS[:20]   # first 20 numeric features for GAT

    for i, nid in enumerate(NODE_IDS):
        s = states[nid]
        # Build a quick feature row (same order as serialize → NUMERIC_COLS)
        row_values = _state_to_feature_row(s)
        node_features[i] = row_values[:20]
        node_labels[i]   = 1 if s.fault_type != "none" else 0

    src_list, dst_list = get_edge_index_coo()
    return {
        "timestamp":     t.isoformat(),
        "node_features": node_features,
        "node_labels":   node_labels,
        "edge_index":    (src_list, dst_list),
    }


def _state_to_feature_row(s) -> list[float]:
    """Extract NUMERIC_COLS features from NodeState using per-interface aggregates."""
    ifaces = list(s.interfaces.values())

    if ifaces:
        total_bytes_in = sum(
            iface.util * iface.capacity_mbps * 1e6 / 8 * 2 * 0.6
            for iface in ifaces
        )
        total_bytes_out = sum(
            iface.util * iface.capacity_mbps * 1e6 / 8 * 2 * 0.4
            for iface in ifaces
        )
        errors   = sum(iface.errors_in for iface in ifaces)
        drops_in = sum(iface.drops_in for iface in ifaces)
        drops_out= sum(iface.drops_out for iface in ifaces)
        util     = max(iface.util * 100.0 for iface in ifaces)
        queue    = max(iface.queue for iface in ifaces)
    else:
        total_bytes_in = 0.0
        total_bytes_out = 0.0
        errors = drops_in = drops_out = 0.0
        util = queue = 0.0

    return [
        total_bytes_in, total_bytes_out,
        0, 0,   # packets (derived, not critical for GAT)
        errors, drops_in, drops_out,
        util,
        float(s.bgp_active), s.bgp_prefixes,
        s.bgp_updates, s.bgp_withdrawals, s.ospf_spf,
        float(s.ldp_active), float(s.lsp_count),
        float(s.label_table_size), float(s.vpn_routes),
        s.cpu, s.memory, queue,
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate MPLS telemetry dataset")
    parser.add_argument("--train-days", type=int, default=TRAIN_DAYS)
    parser.add_argument("--val-days",   type=int, default=VAL_DAYS)
    parser.add_argument("--seed",       type=int, default=SEED)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_start = START_DATE
    train_end   = train_start + timedelta(days=args.train_days)
    val_start   = train_end
    val_end     = val_start + timedelta(days=args.val_days)

    log.info(f"Training window: {train_start} → {train_end} ({args.train_days} days)")
    log.info(f"Validation window: {val_start} → {val_end} ({args.val_days} days)")
    log.info(f"Expected rows: train={args.train_days * 432000:,}, val={args.val_days * 432000:,}")
    log.info(f"Seed: {args.seed}")

    # ---- Train set ----
    log.info("=== Generating training set ===")
    gen_train  = NetworkTelemetryGenerator(seed=args.seed)
    inj_train  = AnomalyInjector()
    schedule   = build_fault_schedule(train_start, train_end, seed=args.seed)
    for f in schedule:
        inj_train.schedule_fault(f["fault_type"], f["primary_node"], f["start_time"])

    train_path = os.path.join(args.output_dir, "telemetry_train.csv")
    train_snapshots, train_count, train_anomalies, train_faults = simulate(
        train_start, train_end, gen_train, inj_train, train_path,
    )

    # Save training graph snapshots
    snap_path = os.path.join(args.output_dir, "graph_snapshots_train.pkl")
    with open(snap_path, "wb") as f:
        pickle.dump(train_snapshots, f, protocol=4)
    log.info(f"Saved training telemetry: {train_path} ({train_count:,} rows)")
    log.info(f"Saved training snapshots: {snap_path} ({len(train_snapshots):,} snapshots)")

    # Print anomaly stats
    log.info(f"Anomaly rate in training set: {100 * train_anomalies / train_count:.1f}%")
    log.info("Fault type breakdown:")
    for ft, cnt in sorted(train_faults.items()):
        log.info(f"  {ft}: {cnt}")

    # ---- Validation set ----
    log.info("=== Generating validation set ===")
    gen_val   = NetworkTelemetryGenerator(seed=args.seed + 1)
    inj_val   = AnomalyInjector()
    schedule_v = build_fault_schedule(val_start, val_end, seed=args.seed + 1)
    for f in schedule_v:
        inj_val.schedule_fault(f["fault_type"], f["primary_node"], f["start_time"])

    val_path = os.path.join(args.output_dir, "telemetry_val.csv")
    val_snapshots, val_count, val_anomalies, val_faults = simulate(
        val_start, val_end, gen_val, inj_val, val_path,
    )

    snap_val_path = os.path.join(args.output_dir, "graph_snapshots_val.pkl")
    with open(snap_val_path, "wb") as f:
        pickle.dump(val_snapshots, f, protocol=4)
    log.info(f"Saved validation telemetry: {val_path} ({val_count:,} rows)")
    log.info(f"Saved validation snapshots: {snap_val_path} ({len(val_snapshots):,} snapshots)")

    log.info("=== Dataset generation complete ===")
    log.info(f"  Training CSV:         {train_path}")
    log.info(f"  Validation CSV:       {val_path}")
    log.info(f"  Training snapshots:   {snap_path}")
    log.info(f"  Validation snapshots: {snap_val_path}")
    log.info("Next step: upload telemetry_train.csv + graph_snapshots_train.pkl to Kaggle Dataset")


if __name__ == "__main__":
    main()
