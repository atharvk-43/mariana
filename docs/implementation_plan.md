# ISRO PS-13: Rich Telemetry + ML Architecture Plan

## Objective

Build a production-quality telemetry pipeline and ML system that detects **precursor conditions** — not just threshold breaches. The system must answer: *"The network will degrade in 3.2 hours"*, not just *"the CPU is high right now"*.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     TELEMETRY DATA LAYER                        │
│                                                                 │
│  NetworkSimulator (topology-aware, correlated, realistic)       │
│  ├── 10 nodes: PE routers, P routers, CE routers               │
│  ├── SNMP metrics per interface (bytes, packets, errors)        │
│  ├── Routing metrics (BGP sessions, OSPF neighbors, prefixes)   │
│  ├── Tunnel metrics (IPSec SA, jitter, loss, rekey)             │
│  ├── Injected anomaly patterns (DDoS, BGP flap, congestion)     │
│  └── Diurnal patterns (realistic traffic rhythms)              │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                    FEATURE ENGINEERING                          │
│  ├── Rolling windows (30s, 5min, 1hr) → mean, std, slope       │
│  ├── Rate-of-change (delta per second)                          │
│  ├── Graph features (neighbor health, betweenness centrality)   │
│  └── Cross-metric correlations (cpu × bytes, loss × jitter)    │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────────┐
         │               │                   │
┌────────▼────────┐ ┌────▼──────────┐ ┌──────▼────────────┐
│ LSTM Autoencoder│ │    Prophet    │ │  Graph Attention  │
│ (temporal anom) │ │(forecasting + │ │  Network (GAT)    │
│ Multivariate    │ │ precursor     │ │  topology-aware   │
│ sequence model  │ │ detection)    │ │  anomaly spread   │
└────────┬────────┘ └────┬──────────┘ └──────┬────────────┘
         │               │                   │
         └───────────────┼───────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                      ENSEMBLE LAYER                             │
│  ├── Weighted fusion of all 3 model scores + Isolation Forest  │
│  ├── Per-metric health: NORMAL / WARNING / CRITICAL            │
│  ├── Network-level risk score (0-100)                          │
│  └── Time-to-impact estimation (Prophet extrapolation)         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                   FastAPI + WebSocket
```

---

## Model Design Rationale

### Model 1: Isolation Forest (Baseline)
- **Role**: Unsupervised, stateless, per-datapoint anomaly scoring. Fast inference.
- **Input**: Current feature vector (all metrics for one node at one timestep)
- **Output**: `anomaly_score ∈ [-1, 1]`
- **Why keep it**: Already exists, gives instant per-point signal. Good at outlier detection.

### Model 2: LSTM Autoencoder (Temporal)
- **Role**: Learns what "normal" temporal sequences look like. Flags when reconstruction error spikes.
- **Input**: Sliding window of T=60 timesteps × M metrics per node
- **Output**: `reconstruction_error` per metric → anomaly if error > learned threshold
- **Strength**: Catches gradual drift (memory leak, latency creep) before it crosses a threshold
- **Training**: ~5000 windows of synthetic normal data (easy to generate)

### Model 3: Prophet (Forecasting + Precursor Detection)
- **Role**: Per-metric time-series forecasting. Forecasts 1–6 hours ahead with confidence intervals.
- **Input**: Historical values for one metric (e.g., `latency_ms` on `pe-router-1/eth1`)
- **Output**: `forecast_value`, `lower_bound`, `upper_bound` at t+1h, t+3h, t+6h
- **Precursor Logic**: If the forecast line is on a trajectory to breach threshold within N hours → emit early warning
- **Also**: Actual value outside `[lower, upper]` = anomaly signal (complements IF + LSTM)
- **Note**: One Prophet model per (node, metric) pair. Fit incrementally as data arrives.

### Model 4: Graph Attention Network — GAT (Topological)
- **Role**: Models how anomalies propagate through the network. A congested backbone router affects all downstream nodes — GAT captures this.
- **Input**: Node features (current telemetry) + adjacency matrix (network topology)
- **Output**: Per-node anomaly probability, accounting for neighbor state
- **Strength**: Routing instability detection — BGP/OSPF stress often manifests across multiple nodes simultaneously. GAT sees the pattern; per-node models don't.
- **Library**: `torch_geometric` (PyTorch Geometric). Lightweight, no CUDA required for our graph size (~10 nodes).
- **Architecture**: 2-layer GAT → node embeddings → binary anomaly head

### Ensemble Fusion
```python
final_risk_score = (
    0.20 * isolation_forest_score +
    0.30 * lstm_reconstruction_error +
    0.30 * prophet_deviation_from_forecast +
    0.20 * gat_anomaly_probability
)
```
Weights tuned on validation split. Final health = threshold on `final_risk_score`.

---

## PS-13 Objective Mapping

| PS-13 Requirement | Our Solution |
|---|---|
| Congestion buildup forecasting | Prophet per interface `bytes_in`, `utilization_pct` |
| Interface utilization saturation | Prophet forecasts crossing 90% threshold |
| Latency drift | LSTM autoencoder on `latency_ms` + Prophet trend |
| BGP/OSPF convergence stress | GAT — cross-node BGP session drops as graph signal |
| Route flapping precursors | Sliding window `bgp_updates` rate spike → LSTM flags |
| Path asymmetry | Per-path latency diff computed in feature engineering |
| Tunnel health degradation | Prophet on `jitter_ms`, `packet_loss_pct` per tunnel |
| IPSec rekey anomalies | Statistical z-score on `rekey_interval_deviation` |
| Time-to-impact estimation | Prophet forecast + slope extrapolation → ETA in hours |

---

## Telemetry Schema (Rich, Network-Domain)

```python
{
  # Identity
  "timestamp": "2026-06-18T02:15:30Z",
  "node_id": "pe-router-1",
  "node_type": "PE",           # PE | P | CE | DC
  "interface": "eth1",

  # SNMP Interface Metrics
  "bytes_in": 1823456,
  "bytes_out": 943210,
  "packets_in": 12400,
  "packets_out": 8900,
  "errors_in": 2,
  "errors_out": 0,
  "drops_in": 5,
  "drops_out": 1,
  "utilization_pct": 43.2,     # derived: bytes / interface_capacity
  "link_state": "UP",          # UP | DOWN | FLAPPING

  # Routing Plane Metrics
  "bgp_sessions_active": 3,
  "bgp_sessions_total": 3,
  "bgp_prefixes_received": 120,
  "bgp_updates_per_min": 2,    # high value = instability
  "bgp_withdrawals_per_min": 0,
  "ospf_neighbor_count": 2,
  "ospf_spf_runs": 1,          # high value = topology churn
  "route_table_size": 5432,

  # Performance
  "cpu_load_pct": 43.2,
  "memory_used_pct": 61.0,
  "queue_depth": 12,           # packets queued = congestion signal

  # Link Quality
  "latency_ms": 12.4,
  "jitter_ms": 1.1,
  "packet_loss_pct": 0.02,
  "rtt_ms": 24.8,

  # Tunnel / IPSec
  "ipsec_sa_active": 4,
  "ipsec_rekeyed_last_hr": 1,
  "tunnel_packet_loss_pct": 0.01,
  "tunnel_jitter_ms": 0.8,
  "tunnel_uptime_hrs": 142.3,

  # Derived / Engineered
  "latency_trend_slope": 0.12,   # ms/min over last 10 min
  "loss_trend_slope": 0.001,
  "bgp_churn_score": 0.0         # composite routing instability
}
```

---

## Anomaly Injection Patterns

| Pattern | Duration | Affected Metrics | Propagation |
|---|---|---|---|
| **BGP Session Drop** | Instant | `bgp_sessions_active` → 0, `bgp_updates` spikes | Downstream nodes lose routes |
| **Link Flap** | 60–120s | `link_state` alternates, `ospf_spf_runs` spikes | Neighbor routing instability |
| **DDoS Ingress** | 5–30 min | `bytes_in` ×10, `drops_in` ↑, `cpu_load` ↑ | Upstream congestion |
| **Congestion Cascade** | 10+ min | `latency_ms` drifts up, `queue_depth` ↑, `packet_loss` ↑ | Propagates downstream |
| **Memory Leak** | Hours | `memory_used_pct` slow monotonic rise → crash | Node-local only |
| **Hardware Degradation** | Hours | `errors_in` slowly increases → link failure | Local, then link |
| **IPSec Rekey Storm** | Minutes | `ipsec_rekeyed_last_hr` spikes, brief loss ↑ | Tunnel-local |
| **Route Leak** | Minutes | `route_table_size` sudden ×5, `cpu_load` ↑ | CPU stress on all nodes |

Each pattern has a **precursor phase** (2–60 min before peak) — this is what the models should detect.

---

## File Structure

```
backend/
├── data/
│   ├── __init__.py
│   ├── topology.py          # Network graph: nodes, edges, capacities
│   ├── network_gen.py       # Rich synthetic telemetry generator
│   └── anomaly_injector.py  # Realistic pattern injection with precursors
│
├── ml/
│   ├── __init__.py
│   ├── features.py          # Rolling stats, rate-of-change, graph features
│   ├── isolation_forest.py  # Refactored IF: callable, returns score + label
│   ├── lstm_ae.py           # LSTM Autoencoder: definition + inference
│   ├── prophet_model.py     # Per-metric Prophet: fit + forecast + anomaly
│   ├── gat_model.py         # Graph Attention Network: definition + inference
│   └── ensemble.py          # Fusion layer: combines all 4 scores
│
├── training/
│   ├── train_lstm.py        # Generate data → train → save lstm_ae.pt
│   ├── train_gat.py         # Generate graph data → train → save gat.pt
│   └── fit_prophet.py       # Fit Prophet per metric, save model files
│
├── inference/
│   └── pipeline.py          # End-to-end: raw row → risk score + prediction
│
├── api/
│   ├── main.py              # FastAPI: all endpoints + WebSocket stream
│   └── schemas.py           # Pydantic response models
│
├── models/                  # Saved model artifacts (gitignored if large)
│   ├── lstm_ae.pt
│   ├── gat.pt
│   ├── isolation_forest.pkl
│   └── prophet/             # 31 .pkl files (targeted lead metrics per fault)
│
├── data_store/
│   ├── telemetry.csv        # Historical (training) data
│   └── live_telemetry.csv   # Rolling live window
│
└── requirements.txt
```

---

## Training Strategy

### Data Generation
- Generate **7 days** of synthetic telemetry at 2s intervals = ~302,400 rows
- 80% normal operation, 20% with injected anomaly patterns
- Label each row: `anomaly_type`, `anomaly_phase` (none/precursor/active/recovery)
- **Precursor labels are key** — this is what differentiates the system

### LSTM Training
- Train on **normal-only** sequences (autoencoder learns normal = low reconstruction)
- Threshold: 95th percentile of reconstruction error on validation set
- **Kaggle GPU (T4/P100)**: Larger model — hidden_dim=256, 3 LSTM layers, T=120 window
- Expected training time on Kaggle GPU: ~5–10 minutes

### GAT Training
- Supervised: labeled `(graph_snapshot, anomaly_label)` pairs
- One graph per timestep: 10 nodes × 25 features + edge attributes
- Train/val/test: 60/20/20 split by time (no leakage)
- **Kaggle GPU**: 3-layer GAT, hidden_dim=128, 8 attention heads, 100 epochs
- Expected training time on Kaggle GPU: ~3–5 minutes

### Prophet Fitting
- Fit one Prophet model per (node, metric) pair on 7-day history
- Use `add_seasonality(period=24, fourier_order=5)` for diurnal patterns
- Re-fit daily as new data arrives (incremental)

---

## Key API Endpoints (New)

```
GET  /telemetry/latest           → latest snapshot all nodes + health labels + risk scores
GET  /telemetry/history?n=500    → last N rows
GET  /telemetry/node/{node_id}   → single node detail + all model scores
GET  /alerts/active              → current WARNING/CRITICAL events with time-to-impact
GET  /forecast/{node_id}/{metric} → Prophet forecast: next 1h, 3h, 6h + confidence
GET  /topology/health            → full graph with per-node risk scores (for viz)
POST /telemetry/ingest           → receive real data from EC2 InfluxDB
WS   /ws/stream                  → real-time push every 2s
```

---

## Training Workflow (Kaggle GPU)

1. Run `training/generate_dataset.py` locally → produces `telemetry_train.csv` + `graph_snapshots.pkl`
2. Upload both files to a Kaggle dataset
3. Run `training/train_lstm_kaggle.ipynb` on Kaggle T4 GPU → download `lstm_ae.pt`
4. Run `training/train_gat_kaggle.ipynb` on Kaggle T4 GPU → download `gat.pt`
5. Drop both `.pt` files into `backend/models/`
6. Prophet fits locally (100 models × ~3s = ~5 min, parallelized)

Kaggle GPU enables: deeper LSTM (3 layers, hidden=256, seq_len=120), stronger GAT (3 layers, 8 heads, hidden=128).

## Open Questions

> [!NOTE]
> **torch_geometric on Kaggle**: Works out of the box on Kaggle Linux kernels. Use `!pip install torch_geometric` in the notebook. No Windows install pain.

> [!NOTE]
> **Prophet fitting**: 100 models × ~3s = ~5 min locally with `multiprocessing`. Acceptable for one-time fit. Re-fit nightly if doing a live demo.

> [!NOTE]
> **Real EC2 data vs. synthetic**: The system is designed so that `network_gen.py` can be swapped out for a real InfluxDB client with zero changes to the ML layer. The schema is the contract.

---

## Verification Plan

1. **Unit**: Each model class has a `test_inference()` method that runs with dummy data and asserts output shape/types
2. **Integration**: `python inference/pipeline.py --demo` runs the full pipeline on 10 synthetic rows and prints scores
3. **API**: `curl http://localhost:8000/telemetry/latest` returns full JSON with all model outputs
4. **WebSocket**: Browser console test — connect to `ws://localhost:8000/ws/stream`, verify messages arrive every 2s
5. **Anomaly demo**: Inject a BGP drop via API `POST /demo/inject_anomaly`, verify all 4 models flag it within 30s
