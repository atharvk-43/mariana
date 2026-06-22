# ISRO PS-13 — Hackathon Roadmap (Ground Truth Version)

> Last updated after reading the actual problem statement PDF.

---

## What the Problem Statement Actually Is

**"Air-Gapped Predictive Copilot for Secure MPLS Operations"**

An autonomous, **offline AI NOC Copilot** that:
1. Simulates a real SD-WAN/MPLS network
2. Predicts network failures *before* impact (not after)
3. Explains reasoning in natural language via a **local LLM**
4. Runs with **zero outbound internet dependency** (air-gapped)

---

## Actual Judge Evaluation Weights

| Criterion | Weight | What It Means |
|---|---|---|
| **Technical Merit** | 35% | ML prediction accuracy + lead time before fault |
| **Copilot Effectiveness** | 35% | Offline LLM quality — correct, grounded, no hallucination |
| **Security / Air-Gap Compliance** | 20% | Verifiably zero outbound dependency at runtime |
| **Documentation** | 10% | Architecture clarity, design rationale |

> [!CAUTION]
> The previous roadmap had completely wrong scoring weights (30/25/25/10). The **Offline LLM Copilot** is 35% of your score — equal to all ML work — and was not in the plan at all.

---

## Current State vs. What's Needed

| Layer | Status | Gap |
|---|---|---|
| Network Simulation | ✅ Complete | 10-node topology, 92-col telemetry, 4 fault types |
| Telemetry Schema | ✅ Complete | 23 NUMERIC_COLS aggregates + per-interface fields |
| ML Models — GAT | ✅ Trained (0.941 AUROC) | Downloaded to `src/models/` |
| ML Models — IF | ✅ Trained (0.898 AUROC) | Full HPO + final training done |
| ML Models — LSTM AE | ⚠️ Trained (0.664 AUROC) | Re-submitted with 200 epochs |
| ML Models — Prophet | ✅ Trained (30 models) | Downloaded with manifest.json |
| **Ensemble Wiring** | ❌ Not wired | Model files exist, fusion layer pending |
| **Offline LLM Copilot** | ⚠️ Skeleton | Ollama client + RAG pipeline exist, not integrated with real scores |
| **RAG Pipeline** | ⚠️ Exists | 6 runbooks + past incidents, not wired to ensemble |
| **Air-Gap Compliance** | ❌ Not verified | System currently uses internet-dependent packages |
| Frontend Dashboard | ❌ Empty | `src/frontend/` directory exists but no code |

---

## Four Objectives from PS (in order of score impact)

### Obj 3+4 (combined 35%+20% impact): Offline LLM Copilot + Workflow Automation
- Local quantized LLM: **Phi-3-mini** or **Mistral 7B** (via Ollama)
- RAG over: topology maps, runbooks, past incident records
- Structured response per alert:
  - Predicted issue type
  - Confidence score
  - Probable root cause
  - Affected sites/services
  - Estimated time-to-impact
  - Recommended corrective action
- Natural language query interface for NOC operators
- Confidence-scored alert prioritization (reduce alert fatigue)
- Automated playbook suggestion per fault type

### Obj 2 (35% technical merit): Predictive Fault Analytics
- Time-series forecasting: congestion, utilization, latency drift → **Prophet**
- Routing instability: BGP/OSPF convergence stress, route flapping → **GAT (graph AE)**
- Tunnel degradation scoring: packet loss trend, jitter, rekey → **LSTM Autoencoder**
- Time-to-impact estimation: actionable lead times → **Prophet extrapolation + slope**

### Obj 1 (foundation): Simulated SD-WAN/MPLS
- Multi-site topology: branch, hub, datacenter nodes
- CE/PE/P router roles
- MPLS + VPN + BGP/OSPF + IPSec tunnels
- Realistic traffic + fault injection

### Obj 4 (workflow): NOC Automation
- Alert prioritization by confidence score
- Graph-based event correlation (which fault caused which downstream effect)
- Incident summaries with impact and urgency

---

## ML Architecture — Final Grounded Design

All four models are **unsupervised or self-supervised** — no dependency on perfect fault injection labels.

```
Raw Telemetry (23 fields per node, 2s intervals)
          ↓
Feature Engineering
  - Rolling 30s / 5min windows: mean, std, slope
  - Rate-of-change per metric
  - Cross-metric: cpu × bytes, loss × jitter
  - Graph features: neighbor health aggregates
          ↓
┌─────────────────────────────────────────────────┐
│  Model 1: Isolation Forest                      │
│  Role: Stateless per-point outlier scoring      │
│  Mode: Unsupervised (no labels needed)          │
│  Best at: Sudden spikes, obvious outliers        │
├─────────────────────────────────────────────────┤
│  Model 2: LSTM Autoencoder                      │
│  Role: Pattern-level anomaly detection          │
│  Mode: Train on NORMAL sequences only           │
│  Best at: Gradual drift, multi-metric patterns  │
│  Why AE not plain LSTM: Prophet already does    │
│  next-step forecasting. AE fills a different    │
│  gap — it detects abnormal SEQUENCE SHAPES,     │
│  not just surprising next values.               │
├─────────────────────────────────────────────────┤
│  Model 3: Prophet                               │
│  Role: Per-metric forecasting + precursor det.  │
│  Mode: Statistical, no class labels             │
│  Best at: Time-to-impact estimation, trend      │
│  Output: "latency_ms will breach 150ms in 2.4h" │
├─────────────────────────────────────────────────┤
│  Model 4: GAT Graph Autoencoder                 │
│  Role: Topology-aware anomaly propagation       │
│  Mode: Unsupervised graph reconstruction        │
│  Best at: Multi-node correlated events          │
│  (BGP drop on PE-1 → downstream instability)   │
└─────────────────────────────────────────────────┘
          ↓
Ensemble: weighted combination of all 4 scores
  → final_risk_score (0–100)
  → health_state: NORMAL / WARNING / CRITICAL
  → time_to_impact_hours (from Prophet)
          ↓
FastAPI (air-gapped) → Offline LLM Copilot
```

**Why these 4 models cover different failure modes:**
- IF: catches the sudden (DDoS, BGP drop)
- LSTM AE: catches the gradual (latency creep, memory leak)  
- Prophet: forecasts the future (when will it fail)
- GAT: catches the correlated (fault propagating through network)

**Class imbalance handling**: Moot. All 4 are unsupervised/self-supervised. None need fault labels for training.

---

## Telemetry Schema (92 columns)

Each row represents one node at one timestamp. Only fields directly manipulated by fault injectors are kept per-interface — the 5 removed fields (bytes_in/out, packets_in/out, drops_out) are computed internally for NUMERIC_COLS aggregates.

| Tier | Fields | Count |
|------|--------|-------|
| **Identity** | `timestamp`, `node_id`, `node_type`, `site` | 4 |
| **Per-Interface** (×5 ifaces) | `{iface}_errors_in`, `{iface}_drops_in`, `{iface}_utilization_pct`, `{iface}_queue_depth`, `{iface}_latency_ms`, `{iface}_jitter_ms`, `{iface}_packet_loss_pct`, `{iface}_link_state` | 8/iface = 40 |
| **Per-Tunnel** (×6 LSPs) | `{tunnel}_loss_pct`, `{tunnel}_jitter_ms`, `{tunnel}_latency_ms` | 3/tunnel = 18 |
| **NUMERIC_COLS** (23) + extras | `bytes_in`, `bytes_out`, `packets_in`, `packets_out`, `errors_in`, `drops_in`, `drops_out`, `utilization_pct`, `bgp_sessions_active`, `bgp_prefixes_received`, `bgp_updates_per_min`, `bgp_withdrawals_per_min`, `ospf_spf_runs`, `ldp_sessions_active`, `mpls_lsp_count`, `mpls_label_table_size`, `vpn_routes_count`, `cpu_load_pct`, `memory_used_pct`, `queue_depth`, `latency_ms`, `jitter_ms`, `packet_loss_pct` + `link_state`, `tunnel_packet_loss_pct`, `ipsec_rekeyed_last_hr` | 26 |
| **Labels** | `fault_type`, `fault_phase`, `is_anomaly`, `is_precursor` | 4 |

---

## Offline LLM Copilot Stack

```
Ollama (local model runner)
  └── Phi-3-mini-4k-instruct.Q4_K_M.gguf   ← ~2.5GB, fast on CPU
       OR Mistral-7B-v0.3.Q4_K_M.gguf      ← ~4.1GB, better quality

ChromaDB (local vector store)
  └── Documents to RAG over:
        - topology.json (node/link metadata)
        - runbooks/ (BGP recovery, congestion response, etc.)
        - past_incidents.json (simulated incident history)

LangChain (orchestration)
  └── RAGChain: alert_context → retrieve → LLM → structured response

FastAPI endpoint:
  POST /copilot/query
  Body: { "alert": {...}, "question": "What should we do?" }
  Response: {
    "predicted_issue": "BGP session failure on pe-router-1",
    "confidence": 0.87,
    "root_cause": "Sustained packet loss > 3% on eth1 for 8 minutes",
    "affected_scope": ["branch-site-2", "branch-site-3"],
    "time_to_impact_hours": 1.4,
    "recommended_action": "Check physical layer eth1, verify BGP keepalive timers...",
    "runbook_reference": "runbook_bgp_recovery_v2.md"
  }
```

**Why Phi-3-mini over Mistral 7B for hackathon**: Fits comfortably in 8GB RAM, faster inference, still very capable for structured NOC responses. Mistral if EC2 has RAM headroom after Containerlab.

**Air-gap compliance**: Ollama runs entirely local. ChromaDB is embedded (no server). Zero outbound calls. Demo: disable network interface, show the copilot still works.

---

## File Structure (Complete)

```
src/                              # PS-13 NOC Copilot (our team)
├── data/
│   ├── topology.py              # Network graph: 10 nodes, per-interface edges (5-tuple)
│   ├── network_gen.py           # Per-interface + per-tunnel telemetry (92 fields)
│   └── anomaly_injector.py      # Per-interface fault injection with precursor phases
│
├── ml/
│   ├── features.py              # Feature engineering: rolling stats, slopes
│   ├── isolation_forest.py      # Callable IF: returns score
│   ├── lstm_ae.py               # LSTM Autoencoder: train on normal, score sequences
│   ├── prophet_model.py         # Per-metric Prophet: fit + forecast + time-to-breach
│   ├── gat_model.py             # Graph AE: node reconstruction error
│   └── ensemble.py              # Fuse all 4 → risk_score + health_state + ETA
│
├── training/
│   ├── generate_dataset.py      # Run locally: produce telemetry_train.csv
│   ├── train_if.py              # Local IF training script
│   └── train_prophet.py         # Local Prophet training script
│
├── copilot/
│   ├── ollama_client.py         # Wrapper around local Ollama API
│   ├── rag_pipeline.py          # ChromaDB + LangChain RAG chain
│   ├── runbooks/                # 6 markdown runbooks (BGP, congestion, MPLS, etc.)
│   └── past_incidents.json      # Simulated incident history for RAG context
│
├── api/
│   ├── main.py                  # FastAPI: all endpoints + WebSocket
│   └── schemas.py               # Pydantic models for all responses
│
├── models/
│   ├── lstm_ae.pt               # Downloaded from Kaggle (pending re-submit)
│   ├── lstm_scaler.pkl          # Downloaded from Kaggle
│   ├── gat.pt                   # Downloaded from Kaggle
│   ├── gat_scaler.pkl           # Downloaded from Kaggle
│   ├── isolation_forest.pkl     # Trained locally on 300K rows
│   ├── if_scaler.pkl            # Trained locally
│   └── prophet/                 # 30 .pkl files (10 nodes × 3 lead metrics) + manifest.json
│
├── frontend/                    # Empty — pending implementation
│
└── smoke_test.py                # V2: tests per-interface + per-tunnel telemetry

nbs/                              # Kaggle training notebooks
├── train_lstm_ae_kaggle.ipynb   # LSTM AE on T4×2 (25 trials HPO, 200 epochs)
├── train_gat_kaggle.ipynb       # GAT AE on P100 (25 trials HPO, 150 epochs)
├── train_if_kaggle.ipynb        # IF on CPU (15 trials HPO, 70 features)
└── train_prophet_kaggle.ipynb   # Prophet on CPU (5 combos grid search, 30 models)
```

---

## API Endpoints

```
GET  /telemetry/latest                → all nodes: current metrics + risk scores + health
GET  /telemetry/history?node=X&n=300 → time series for charts
GET  /alerts/active                   → current WARNING/CRITICAL with time-to-impact
GET  /forecast/{node}/{metric}        → Prophet: next 1h/3h/6h + confidence bounds
GET  /topology/health                 → graph: nodes + edges with health colors
POST /copilot/query                   → LLM copilot structured response
POST /demo/inject_fault               → trigger anomaly scenario for live demo
WS   /ws/stream                       → 2s push: latest telemetry + active alerts
```

---

## Fault Scenarios for Demo (From PS — Required)

PS explicitly lists test scenarios. Build and rehearse these:

| Scenario | What It Tests | Demo Script |
|---|---|---|
| Progressive congestion on hub-spoke link | Prophet detects utilization trend → time-to-impact | "Model predicted this 3.2 hours before breach" |
| BGP route flap + downstream reroute cascade | GAT: multi-node correlated event | Topology viz lights up cascade path |
| Intermittent MPLS underlay failure + tunnel degradation | LSTM AE: pattern-level degradation | Copilot: "Recommend checking IPSec rekey policy" |
| Controller misconfiguration → policy drift | IF: sudden distributional shift | Copilot gives runbook reference |

---

## Execution Status

### Block 1 — Foundation ✅ Complete
- [x] Lock telemetry schema (23 NUMERIC_COLS + per-interface)
- [x] Build `topology.py` — 10-node network graph
- [x] Build `network_gen.py` — correlated synthetic telemetry (92 cols)
- [x] Build `anomaly_injector.py` — 4 fault scenarios (congestion, bgp_flap, mpls_failure, link_flap)
- [x] Generate `telemetry_train.csv` + `telemetry_val.csv` + `graph_snapshots.pkl`

### Block 2 — ML ⏳ Model training complete, ensemble pending
- [x] `features.py` — rolling windows (30s/5min/1hr), slope features
- [x] `isolation_forest.py` — callable class, trained (0.898 AUROC)
- [x] `prophet_model.py` — 30 models trained (10 nodes × 3 lead metrics)
- [x] `lstm_ae.py` — trained on Kaggle T4×2 (0.664 AUROC, re-submitted with 200 ep)
- [x] `gat_model.py` — trained on Kaggle P100 (0.941 AUROC)
- [ ] `ensemble.py` — fuse all 4 scores (pending)

### Block 3 — LLM Copilot ⚠️ Code exists, real integration pending
- [x] `ollama_client.py` — HTTP wrapper for local Ollama
- [x] `rag_pipeline.py` — ChromaDB RAG with keyword fallback
- [x] `runbooks/` — 6 markdown runbooks
- [x] `past_incidents.json` — 5 realistic incident records
- [ ] Wire ensemble scores → `AlertContext` → copilot queries
- [ ] Confirm Ollama + Phi-3-mini work offline

### Block 4 — API + Frontend ⚠️ API scaffolded, frontend empty
- [x] `api/main.py` — FastAPI with 9 endpoints + WebSocket (needs rewiring per `spec_api.md`)
- [x] `api/schemas.py` — Pydantic models (missing `AlertContext`)
- [ ] `src/frontend/` — empty, dashboard pending
- [ ] End-to-end wiring all components

### Block 5 — Air-Gap + Demo Prep ❌ Not started
- [ ] Verify zero outbound calls
- [ ] Rehearse all 4 fault scenarios
- [ ] Architecture diagram + README
- [ ] `docs/training_results.md` — comprehensive model results

---

## Hard Warnings

> [!CAUTION]
> **The LLM Copilot is not optional.** It's 35% of your score. Even a simple Phi-3-mini + 3 runbooks + basic RAG chain that produces one structured response is infinitely better than nothing. Do this before polishing the frontend.

> [!CAUTION]
> **Air-gap compliance is 20% of your score.** During the demo, show the system working with no internet. Disable the network interface and demonstrate it. Judges will check.

> [!NOTE]
> **Don't over-engineer the LLM.** You don't need perfect RAG. You need: local model running, 5 runbooks indexed, structured JSON response from `/copilot/query`. That's a winning demo for this component.
