# Project Mariana — ISRO PS-13 NOC Copilot

Air-gapped predictive NOC operations platform for MPLS/SD-WAN networks. Generates synthetic telemetry, detects anomalies via 4 unsupervised ML models, and provides actionable guidance via an offline LLM copilot.

## Architecture

```
src/
├── data/              ← Synthetic telemetry generation
│   ├── topology.py         10-node MPLS topology with per-interface edges
│   ├── network_gen.py      Per-interface/per-tunnel telemetry (92 columns)
│   └── anomaly_injector.py 4 fault scenarios with precursor phases
│
├── ml/                ← Unsupervised anomaly detection
│   ├── features.py         Feature engineering
│   ├── isolation_forest.py Point anomalies
│   ├── lstm_ae.py          Temporal reconstruction (sequence anomalies)
│   ├── prophet_model.py    Forecast-based breach prediction (~31 targeted models)
│   ├── gat_model.py        Graph autoencoder (topology-aware anomalies)
│   └── ensemble.py         Fuses 4 models → risk_score + health_state
│
├── training/           ← Dataset generation & model training
│   └── generate_dataset.py  Produces telemetry_train.csv + graph_snapshots
│
├── copilot/            ← Offline RAG copilot
│   ├── ollama_client.py    Wrapper for local LLM (Phi-3 / Mistral)
│   ├── rag_pipeline.py     ChromaDB + LangChain
│   └── runbooks/           Markdown runbooks per fault type
│
├── api/                ← FastAPI backend
│   ├── main.py             REST endpoints + WebSocket streaming
│   └── schemas.py          Pydantic models
│
├── models/             ← Trained model artifacts
│   └── prophet/            ~31 .pkl files (targeted lead metrics)
│
└── smoke_test.py       Validates data layer

backend/                ← Original telemetry dashboard (FastAPI + IF)
frontend/               ← Dashboard HTML/CSS/JS
```

## Quick Start

```bash
# Generate training data (~10 min)
python -m src.training.generate_dataset

# Run smoke test
python -m src.smoke_test
```

## Fault Scenarios

| Fault | Target | Precursor Signal |
|-------|--------|-----------------|
| Congestion | CE wan0 access link | Utilization rising, queue building |
| BGP Flap | PE router | BGP updates spiking |
| MPLS Failure | P-router core interfaces | Errors accumulating |
| Policy Drift | PE router | CPU spiking, VPN routes growing |

## Telemetry Schema

92 columns per row: per-interface (8 fields × 5 ifaces) + per-tunnel (3 fields × 6 LSPs) + NUMERIC_COLS aggregates (23) + labels (4). ML models consume only the 23 NUMERIC_COLS.

## Documentation

- `docs/ISRO_PS13_Roadmap.md` — Full architecture plan
- `docs/data_layer_explained.md` — Data layer walkthrough
- `docs/spec_data_layer.md` — Data layer specification
- `docs/spec_ml_models.md` — ML model specification
- `docs/spec_api.md` — API specification
