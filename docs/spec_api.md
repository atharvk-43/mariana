# Spec: FastAPI Backend
## Files: `backend/api/main.py`, `backend/api/schemas.py`

---

## Overview

The FastAPI backend is the integration hub. It:
1. Loads all 4 ML models at startup
2. Runs the telemetry generator in a background thread
3. Serves REST endpoints for dashboard and copilot
4. Pushes real-time data via WebSocket

---

## `backend/api/schemas.py` — All Pydantic Models

```python
from pydantic import BaseModel
from typing import Optional

# --- Telemetry ---

class TelemetryRow(BaseModel):
    timestamp: str
    node_id: str
    node_type: str
    interface: str
    bytes_in: int
    bytes_out: int
    packets_in: int
    packets_out: int
    errors_in: int
    drops_in: int
    drops_out: int
    utilization_pct: float
    link_state: str
    bgp_sessions_active: int
    bgp_prefixes_received: int
    bgp_updates_per_min: int
    bgp_withdrawals_per_min: int
    ospf_spf_runs: int
    cpu_load_pct: float
    memory_used_pct: float
    queue_depth: int
    latency_ms: float
    jitter_ms: float
    packet_loss_pct: float
    tunnel_packet_loss_pct: float
    ipsec_rekeyed_last_hr: int

# --- ML Outputs ---

class ModelScores(BaseModel):
    isolation_forest: float
    lstm_ae: float
    prophet: float
    gat: float

class NodeHealthResult(BaseModel):
    node_id: str
    timestamp: str
    risk_score: float               # 0–100
    health_state: str               # NORMAL | WARNING | CRITICAL
    time_to_impact_hours: Optional[float]
    breach_metric: Optional[str]
    model_scores: ModelScores
    metric_health: dict[str, str]   # metric_name → NORMAL|WARNING|CRITICAL
    telemetry: TelemetryRow

# --- Alerts ---

class Alert(BaseModel):
    alert_id: str
    node_id: str
    timestamp: str
    health_state: str
    risk_score: float
    time_to_impact_hours: Optional[float]
    breach_metric: Optional[str]
    message: str                    # human-readable summary

# --- Forecasts ---

class HorizonForecast(BaseModel):
    yhat: float
    yhat_lower: float
    yhat_upper: float

class ForecastResult(BaseModel):
    node_id: str
    metric: str
    horizon: dict[str, HorizonForecast]   # "1h", "3h", "6h"
    time_to_breach_hours: Optional[float]
    breach_threshold: float

# --- Topology ---

class TopologyNode(BaseModel):
    id: str
    type: str
    site: str
    health_state: str           # current health
    risk_score: float

class TopologyEdge(BaseModel):
    source: str
    target: str
    capacity_mbps: int

class TopologyHealth(BaseModel):
    nodes: list[TopologyNode]
    edges: list[TopologyEdge]

# --- Copilot ---

class AlertContext(BaseModel):
    node_id: str
    risk_score: float
    health_state: str
    time_to_impact_hours: Optional[float]
    breach_metric: Optional[str]
    model_scores: ModelScores
    metric_health: dict[str, str]
    current_metrics: dict

class CopilotQueryRequest(BaseModel):
    alert_context: AlertContext
    question: Optional[str] = None

class CopilotResponse(BaseModel):
    predicted_issue: str
    confidence: float
    root_cause_hypothesis: str
    affected_scope: list[str]
    estimated_time_to_impact_hours: Optional[float]
    recommended_actions: list[str]
    runbook_reference: Optional[str]
    urgency: str
    available: bool = True

# --- Demo Control ---

class FaultInjectionRequest(BaseModel):
    fault_type: str     # "congestion" | "bgp_flap" | "mpls_failure" | "policy_drift"
    node_id: str        # primary node to inject on

class WebSocketMessage(BaseModel):
    type: str           # "telemetry" | "alert" | "heartbeat"
    data: dict
```

---

## `backend/api/main.py` — Complete Structure

### Imports & Global State

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import threading
import pandas as pd
import time
from datetime import datetime
from collections import deque

from data.topology import get_graph_as_dict, NODES
from data.network_gen import NetworkTelemetryGenerator
from data.anomaly_injector import AnomalyInjector
from ml.isolation_forest import IsolationForestDetector
from ml.lstm_ae import LSTMAnomalyDetector
from ml.prophet_model import ProphetForecaster
from ml.gat_model import GATAnomalyDetector
from ml.ensemble import EnsembleDetector
from ml.features import get_feature_vector_for_if, get_sequence_for_lstm, get_graph_features
from copilot.rag_pipeline import NOCCopilot
from copilot.ollama_client import OllamaClient
from api.schemas import *

# --- Global State ---
# Telemetry ring buffer: keeps last 1800 rows per node (1 hour at 2s)
# dict[node_id → deque(maxlen=1800)]
telemetry_buffer: dict[str, deque] = {}

# Latest row per node
latest_rows: dict[str, dict] = {}

# Active alerts (only WARNING and CRITICAL)
active_alerts: dict[str, Alert] = {}   # node_id → latest alert

# WebSocket connections
ws_connections: list[WebSocket] = []

# ML model instances (loaded at startup)
if_detector: IsolationForestDetector = None
lstm_detector: LSTMAnomalyDetector = None
prophet_forecaster: ProphetForecaster = None
gat_detector: GATAnomalyDetector = None
ensemble: EnsembleDetector = None
copilot: NOCCopilot = None
ollama_client: OllamaClient = None

# Telemetry generator + injector
generator: NetworkTelemetryGenerator = None
injector: AnomalyInjector = None
```

### Startup / Shutdown

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and start background threads on startup."""
    global if_detector, lstm_detector, prophet_forecaster, gat_detector
    global ensemble, copilot, ollama_client, generator, injector, telemetry_buffer
    
    # Initialize telemetry buffer per node
    for node in NODES:
        telemetry_buffer[node["id"]] = deque(maxlen=1800)
    
    # Load ML models
    print("Loading ML models...")
    if_detector = IsolationForestDetector.load("backend/models/isolation_forest.pkl")
    lstm_detector = LSTMAnomalyDetector("backend/models/lstm_ae.pt")
    prophet_forecaster = ProphetForecaster.load("backend/models/")
    gat_detector = GATAnomalyDetector("backend/models/gat.pt")
    ensemble = EnsembleDetector(if_detector, lstm_detector, prophet_forecaster, gat_detector)
    print("ML models loaded.")
    
    # Load copilot
    ollama_client = OllamaClient()
    copilot = NOCCopilot()
    if ollama_client.is_available():
        copilot.load_index()
        print("NOC Copilot ready.")
    else:
        print("WARNING: Ollama not running. Copilot endpoints will return error.")
    
    # Start generator
    generator = NetworkTelemetryGenerator(seed=42)
    injector = AnomalyInjector()
    
    # Start background telemetry loop
    thread = threading.Thread(target=telemetry_loop, daemon=True)
    thread.start()
    print("Telemetry loop started.")
    
    yield  # App runs here
    
    print("Shutting down.")

app = FastAPI(title="MARTIAN PS13 — NOC Copilot", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
```

### Background Telemetry Loop

```python
def telemetry_loop():
    """
    Runs in a background thread. Every 2 seconds:
    1. Generate next tick of telemetry (10 rows, one per node)
    2. Apply anomaly injections
    3. Run ML inference on each node
    4. Update telemetry_buffer, latest_rows, active_alerts
    5. Push WebSocket message to all connected clients
    """
    global latest_rows, active_alerts
    
    while True:
        tick_start = time.time()
        current_time = datetime.utcnow()
        
        # Generate telemetry
        rows = generator.next_tick(current_time)
        injector.apply(generator, current_time)
        
        # Update buffer
        for row in rows:
            node_id = row["node_id"]
            telemetry_buffer[node_id].append(row)
            latest_rows[node_id] = row
        
        # Build history DataFrame for LSTM
        history_dfs = {}
        for node_id in telemetry_buffer:
            history_dfs[node_id] = pd.DataFrame(list(telemetry_buffer[node_id]))
        
        # Run ensemble inference for all nodes
        try:
            results = ensemble.predict_all_nodes(rows, history_dfs)
        except Exception as e:
            print(f"Inference error: {e}")
            results = []
        
        # Update alerts
        new_alerts = []
        for result in results:
            node_id = result["node_id"]
            if result["health_state"] in ("WARNING", "CRITICAL"):
                alert = Alert(
                    alert_id=f"{node_id}-{current_time.isoformat()}",
                    node_id=node_id,
                    timestamp=current_time.isoformat(),
                    health_state=result["health_state"],
                    risk_score=result["risk_score"],
                    time_to_impact_hours=result.get("time_to_impact_hours"),
                    breach_metric=result.get("breach_metric"),
                    message=_build_alert_message(result),
                )
                active_alerts[node_id] = alert
                new_alerts.append(alert)
            elif node_id in active_alerts:
                # Node recovered
                del active_alerts[node_id]
        
        # Push WebSocket update
        ws_payload = {
            "type": "telemetry",
            "timestamp": current_time.isoformat(),
            "nodes": results,
            "active_alerts": [a.model_dump() for a in active_alerts.values()],
        }
        asyncio.run(_broadcast_ws(ws_payload))
        
        # Sleep remainder of 2-second interval
        elapsed = time.time() - tick_start
        sleep_time = max(0, 2.0 - elapsed)
        time.sleep(sleep_time)


def _build_alert_message(result: dict) -> str:
    """Build human-readable alert message from result dict."""
    breach = result.get("breach_metric", "multiple metrics")
    eta = result.get("time_to_impact_hours")
    eta_str = f" — estimated impact in {eta:.1f}h" if eta else ""
    return f"{result['health_state']}: {result['node_id']} — {breach} anomalous{eta_str}"


async def _broadcast_ws(payload: dict):
    """Send payload to all connected WebSocket clients."""
    disconnected = []
    for ws in ws_connections:
        try:
            await ws.send_json(payload)
        except:
            disconnected.append(ws)
    for ws in disconnected:
        ws_connections.remove(ws)
```

### REST Endpoints

```python
@app.get("/")
def root():
    return {"project": "MARTIAN PS13", "status": "running", "air_gapped": True}

@app.get("/health")
def health():
    return {
        "healthy": True,
        "models_loaded": all([if_detector, lstm_detector, prophet_forecaster, gat_detector]),
        "copilot_available": ollama_client.is_available() if ollama_client else False,
        "active_nodes": len(latest_rows),
    }

@app.get("/telemetry/latest")
def telemetry_latest() -> list[NodeHealthResult]:
    """
    Return the latest inference result for all 10 nodes.
    Includes telemetry + ML scores + health state.
    Run inference on latest_rows if results are stale.
    """

@app.get("/telemetry/history")
def telemetry_history(
    node_id: str,
    n: int = 300,   # last N rows
) -> list[TelemetryRow]:
    """
    Return last N telemetry rows for a specific node.
    Read from telemetry_buffer[node_id].
    """

@app.get("/alerts/active")
def alerts_active() -> list[Alert]:
    """Return all currently active WARNING/CRITICAL alerts."""
    return list(active_alerts.values())

@app.get("/forecast/{node_id}/{metric}")
def get_forecast(node_id: str, metric: str) -> ForecastResult:
    """
    Return Prophet forecast for a specific node + metric.
    Horizons: 1h, 3h, 6h.
    Returns 404 if node_id or metric not recognized.
    """

@app.get("/topology/health")
def topology_health() -> TopologyHealth:
    """
    Return the full network topology with current health state per node.
    Uses latest inference results.
    """
    graph = get_graph_as_dict()
    # Enrich nodes with current health state
    # ...
    return TopologyHealth(nodes=[...], edges=[...])

@app.post("/copilot/query")
async def copilot_query(request: CopilotQueryRequest) -> CopilotResponse:
    """Offline LLM copilot structured response."""
    if not ollama_client or not ollama_client.is_available():
        return CopilotResponse(
            predicted_issue="Copilot offline",
            confidence=0.0, root_cause_hypothesis="Ollama not running",
            affected_scope=[], recommended_actions=["Start Ollama: ollama serve"],
            urgency="LOW", available=False,
            estimated_time_to_impact_hours=None, runbook_reference=None,
        )
    result = copilot.query(
        alert_context=request.alert_context.model_dump(),
        operator_question=request.question,
    )
    return CopilotResponse(**result, available=True)

@app.get("/copilot/status")
def copilot_status():
    return {
        "available": ollama_client.is_available() if ollama_client else False,
        "model": "phi3:mini",
        "air_gapped": True,
        "chroma_indexed": True,
    }

@app.post("/demo/inject_fault")
def inject_fault(request: FaultInjectionRequest):
    """
    Trigger a fault injection immediately (for live demo).
    Returns the FaultEvent ID so the demo can track it.
    """
    event = injector.schedule_fault(
        fault_type=request.fault_type,
        primary_node=request.node_id,
        start_time=datetime.utcnow(),
    )
    return {"fault_id": event.fault_id, "status": "scheduled", "node_id": request.node_id}

@app.post("/demo/clear_faults")
def clear_faults():
    """Clear all active fault injections. Reset to normal state."""
    injector.active_faults.clear()
    return {"status": "cleared"}

# --- WebSocket ---

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint. Client connects and receives JSON push every 2 seconds.
    Message format: WebSocketMessage schema.
    """
    await websocket.accept()
    ws_connections.append(websocket)
    try:
        while True:
            # Keep connection alive — telemetry_loop handles pushes
            await asyncio.sleep(30)
            await websocket.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        ws_connections.remove(websocket)
```

---

## Running the Backend

```bash
cd backend
pip install -r requirements.txt

# Ensure models are in backend/models/
# Ensure Ollama is running: ollama serve

uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Key Notes for Implementation

1. **Thread safety**: `telemetry_buffer` and `latest_rows` are written from the background thread and read from FastAPI request handlers. Use `threading.Lock()` around all reads/writes to these shared structures.

2. **Model loading**: If `lstm_ae.pt` or `gat.pt` don't exist (not yet trained), gracefully degrade:
   - Log a warning
   - Return score of 0.0 for that model in ensemble
   - This lets the API run even before Kaggle training is done

3. **History DataFrame**: The LSTM needs 60 rows of history per node. For the first 60 ticks (2 minutes after startup), zero-pad the sequence.

4. **Port**: Backend on 8000. Frontend on 5500 (static serve). Ollama on 11434. All local.
