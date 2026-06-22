import asyncio
import json
import os
import logging
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware

from ..data.topology import NODE_IDS, get_graph_as_dict
from ..data.network_gen import NetworkTelemetryGenerator, NUMERIC_COLS, TELEMETRY_SCHEMA
from ..data.anomaly_injector import AnomalyInjector, FAULT_TYPES
from ..ml.ensemble import EnsembleDetector
from ..ml.isolation_forest import IsolationForestDetector
from ..ml.lstm_ae import LSTMAnomalyDetector
from ..ml.prophet_model import ProphetForecaster
from ..ml.gat_model import GATAnomalyDetector

from ..copilot.rag_pipeline import RAGPipeline
from .schemas import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class AppState:
    def __init__(self):
        self.gen = NetworkTelemetryGenerator(seed=42)
        self.injector = AnomalyInjector()
        self.ensemble = self._load_models()
        self.copilot = RAGPipeline()
        self.telemetry_history: list[dict] = []
        self.max_history = 3600
        self.current_rows: list[dict] = []
        self.current_timestamp: str = ""
        self.demotime_enabled = False
        self.demo_scenario: str | None = None
        self.demo_fault = None
        self.demo_start: datetime | None = None
        self.demo_duration_min: float = 10.0

    def _load_models(self) -> EnsembleDetector:
        if_model = self._load_if()
        lstm_model = self._load_lstm()
        prophet_model = self._load_prophet()
        gat_model = self._load_gat()
        return EnsembleDetector(if_model, lstm_model, prophet_model, gat_model)

    def _load_if(self) -> IsolationForestDetector:
        path = os.path.join(MODELS_DIR, "isolation_forest.pkl")
        if os.path.exists(path):
            return IsolationForestDetector(path)
        logger.warning("IsolationForest model not found, using untrained detector")
        return IsolationForestDetector()

    def _load_lstm(self) -> LSTMAnomalyDetector:
        path = os.path.join(MODELS_DIR, "lstm_ae.pt")
        if os.path.exists(path):
            return LSTMAnomalyDetector(path)
        logger.warning("LSTM model not found, using untrained detector")
        return LSTMAnomalyDetector()

    def _load_prophet(self) -> ProphetForecaster:
        path = os.path.join(MODELS_DIR, "prophet")
        if os.path.exists(path):
            return ProphetForecaster.load(path)
        logger.warning("Prophet models not found, using untrained forecaster")
        return ProphetForecaster()

    def _load_gat(self) -> GATAnomalyDetector:
        path = os.path.join(MODELS_DIR, "gat.pt")
        if os.path.exists(path):
            return GATAnomalyDetector(path)
        logger.warning("GAT model not found, using untrained detector")
        return GATAnomalyDetector()

    def tick(self):
        now = datetime.utcnow()
        if self.demotime_enabled and self.demo_fault is not None:
            elapsed = (now - self.demo_start).total_seconds() / 60.0 if self.demo_start else 0
            if elapsed < self.demo_duration_min:
                self.injector.apply(self.gen, now)
            else:
                self.injector.active_faults.clear()
                self.demotime_enabled = False
                self.demo_scenario = None
                self.demo_fault = None
                logger.info("Demo scenario completed, cleared all faults")
        rows = self.gen.next_tick(now)
        self.current_rows = rows
        self.current_timestamp = now.strftime("%Y-%m-%dT%H:%M:%S")
        rows_with_ts = []
        for r in rows:
            r_copy = dict(r)
            r_copy["__timestamp_dt"] = now
            rows_with_ts.append(r_copy)
        self.telemetry_history.extend(rows_with_ts)
        if len(self.telemetry_history) > self.max_history:
            self.telemetry_history = self.telemetry_history[-self.max_history:]
        return rows

    def get_health(self, node_rows: list[dict]) -> list[NodeHealth]:
        all_latest = {r["node_id"]: r for r in node_rows}
        results = []
        for row in node_rows:
            result = self.ensemble.predict(row, all_latest)
            results.append(NodeHealth(
                node_id=row["node_id"],
                risk_score=result["risk_score"],
                health_state=result["health_state"],
                is_anomaly=result["is_anomaly"],
                scores=ScoreDetail(**result["scores"]),
                metric_health=result["metric_health"],
            ))
        return results


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting PS-13 NOC Copilot API")
    yield
    logger.info("Shutting down PS-13 NOC Copilot API")


app = FastAPI(
    title="PS-13 NOC Copilot",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/telemetry/latest")
def get_latest():
    rows = app_state.tick()
    health = app_state.get_health(rows)
    telemetry_rows = [
        TelemetryRow(
            timestamp=app_state.current_timestamp,
            node_id=r["node_id"],
            node_type=r["node_type"],
            site=r["site"],
            data={k: v for k, v in r.items() if k in TELEMETRY_SCHEMA},
        )
        for r in rows
    ]
    return TelemetryLatest(
        timestamp=app_state.current_timestamp,
        nodes=telemetry_rows,
        health=health,
    )


@app.get("/telemetry/history")
def get_history(node: str = Query(...), n: int = Query(300, le=3600)):
    all_rows = app_state.telemetry_history
    node_rows = [r for r in all_rows if r.get("node_id") == node][-n:]
    telemetry_rows = [
        TelemetryRow(
            timestamp=r.get("__timestamp").strftime("%Y-%m-%dT%H:%M:%S") if isinstance(r.get("__timestamp_dt"), datetime) else str(r.get("timestamp", "")),
            node_id=r.get("node_id", node),
            node_type=r.get("node_type", ""),
            site=r.get("site", ""),
            data={k: v for k, v in r.items() if k in TELEMETRY_SCHEMA},
        )
        for r in node_rows
    ]
    return TelemetryHistory(node_id=node, n=len(telemetry_rows), rows=telemetry_rows)


@app.get("/alerts/active")
def get_active_alerts():
    rows = app_state.current_rows
    if not rows:
        return ActiveAlerts(alerts=[], count=0)
    health = app_state.get_health(rows)
    alerts = []
    for h in health:
        if h.health_state in ("WARNING", "CRITICAL"):
            for metric, state in h.metric_health.items():
                if state in ("WARNING", "CRITICAL"):
                    val = next((r.get(metric, 0) for r in rows if r["node_id"] == h.node_id), 0)
                    alerts.append(Alert(
                        node_id=h.node_id,
                        metric=metric,
                        severity=state,
                        value=float(val) if val else 0.0,
                        threshold=0.0,
                        message=f"{h.node_id} {metric} is {state}",
                        time_to_impact_hours=None,
                    ))
    return ActiveAlerts(alerts=alerts[:50], count=len(alerts))


@app.get("/forecast/{node_id}/{metric}")
def get_forecast(node_id: str, metric: str):
    result = app_state.ensemble.prophet_forecaster.forecast(node_id, metric)
    return ForecastResponse(**result)


@app.get("/topology/health")
def get_topology_health():
    graph = get_graph_as_dict()
    rows = app_state.current_rows
    health_rows = {h.node_id: h for h in app_state.get_health(rows)} if rows else {}
    nodes = [
        TopologyNode(
            id=n["id"],
            type=n["type"],
            site=n["site"],
            health=health_rows.get(n["id"], NodeHealth(
                node_id=n["id"], risk_score=0.0, health_state="NORMAL", is_anomaly=False,
                scores=ScoreDetail(isolation_forest=0.0, lstm=0.0, prophet=0.0, gat=0.0),
                metric_health={},
            )).health_state,
            risk_score=health_rows.get(n["id"], NodeHealth(
                node_id=n["id"], risk_score=0.0, health_state="NORMAL", is_anomaly=False,
                scores=ScoreDetail(isolation_forest=0.0, lstm=0.0, prophet=0.0, gat=0.0),
                metric_health={},
            )).risk_score,
        )
        for n in graph["nodes"]
    ]
    edges = [
        TopologyEdge(
            source=e["source"],
            target=e["target"],
            capacity_mbps=e["capacity_mbps"],
            health="NORMAL",
        )
        for e in graph["edges"]
    ]
    return TopologyHealth(nodes=nodes, edges=edges)


@app.post("/copilot/query")
def copilot_query(req: CopilotQueryRequest):
    result = app_state.copilot.copilot_query(req.node_id, req.alert, req.question)
    return CopilotQueryResponse(
        predicted_issue=result.get("predicted_issue", "Unknown"),
        confidence=result.get("confidence", 0.0),
        root_cause=result.get("root_cause", ""),
        affected_scope=result.get("affected_scope", []),
        time_to_impact_hours=result.get("time_to_impact_hours"),
        recommended_action=result.get("recommended_action", "Manual investigation required"),
        runbook_reference=result.get("runbook_reference"),
    )


@app.post("/demo/inject_fault")
def inject_fault(req: InjectFaultRequest):
    scenario = req.scenario
    primary = {
        "congestion": "CE-B1",
        "bgp_flap": "PE-1",
        "mpls_failure": "P-1",
        "policy_drift": "PE-1",
    }.get(scenario, "PE-1")
    if scenario not in FAULT_TYPES:
        return {"status": "error", "message": f"Unknown scenario: {scenario}"}
    fault = app_state.injector.schedule_fault(
        scenario, primary, datetime.utcnow(),
        precursor_min=3, active_min=5, recovery_min=2,
    )
    app_state.demotime_enabled = True
    app_state.demo_scenario = scenario
    app_state.demo_fault = fault
    app_state.demo_start = datetime.utcnow()
    app_state.demo_duration_min = req.duration_minutes
    logger.info(f"Demo fault injected: {scenario} on {primary} for {req.duration_minutes} min")
    return {"status": "ok", "scenario": scenario, "primary_node": primary, "duration_minutes": req.duration_minutes}


@app.get("/health")
def health():
    return {"status": "ok", "uptime_seconds": 0}


active_connections: list[WebSocket] = []


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            rows = app_state.tick()
            health = app_state.get_health(rows)
            msg = TelemetryLatest(
                timestamp=app_state.current_timestamp,
                nodes=[
                    TelemetryRow(
                        timestamp=app_state.current_timestamp,
                        node_id=r["node_id"],
                        node_type=r["node_type"],
                        site=r["site"],
                        data={k: v for k, v in r.items() if k in TELEMETRY_SCHEMA},
                    )
                    for r in rows
                ],
                health=health,
            )
            await websocket.send_text(json.dumps(msg.model_dump(), default=str))
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WS error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)
