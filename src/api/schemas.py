from pydantic import BaseModel
from typing import Any


class TelemetryRow(BaseModel):
    timestamp: str
    node_id: str
    node_type: str
    site: str
    data: dict[str, Any]


class ScoreDetail(BaseModel):
    isolation_forest: float
    lstm: float
    prophet: float
    gat: float


class NodeHealth(BaseModel):
    node_id: str
    risk_score: float
    health_state: str
    is_anomaly: bool
    scores: ScoreDetail
    metric_health: dict[str, str]


class TelemetryLatest(BaseModel):
    timestamp: str
    nodes: list[TelemetryRow]
    health: list[NodeHealth]


class TelemetryHistory(BaseModel):
    node_id: str
    n: int
    rows: list[TelemetryRow]


class Alert(BaseModel):
    node_id: str
    metric: str
    severity: str
    value: float
    threshold: float
    message: str
    time_to_impact_hours: float | None = None


class ActiveAlerts(BaseModel):
    alerts: list[Alert]
    count: int


class ForecastResponse(BaseModel):
    metric: str
    node_id: str
    horizon: dict[str, dict]
    time_to_breach_hours: float | None
    breach_threshold: float


class TopologyNode(BaseModel):
    id: str
    type: str
    site: str
    health: str
    risk_score: float


class TopologyEdge(BaseModel):
    source: str
    target: str
    capacity_mbps: int
    health: str


class TopologyHealth(BaseModel):
    nodes: list[TopologyNode]
    edges: list[TopologyEdge]


class CopilotQueryRequest(BaseModel):
    node_id: str
    alert: str
    question: str


class CopilotQueryResponse(BaseModel):
    predicted_issue: str
    confidence: float
    root_cause: str
    affected_scope: list[str]
    time_to_impact_hours: float | None
    recommended_action: str
    runbook_reference: str | None


class InjectFaultRequest(BaseModel):
    scenario: str
    duration_minutes: float = 10.0


class WSMessage(BaseModel):
    type: str
    data: dict[str, Any]
