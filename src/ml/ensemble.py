import numpy as np
import pandas as pd

from .features import get_feature_vector_for_if, get_sequence_for_lstm, get_graph_features
from .isolation_forest import IsolationForestDetector
from .lstm_ae import LSTMAnomalyDetector
from .prophet_model import ProphetForecaster
from .gat_model import GATAnomalyDetector

HEALTH_THRESHOLDS = {
    "risk_score": [(0.35, "NORMAL"), (0.65, "WARNING")],
    "latency_ms": [(50.0, "WARNING"), (150.0, "CRITICAL")],
    "packet_loss_pct": [(1.0, "WARNING"), (5.0, "CRITICAL")],
    "utilization_pct": [(75.0, "WARNING"), (90.0, "CRITICAL")],
    "cpu_load_pct": [(70.0, "WARNING"), (85.0, "CRITICAL")],
    "bgp_updates_per_min": [(15.0, "WARNING"), (50.0, "CRITICAL")],
    "bgp_sessions_active": lambda v: "CRITICAL" if v == 0 else ("WARNING" if v < 3 else "NORMAL"),
    "errors_in": [(5.0, "WARNING"), (20.0, "CRITICAL")],
    "memory_used_pct": [(80.0, "WARNING"), (90.0, "CRITICAL")],
    "queue_depth": [(40.0, "WARNING"), (100.0, "CRITICAL")],
    "link_state": lambda v: {"FLAPPING": "WARNING", "DOWN": "CRITICAL"}.get(v, "NORMAL"),
}

ENSEMBLE_WEIGHTS = {
    "isolation_forest": 0.30,
    "lstm": 0.30,
    "prophet": 0.25,
    "gat": 0.15,
}


class EnsembleDetector:
    def __init__(self, if_detector, lstm_detector, prophet_forecaster, gat_detector):
        self.if_detector = if_detector
        self.lstm_detector = lstm_detector
        self.prophet_forecaster = prophet_forecaster
        self.gat_detector = gat_detector
        self.history: dict[str, pd.DataFrame] = {}

    def _update_history(self, row: dict):
        nid = row["node_id"]
        df_row = pd.DataFrame([row])
        if nid in self.history:
            self.history[nid] = pd.concat([self.history[nid], df_row], ignore_index=True)
            max_rows = 3600
            if len(self.history[nid]) > max_rows:
                self.history[nid] = self.history[nid].iloc[-max_rows:]
        else:
            self.history[nid] = df_row

    def predict(self, current_row: dict, all_node_latest: dict[str, dict] = None) -> dict:
        self._update_history(current_row)
        node_id = current_row["node_id"]
        history = self.history.get(node_id, pd.DataFrame([current_row]))

        if_result = self.if_detector.score(current_row)

        lstm_result = {"lstm_score": 0.0, "reconstruction_error": 0.0, "is_anomaly": False}
        if len(history) >= 60:
            seq = get_sequence_for_lstm(history, node_id)
            lstm_result = self.lstm_detector.score(seq)

        prophet_result = self.prophet_forecaster.get_overall_score(node_id, current_row)

        gat_result = {"per_node_scores": {}, "per_node_anomaly": {}, "network_score": 0.0}
        if all_node_latest:
            gf = get_graph_features(all_node_latest)
            gat_result = self.gat_detector.score(gf)

        risk_score = (
            ENSEMBLE_WEIGHTS["isolation_forest"] * if_result["if_score"]
            + ENSEMBLE_WEIGHTS["lstm"] * lstm_result["lstm_score"]
            + ENSEMBLE_WEIGHTS["prophet"] * prophet_result["prophet_score"]
            + ENSEMBLE_WEIGHTS["gat"] * gat_result.get("network_score", 0.0)
        )

        health = self._classify_health(risk_score)
        metric_health = self._classify_metric_health(current_row)

        return {
            "risk_score": round(risk_score, 4),
            "health_state": health,
            "is_anomaly": risk_score > 0.5,
            "scores": {
                "isolation_forest": round(if_result["if_score"], 4),
                "lstm": round(lstm_result["lstm_score"], 4),
                "prophet": round(prophet_result["prophet_score"], 4),
                "gat": round(gat_result.get("network_score", 0.0), 4),
            },
            "metric_health": metric_health,
        }

    def _classify_health(self, risk_score: float) -> str:
        if risk_score < 0.35:
            return "NORMAL"
        if risk_score < 0.65:
            return "WARNING"
        return "CRITICAL"

    def _classify_metric_health(self, row: dict) -> dict[str, str]:
        result = {}
        for metric, rule in HEALTH_THRESHOLDS.items():
            val = row.get(metric)
            if val is None:
                continue
            if callable(rule):
                result[metric] = rule(val)
            else:
                try:
                    v = float(val)
                    state = "NORMAL"
                    for threshold, label in rule:
                        if v >= threshold:
                            state = label
                    result[metric] = state
                except (ValueError, TypeError):
                    result[metric] = "NORMAL"
        return result

    def predict_all_nodes(
        self,
        current_rows: list[dict],
    ) -> list[dict]:
        all_latest = {r["node_id"]: r for r in current_rows}
        results = []
        for row in current_rows:
            result = self.predict(row, all_latest)
            result["node_id"] = row["node_id"]
            results.append(result)
        return results
