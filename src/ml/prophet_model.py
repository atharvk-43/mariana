import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import timedelta
import joblib
from joblib import Parallel, delayed

try:
    from prophet import Prophet
except ImportError:
    Prophet = None

log = logging.getLogger(__name__)

PROPHET_METRICS = [
    ("CE-B1", "utilization_pct", 90.0), ("CE-B1", "queue_depth", 100.0),
    ("CE-B2", "utilization_pct", 90.0), ("CE-B2", "queue_depth", 100.0),
    ("CE-B3", "utilization_pct", 90.0), ("CE-B3", "queue_depth", 100.0),
    ("PE-1", "bgp_updates_per_min", 50.0), ("PE-2", "bgp_updates_per_min", 50.0),
    ("P-1", "errors_in", 20.0), ("P-1", "cpu_load_pct", 85.0), ("P-1", "tunnel_packet_loss_pct", 2.0),
    ("P-2", "errors_in", 20.0), ("P-2", "cpu_load_pct", 85.0), ("P-2", "tunnel_packet_loss_pct", 2.0),
    ("P-3", "errors_in", 20.0), ("P-3", "cpu_load_pct", 85.0), ("P-3", "tunnel_packet_loss_pct", 2.0),
    ("PE-1", "cpu_load_pct", 85.0), ("PE-1", "vpn_routes_count", 500.0),
    ("PE-2", "cpu_load_pct", 85.0), ("PE-2", "vpn_routes_count", 500.0),
    ("PE-1", "latency_ms", 150.0), ("PE-2", "latency_ms", 150.0),
    ("P-1", "latency_ms", 50.0), ("P-2", "latency_ms", 50.0), ("P-3", "latency_ms", 50.0),
    ("PE-1", "packet_loss_pct", 5.0), ("PE-2", "packet_loss_pct", 5.0),
    ("P-1", "packet_loss_pct", 3.0), ("P-2", "packet_loss_pct", 3.0), ("P-3", "packet_loss_pct", 3.0),
]

THRESHOLDS = {m: t for _, m, t in PROPHET_METRICS}


class ProphetForecaster:
    def __init__(self):
        self.models: dict[tuple, Prophet] = {}
        self.means: dict[tuple, float] = {}

    def fit_all(self, df: pd.DataFrame, parallel: bool = True, n_jobs: int = -1) -> None:
        if Prophet is None:
            raise ImportError("prophet package is required to fit models. Install with: pip install prophet")
        if parallel:
            self._fit_parallel(df, n_jobs)
        else:
            self._fit_sequential(df)

    def _fit_sequential(self, df: pd.DataFrame) -> None:
        for node_id, metric, threshold in PROPHET_METRICS:
            self._fit_one(df, node_id, metric)

    def _fit_parallel(self, df: pd.DataFrame, n_jobs: int = -1) -> None:
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(self._fit_one)(df, node_id, metric)
            for node_id, metric, _ in PROPHET_METRICS
        )
        for model, mean_val, key in results:
            if model is not None:
                self.models[key] = model
                self.means[key] = mean_val

    def _fit_one(self, df: pd.DataFrame, node_id: str, metric: str) -> tuple:
        key = (node_id, metric)
        node_df = df[df["node_id"] == node_id][["timestamp", metric]].copy()
        node_df = node_df.rename(columns={"timestamp": "ds", metric: "y"})
        node_df["ds"] = pd.to_datetime(node_df["ds"])
        if len(node_df) < 10:
            return None, None, key
        mean_val = float(node_df["y"].mean())
        model = Prophet(
            seasonality_mode="multiplicative",
            daily_seasonality=False,
            weekly_seasonality=False,
            changepoint_prior_scale=0.05,
        )
        model.add_seasonality(name="diurnal", period=1, fourier_order=10)
        model.fit(node_df)
        return model, mean_val, key

    def forecast(
        self, node_id: str, metric: str, horizon_hours: list[float] = None
    ) -> dict:
        if horizon_hours is None:
            horizon_hours = [1.0, 3.0, 6.0]
        key = (node_id, metric)
        if key not in self.models:
            return {"metric": metric, "node_id": node_id, "horizon": {}, "time_to_breach_hours": None}
        model = self.models[key]
        max_h = max(horizon_hours)
        periods = int(max_h * 60)
        future = model.make_future_dataframe(periods=periods, freq="min")
        forecast = model.predict(future)
        horizon = {}
        for h in horizon_hours:
            idx = min(int(h * 60), len(forecast) - 1)
            row = forecast.iloc[idx]
            horizon[f"{h:g}h"] = {
                "yhat": float(row["yhat"]),
                "yhat_lower": float(row["yhat_lower"]),
                "yhat_upper": float(row["yhat_upper"]),
            }
        threshold = THRESHOLDS.get(metric, float("inf"))
        ttb = self._estimate_time_to_breach(forecast, threshold, periods)
        return {
            "metric": metric,
            "node_id": node_id,
            "horizon": horizon,
            "time_to_breach_hours": ttb,
            "breach_threshold": threshold,
        }

    def _estimate_time_to_breach(self, forecast: pd.DataFrame, threshold: float, periods: int) -> float | None:
        latest = forecast[forecast["yhat"] >= threshold]
        if latest.empty:
            return None
        first_breach = latest.iloc[0]
        last_hist_idx = max(0, len(forecast) - periods - 1)
        ref_ds = forecast.iloc[last_hist_idx]["ds"]
        delta = first_breach["ds"] - ref_ds
        return max(0.0, delta.total_seconds() / 3600.0)

    def anomaly_score(self, node_id: str, metric: str, actual_value: float) -> float:
        key = (node_id, metric)
        if key not in self.models:
            return 0.0
        model = self.models[key]
        future = model.make_future_dataframe(periods=1, freq="min")
        forecast = model.predict(future)
        latest = forecast.iloc[-1]
        ci = latest["yhat_upper"] - latest["yhat_lower"]
        if ci < 1e-6:
            return 0.0
        deviation = abs(actual_value - latest["yhat"])
        return min(deviation / ci, 1.0)

    def get_overall_score(self, node_id: str, current_row: dict) -> dict:
        scores = {}
        earliest_breach = None
        breach_metric = None
        for nid, metric, threshold in PROPHET_METRICS:
            if nid != node_id:
                continue
            val = float(current_row.get(metric, 0.0))
            scores[metric] = self.anomaly_score(node_id, metric, val)
            fc = self.forecast(node_id, metric)
            ttb = fc.get("time_to_breach_hours")
            if ttb is not None and (earliest_breach is None or ttb < earliest_breach):
                earliest_breach = ttb
                breach_metric = metric
        prophet_score = max(scores.values()) if scores else 0.0
        return {
            "prophet_score": prophet_score,
            "per_metric_scores": scores,
            "earliest_breach_hours": earliest_breach,
            "breach_metric": breach_metric,
        }

    def save(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        manifest = []
        for (node_id, metric), model in self.models.items():
            path = os.path.join(save_dir, f"{node_id}__{metric}.pkl")
            joblib.dump(model, path)
            manifest.append({"node_id": node_id, "metric": metric, "path": path})
        with open(os.path.join(save_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def load(cls, save_dir: str) -> "ProphetForecaster":
        forecaster = cls()
        manifest_path = os.path.join(save_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            return forecaster
        with open(manifest_path) as f:
            manifest = json.load(f)
        for entry in manifest:
            path = entry["path"]
            if not os.path.isabs(path):
                path = os.path.join(save_dir, path)
            model = joblib.load(path)
            forecaster.models[(entry["node_id"], entry["metric"])] = model
        return forecaster
