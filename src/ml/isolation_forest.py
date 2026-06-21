import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

from ml.features import get_feature_vector_for_if, normalize_features


class IsolationForestDetector:
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = None
        self.threshold = None
        if model_path:
            self.load(model_path)

    def fit(self, normal_rows: list[dict]) -> None:
        X = np.array([get_feature_vector_for_if(r) for r in normal_rows], dtype=np.float32)
        X_norm, self.scaler = normalize_features(X, fit=True)
        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_norm)
        scores = -self.model.score_samples(X_norm)
        self.threshold = float(np.percentile(scores, 95))

    def score(self, row: dict) -> dict:
        if self.model is None:
            return {"if_score": 0.0, "is_anomaly": False}
        x = get_feature_vector_for_if(row).reshape(1, -1)
        x_norm, _ = normalize_features(x, scaler=self.scaler)
        raw = -float(self.model.score_samples(x_norm)[0])
        score = min(raw / (3 * self.threshold + 1e-8), 1.0)
        return {"if_score": score, "is_anomaly": score > 0.5}

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "scaler": self.scaler, "threshold": self.threshold}, path)

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.threshold = data["threshold"]
