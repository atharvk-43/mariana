import logging
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
import joblib

from ml.features import get_feature_matrix_for_if, get_feature_vector_for_if, normalize_features

log = logging.getLogger(__name__)


class IsolationForestDetector:
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = None
        self.threshold = None
        if model_path:
            self.load(model_path)

    # Cap training rows — IF subsamples internally per tree anyway,
    # so quality plateaus well below 1M rows. 200K is more than sufficient.
    _MAX_FIT_ROWS = 200_000
    # How many trees to add per warm-start batch — smaller = more frequent logs
    _BATCH_SIZE = 50
    _TOTAL_TREES = 300

    def fit(self, normal_df: pd.DataFrame) -> None:
        # Subsample if needed to stay within memory budget
        n = len(normal_df)
        if n > self._MAX_FIT_ROWS:
            log.info("Subsampling %d → %d rows for IF training (memory cap).",
                     n, self._MAX_FIT_ROWS)
            normal_df = normal_df.sample(n=self._MAX_FIT_ROWS, random_state=42)

        log.info("Building feature matrix (%d rows, vectorised)...", len(normal_df))
        X = get_feature_matrix_for_if(normal_df)
        log.info("Feature matrix shape: %s  (%.1f MiB) — normalising...",
                 X.shape, X.nbytes / 1024 / 1024)
        X_norm, self.scaler = normalize_features(X, fit=True)

        # Fit in batches using warm_start so we can log progress between rounds.
        # contamination='auto' because training data is pre-filtered normal-only.
        n_batches = self._TOTAL_TREES // self._BATCH_SIZE
        log.info("Fitting IsolationForest (%d trees, %d batches of %d)...",
                 self._TOTAL_TREES, n_batches, self._BATCH_SIZE)

        self.model = IsolationForest(
            n_estimators=self._BATCH_SIZE,
            contamination="auto",
            max_samples="auto",
            warm_start=True,
            random_state=42,
            n_jobs=-1,
        )

        t0 = time.time()
        for batch_idx in range(1, n_batches + 1):
            self.model.n_estimators = batch_idx * self._BATCH_SIZE
            self.model.fit(X_norm)
            elapsed = time.time() - t0
            pct = 100 * batch_idx / n_batches
            log.info("  [%3.0f%%] %d / %d trees fitted  (%.1fs elapsed)",
                     pct, self.model.n_estimators, self._TOTAL_TREES, elapsed)

        log.info("Computing anomaly score threshold on normal data...")
        scores = -self.model.score_samples(X_norm)
        # threshold = 95th-pct anomaly score on *normal* data
        self.threshold = float(np.percentile(scores, 95))
        log.info("Threshold set to %.6f (p95 of normal scores)", self.threshold)


    def score(self, row: dict) -> dict:
        if self.model is None:
            return {"if_score": 0.0, "is_anomaly": False}
        x = get_feature_vector_for_if(row).reshape(1, -1)
        x_norm, _ = normalize_features(x, scaler=self.scaler)
        raw = -float(self.model.score_samples(x_norm)[0])
        # Normalize so that score=0.5 at the threshold boundary,
        # and score=1.0 at 2x the threshold.  Anomalies should clearly exceed 0.5.
        score = float(np.clip(raw / (2.0 * self.threshold + 1e-8), 0.0, 1.0))
        return {"if_score": round(score, 4), "is_anomaly": score > 0.5}

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "scaler": self.scaler, "threshold": self.threshold}, path)

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.threshold = data["threshold"]
