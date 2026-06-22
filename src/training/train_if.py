"""
Train Isolation Forest
======================
Fits IsolationForestDetector on normal data from telemetry_train.csv.
Saves to src/models/isolation_forest.pkl

Usage:
    python -m training.train_if
"""

import os
import sys
import logging
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)

from ml.isolation_forest import IsolationForestDetector
from ml.features import build_feature_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "src", "models")


def main():
    log.info("Loading telemetry_train.csv...")
    df = pd.read_csv(os.path.join(DATA_DIR, "telemetry_train.csv"))

    log.info("Dataset shape: %s", df.shape)

    # Enrich with rolling means/slopes BEFORE splitting on is_anomaly.
    # This is safe because build_feature_matrix only looks at per-node history
    # and does not use the label columns — no leakage risk.
    log.info("Building rolling feature matrix (this may take ~30s)...")
    t0 = time.time()
    df = build_feature_matrix(df, per_node=True)
    log.info("Feature matrix built — columns: %d  (%.1fs)", df.shape[1], time.time() - t0)

    normal_df = df[df["is_anomaly"] == False].copy()
    log.info("Normal rows: %d (%.1f%%)", len(normal_df), 100 * len(normal_df) / len(df))

    log.info("Training Isolation Forest on %d normal rows...", len(normal_df))
    detector = IsolationForestDetector()
    detector.fit(normal_df)

    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, "isolation_forest.pkl")
    detector.save(path)
    log.info("Saved to %s", path)

    log.info("--- Evaluating separability ---")
    sample_normal = normal_df.iloc[:2000].to_dict("records")
    scores_n = [detector.score(r)["if_score"]
                for r in tqdm(sample_normal, desc="  scoring normal", ncols=80)]
    log.info("Normal (2000 rows): mean=%.4f, max=%.4f, p95=%.4f",
             np.mean(scores_n), max(scores_n), np.percentile(scores_n, 95))

    for phase in ["precursor", "active", "recovery"]:
        phase_df = df[df["fault_phase"] == phase]
        if len(phase_df) == 0:
            continue
        sample = phase_df.iloc[:min(500, len(phase_df))].to_dict("records")
        scores = [detector.score(r)["if_score"]
                  for r in tqdm(sample, desc=f"  scoring {phase}", ncols=80)]
        log.info("  %s phase (%d rows): mean=%.4f, max=%.4f, p95=%.4f",
                 phase, len(sample), np.mean(scores), max(scores), np.percentile(scores, 95))

    log.info("Done.")


if __name__ == "__main__":
    main()
