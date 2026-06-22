"""
Train Prophet Models
====================
Fits 31 targeted Prophet models on telemetry_train.csv.
Saves each as .pkl to src/models/prophet/ with a manifest.json.

Usage:
    python -m training.train_prophet
"""

import os
import sys
import logging

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)

from ml.prophet_model import ProphetForecaster, PROPHET_METRICS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "src", "models")
N_JOBS = max(1, os.cpu_count() // 4 or 1)


def main():
    log.info("Loading telemetry_train.csv...")
    df = pd.read_csv(os.path.join(DATA_DIR, "telemetry_train.csv"))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.groupby("node_id").apply(lambda g: g.iloc[::6]).reset_index(level=0).reset_index(drop=True)
    log.info("Dataset shape: %s (subsampled per-node to 12s intervals), date range: %s to %s",
             df.shape, df["timestamp"].min(), df["timestamp"].max())

    log.info("Fitting %d Prophet models...", len(PROPHET_METRICS))
    log.info("Using %d parallel workers (cpu_count=%d, 4 chains per model)", N_JOBS, os.cpu_count())
    forecaster = ProphetForecaster()
    forecaster.fit_all(df, parallel=True, n_jobs=N_JOBS)
    log.info("Fitted %d models successfully.", len(forecaster.models))

    save_dir = os.path.join(MODELS_DIR, "prophet")
    os.makedirs(save_dir, exist_ok=True)
    forecaster.save(save_dir)
    log.info("Saved %d models to %s", len(forecaster.models), save_dir)

    log.info("--- Evaluating on validation set ---")
    val_df = pd.read_csv(os.path.join(DATA_DIR, "telemetry_val.csv"))
    val_df["timestamp"] = pd.to_datetime(val_df["timestamp"])
    mape_scores = []
    for (nid, metric), model in forecaster.models.items():
        actuals = val_df[(val_df["node_id"] == nid)][metric].values[:100]
        if len(actuals) < 2:
            continue
        fc = forecaster.forecast(nid, metric, horizon_hours=[1.0])
        pred = fc.get("horizon", {}).get("1h", {}).get("yhat", 0)
        mae = float(np.mean(np.abs(actuals - pred)))
        mean_actual = float(np.mean(actuals))
        mape = (mae / max(mean_actual, 1e-6)) * 100 if mean_actual > 1e-6 else 0
        mape_scores.append(mape)
        if len(forecaster.models) <= 10 or "PE-1" in nid or mape > 50:
            log.info("  %s / %s: MAPE=%.1f%%, ttb=%s",
                     nid, metric, mape,
                     fc.get("time_to_breach_hours", "N/A"))
    if mape_scores:
        log.info("MAPE across %d models: median=%.1f%%, mean=%.1f%%",
                 len(mape_scores), np.median(mape_scores), np.mean(mape_scores))

    log.info("Done.")


if __name__ == "__main__":
    main()
