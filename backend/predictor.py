import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.read_csv("telemetry.csv")

features = df[
    [
        "cpu_load_pct",
        "memory_used_pct",
        "utilization_pct",
        "latency_ms",
        "jitter_ms",
        "packet_loss_pct",
        "bgp_sessions_active",
        "bgp_updates_per_min",
        "queue_depth"
    ]
]

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

predictions = model.fit_predict(features)

df["anomaly"] = predictions

df.to_csv(
    "telemetry.csv",
    index=False
)

print(df["anomaly"].value_counts())