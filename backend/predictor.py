import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.read_csv("telemetry.csv")

features = df[
    [
        "battery_voltage",
        "temperature",
        "cpu_load",
        "signal_strength"
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

print(
    df["anomaly"].value_counts()
)
