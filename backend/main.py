from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "project": "MARTIAN PS13",
        "status": "running"
    }

@app.get("/health")
def health():
    return {
        "healthy": True
    }

@app.get("/anomalies")
def anomalies():

    df = pd.read_csv("telemetry.csv")

    anomaly_count = len(
        df[df["anomaly"] == -1]
    )

    total_records = len(df)

    return {
        "total_records": total_records,
        "anomalies_detected": anomaly_count
    }

@app.get("/latest")
def latest():

    df = pd.read_csv("live_telemetry.csv")

    row = df.iloc[-1]

    return row.to_dict()

