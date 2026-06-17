# ML Architecture & Predictive Models (Local Environment)

**Assignee:** ML & AI Engineer (You)
**Goal:** Predict anomalies before they hit thresholds, and power the offline Copilot.

## 1. Time-Series Forecaster (Prediction)
* **Model:** `Prophet` or `LSTM` (PyTorch).
* **Input:** The last 60 minutes of SNMP byte counters and latency metrics pulled from the EC2 InfluxDB.
* **Output:** A forecast for the next 15 minutes.
* **Objective:** If the forecasted line crosses 90% utilization or 150ms latency, trigger a **Pre-emptive Alert** (Scenario 1 & 3).

## 2. Health State Classifier (Risk Scoring)
* **Model:** `XGBoost` Classifier.
* **Input:** A flattened vector of recent syslog events (e.g., flap count), NetFlow anomalies, and current CPU/bandwidth.
* **Output:** 
  - Risk Score (0-100)
  - State classification: `NORMAL`, `WARNING`, `CRITICAL`.
* **Objective:** Correlate minor routing events to determine if a major outage is imminent (Scenario 2).

## 3. The Offline RAG Copilot
* **LLM Engine:** `Ollama` running locally serving the `phi3:mini` model (quantized 4-bit, takes ~2.5GB disk).
* **Vector Store:** `FAISS` running locally.
* **Knowledge Base:** Ingest `.txt` files containing NOC troubleshooting steps (e.g., "If BGP flaps, check physical link layer").
* **Execution Flow:** 
  1. ML Classifier flags `WARNING` (e.g., Risk Score 85 due to Jitter).
  2. Backend queries FAISS for "Jitter on IPSec Tunnel".
  3. Backend constructs prompt: *"You are an AI NOC Copilot. Alert: IPSec Tunnel Jitter is high. Runbook suggests checking QoS policies. Provide a summary."*
  4. Ollama returns a natural language response.
