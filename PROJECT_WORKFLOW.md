# MARTIAN PS13 - Developer Handover & Workflow

## Project Overview

MARTIAN PS13 is a Network Telemetry Monitoring and Anomaly Detection Platform.

The objective is to:

* Collect network telemetry
* Detect anomalies
* Estimate operational risk
* Predict future failures
* Visualize network health through a dashboard

The project uses synthetic network telemetry to simulate real network devices and monitoring systems.

---

# Current Project Status

## Completed

### Data Generation

* Synthetic network telemetry generator implemented
* Historical telemetry dataset generation implemented
* Live telemetry stream implemented

### Machine Learning

* Isolation Forest anomaly detection implemented
* Anomaly labeling pipeline operational

### Backend

* FastAPI backend implemented
* REST API endpoints operational
* Telemetry serving operational

### Frontend

* Dashboard implemented
* Live telemetry display operational
* Anomaly statistics display operational

### Infrastructure

* GitHub repository configured
* Local development workflow established
* AWS testing completed
* Project no longer depends on EC2

---

# Current Telemetry Schema

Historical telemetry:

backend/telemetry.csv

Live telemetry:

backend/live_telemetry.csv

Fields:

* timestamp
* node_id
* cpu_load_pct
* memory_used_pct
* utilization_pct
* latency_ms
* jitter_ms
* packet_loss_pct
* bgp_sessions_active
* bgp_updates_per_min
* queue_depth

---

# Current ML Pipeline

telemetry.py

↓

telemetry.csv

↓

predictor.py

↓

Isolation Forest

↓

anomaly column

↓

FastAPI API

↓

Dashboard

---

# Repository Structure

mariana/

backend/

* main.py
* telemetry.py
* predictor.py
* stream.py
* telemetry.csv
* live_telemetry.csv
* requirements.txt

frontend/

* index.html
* style.css
* script.js

README.md

PROJECT_WORKFLOW.md

---

# API Endpoints

GET /

Returns project status.

GET /health

Returns health status.

GET /latest

Returns latest live telemetry row.

GET /anomalies

Returns:

* total_records
* anomalies_detected

---

# Running Locally

Clone repository:

git clone https://github.com/atharvk-43/mariana.git

Enter project:

cd mariana

Install dependencies:

pip install -r backend/requirements.txt

---

# Running Backend

cd backend

uvicorn main:app --reload

Backend URL:

http://localhost:8000

---

# Running Frontend

Open another terminal:

cd frontend

python -m http.server 5500

Frontend URL:

http://localhost:5500

Important:

Frontend expects backend at:

http://localhost:8000

No AWS infrastructure is required.

---

# ML Development Scope

Current model:

Isolation Forest

File:

backend/predictor.py

---

# Recommended Next Steps

## Phase 1

Risk Scoring

Example:

{
"risk_score": 82
}

---

## Phase 2

Health Classification

Possible statuses:

* NORMAL
* WARNING
* CRITICAL

---

## Phase 3

Failure Prediction

Predict:

* Congestion
* Packet loss spikes
* BGP instability
* Resource exhaustion

---

## Phase 4

Dashboard Improvements

Add:

* Latency graphs
* Packet loss graphs
* CPU utilization graphs
* Health indicators
* Risk score cards

---

# Files Safe To Modify

* predictor.py
* telemetry.py
* stream.py

---

# Files To Modify Carefully

* main.py
* frontend/index.html
* frontend/script.js

These are currently working.

---

# Design Philosophy

The platform should answer:

1. What is happening?
2. Is it abnormal?
3. How severe is it?
4. What may happen next?
5. What action should be taken?

---

# Current State

The project is fully portable.

No EC2 dependency exists.

All code, datasets, and dependencies are available in GitHub.

A new developer should be able to clone the repository and continue development locally.
