# MARTIAN PS13 - Project Workflow and Development Guide

## Project Objective

Build a telemetry monitoring and anomaly detection platform inspired by spacecraft mission-control systems.

The system should:

* Ingest telemetry data
* Detect anomalies
* Predict failures before they occur
* Display telemetry and alerts through a dashboard
* Run locally without requiring AWS

---

## Current Architecture

Frontend

* index.html
* style.css
* script.js

Backend

* main.py
* telemetry.py
* predictor.py
* stream.py

Data

* telemetry.csv
* live_telemetry.csv

Deployment

* FastAPI
* Gunicorn
* AWS EC2

---

## Working APIs

GET /health

Returns:

{
"healthy": true
}

GET /latest

Returns latest telemetry row.

GET /anomalies

Returns:

{
"total_records": X,
"anomalies_detected": Y
}

---

## Current Telemetry Fields

* battery_voltage
* temperature
* cpu_load
* signal_strength

---

## ML Team Responsibilities

Primary file:

predictor.py

Additional files allowed:

* model.py
* train.py
* inference.py
* evaluation.py

Avoid modifying:

* frontend/
* main.py

unless required.

---

## ML Goals

### Goal 1

Improve anomaly detection.

Current model:

Isolation Forest

Desired output:

{
"anomaly": true,
"risk_score": 87
}

---

### Goal 2

Health Classification

States:

* NORMAL
* WARNING
* CRITICAL

Example:

Battery = NORMAL

Temperature = WARNING

Signal = CRITICAL

---

### Goal 3

Failure Prediction

Examples:

* Battery degradation
* Signal degradation
* Thermal overload
* CPU overload

Desired output:

{
"predicted_failure_hours": 4.2,
"risk_level": "HIGH"
}

---

## Running Locally

Clone repository:

git clone https://github.com/atharvk-43/mariana.git

Create environment:

Windows:

python -m venv venv

venv\Scripts\activate

Linux:

python3 -m venv venv

source venv/bin/activate

Install:

pip install -r backend/requirements.txt

Backend:

cd backend

python main.py

Frontend:

Serve frontend folder and open:

http://localhost:5500

---

## Data Assumptions

Battery Voltage:

24V - 30V

Temperature:

-20°C to 40°C

CPU Load:

0% - 100%

Signal Strength:

40 - 100

---

## Project Philosophy

The system should answer:

1. What is happening?
2. Is it abnormal?
3. Why is it abnormal?
4. What happens next?
5. How urgent is the issue?

Every ML contribution should move the project toward answering these questions.
