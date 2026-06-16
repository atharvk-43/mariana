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
# MARTIAN PS13 - Developer Handover & Project Workflow

## Project Overview

MARTIAN PS13 is a telemetry monitoring and anomaly detection platform inspired by spacecraft mission-control systems.

The goal is to monitor telemetry data, identify abnormal behavior, predict potential failures, and present the information through a dashboard.

The project is intentionally designed so that development can continue completely locally without requiring AWS or EC2.

---

# Current Project Status

## Completed

### Backend

* FastAPI backend created
* API endpoints implemented
* Telemetry ingestion implemented
* CSV-based telemetry storage implemented
* Live telemetry endpoint implemented
* Anomaly endpoint implemented

### Data

* telemetry.csv generated
* live_telemetry.csv generated

### ML Baseline

* Isolation Forest anomaly detection implemented
* Initial anomaly labeling working

### Frontend

* Dashboard created
* Live telemetry display implemented
* Anomaly statistics display implemented

### Infrastructure

* GitHub repository configured
* Local development workflow established
* EC2 deployment tested successfully

---

# Current Repository Structure

```text
mariana/

backend/
├── main.py
├── telemetry.py
├── predictor.py
├── stream.py
├── telemetry.csv
├── live_telemetry.csv
├── requirements.txt

frontend/
├── index.html
├── style.css
├── script.js

README.md
PROJECT_WORKFLOW.md
```

---

# API Endpoints

## GET /

Returns service status.

Example:

```json
{
  "project": "MARTIAN PS13",
  "status": "running"
}
```

---

## GET /health

Returns health status.

Example:

```json
{
  "healthy": true
}
```

---

## GET /latest

Returns latest telemetry record from:

live_telemetry.csv

---

## GET /anomalies

Returns:

```json
{
  "total_records": 1000,
  "anomalies_detected": 50
}
```

---

# Telemetry Fields

Current telemetry contains:

## battery_voltage

Expected range:

24V - 30V

---

## temperature

Expected range:

-20°C to 40°C

---

## cpu_load

Expected range:

0% - 100%

---

## signal_strength

Expected range:

40 - 100

---

# Local Development Setup

Clone:

```bash
git clone https://github.com/atharvk-43/mariana.git
```

Enter project:

```bash
cd mariana
```

Create environment:

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r backend/requirements.txt
```

---

# Running Backend

```bash
cd backend
uvicorn main:app --reload
```

Backend URL:

```text
http://localhost:8000
```

---

# Running Frontend

Open another terminal:

```bash
cd frontend
python -m http.server 5500
```

Frontend URL:

```text
http://localhost:5500
```

Important:

The frontend expects the backend to be available at:

```text
http://localhost:8000
```

No AWS or EC2 instance is required.

---

# ML Development Scope

Primary file:

```text
backend/predictor.py
```

Additional files that may be added:

```text
backend/model.py
backend/train.py
backend/inference.py
backend/evaluation.py
```

---

# Files To Avoid Modifying

Avoid modifying unless necessary:

```text
frontend/
backend/main.py
```

These components are currently working.

---

# ML Objectives

## Objective 1

Improve anomaly detection quality.

Current implementation:

Isolation Forest

Potential improvements:

* Hyperparameter tuning
* Feature engineering
* Better anomaly scoring

Desired output:

```json
{
  "anomaly": true,
  "risk_score": 87
}
```

---

## Objective 2

Health Classification

Introduce states:

* NORMAL
* WARNING
* CRITICAL

Example:

```json
{
  "battery_status": "NORMAL",
  "temperature_status": "WARNING",
  "signal_status": "CRITICAL"
}
```

---

## Objective 3

Failure Prediction

Predict future system failures.

Examples:

* Battery degradation
* Signal degradation
* Thermal overload
* CPU overload

Desired output:

```json
{
  "predicted_failure_hours": 4.2,
  "risk_level": "HIGH"
}
```

---

# Design Philosophy

This project should answer:

1. What is happening?
2. Is it abnormal?
3. Why is it abnormal?
4. What will happen next?
5. How urgent is the issue?

The ML component should move the system toward answering all five questions.

---

# Important Note

AWS EC2 was used only for deployment testing.

The project is now fully portable.

All required code, datasets, and dependencies are available in GitHub.

Development should continue locally unless deployment testing is required.
