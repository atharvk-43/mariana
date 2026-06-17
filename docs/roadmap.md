# Hackathon Master Roadmap: MARTIAN PS13

**Objective:** Build an Air-Gapped Predictive Copilot for Secure MPLS Operations (ISRO PS13).
**Architecture Split:** EC2 (Heavy Networking & Data) + Local (ML Inference & AI Copilot).

## Team Roles & Assignments
* **Networking & DevOps (Teammate - EC2):** Responsible for the Containerlab topology, routing protocols, fault injection, and the telemetry collection pipeline.
* **ML & AI Engineer (You - Local):** Responsible for the predictive analytics, the local RAG copilot, and the backend/frontend integration.

---

## Project Phases & Timeline

### Phase 1: Network Simulator (EC2)
**Owner:** DevOps/Networking
**Goal:** Establish the physical and logical underlay/overlay network using FRRouting.
**Key Deliverables:** 
- OSPF/BGP and MPLS configurations.
- SD-WAN IPSec Tunnels.
- Fault Injection scripts (latency, packet loss, route flapping).
**Documentation:** See `network_simulator.md`

### Phase 2: Telemetry Pipeline (EC2)
**Owner:** DevOps/Networking
**Goal:** Capture high-fidelity data and store it in a time-series database.
**Key Deliverables:** 
- SNMP and NetFlow exporters.
- Syslog parsing.
- Telegraf + InfluxDB deployment on the EC2 instance.
**Documentation:** See `telemetry.md`

### Phase 3: ML Architecture & Predictive Models (Local)
**Owner:** ML Engineer
**Goal:** Build the forecasting and classification engines that predict failures before they happen.
**Key Deliverables:** 
- Time-series forecaster (LSTM/Prophet).
- Risk-score classifier (XGBoost).
**Documentation:** See `ml_architecture.md`

### Phase 4: Air-Gapped RAG Copilot (Local)
**Owner:** ML Engineer
**Goal:** Deploy a local LLM that grounds its advice in NOC runbooks.
**Key Deliverables:** 
- Local FAISS vector database.
- Ollama + `phi3` inference pipeline.

### Phase 5: System Integration & Dashboard
**Owner:** Fullstack
**Goal:** Tie the EC2 database to the local ML models and visualize it.
**Key Deliverables:** 
- FastAPI backend polling InfluxDB.
- React/HTML dashboard showing network graphs and Copilot insights.
**Documentation:** See `integration.md`
