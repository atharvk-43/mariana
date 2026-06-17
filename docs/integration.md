# System Integration & Dashboard (Local Environment)

**Assignee:** Fullstack Developer (You / Shared)
**Goal:** Tie the distributed system together securely and present it to the operators.

## The Backend (FastAPI)
The central nervous system running locally on your laptop:
* **Background Worker:** A scheduler (e.g., `APScheduler`) that queries the EC2 InfluxDB `http://<EC2-IP>:8086` every 5 seconds.
* **Inference Pipeline:** Feeds the freshly polled data into the XGBoost and Prophet models loaded in memory.
* **LLM Invocation:** If Risk Score > 75, trigger the `copilot.py` script.
* **Endpoints:**
  - `GET /api/network/graph`: Returns the topology and edge status (Green/Red).
  - `GET /api/metrics/live`: Returns the latest forecasts.
  - `POST /api/copilot/chat`: Allows the human operator to type a question to the Ollama model.

## The Frontend (React or HTML/JS)
* **Topology Visualization:** Use a library like `Cytoscape.js` or `React Flow` to draw the 6 routers. Bind the link colors to the XGBoost Risk Scores.
* **Alert Feed:** A scrolling list of pre-emptive alerts ("Forecasting branch1_to_hub link will reach 100% capacity in 12 minutes").
* **Copilot Sidebar:** A chat interface where the NOC operator can see the AI's automated root-cause analysis and ask follow-up questions.
