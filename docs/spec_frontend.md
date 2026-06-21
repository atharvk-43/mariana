# Spec: Frontend Dashboard
## Files: `frontend/index.html`, `frontend/style.css`, `frontend/script.js`

---

## Design Requirements

- **Zero build step**: Vanilla HTML + CSS + JS only. Chart.js via CDN.
- **Dark mode only**: Space/mission-control aesthetic fits ISRO context.
- **Real-time**: Connect to WebSocket `/ws/stream` for 2-second updates.
- **6 panels**: All visible on one screen (no scrolling needed for demo).

---

## Layout (CSS Grid)

```
┌────────────────────────────────────────────────────────┐
│  MARTIAN PS13 — NOC Mission Control    ● LIVE  [time]  │  ← Header bar
├──────────────────┬────────────────┬────────────────────┤
│                  │  Timeseries    │   Alert Feed       │
│  Network         │  Charts        │   (live scroll)    │
│  Topology        │  (Chart.js)    │                    │
│  (SVG graph)     │                │                    │
│                  ├────────────────┤                    │
│                  │  Prediction    │                    │
│                  │  Widget        │                    │
├──────────────────┴────────────────┴────────────────────┤
│   Anomaly Heatmap (time × node grid)                   │  ← Bottom strip
├────────────────────────────────────────────────────────┤
│   NOC Copilot Panel (query box + response)             │  ← Expandable
└────────────────────────────────────────────────────────┘
```

Grid definition:
```css
.dashboard {
  display: grid;
  grid-template-rows: 60px 1fr 180px auto;
  grid-template-columns: 380px 1fr 320px;
  gap: 12px;
  height: 100vh;
  padding: 12px;
  background: #0a0e1a;
}
```

---

## Color System

```css
:root {
  /* Background */
  --bg-primary:    #0a0e1a;
  --bg-card:       #111827;
  --bg-card-hover: #1a2235;
  --border:        #1e2d42;
  
  /* Status Colors */
  --normal:   #22c55e;   /* green */
  --warning:  #f59e0b;   /* amber */
  --critical: #ef4444;   /* red */
  --info:     #3b82f6;   /* blue */
  
  /* Text */
  --text-primary:   #e2e8f0;
  --text-secondary: #64748b;
  --text-accent:    #38bdf8;  /* sky blue — for values/numbers */
  
  /* Glow effects (use sparingly) */
  --glow-normal:   0 0 12px rgba(34, 197, 94, 0.3);
  --glow-warning:  0 0 12px rgba(245, 158, 11, 0.3);
  --glow-critical: 0 0 12px rgba(239, 68, 68, 0.4);
  
  /* Typography */
  --font-mono: 'JetBrains Mono', 'Courier New', monospace;
  --font-ui:   'Inter', system-ui, sans-serif;
}
```

Import in `<head>`:
```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
```

Note: These font imports are the ONLY external URLs in the frontend. All data is local.

---

## Panel 1: Network Topology SVG

### Implementation
Build an inline SVG in `index.html`. Hardcode node positions (static topology, 10 nodes).

```html
<div class="card panel-topology">
  <div class="card-header">
    <span class="card-title">Network Topology</span>
    <span class="card-subtitle" id="topo-status">10 nodes</span>
  </div>
  <svg id="topology-svg" viewBox="0 0 360 320" xmlns="http://www.w3.org/2000/svg">
    <!-- Edges (drawn first, behind nodes) -->
    <g id="edges-layer">
      <!-- Each edge: <line> with class "topo-edge" -->
      <line class="topo-edge" x1="80" y1="200" x2="160" y2="140" .../>
      <!-- ... all 10 edges from topology.EDGES ... -->
    </g>
    <!-- Nodes -->
    <g id="nodes-layer">
      <!-- Each node: <circle> + <text> label -->
      <!-- Circle radius 18, color based on node_type, border on health -->
      <circle id="node-pe-router-1" class="topo-node" cx="160" cy="140" r="18" .../>
      <text class="node-label" x="160" y="170">PE-1</text>
      <!-- ... all 10 nodes ... -->
    </g>
  </svg>
</div>
```

### Suggested Node Positions (SVG viewBox 360×320)
```javascript
const NODE_POSITIONS = {
  "pe-router-1":  {x: 160, y: 120},
  "pe-router-2":  {x: 260, y: 120},
  "p-router-1":   {x: 180, y: 190},
  "p-router-2":   {x: 270, y: 230},
  "p-router-3":   {x: 120, y: 230},
  "ce-branch-1":  {x:  60, y:  70},
  "ce-branch-2":  {x: 160, y:  50},
  "ce-branch-3":  {x: 300, y:  70},
  "ce-dc-1":      {x: 290, y: 290},
  "ce-dc-2":      {x: 100, y: 290},
};
```

### JS: Update topology colors

```javascript
function updateTopology(nodes) {
  // nodes = array of NodeHealthResult from /telemetry/latest or WS
  nodes.forEach(node => {
    const circle = document.getElementById(`node-${node.node_id}`);
    if (!circle) return;
    
    const colorMap = {
      "NORMAL":   "#22c55e",
      "WARNING":  "#f59e0b",
      "CRITICAL": "#ef4444",
    };
    const glowMap = {
      "NORMAL":   "0 0 10px rgba(34,197,94,0.4)",
      "WARNING":  "0 0 10px rgba(245,158,11,0.5)",
      "CRITICAL": "0 0 14px rgba(239,68,68,0.7)",
    };
    
    circle.setAttribute("fill", colorMap[node.health_state] || "#64748b");
    circle.style.filter = `drop-shadow(${glowMap[node.health_state] || "none"})`;
    
    // Pulse animation for CRITICAL nodes
    if (node.health_state === "CRITICAL") {
      circle.classList.add("pulse-critical");
    } else {
      circle.classList.remove("pulse-critical");
    }
  });
}

// CSS for pulse animation:
// @keyframes pulse-critical {
//   0%, 100% { opacity: 1; }
//   50% { opacity: 0.5; }
// }
// .pulse-critical { animation: pulse-critical 1s infinite; }
```

---

## Panel 2: Timeseries Charts (Chart.js)

### CDN import
```html
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
```

### Charts to Show (use tab switcher or stacked small charts)
1. **Latency (ms)** — line chart, rolling 5 minutes
2. **Packet Loss (%)** — line chart
3. **Utilization (%)** — line chart

### JS: Chart initialization

```javascript
const CHART_COLORS = {
  "pe-router-1": "#38bdf8",   // sky blue
  "pe-router-2": "#a78bfa",   // purple
  "p-router-1":  "#34d399",   // emerald
  // etc. — one color per node
};

const MAX_DATAPOINTS = 150; // 5 minutes at 2s

function initChart(canvasId, label, color) {
  const ctx = document.getElementById(canvasId).getContext("2d");
  return new Chart(ctx, {
    type: "line",
    data: {
      labels: [],  // timestamps (last N)
      datasets: []  // one dataset per node (visible nodes only)
    },
    options: {
      animation: false,           // no animation — too slow for 2s updates
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true, labels: { color: "#94a3b8", font: { size: 10 } } },
      },
      scales: {
        x: {
          ticks: { color: "#64748b", maxTicksLimit: 6, maxRotation: 0 },
          grid: { color: "#1e2d42" },
        },
        y: {
          ticks: { color: "#64748b" },
          grid: { color: "#1e2d42" },
        }
      }
    }
  });
}

// Charts:
const charts = {
  latency:  initChart("chart-latency", "Latency (ms)", "#38bdf8"),
  loss:     initChart("chart-loss", "Packet Loss (%)", "#ef4444"),
  util:     initChart("chart-util", "Utilization (%)", "#f59e0b"),
};

function updateCharts(historyData) {
  // historyData: array of TelemetryRow for selected nodes
  // Push to chart datasets, trim to MAX_DATAPOINTS
}
```

### Node selector for charts
```html
<div class="chart-node-selector">
  <button class="node-btn active" data-node="pe-router-1">PE-1</button>
  <button class="node-btn" data-node="pe-router-2">PE-2</button>
  <button class="node-btn" data-node="p-router-1">P-1</button>
  <!-- etc -->
</div>
```

---

## Panel 3: Alert Feed

```html
<div class="card panel-alerts">
  <div class="card-header">
    <span class="card-title">Live Alerts</span>
    <span class="alert-badge" id="alert-count">0</span>
  </div>
  <div class="alert-list" id="alert-list">
    <!-- Dynamically populated -->
  </div>
</div>
```

### JS: Render alert item

```javascript
function renderAlert(alert) {
  // alert = Alert schema
  const div = document.createElement("div");
  div.className = `alert-item alert-${alert.health_state.toLowerCase()}`;
  div.innerHTML = `
    <div class="alert-header">
      <span class="alert-icon">${alert.health_state === "CRITICAL" ? "🔴" : "⚠️"}</span>
      <span class="alert-node">${alert.node_id}</span>
      <span class="alert-time">${formatTime(alert.timestamp)}</span>
    </div>
    <div class="alert-message">${alert.message}</div>
    ${alert.time_to_impact_hours ? 
      `<div class="alert-eta">Impact in: <strong>${alert.time_to_impact_hours.toFixed(1)}h</strong></div>` : ""}
    <button class="btn-copilot" onclick="querycopilotForAlert('${alert.node_id}')">
      Ask Copilot →
    </button>
  `;
  return div;
}

function updateAlertFeed(alerts) {
  const list = document.getElementById("alert-list");
  list.innerHTML = "";  // clear
  document.getElementById("alert-count").textContent = alerts.length;
  
  if (alerts.length === 0) {
    list.innerHTML = '<div class="no-alerts">✅ All systems nominal</div>';
    return;
  }
  
  // Sort: CRITICAL first, then by risk_score descending
  alerts.sort((a, b) => {
    if (a.health_state === "CRITICAL" && b.health_state !== "CRITICAL") return -1;
    return b.risk_score - a.risk_score;
  });
  
  alerts.forEach(alert => list.appendChild(renderAlert(alert)));
}
```

---

## Panel 4: Prediction Widget

```html
<div class="card panel-predictions">
  <div class="card-header">
    <span class="card-title">Failure Predictions</span>
  </div>
  <div id="prediction-list">
    <!-- Rows for nodes with non-null time_to_impact_hours -->
  </div>
</div>
```

### JS: Render prediction row

```javascript
function renderPrediction(node) {
  // node = NodeHealthResult with time_to_impact_hours != null
  const urgencyColor = node.time_to_impact_hours < 1 ? "var(--critical)" :
                       node.time_to_impact_hours < 3 ? "var(--warning)" : "var(--info)";
  return `
    <div class="prediction-row">
      <div class="pred-node">${node.node_id}</div>
      <div class="pred-metric">${node.breach_metric || "unknown metric"}</div>
      <div class="pred-eta" style="color: ${urgencyColor}">
        ${node.time_to_impact_hours.toFixed(1)}h
      </div>
      <div class="pred-bar">
        <div class="pred-bar-fill" style="
          width: ${Math.min(100, (6 - node.time_to_impact_hours) / 6 * 100)}%;
          background: ${urgencyColor};
        "></div>
      </div>
    </div>
  `;
}
```

---

## Panel 5: Anomaly Heatmap

A time × node grid where each cell is colored by anomaly risk score.

```html
<div class="card panel-heatmap">
  <div class="card-header"><span class="card-title">Anomaly Heatmap (5 min × nodes)</span></div>
  <div class="heatmap-container">
    <canvas id="heatmap-canvas"></canvas>
  </div>
</div>
```

### JS: Draw heatmap on canvas

```javascript
// Stores: risk_score history per node
// shape: dict[node_id → array of last 150 risk_scores]
const riskHistory = {};
NODES.forEach(n => riskHistory[n] = []);

function drawHeatmap() {
  const canvas = document.getElementById("heatmap-canvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  
  const nNodes = NODES.length;           // 10
  const nTime  = 150;                    // 5 minutes at 2s
  const cellW  = W / nTime;
  const cellH  = H / nNodes;
  
  ctx.clearRect(0, 0, W, H);
  
  NODES.forEach((nodeId, nodeIdx) => {
    const history = riskHistory[nodeId] || [];
    history.forEach((score, timeIdx) => {
      const x = timeIdx * cellW;
      const y = nodeIdx * cellH;
      ctx.fillStyle = riskScoreToColor(score);  // 0=dark, 100=red
      ctx.fillRect(x, y, cellW - 1, cellH - 1);
    });
  });
}

function riskScoreToColor(score) {
  // 0–35: dark green, 35–65: amber, 65–100: red
  if (score < 35) return `rgba(34, 197, 94, ${score/35 * 0.6 + 0.1})`;
  if (score < 65) return `rgba(245, 158, 11, ${(score-35)/30 * 0.7 + 0.3})`;
  return `rgba(239, 68, 68, ${(score-65)/35 * 0.5 + 0.5})`;
}
```

---

## Panel 6: NOC Copilot Panel

```html
<div class="card panel-copilot" id="copilot-panel">
  <div class="card-header">
    <span class="card-title">🤖 NOC Copilot</span>
    <span class="copilot-status" id="copilot-status">● OFFLINE</span>
  </div>
  <div class="copilot-response" id="copilot-response">
    <div class="copilot-placeholder">Select an alert and click "Ask Copilot →"</div>
  </div>
  <div class="copilot-input-row">
    <input type="text" id="copilot-question" 
           placeholder="Ask a question... e.g. What caused this?" />
    <button id="copilot-submit" onclick="submitCopilotQuery()">Ask</button>
  </div>
</div>
```

### JS: Copilot query

```javascript
let pendingAlertContext = null;

function querycopilotForAlert(nodeId) {
  // Find the current alert/result for this node
  const result = currentNodeResults[nodeId];
  if (!result) return;
  
  pendingAlertContext = {
    node_id: result.node_id,
    risk_score: result.risk_score,
    health_state: result.health_state,
    time_to_impact_hours: result.time_to_impact_hours,
    breach_metric: result.breach_metric,
    model_scores: result.model_scores,
    metric_health: result.metric_health,
    current_metrics: result.telemetry,
  };
  
  document.getElementById("copilot-panel").scrollIntoView({ behavior: "smooth" });
  document.getElementById("copilot-question").focus();
}

async function submitCopilotQuery() {
  if (!pendingAlertContext) return;
  
  const question = document.getElementById("copilot-question").value;
  const responseDiv = document.getElementById("copilot-response");
  
  responseDiv.innerHTML = '<div class="copilot-loading">⏳ Thinking...</div>';
  
  try {
    const response = await fetch("http://localhost:8000/copilot/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        alert_context: pendingAlertContext,
        question: question || null,
      }),
    });
    const data = await response.json();
    
    if (!data.available) {
      responseDiv.innerHTML = `<div class="copilot-error">Copilot offline: ${data.predicted_issue}</div>`;
      return;
    }
    
    responseDiv.innerHTML = `
      <div class="copilot-result">
        <div class="copilot-field">
          <span class="field-label">Issue</span>
          <span class="field-value">${data.predicted_issue}</span>
        </div>
        <div class="copilot-field">
          <span class="field-label">Confidence</span>
          <span class="field-value">${(data.confidence * 100).toFixed(0)}%</span>
        </div>
        <div class="copilot-field">
          <span class="field-label">Root Cause</span>
          <span class="field-value">${data.root_cause_hypothesis}</span>
        </div>
        <div class="copilot-field">
          <span class="field-label">Time to Impact</span>
          <span class="field-value urgency-${data.urgency.toLowerCase()}">
            ${data.estimated_time_to_impact_hours ? data.estimated_time_to_impact_hours.toFixed(1) + "h" : "N/A"}
          </span>
        </div>
        <div class="copilot-actions">
          <span class="field-label">Actions</span>
          <ol>
            ${data.recommended_actions.map(a => `<li>${a}</li>`).join("")}
          </ol>
        </div>
        ${data.runbook_reference ? 
          `<div class="copilot-runbook">📄 Runbook: ${data.runbook_reference}</div>` : ""}
      </div>
    `;
  } catch (err) {
    responseDiv.innerHTML = `<div class="copilot-error">Request failed: ${err.message}</div>`;
  }
}
```

---

## WebSocket Integration

```javascript
const BACKEND = "http://localhost:8000";
const WS_URL  = "ws://localhost:8000/ws/stream";

let ws = null;
let currentNodeResults = {};   // node_id → latest NodeHealthResult

function connectWebSocket() {
  ws = new WebSocket(WS_URL);
  
  ws.onopen = () => {
    console.log("WebSocket connected");
    document.getElementById("ws-status").textContent = "● LIVE";
    document.getElementById("ws-status").style.color = "var(--normal)";
  };
  
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "telemetry") {
      handleTelemetryUpdate(msg);
    }
  };
  
  ws.onerror = () => {
    document.getElementById("ws-status").textContent = "● DISCONNECTED";
    document.getElementById("ws-status").style.color = "var(--critical)";
  };
  
  ws.onclose = () => {
    // Auto-reconnect after 3 seconds
    setTimeout(connectWebSocket, 3000);
  };
}

function handleTelemetryUpdate(msg) {
  // msg.nodes: list of NodeHealthResult
  // msg.active_alerts: list of Alert
  
  msg.nodes.forEach(node => {
    currentNodeResults[node.node_id] = node;
    
    // Update risk history for heatmap
    if (!riskHistory[node.node_id]) riskHistory[node.node_id] = [];
    riskHistory[node.node_id].push(node.risk_score);
    if (riskHistory[node.node_id].length > 150) riskHistory[node.node_id].shift();
  });
  
  updateTopology(msg.nodes);
  updateAlertFeed(msg.active_alerts);
  updatePredictions(msg.nodes.filter(n => n.time_to_impact_hours != null));
  drawHeatmap();
  
  // Update timestamp
  document.getElementById("last-update").textContent = new Date().toLocaleTimeString();
}

// Init on page load
window.onload = () => {
  connectWebSocket();
  checkCopilotStatus();
};

async function checkCopilotStatus() {
  try {
    const r = await fetch(`${BACKEND}/copilot/status`);
    const data = await r.json();
    const el = document.getElementById("copilot-status");
    el.textContent = data.available ? "● ONLINE (Air-Gapped)" : "● OFFLINE";
    el.style.color = data.available ? "var(--normal)" : "var(--critical)";
  } catch {}
}
```

---

## CSS Key Snippets

```css
/* Cards */
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
  overflow: hidden;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}

.card-title {
  font-family: var(--font-ui);
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-secondary);
}

/* Alert items */
.alert-item {
  padding: 10px 12px;
  border-radius: 6px;
  margin-bottom: 6px;
  border-left: 3px solid;
  font-size: 13px;
}

.alert-warning  { border-color: var(--warning); background: rgba(245,158,11,0.08); }
.alert-critical { border-color: var(--critical); background: rgba(239,68,68,0.08); }

/* Topology SVG nodes */
.topo-node {
  stroke: var(--border);
  stroke-width: 2;
  cursor: pointer;
  transition: filter 0.3s ease;
}

.topo-edge {
  stroke: #1e2d42;
  stroke-width: 2;
}

/* Copilot panel */
.copilot-result { font-size: 13px; line-height: 1.6; }
.field-label { color: var(--text-secondary); font-size: 11px; text-transform: uppercase; }
.field-value { color: var(--text-primary); font-family: var(--font-mono); }

/* Btn */
.btn-copilot {
  font-size: 11px;
  padding: 3px 8px;
  background: rgba(59,130,246,0.15);
  border: 1px solid rgba(59,130,246,0.3);
  color: var(--info);
  border-radius: 4px;
  cursor: pointer;
  margin-top: 4px;
}
.btn-copilot:hover { background: rgba(59,130,246,0.25); }
```
