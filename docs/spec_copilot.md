# Spec: Offline LLM NOC Copilot
## Files: `backend/copilot/ollama_client.py`, `rag_pipeline.py`, `runbooks/`, `past_incidents.json`

---

## Overview

The Copilot is **35% of the judge score**. It must:
1. Run completely offline (no OpenAI, no cloud APIs)
2. Respond to NOC operator queries with structured, grounded answers
3. Reference internal runbooks (not hallucinate generic advice)
4. Be wired into the FastAPI via `POST /copilot/query`

---

## Stack

| Component | Tool | Notes |
|---|---|---|
| Local LLM | Phi-3-mini-4k-instruct (Q4_K_M) via Ollama | ~2.5GB, fast on CPU, runs on EC2 |
| Fallback LLM | Mistral-7B-v0.3 (Q4_K_M) | ~4.1GB, use if EC2 has RAM headroom |
| Vector Store | ChromaDB (embedded, no server) | Saves to local disk |
| Orchestration | LangChain Community | pip install langchain langchain-community |
| Embeddings | nomic-embed-text via Ollama | Local embedding model, no internet |

---

## Setup Instructions (Run on EC2)

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull the LLM
ollama pull phi3:mini        # Phi-3-mini 4k instruct (recommended)
ollama pull nomic-embed-text # local embeddings model

# 3. Verify Ollama is running
ollama list   # should show phi3:mini and nomic-embed-text

# 4. Ollama runs at http://localhost:11434 (default, no auth needed)
# It is LAN-accessible but has no outbound calls — air-gap compliant.
```

---

## File 1: `backend/copilot/ollama_client.py`

### Purpose
Thin wrapper around the Ollama HTTP API. All LLM calls go through this class.

```python
import requests
import json

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "phi3:mini"

class OllamaClient:
    
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: int = 120,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.1,  # Low temperature for factual NOC responses
        max_tokens: int = 512,
    ) -> str:
        """
        Call Ollama /api/generate endpoint.
        
        Request body:
        {
          "model": self.model,
          "prompt": prompt,
          "system": system_prompt,
          "options": {"temperature": temperature, "num_predict": max_tokens},
          "stream": false
        }
        
        Returns the generated text string.
        Raises requests.exceptions.ConnectionError if Ollama is not running.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.9,
            }
        }
        if system_prompt:
            payload["system"] = system_prompt
        
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["response"]
    
    def is_available(self) -> bool:
        """Ping Ollama to check if it's running. Returns True/False."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except:
            return False
    
    def list_models(self) -> list[str]:
        """Return list of locally available model names."""
```

---

## File 2: `backend/copilot/rag_pipeline.py`

### Purpose
ChromaDB vector store + LangChain retriever + Ollama LLM → structured NOC response.

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import chromadb
import json

CHROMA_PERSIST_DIR = "backend/copilot/chroma_db"
RUNBOOKS_DIR = "backend/copilot/runbooks"
INCIDENTS_FILE = "backend/copilot/past_incidents.json"

# System prompt for the NOC Copilot
SYSTEM_PROMPT = """You are an expert NOC (Network Operations Center) AI Copilot for a secure MPLS SD-WAN network.
You operate in an air-gapped environment. You have access to network topology information, runbooks, and past incident records.
Your responses must be factual, concise, and based only on the provided context.
Do not hallucinate. If you don't know something, say so.
Always structure your response as valid JSON matching the schema provided."""


class NOCCopilot:
    """
    RAG-based NOC Copilot using ChromaDB + Ollama.
    Call build_index() once to populate the vector store.
    Call query() for each operator request.
    """
    
    def __init__(
        self,
        ollama_model: str = "phi3:mini",
        embed_model: str = "nomic-embed-text",
        chroma_dir: str = CHROMA_PERSIST_DIR,
    ):
        self.llm = Ollama(model=ollama_model, temperature=0.1, num_predict=600)
        self.embeddings = OllamaEmbeddings(model=embed_model)
        self.chroma_dir = chroma_dir
        self.vectorstore = None
        self.retriever = None
    
    def build_index(self) -> None:
        """
        Load all runbooks and past incidents into ChromaDB.
        Call once at startup if chroma_dir doesn't exist yet, or to rebuild.
        
        Steps:
        1. Load each .md file from RUNBOOKS_DIR → one Document per file
           - metadata: {"source": filename, "type": "runbook"}
        2. Load INCIDENTS_FILE → each incident becomes one Document
           - metadata: {"source": incident["id"], "type": "incident", "fault_type": incident["fault_type"]}
        3. Load topology metadata → one Document
           - metadata: {"source": "topology", "type": "topology"}
        4. Create ChromaDB collection via Chroma.from_documents(docs, embeddings, persist_directory=chroma_dir)
        5. Log: "Indexed N documents into ChromaDB"
        """
    
    def load_index(self) -> None:
        """Load existing ChromaDB index from chroma_dir."""
        self.vectorstore = Chroma(
            persist_directory=self.chroma_dir,
            embedding_function=self.embeddings,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},  # retrieve top-4 most relevant docs
        )
    
    def query(
        self,
        alert_context: dict,
        operator_question: str = None,
    ) -> dict:
        """
        Main entry point. Takes a structured alert and returns a structured copilot response.
        
        alert_context structure:
        {
          "node_id": str,
          "risk_score": float,
          "health_state": str,
          "time_to_impact_hours": float | None,
          "breach_metric": str | None,
          "model_scores": dict,
          "metric_health": dict,
          "current_metrics": dict,  # subset of current telemetry row
        }
        
        operator_question: optional natural language question, e.g.
          "What should we do to prevent this failure?"
        
        Steps:
        1. Build query string from alert_context (summarize the alert in natural language)
        2. Retrieve top-4 relevant documents from ChromaDB
        3. Build prompt (see PROMPT_TEMPLATE below)
        4. Call LLM, parse JSON response
        5. Return structured dict
        
        Returns:
        {
          "predicted_issue": str,
          "confidence": float,
          "root_cause_hypothesis": str,
          "affected_scope": list[str],
          "estimated_time_to_impact_hours": float | None,
          "recommended_actions": list[str],   # ordered list of steps
          "runbook_reference": str | None,    # filename of most relevant runbook
          "urgency": str,                     # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
          "raw_llm_response": str,            # for debugging
        }
        """
    
    def _build_query_string(self, alert_context: dict) -> str:
        """
        Convert alert_context to a natural language query for retrieval.
        Example: "BGP route flap on pe-router-1, risk score 87, 
                  bgp_updates_per_min=450, time_to_impact=1.2 hours"
        """
    
    def _build_prompt(
        self,
        alert_context: dict,
        retrieved_docs: list,
        operator_question: str,
    ) -> str:
        """Build the final prompt from context + retrieved docs + question."""


PROMPT_TEMPLATE = """
You are a NOC AI Copilot. Analyze the following network alert and provide a structured JSON response.

=== NETWORK ALERT ===
{alert_summary}

=== RELEVANT RUNBOOKS AND PAST INCIDENTS ===
{retrieved_context}

=== OPERATOR QUESTION (if any) ===
{operator_question}

=== INSTRUCTIONS ===
Based ONLY on the above context, provide your analysis as valid JSON in this exact format:
{{
  "predicted_issue": "<brief description of what is happening>",
  "confidence": <float 0.0-1.0>,
  "root_cause_hypothesis": "<most likely root cause based on signals>",
  "affected_scope": ["<node or site 1>", "<node or site 2>"],
  "estimated_time_to_impact_hours": <float or null>,
  "recommended_actions": [
    "<step 1>",
    "<step 2>",
    "<step 3>"
  ],
  "runbook_reference": "<filename of most relevant runbook or null>",
  "urgency": "<LOW|MEDIUM|HIGH|CRITICAL>"
}}

Respond with ONLY the JSON object, no additional text.
"""
```

---

## Runbooks (create these 6 files in `backend/copilot/runbooks/`)

### `runbook_bgp_session_recovery.md`
```markdown
# Runbook: BGP Session Recovery

## Trigger Conditions
- bgp_sessions_active drops below bgp_sessions_total
- bgp_updates_per_min > 50
- bgp_withdrawals_per_min > 10

## Immediate Actions
1. Check physical connectivity on affected interface: `show interface eth1`
2. Verify BGP neighbor state: `show bgp summary`
3. Check BGP keepalive timers — default is 60s hold-time; if mismatched, sessions drop
4. Review recent configuration changes on the affected PE router
5. If flapping: apply BGP route dampening to reduce churn propagation

## Escalation
- If bgp_sessions_active = 0 for > 5 minutes: escalate to Tier-2 NOC
- Notify downstream sites (branches connected via affected PE)

## Recovery Validation
- bgp_sessions_active returns to total
- bgp_updates_per_min < 5
- bgp_prefixes_received returns to baseline
```

### `runbook_congestion_response.md`
```markdown
# Runbook: Link Congestion Response

## Trigger Conditions
- utilization_pct > 75% (WARNING), > 90% (CRITICAL)
- queue_depth > 40
- packet_loss_pct > 1%

## Immediate Actions
1. Identify congested interface: check bytes_in vs link capacity
2. Check QoS policy — verify traffic shaping and priority queues are applied
3. Traffic engineering: if MPLS TE available, shift traffic to alternate LSP
4. If DDoS suspected (sudden spike in bytes_in): apply rate limiting or ACL on ingress
5. Notify impacted sites of potential degradation

## Preventive Steps
- If utilization trending > 70% over 1 hour: pre-emptively reroute
- Review provisioned capacity vs current peak demand

## Recovery Validation
- utilization_pct drops below 60%
- queue_depth < 20
- packet_loss_pct < 0.5%
```

### `runbook_mpls_underlay_failure.md`
```markdown
# Runbook: MPLS Underlay Failure / Link Flap

## Trigger Conditions
- link_state = FLAPPING or DOWN on P-router
- errors_in increasing over time
- tunnel_packet_loss_pct > 1%
- ipsec_rekeyed_last_hr > 6

## Immediate Actions
1. Check physical layer: verify fiber/cable integrity on P-router interfaces
2. Check for CRC errors on interface (errors_in counter)
3. Verify MPLS label stack is intact: `show mpls ldp neighbors`
4. Check IPSec tunnel status: `show crypto ipsec sa`
5. If tunnel degraded: force IPSec rekey manually to refresh SA

## Traffic Impact Assessment
- Identify all VPNs traversing the affected P-router
- Estimate affected branch sites and services

## Recovery Validation
- link_state = UP stable for > 5 minutes
- errors_in = 0
- tunnel_packet_loss_pct < 0.3%
```

### `runbook_policy_drift_response.md`
```markdown
# Runbook: Routing Policy Misconfiguration / Policy Drift

## Trigger Conditions
- bgp_prefixes_received increases suddenly (> 20% above baseline)
- cpu_load_pct > 80%
- memory_used_pct > 85%
- ospf_spf_runs elevated

## Immediate Actions
1. Compare current route table size to baseline: if >> expected, route leak suspected
2. Check recent BGP policy changes: review route-map and prefix-list configs
3. Apply emergency prefix filtering on affected PE: block unexpected prefixes
4. Check memory usage — if routing table too large, risk of OOM crash
5. Roll back configuration change if root cause identified

## Escalation
- If cpu_load_pct > 90% sustained: risk of control-plane crash — escalate immediately
- Isolate affected PE if necessary to prevent cascade to other PEs

## Recovery Validation
- bgp_prefixes_received returns to baseline
- cpu_load_pct drops below 60%
- memory_used_pct drops below 75%
```

### `runbook_tunnel_degradation.md`
```markdown
# Runbook: SD-WAN Tunnel Health Degradation

## Trigger Conditions
- tunnel_packet_loss_pct > 1%
- jitter_ms > 5ms
- ipsec_rekeyed_last_hr > 8 (excessive rekeying)

## Immediate Actions
1. Run active probe across tunnel to measure current jitter/loss
2. Check underlay path MTU — IPSec adds overhead, fragmentation causes loss
3. Verify IPSec proposal (encryption/hash algorithms) — mismatch causes rekey failures
4. If jitter is primary concern: check QoS marking on tunnel traffic
5. Consider switching to backup tunnel if available

## Recovery Validation
- tunnel_packet_loss_pct < 0.3%
- jitter_ms < 2ms
- ipsec_rekeyed_last_hr normalizes to 1–4
```

### `runbook_general_escalation.md`
```markdown
# Runbook: General Escalation Criteria

## Escalate to Tier-2 Immediately If:
- Any node health_state = CRITICAL for > 3 minutes
- bgp_sessions_active = 0 on any PE router
- link_state = DOWN on any core P-router for > 2 minutes
- Multiple nodes simultaneously showing CRITICAL state

## Information to Provide When Escalating:
1. Affected node IDs and their health states
2. Time of first alert
3. Metrics that triggered the alert
4. Estimated time-to-impact from ML prediction
5. Runbooks already attempted

## Contact Information
- NOC Tier-2: [internal escalation path]
- Network Engineering On-call: [internal contact]
```

---

## `backend/copilot/past_incidents.json`

```json
[
  {
    "id": "INC-001",
    "fault_type": "bgp_flap",
    "date": "2026-05-12",
    "primary_node": "pe-router-1",
    "duration_minutes": 18,
    "root_cause": "BGP keepalive timer mismatch after config push",
    "resolution": "Corrected hold-time to 90s on both sides. BGP recovered in 3 minutes.",
    "affected_sites": ["branch-site-1", "branch-site-2"],
    "precursor_signals": "bgp_updates_per_min elevated for 8 minutes before session drop",
    "impact": "Service degradation to 2 branch sites for 18 minutes"
  },
  {
    "id": "INC-002",
    "fault_type": "congestion",
    "date": "2026-05-20",
    "primary_node": "ce-branch-1",
    "duration_minutes": 35,
    "root_cause": "Bulk backup job saturated uplink during business hours",
    "resolution": "Rescheduled backup job to 02:00 UTC. Applied QoS to deprioritize backup traffic.",
    "affected_sites": ["branch-site-1"],
    "precursor_signals": "utilization_pct rising steadily for 22 minutes before drops_in spike",
    "impact": "Branch site application latency 3x normal for 35 minutes"
  },
  {
    "id": "INC-003",
    "fault_type": "mpls_failure",
    "date": "2026-06-01",
    "primary_node": "p-router-1",
    "duration_minutes": 52,
    "root_cause": "Physical fiber cut on p-router-1 eth2. Link flapped before failing.",
    "resolution": "Physical repair + MPLS path failover to backup LSP within 4 minutes of failure.",
    "affected_sites": ["branch-site-1", "branch-site-2", "branch-site-3"],
    "precursor_signals": "errors_in increasing for 30 minutes, tunnel_packet_loss rising",
    "impact": "Partial service degradation across 3 sites. Full failover in 4 minutes."
  },
  {
    "id": "INC-004",
    "fault_type": "policy_drift",
    "date": "2026-06-08",
    "primary_node": "pe-router-2",
    "duration_minutes": 28,
    "root_cause": "Incorrect route-map applied during maintenance window caused route leak",
    "resolution": "Rolled back route-map. Cleared BGP sessions. Table normalized.",
    "affected_sites": ["branch-site-3", "datacenter"],
    "precursor_signals": "bgp_prefixes_received increased 40% over 12 minutes, cpu_load rising",
    "impact": "Routing instability on pe-router-2. CPU at 94% briefly."
  }
]
```

---

## FastAPI Integration

### In `backend/api/main.py`, add:

```python
from copilot.rag_pipeline import NOCCopilot
from copilot.ollama_client import OllamaClient

# Global instances (load at startup)
copilot = NOCCopilot()
ollama_client = OllamaClient()

@app.on_event("startup")
async def startup_event():
    if ollama_client.is_available():
        copilot.load_index()  # Load ChromaDB from disk
        print("NOC Copilot ready")
    else:
        print("WARNING: Ollama not running. Copilot disabled.")

@app.post("/copilot/query")
async def copilot_query(request: CopilotQueryRequest) -> CopilotResponse:
    """
    Accepts an alert context + optional operator question.
    Returns structured copilot response.
    
    If Ollama is not available: return a mock/error response, not a 500.
    """
    if not ollama_client.is_available():
        return {"error": "Copilot offline — Ollama not running", "available": False}
    
    result = copilot.query(
        alert_context=request.alert_context,
        operator_question=request.question,
    )
    return result

@app.get("/copilot/status")
async def copilot_status():
    """Check if Ollama is running and which model is loaded."""
    return {
        "ollama_available": ollama_client.is_available(),
        "model": DEFAULT_MODEL,
        "air_gapped": True,
    }
```

### Pydantic schemas (add to `backend/api/schemas.py`):

```python
class AlertContext(BaseModel):
    node_id: str
    risk_score: float
    health_state: str
    time_to_impact_hours: float | None
    breach_metric: str | None
    model_scores: dict[str, float]
    metric_health: dict[str, str]
    current_metrics: dict

class CopilotQueryRequest(BaseModel):
    alert_context: AlertContext
    question: str | None = None

class CopilotResponse(BaseModel):
    predicted_issue: str
    confidence: float
    root_cause_hypothesis: str
    affected_scope: list[str]
    estimated_time_to_impact_hours: float | None
    recommended_actions: list[str]
    runbook_reference: str | None
    urgency: str
    available: bool = True
```

---

## Air-Gap Compliance Steps

To demonstrate air-gap compliance to judges:

1. **During demo**: disable EC2 outbound internet access (`sudo iptables -I OUTPUT -d 0.0.0.0/0 ! -d 10.0.0.0/8 -j DROP`)
2. Show `/copilot/status` returns `"air_gapped": true`
3. Show Ollama running locally: `ollama list`
4. Show ChromaDB is local directory: `ls backend/copilot/chroma_db/`
5. Re-enable internet after demo

---

## Python Dependencies (add to requirements.txt)
```
ollama
langchain>=0.2.0
langchain-community>=0.2.0
chromadb>=0.5.0
```
