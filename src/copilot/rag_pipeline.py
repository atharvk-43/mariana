import json
import logging
import os

logger = logging.getLogger("copilot")

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_OK = True
except ImportError:
    CHROMA_OK = False

try:
    import numpy as np
    NP_OK = True
except ImportError:
    NP_OK = False


class RAGPipeline:
    def __init__(self, persist_dir: str = None):
        self.persist_dir = persist_dir
        self.collection = None
        self.client = None
        self.runbooks: dict[str, str] = {}
        self.incidents: list[dict] = []

        if persist_dir is None:
            persist_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
        self.persist_dir = persist_dir

        if CHROMA_OK and NP_OK:
            self._init_chromadb()
        else:
            logger.warning("chromadb not available, using fallback keyword matcher")

    def _init_chromadb(self):
        try:
            os.makedirs(self.persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            self.collection = self.client.get_or_create_collection(
                name="noc_runbooks",
                metadata={"hnsw:space": "cosine"},
            )
            self._index_runbooks()
            self._index_incidents()
        except Exception as e:
            logger.error("ChromaDB init failed: %s", e)
            self.client = None
            self.collection = None

    def _index_runbooks(self):
        runbooks_dir = os.path.join(os.path.dirname(__file__), "runbooks")
        if not os.path.isdir(runbooks_dir):
            return
        for fname in os.listdir(runbooks_dir):
            if fname.endswith(".md"):
                path = os.path.join(runbooks_dir, fname)
                with open(path, encoding="utf-8") as f:
                    content = f.read()
                self.runbooks[fname] = content

        if self.collection:
            existing = self.collection.get()
            if existing and existing.get("ids"):
                return
            for fname, content in self.runbooks.items():
                self.collection.add(
                    documents=[content],
                    metadatas=[{"source": fname, "type": "runbook"}],
                    ids=[f"runbook:{fname}"],
                )

    def _index_incidents(self):
        incidents_path = os.path.join(os.path.dirname(__file__), "past_incidents.json")
        if not os.path.exists(incidents_path):
            return
        with open(incidents_path, encoding="utf-8") as f:
            self.incidents = json.load(f)

        if self.collection:
            for inc in self.incidents:
                doc = json.dumps(inc, indent=2)
                self.collection.add(
                    documents=[doc],
                    metadatas=[{"source": inc.get("incident_id", ""), "type": "incident"}],
                    ids=[f"incident:{inc.get('incident_id', '')}"],
                )

    def retrieve(self, query: str, n_results: int = 3) -> str:
        if self.collection:
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                )
                contexts = []
                for i, doc in enumerate(results.get("documents", [[]])[0]):
                    meta = results.get("metadatas", [[]])[0][i]
                    source = meta.get("source", "unknown")
                    contexts.append(f"[Source: {source}]\n{doc}")
                return "\n\n---\n\n".join(contexts)
            except Exception as e:
                logger.error("ChromaDB query failed: %s", e)

        return self._fallback_retrieve(query, n_results)

    def _fallback_retrieve(self, query: str, n_results: int = 3) -> str:
        query_lower = query.lower()
        contexts = []

        keywords_runbook = {
            "bgp": "runbook_bgp_recovery.md",
            "congestion": "runbook_congestion.md",
            "mpls": "runbook_mpls_tunnel.md",
            "tunnel": "runbook_mpls_tunnel.md",
            "lsp": "runbook_mpls_tunnel.md",
            "policy": "runbook_policy_drift.md",
            "drift": "runbook_policy_drift.md",
            "flap": "runbook_link_flap.md",
            "link": "runbook_link_flap.md",
            "escalation": "runbook_noc_escalation.md",
        }

        matched = set()
        for keyword, fname in keywords_runbook.items():
            if keyword in query_lower:
                matched.add(fname)

        for fname in list(matched)[:n_results]:
            content = self.runbooks.get(fname, "")
            if content:
                contexts.append(f"[Source: {fname}]\n{content[:1000]}")

        for inc in self.incidents[:n_results]:
            if any(kw in query_lower for kw in [inc.get("fault_type", ""), inc.get("node", "").lower()]):
                contexts.append(f"[Source: {inc.get('incident_id', '')}]\n{json.dumps(inc, indent=2)}")

        return "\n\n---\n\n".join(contexts) if contexts else "No relevant context found."

    def copilot_query(self, node_id: str, alert: str, question: str) -> dict:
        context = self.retrieve(f"{alert} {question} {node_id}", n_results=3)

        from .ollama_client import OllamaClient
        ollama = OllamaClient()
        if ollama.is_available():
            result = ollama.structured_query(context, question)
        else:
            result = self._fallback_response(node_id, alert, context)

        return result

    def _fallback_response(self, node_id: str, alert: str, context: str) -> dict:
        alert_lower = alert.lower()
        if "bgp" in alert_lower or "flap" in alert_lower:
            return {
                "predicted_issue": f"BGP session instability on {node_id}",
                "confidence": 0.82,
                "root_cause": "Likely BGP hold timer expiry or peer reachability issue",
                "affected_scope": [node_id] + ["PE-2" if "PE-1" in node_id else "PE-1"],
                "time_to_impact_hours": 0.3,
                "recommended_action": "Verify BGP sessions, check TCP/179 reachability, soft reset peer",
                "runbook_reference": "runbook_bgp_recovery.md",
            }
        if "congestion" in alert_lower or "utilization" in alert_lower:
            return {
                "predicted_issue": f"Link congestion on {node_id}",
                "confidence": 0.88,
                "root_cause": "Bandwidth saturation during peak traffic hours",
                "affected_scope": [node_id],
                "time_to_impact_hours": 0.8,
                "recommended_action": "Apply QoS policy, shape non-critical traffic, consider bandwidth upgrade",
                "runbook_reference": "runbook_congestion.md",
            }
        if "mpls" in alert_lower or "tunnel" in alert_lower:
            return {
                "predicted_issue": f"MPLS tunnel degradation through {node_id}",
                "confidence": 0.85,
                "root_cause": "Likely LDP session flapping or label table inconsistency on P router",
                "affected_scope": [node_id] + ["lsp-b1-dc1", "lsp-b2-dc1", "lsp-b3-dc1"],
                "time_to_impact_hours": 0.5,
                "recommended_action": "Check LDP neighbors, verify MPLS forwarding table, check optical interfaces",
                "runbook_reference": "runbook_mpls_tunnel.md",
            }
        return {
            "predicted_issue": f"Anomaly detected on {node_id}",
            "confidence": 0.75,
            "root_cause": "Multiple metrics deviating from baseline",
            "affected_scope": [node_id],
            "time_to_impact_hours": 1.0,
            "recommended_action": "Check node health metrics and recent configuration changes",
            "runbook_reference": "runbook_noc_escalation.md",
        }
