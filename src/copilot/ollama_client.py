import json
import logging
import urllib.request
import urllib.error

logger = logging.getLogger("copilot")

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "phi3:mini"


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE, model: str = DEFAULT_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._available = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=2) as resp:
                self._available = resp.status == 200
        except Exception:
            self._available = False
            logger.warning("Ollama not available at %s", self.base_url)
        return self._available

    def generate(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "temperature": temperature,
            "max_tokens": 512,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            return result.get("response", "")

    def structured_query(
        self,
        context: str,
        question: str,
    ) -> dict:
        system = (
            "You are an expert NOC engineer for an MPLS network. "
            "Analyze the alert context and return a structured JSON response with keys: "
            "predicted_issue, confidence, root_cause, affected_scope, time_to_impact_hours, recommended_action, runbook_reference. "
            "Be concise and specific. Use only information present in the context."
        )
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nRespond in valid JSON only."
        raw = self.generate(prompt, system=system)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return {
                    "predicted_issue": f"Error parsing LLM response for {question}",
                    "confidence": 0.0,
                    "root_cause": raw[:500],
                    "affected_scope": [],
                    "time_to_impact_hours": None,
                    "recommended_action": "Manual investigation required",
                    "runbook_reference": None,
                }
