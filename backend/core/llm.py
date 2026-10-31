import json
import os
from typing import Dict, Any

class LocalLLM:
    """
    Optional local Llama (Ollama) wrapper. Set SIGNALFORGE_USE_LLM=0 to disable.
    If Ollama is missing/unreachable, callers fall back to deterministic defaults.
    """
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        self.enabled = os.getenv("SIGNALFORGE_USE_LLM", "1") == "1"
        self._client = None

    def _get_client(self):
        if self._client is None:
            import ollama
            self._client = ollama.Client(timeout=float(os.getenv("SIGNALFORGE_LLM_TIMEOUT", "40")))
        return self._client

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        resp = self._get_client().chat(
            model=self.model_name,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            format="json",                 # constrained JSON output
            options={"temperature": 0},
        )
        text = resp["message"]["content"].strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        return json.loads(text)
