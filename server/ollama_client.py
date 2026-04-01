from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, List, Sequence
from urllib import error, request


@dataclass
class OllamaChatResult:
    model: str
    reply: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class OllamaClient:
    def __init__(self, base_url: str, timeout_seconds: float = 120.0, keep_alive: str = "10m"):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.keep_alive = keep_alive

    def _request_json(self, path: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        method = "POST" if payload is not None else "GET"
        req = request.Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise ConnectionError(body or f"Ollama returned HTTP {exc.code}.") from exc
        except error.URLError as exc:
            raise ConnectionError(f"Could not reach Ollama at {self.base_url}.") from exc

        if not body.strip():
            return {}
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise ConnectionError("Ollama returned invalid JSON.") from exc

    def list_models(self) -> List[str]:
        payload = self._request_json("/api/tags")
        names = []
        for item in payload.get("models", []):
            name = str(item.get("name", "")).strip()
            if name:
                names.append(name)
        return names

    def resolve_model(self, preferred: str = "") -> str:
        names = self.list_models()
        if preferred and preferred in names:
            return preferred
        for candidate in ("qwen2.5:14b", "qwen2.5:7b", "llama3.1:8b", "mistral:7b", "qwen2.5-coder:14b"):
            if candidate in names:
                return candidate
        if names:
            return names[0]
        raise RuntimeError("No local Ollama models are available.")

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Dict[str, str]],
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        max_new_tokens: int,
    ) -> OllamaChatResult:
        options = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repetition_penalty,
            "num_predict": max_new_tokens,
        }
        payload = self._request_json(
            "/api/chat",
            {
                "model": model,
                "messages": list(messages),
                "stream": False,
                "keep_alive": self.keep_alive,
                "options": options,
            },
        )
        message = payload.get("message", {})
        reply = str(message.get("content", "")).strip()
        return OllamaChatResult(
            model=str(payload.get("model", model)),
            reply=reply,
            prompt_tokens=int(payload.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(payload.get("eval_count", 0) or 0),
        )

    def embed(self, *, model: str, inputs: Sequence[str]) -> List[List[float]]:
        cleaned_inputs = [str(item).strip() for item in inputs if str(item).strip()]
        if not cleaned_inputs:
            return []
        try:
            payload = self._request_json(
                "/api/embed",
                {
                    "model": model,
                    "input": cleaned_inputs,
                    "keep_alive": self.keep_alive,
                },
            )
        except ConnectionError:
            if len(cleaned_inputs) != 1:
                raise
            payload = self._request_json(
                "/api/embeddings",
                {
                    "model": model,
                    "prompt": cleaned_inputs[0],
                    "keep_alive": self.keep_alive,
                },
            )
            embedding = payload.get("embedding")
            return [embedding] if isinstance(embedding, list) else []
        embeddings = payload.get("embeddings", [])
        if isinstance(embeddings, list):
            return [vector for vector in embeddings if isinstance(vector, list)]
        return []
