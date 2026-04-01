from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, Sequence
from urllib import error, request


@dataclass
class HostedChatResult:
    model: str
    reply: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class HostedProviderClient:
    def __init__(self, base_url: str, api_key: str, timeout_seconds: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.timeout_seconds = timeout_seconds

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key)

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Dict[str, str]],
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> HostedChatResult:
        if not self.configured:
            raise ConnectionError("Hosted provider is not configured.")

        payload = {
            "model": model,
            "messages": list(messages),
            "stream": False,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_new_tokens),
        }
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise ConnectionError(body or f"Hosted provider returned HTTP {exc.code}.") from exc
        except error.URLError as exc:
            raise ConnectionError("Could not reach the hosted model provider.") from exc

        try:
            payload_json = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ConnectionError("Hosted provider returned invalid JSON.") from exc

        choices = payload_json.get("choices", [])
        choice = choices[0] if isinstance(choices, list) and choices else {}
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        usage = payload_json.get("usage", {}) if isinstance(payload_json, dict) else {}
        reply = str(message.get("content", "")).strip()
        return HostedChatResult(
            model=str(payload_json.get("model") or model),
            reply=reply,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        )
