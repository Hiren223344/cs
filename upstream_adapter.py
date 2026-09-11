"""
upstream_adapter.py — OpenAI-compatible upstream adapter for CS API.

Enables routing to external OpenAI-compatible endpoints (such as
http://43.153.6.116:8000/v1 or official DeepSeek API endpoints) while preserving:
- White-label brand sanitization
- DSML-based tool calling with StreamSieve
- Reasoning tokens (thinking / reasoning_content)
- Streaming and non-streaming responses
"""
import json
import os
import secrets
from typing import Generator, Tuple, Optional
import httpx
from logger import get_logger

log = get_logger("upstream_adapter")


def _prompt_to_messages(prompt: str) -> list[dict]:
    """Convert proxy's flattened prompt back into OpenAI-compatible messages."""
    lines = prompt.split("\n")
    messages = []
    current_role = None
    current_content = []

    role_prefixes = [
        ("System: ", "system"),
        ("User: ", "user"),
        ("Assistant: ", "assistant"),
        ("Tool result: ", "user"),
        ("Tool result (", "user"),
    ]

    for line in lines:
        matched = False
        for prefix, role in role_prefixes:
            if line.startswith(prefix):
                if current_role and current_content:
                    messages.append({
                        "role": current_role,
                        "content": "\n".join(current_content).strip()
                    })
                    current_content = []
                current_role = role
                if prefix.startswith("Tool result"):
                    current_content.append(line)
                else:
                    current_content.append(line[len(prefix):])
                matched = True
                break
        if not matched:
            if current_role is None:
                current_role = "user"
            current_content.append(line)

    if current_role and current_content:
        messages.append({
            "role": current_role,
            "content": "\n".join(current_content).strip()
        })

    if not messages:
        messages.append({"role": "user", "content": prompt})

    merged = []
    for m in messages:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n\n" + m["content"]
        else:
            merged.append(m)

    return merged


class OpenAIUpstreamAdapter:
    """Adapter for an OpenAI-compatible upstream API."""

    def __init__(self, base_url: str, api_key: str = "", model: str = "deepseek-chat", timeout: float = 120.0):
        url = base_url.strip().rstrip("/")
        if not url.endswith("/v1") and not url.endswith("/chat/completions"):
            url = f"{url}/v1"
        self.base_url = url
        self.api_key = api_key.strip() if api_key else ""
        self.model = model.strip() if model else "deepseek-chat"
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def create_session(self) -> str:
        """Stateless backend session id."""
        return f"upstream_{secrets.token_urlsafe(8)}"

    def _headers(self) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def chat(self, session_id: str, prompt: str,
             model_type: str | None = None,
             thinking_enabled: bool = False,
             search_enabled: bool = False,
             parent_message_id: int | None = None,
             ready_out: dict | None = None) -> Tuple[str, Optional[str]]:
        """Non-streaming chat request."""
        messages = _prompt_to_messages(prompt)
        target_url = f"{self.base_url}/chat/completions" if not self.base_url.endswith("/chat/completions") else self.base_url
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        resp = self._client.post(target_url, json=payload, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content") or ""
        thinking = msg.get("reasoning_content") or None

        if ready_out is not None:
            ready_out["response_message_id"] = 1
            ready_out["session_id"] = session_id

        return content, thinking

    def chat_stream(self, session_id: str, prompt: str,
                    model_type: str | None = None,
                    thinking_enabled: bool = False,
                    search_enabled: bool = False,
                    parent_message_id: int | None = None,
                    ready_out: dict | None = None) -> Generator:
        """Streaming chat request, yields content strings or control dicts."""
        messages = _prompt_to_messages(prompt)
        target_url = f"{self.base_url}/chat/completions" if not self.base_url.endswith("/chat/completions") else self.base_url
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }

        if ready_out is not None:
            ready_out["response_message_id"] = 1
            ready_out["session_id"] = session_id

        with self._client.stream("POST", target_url, json=payload, headers=self._headers()) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                raw = line[6:].strip()
                if raw == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw)
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    reasoning = delta.get("reasoning_content")
                    if reasoning:
                        yield {"__type": "thinking", "content": reasoning}
                    text = delta.get("content")
                    if text:
                        yield text
                except Exception as e:
                    log.debug("upstream_stream_chunk_parse_error", extra={"error": str(e)})

        yield {"__type": "status", "status": "FINISHED"}
