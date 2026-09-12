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
from typing import Generator, Tuple, Optional, Any
import httpx
from logger import get_logger
from adapter import RateLimitError

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


def _parse_sse_response_to_chat(text: str) -> Tuple[str, Optional[str], list[dict]]:
    """Parse an SSE formatted text into (content, thinking, tool_calls)."""
    content_parts = []
    thinking_parts = []
    tool_calls_dict: dict[int, dict] = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            try:
                chunk = json.loads(line[6:].strip())
                choices = chunk.get("choices", [])
                if not choices or not isinstance(choices[0], dict):
                    continue
                delta = choices[0].get("delta", {}) or {}
                if delta.get("content"):
                    content_parts.append(delta["content"])
                r = delta.get("reasoning_content") or delta.get("reasoning")
                if r:
                    thinking_parts.append(r)
                if delta.get("tool_calls"):
                    for tc in delta["tool_calls"]:
                        idx = tc.get("index", 0)
                        if idx not in tool_calls_dict:
                            tool_calls_dict[idx] = {
                                "id": tc.get("id", f"call_{secrets.token_hex(8)}"),
                                "type": tc.get("type", "function"),
                                "function": {
                                    "name": tc.get("function", {}).get("name", ""),
                                    "arguments": tc.get("function", {}).get("arguments", ""),
                                }
                            }
                        else:
                            fn = tc.get("function", {})
                            if fn.get("name"):
                                tool_calls_dict[idx]["function"]["name"] = fn["name"]
                            if fn.get("arguments"):
                                tool_calls_dict[idx]["function"]["arguments"] += fn["arguments"]
            except Exception:
                pass
    return "".join(content_parts), ("".join(thinking_parts) or None), list(tool_calls_dict.values())


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

    def chat(self, session_id: str, prompt: str = "",
             model_type: str | None = None,
             thinking_enabled: bool = False,
             search_enabled: bool = False,
             parent_message_id: int | None = None,
             ready_out: dict | None = None,
             tools: list[dict] | None = None,
             tool_choice: Any = None,
             messages: list[dict] | None = None) -> Tuple[str, Optional[str]]:
        """Non-streaming chat request."""
        msgs = messages if messages is not None else _prompt_to_messages(prompt)
        target_url = f"{self.base_url}/chat/completions" if not self.base_url.endswith("/chat/completions") else self.base_url

        req_model = self.model
        if req_model == "orcarouter/free":
            req_model = "deepseek/deepseek-v4-flash-free"

        payload = {
            "model": req_model,
            "messages": msgs,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        resp = self._client.post(target_url, json=payload, headers=self._headers())
        if resp.status_code in (402, 429) and req_model == "orcarouter/free":
            log.warning("orcarouter_free_quota_fallback", extra={"fallback_model": "deepseek/deepseek-v4-flash-free"})
            payload["model"] = "deepseek/deepseek-v4-flash-free"
            resp = self._client.post(target_url, json=payload, headers=self._headers())

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            err_body = resp.text
            log.error("upstream_http_error", extra={"status": resp.status_code, "body": err_body})
            if resp.status_code in (429, 402):
                raise RateLimitError(f"Upstream rate limit ({resp.status_code}): {err_body}") from e
            raise RuntimeError(f"Upstream HTTP {resp.status_code}: {err_body}") from e
        raw_text = getattr(resp, "text", "")
        if raw_text and (raw_text.strip().startswith("data: ") or getattr(resp, "headers", {}).get("content-type", "").startswith("text/event-stream")):
            content, thinking, tcs = _parse_sse_response_to_chat(raw_text)
        else:
            data = resp.json()
            choice = data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            content = msg.get("content") or ""
            thinking = msg.get("reasoning_content") or msg.get("reasoning") or None
            tcs = msg.get("tool_calls") or []

        if ready_out is not None:
            ready_out["response_message_id"] = 1
            ready_out["session_id"] = session_id
            if tcs:
                ready_out["tool_calls"] = tcs

        return content, thinking

    def chat_stream(self, session_id: str, prompt: str = "",
                    model_type: str | None = None,
                    thinking_enabled: bool = False,
                    search_enabled: bool = False,
                    parent_message_id: int | None = None,
                    ready_out: dict | None = None,
                    tools: list[dict] | None = None,
                    tool_choice: Any = None,
                    messages: list[dict] | None = None) -> Generator:
        """Streaming chat request, yields content strings or control dicts."""
        msgs = messages if messages is not None else _prompt_to_messages(prompt)
        target_url = f"{self.base_url}/chat/completions" if not self.base_url.endswith("/chat/completions") else self.base_url

        req_model = self.model
        if req_model == "orcarouter/free":
            req_model = "deepseek/deepseek-v4-flash-free"

        payload = {
            "model": req_model,
            "messages": msgs,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        if ready_out is not None:
            ready_out["response_message_id"] = 1
            ready_out["session_id"] = session_id

        stream_ctx = self._client.stream("POST", target_url, json=payload, headers=self._headers())
        resp = stream_ctx.__enter__()
        try:
            if resp.status_code in (402, 429) and req_model == "orcarouter/free":
                stream_ctx.__exit__(None, None, None)
                log.warning("orcarouter_free_quota_fallback", extra={"fallback_model": "deepseek/deepseek-v4-flash-free"})
                payload["model"] = "deepseek/deepseek-v4-flash-free"
                stream_ctx = self._client.stream("POST", target_url, json=payload, headers=self._headers())
                resp = stream_ctx.__enter__()

            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                err_body = resp.read().decode(errors="replace")
                log.error("upstream_stream_http_error", extra={"status": resp.status_code, "body": err_body})
                if resp.status_code in (429, 402):
                    raise RateLimitError(f"Upstream rate limit ({resp.status_code}): {err_body}") from e
                raise RuntimeError(f"Upstream HTTP {resp.status_code}: {err_body}") from e
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                raw = line[6:].strip()
                if raw == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw)
                    choices = chunk.get("choices", [])
                    if not choices or not isinstance(choices[0], dict):
                        continue
                    delta = choices[0].get("delta", {}) or {}
                    tcs = delta.get("tool_calls")
                    if tcs:
                        yield {"__type": "tool_calls", "tool_calls": tcs}
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if reasoning:
                        yield {"__type": "thinking", "content": reasoning}
                    text = delta.get("content")
                    if text:
                        yield text
                except Exception as e:
                    log.debug("upstream_stream_chunk_parse_error", extra={"error": str(e)})
        finally:
            stream_ctx.__exit__(None, None, None)

        yield {"__type": "status", "status": "FINISHED"}

