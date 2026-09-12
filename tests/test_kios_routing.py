"""Tests for Fable 5.1 and opus 5 routing via Kios API with full brand masking."""
from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock

import server
from server import (
    _resolve_model_target,
    _sanitize_brand,
    _sanitize_error_message,
    KIOS_FABLE_ACCT,
    KIOS_OPUS_ACCT,
)


@pytest.mark.anyio
async def test_models_endpoint_exposes_fable_and_opus(monkeypatch):
    """Verify list_models includes Fable 5.1 and opus 5 aliases."""
    monkeypatch.setattr(server, "ALLOW_UNAUTHENTICATED_API", True)
    mock_request = MagicMock()
    mock_request.headers = {}
    mock_request.client = MagicMock(host="127.0.0.1")
    res = await server.list_models(mock_request)
    data = json.loads(res.body.decode("utf-8"))
    model_ids = [m["id"] for m in data["data"]]
    assert "Fable 5.1" in model_ids
    assert "fable-5.1" in model_ids
    assert "opus 5" in model_ids
    assert "opus-5" in model_ids


def test_resolve_model_target_fable():
    """Verify Fable 5.1 resolves to KIOS_FABLE_ACCT with glm-5.3."""
    acct, resp_model, mode = _resolve_model_target("Fable 5.1")
    assert acct is not None
    assert acct.id == "kios_fable"
    assert acct.model == "glm-5.3"
    assert resp_model == "Fable 5.1"
    assert mode == "default"

    # Variant casing/dash
    acct2, resp_model2, mode2 = _resolve_model_target("fable-5.1")
    assert acct2 is not None
    assert acct2.id == "kios_fable"
    assert resp_model2 == "fable-5.1"


def test_resolve_model_target_opus():
    """Verify opus 5 resolves to KIOS_OPUS_ACCT with kilo-auto in expert mode."""
    acct, resp_model, mode = _resolve_model_target("opus 5")
    assert acct is not None
    assert acct.id == "kios_opus"
    assert acct.model == "kilo-auto"
    assert resp_model == "opus 5"
    assert mode == "expert"

    # Variant casing/dash
    acct2, resp_model2, mode2 = _resolve_model_target("OPUS-5")
    assert acct2 is not None
    assert acct2.id == "kios_opus"
    assert resp_model2 == "OPUS-5"
    assert mode2 == "expert"


def test_resolve_model_target_unmatched():
    """Verify standard/unmatched models return None for target account."""
    acct, resp_model, mode = _resolve_model_target("gpt-4o")
    assert acct is None
    assert resp_model == "gpt-4o"
    assert mode == "default"


def test_brand_sanitization():
    """Verify upstream brand tokens and URLs are scrubbed completely."""
    leak_sample = "Connected to router.kiosapi.com via kios backend running glm-5.3 and kilo-auto by zhipu/chatglm (智谱)."
    cleaned = _sanitize_brand(leak_sample)
    assert "kiosapi.com" not in cleaned
    assert "kios" not in cleaned.lower()
    assert "glm-5.3" not in cleaned
    assert "kilo-auto" not in cleaned
    assert "chatglm" not in cleaned.lower()
    assert "zhipu" not in cleaned.lower()
    assert "智谱" not in cleaned

    err_sample = "HTTP 502 Bad Gateway: failed to reach https://router.kiosapi.com/v1 for model glm-5.3 kilo-auto"
    cleaned_err = _sanitize_error_message(err_sample)
    assert "kiosapi.com" not in cleaned_err
    assert "kios" not in cleaned_err.lower()
    assert "glm-5.3" not in cleaned_err
    assert "kilo-auto" not in cleaned_err


def test_nonstream_openai_model_preservation(monkeypatch):
    """Verify non-streaming OpenAI completion preserves client-requested model name."""
    class DummyAcq:
        parent_message_id = None
        def __init__(self):
            self.acct = KIOS_FABLE_ACCT
            self.adapter = self
        def create_session(self):
            return "dummy_sess"
        def release(self):
            pass
        def prepare_prompt(self, p):
            return p
        def record_message_id(self, *args, **kwargs):
            pass
        def chat(self, *args, **kwargs):
            return "Fable response", None

    monkeypatch.setattr(server, "_acquire_safe", lambda *a, **kw: DummyAcq())

    resp = server._handle_nonstream(
        proxy_id="test-id",
        prompt="Hi",
        resp_model="Fable 5.1",
        target_account=KIOS_FABLE_ACCT,
    )
    assert resp["model"] == "Fable 5.1"
    assert resp["choices"][0]["message"]["content"] == "Fable response"


@pytest.mark.anyio
async def test_stream_openai_model_preservation(monkeypatch):
    """Verify streaming OpenAI completion preserves client-requested model name in all chunks."""
    class DummyAcq:
        parent_message_id = None
        def __init__(self):
            self.acct = KIOS_OPUS_ACCT
            self.adapter = self
        def create_session(self):
            return "dummy_sess"
        def release(self):
            pass
        def chat_stream(self, *args, **kwargs):
            yield "Opus "
            yield "thinking"

    monkeypatch.setattr(server, "_acquire_safe", lambda *a, **kw: DummyAcq())

    stream = await server._handle_stream(
        proxy_id="test-id",
        prompt="Hi",
        resp_model="opus 5",
        target_account=KIOS_OPUS_ACCT,
    )
    chunks = [c async for c in stream.body_iterator]
    parsed_chunks = []
    for line in chunks:
        for subline in line.split("\n"):
            if subline.startswith("data: ") and not subline.endswith("[DONE]"):
                parsed_chunks.append(json.loads(subline[6:]))

    assert len(parsed_chunks) > 0
    for chunk in parsed_chunks:
        assert chunk["model"] == "opus 5"


def test_anthropic_nonstream_model_preservation(monkeypatch):
    """Verify Anthropic non-streaming completion preserves client-requested model name."""
    class DummyAcq:
        parent_message_id = None
        def __init__(self):
            self.acct = KIOS_OPUS_ACCT
            self.adapter = self
        def create_session(self):
            return "dummy_sess"
        def release(self):
            pass
        def prepare_prompt(self, p):
            return p
        def record_message_id(self, *args, **kwargs):
            pass
        def chat(self, *args, **kwargs):
            return "Anthropic response", None

    monkeypatch.setattr(server, "_acquire_safe", lambda *a, **kw: DummyAcq())

    resp = server._anthropic_nonstream(
        msg_id="msg_test",
        prompt="User: hi",
        tool_names=[],
        resp_model="opus 5",
        target_account=KIOS_OPUS_ACCT,
    )
    assert resp["model"] == "opus 5"
