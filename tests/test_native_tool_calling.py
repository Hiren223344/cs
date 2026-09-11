"""
tests/test_native_tool_calling.py — Regression and integration tests for native tool calling.
"""
import json
import pytest
from anthropic_format import (
    anthropic_to_openai_messages,
    anthropic_to_openai_tools,
    stream_response,
    build_nonstream_response,
    AnthropicMessage,
    AnthropicToolDef,
    ContentBlock,
)


def test_anthropic_to_openai_tools():
    anth_tools = [
        AnthropicToolDef(
            name="Bash",
            description="Run shell commands",
            input_schema={"type": "object", "properties": {"command": {"type": "string"}}},
        )
    ]
    res = anthropic_to_openai_tools(anth_tools)
    assert len(res) == 1
    assert res[0]["type"] == "function"
    assert res[0]["function"]["name"] == "Bash"
    assert res[0]["function"]["description"] == "Run shell commands"
    assert res[0]["function"]["parameters"]["properties"]["command"]["type"] == "string"


def test_anthropic_to_openai_messages_multi_turn():
    system = "You are a helpful coding assistant."
    messages = [
        AnthropicMessage(role="user", content="List my files"),
        AnthropicMessage(
            role="assistant",
            content=[
                ContentBlock(type="text", text="Checking files..."),
                ContentBlock(
                    type="tool_use",
                    id="toolu_12345",
                    name="Bash",
                    input={"command": "ls -la"},
                ),
            ],
        ),
        AnthropicMessage(
            role="user",
            content=[
                ContentBlock(
                    type="tool_result",
                    tool_use_id="toolu_12345",
                    content="file1.txt\nfile2.txt",
                )
            ],
        ),
    ]
    res = anthropic_to_openai_messages(messages, system=system)
    assert len(res) == 4
    assert res[0] == {"role": "system", "content": "You are a helpful coding assistant."}
    assert res[1] == {"role": "user", "content": "List my files"}
    assert res[2]["role"] == "assistant"
    assert res[2]["content"] == "Checking files..."
    assert len(res[2]["tool_calls"]) == 1
    assert res[2]["tool_calls"][0]["function"]["name"] == "Bash"
    assert res[2]["tool_calls"][0]["id"] == "call_12345"
    assert res[3] == {
        "role": "tool",
        "tool_call_id": "call_12345",
        "content": "file1.txt\nfile2.txt",
    }


def test_anthropic_stream_response_native_tool_calls():
    tokens = [
        {"__type": "thinking", "content": "The user wants to list files."},
        "Let me run ",
        "the command.",
        {
            "__type": "tool_calls",
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_abc123",
                    "type": "function",
                    "function": {"name": "Bash", "arguments": ""},
                }
            ],
        },
        {
            "__type": "tool_calls",
            "tool_calls": [
                {
                    "index": 0,
                    "function": {"arguments": '{"command": "ls -la"}'},
                }
            ],
        },
        {"__type": "status", "status": "FINISHED"},
    ]

    events = list(stream_response("msg_1", "gpt-6-astra", iter(tokens), ["Bash"]))
    raw_events = "".join(events)

    # 1. message_start present
    assert "event: message_start" in raw_events

    # 2. thinking block
    assert "content_block_start" in raw_events
    assert "thinking" in raw_events

    # 3. text block
    assert "Let me run " in raw_events
    assert "the command." in raw_events

    # 4. tool_use block
    assert "event: content_block_start" in raw_events
    assert '"type": "tool_use"' in raw_events
    assert '"id": "toolu_abc123"' in raw_events
    assert '"name": "Bash"' in raw_events

    # 5. input_json_delta
    assert "event: content_block_delta" in raw_events
    assert "input_json_delta" in raw_events
    assert "ls -la" in raw_events

    # 6. tool_use stop reason in message_delta
    assert '"stop_reason": "tool_use"' in raw_events

    # 7. message_stop is valid non-empty json object
    assert 'data: {"type": "message_stop"}' in raw_events


def test_build_nonstream_response_tool_calls():
    tool_calls = [
        {
            "id": "call_999",
            "type": "function",
            "function": {"name": "test_tool", "arguments": '{"a": 1}'},
        }
    ]
    resp = build_nonstream_response(
        "msg_test",
        "gpt-6-astra",
        content_text="I will call the tool",
        tool_calls=tool_calls,
        thinking_text="Thinking process",
    )
    assert resp["stop_reason"] == "tool_use"
    assert resp["usage"]["input_tokens"] > 0
    assert resp["usage"]["output_tokens"] > 0
    blocks = resp["content"]
    assert len(blocks) == 3
    assert blocks[0]["type"] == "thinking"
    assert blocks[1]["type"] == "text"
    assert blocks[2]["type"] == "tool_use"
    assert blocks[2]["id"] == "toolu_999"
    assert blocks[2]["name"] == "test_tool"
    assert blocks[2]["input"] == {"a": 1}
