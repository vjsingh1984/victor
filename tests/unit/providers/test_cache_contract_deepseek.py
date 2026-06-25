# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""TDD RED-PHASE: DeepSeek prompt-caching contract tests.

These tests encode the desired caching contract for DeepSeek. They are
expected to FAIL until the implementation is completed (GREEN phase).

Contract under test:
1. DeepSeekProvider.supports_prompt_caching() == True
2. DeepSeek serializer places tools[] BEFORE messages[] in the payload dict
   so that DeepSeek's server-side auto-prefix cache keys on a stable
   tool-definition prefix.
"""

from typing import Any, Dict, List, Optional

import pytest

from victor.providers.base import Message, ToolDefinition
from victor.providers.deepseek_provider import DeepSeekProvider

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_tools(n: int = 3) -> List[ToolDefinition]:
    return [
        ToolDefinition(
            name=f"tool_{i}",
            description=f"Tool number {i}",
            parameters={
                "type": "object",
                "properties": {"arg": {"type": "string"}},
            },
        )
        for i in range(n)
    ]


def _make_provider(api_key: str = "sk-test-fake") -> DeepSeekProvider:
    return DeepSeekProvider(api_key=api_key, base_url="https://api.deepseek.com/v1")


# ── 1. supports_prompt_caching ─────────────────────────────────────────────────


class TestDeepSeekPromptCachingFlag:
    """Contract: DeepSeek must declare API-level prompt caching support."""

    def test_supports_prompt_caching_returns_true(self):
        """DeepSeek offers cached-token billing discounts (context caching)."""
        provider = _make_provider()
        assert provider.supports_prompt_caching() is True

    def test_supports_kv_prefix_caching_returns_true(self):
        """DeepSeek reuses KV cache for stable prompt prefixes."""
        provider = _make_provider()
        assert provider.supports_kv_prefix_caching() is True


# ── 2. tools[] before messages[] for auto-prefix ───────────────────────────────


class TestDeepSeekSerializerToolOrdering:
    """Contract: DeepSeek payload must place tools[] before messages[].

    DeepSeek's context caching (``deepseek api`` docs) auto-detects a stable
    prefix. By serializing tools before messages, the tools block becomes part
    of the cacheable prefix and is identical across turns, enabling cache hits.
    """

    def test_tools_appear_before_messages_in_payload_keys(self):
        """list(payload) ordering: 'tools' index < 'messages' index."""
        provider = _make_provider()
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello"),
        ]
        tools = _make_tools(3)

        payload = provider._build_request_payload(
            messages=messages,
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=4096,
            tools=tools,
            stream=False,
        )

        keys = list(payload.keys())
        assert "tools" in keys, f"tools missing from payload keys: {keys}"
        assert "messages" in keys, f"messages missing from payload keys: {keys}"
        tools_idx = keys.index("tools")
        messages_idx = keys.index("messages")
        assert tools_idx < messages_idx, (
            f"tools (idx={tools_idx}) must appear BEFORE messages "
            f"(idx={messages_idx}) in payload key order: {keys}"
        )

    def test_tool_choice_precedes_messages(self):
        """tool_choice must also appear before messages for prefix stability."""
        provider = _make_provider()
        messages = [Message(role="user", content="Hi")]
        tools = _make_tools(2)

        payload = provider._build_request_payload(
            messages=messages,
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=4096,
            tools=tools,
            stream=False,
        )

        keys = list(payload.keys())
        if "tool_choice" in keys and "messages" in keys:
            assert keys.index("tool_choice") < keys.index(
                "messages"
            ), f"tool_choice must precede messages: {keys}"
