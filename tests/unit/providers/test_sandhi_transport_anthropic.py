"""Unit tests for sandhi transport wave 2a: Anthropic NON-STREAMING chat (FEP-0020 Phase 4 T6).

All tests run without the real binding by monkeypatching the module seam ``_sg`` (both in
``sandhi_transport`` and, for deterministic cache-split assertions, ``usage_parsing``).

Scope contract pinned here:
- only the non-streaming wire seam (``_create_message_raw``) is overridden;
- streaming stays 100% native (wave 2b is a separate go/no-go);
- OAuth-mode providers are excluded at the resolver (binding sends ``x-api-key`` only).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import victor.providers.sandhi_transport as st
import victor.providers.usage_parsing as up
from victor.providers.anthropic_provider import AnthropicProvider
from victor.providers.base import (
    Message,
    ProviderAuthError,
    ProviderRateLimitError,
    ToolDefinition,
)

CORPUS = json.loads(
    (
        Path(__file__).parent
        / "fixtures"
        / "sandhi_usage"
        / "anthropic"
        / "complete_cache_split.json"
    ).read_text()
)


@pytest.fixture(autouse=True)
def _reset_transport_state(monkeypatch):
    monkeypatch.delenv("VICTOR_SANDHI_TRANSPORT_PROVIDERS", raising=False)
    st.set_sandhi_transport_providers(None)
    st._warned_binding_missing = False
    yield
    st.set_sandhi_transport_providers(None)
    st._warned_binding_missing = False


@pytest.fixture(autouse=True)
def _deterministic_usage_parser(monkeypatch):
    """Stub the usage-parsing binding so cache-split assertions hold without victor[sandhi]."""

    def fake_parse_usage(slug: str, payload_json: str):
        block = json.loads(payload_json)["usage"]
        return {
            "tokens_in": block.get("input_tokens", 0),
            "tokens_out": block.get("output_tokens", 0),
            "cache_creation_tokens": block.get("cache_creation_input_tokens", 0) or 0,
            "cache_read_tokens": block.get("cache_read_input_tokens", 0) or 0,
        }

    monkeypatch.setattr(up, "_sg", SimpleNamespace(parse_usage=fake_parse_usage))


def stub_complete(result=None, exc: BaseException | None = None, calls: list | None = None):
    async def _complete(*args, **kwargs):
        if calls is not None:
            calls.append((args, kwargs))
        if exc is not None:
            raise exc
        return result

    return _complete


def make_native(**overrides) -> AnthropicProvider:
    kwargs = {"api_key": "test-key", "max_retries": 0}
    kwargs.update(overrides)
    return AnthropicProvider(**kwargs)


def make_sandhi(**overrides) -> "st.SandhiAnthropicProvider":
    kwargs = {"api_key": "test-key", "max_retries": 0}
    kwargs.update(overrides)
    return st.SandhiAnthropicProvider(**kwargs)


def corpus_message():
    from anthropic.types import Message as AnthropicMessage

    return AnthropicMessage.model_validate(CORPUS)


# ---------------------------------------------------------------------------
# Golden pin for the _build_request_params extraction (behavior-preserving)
# ---------------------------------------------------------------------------

GOLDEN_MESSAGES = [
    Message(role="system", content="You are terse."),
    Message(role="user", content="What is 2+2?"),
    Message(role="assistant", content="4"),
    Message(role="user", content="And 3+3?"),
]

GOLDEN_TOOLS = [
    ToolDefinition(
        name="read_file",
        description="Read a file",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    ),
    ToolDefinition(
        name="grep",
        description="Search for a pattern",
        parameters={
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
        schema_level="stub",
    ),
]

# The exact request params the CURRENT inline chat() block produces for the scenario
# above (system w/ cache_control, multi-message, tools w/ cache boundary before the
# first stub tool). Pinned pre-refactor by capturing messages.create kwargs.
GOLDEN_PARAMS = {
    "model": "claude-sonnet-4-6",
    "messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
    ],
    "max_tokens": 512,
    "temperature": 0.2,
    "system": [
        {
            "type": "text",
            "text": "You are terse.",
            "cache_control": {"type": "ephemeral"},
        }
    ],
    "tools": [
        {
            "name": "read_file",
            "description": "Read a file",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            "cache_control": {"type": "ephemeral"},
        },
        {
            "name": "grep",
            "description": "Search for a pattern",
            "input_schema": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    ],
}


class _EmptyStream:
    """Async CM mimicking client.messages.stream(...) yielding no events."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class TestBuildRequestParamsGolden:
    async def test_chat_call_site_matches_golden(self):
        """Pins the CURRENT inline behavior: chat() must send exactly GOLDEN_PARAMS."""
        provider = make_native()
        create = AsyncMock(return_value=corpus_message())
        provider.client = SimpleNamespace(messages=SimpleNamespace(create=create))

        await provider.chat(
            GOLDEN_MESSAGES,
            model="claude-sonnet-4-6",
            temperature=0.2,
            max_tokens=512,
            tools=GOLDEN_TOOLS,
        )

        assert create.call_args.kwargs == GOLDEN_PARAMS

    async def test_stream_call_site_matches_golden(self):
        provider = make_native()
        captured: dict = {}

        def fake_stream(**kwargs):
            captured.update(kwargs)
            return _EmptyStream()

        provider.client = SimpleNamespace(messages=SimpleNamespace(stream=fake_stream))
        async for _ in provider.stream(
            GOLDEN_MESSAGES,
            model="claude-sonnet-4-6",
            temperature=0.2,
            max_tokens=512,
            tools=GOLDEN_TOOLS,
        ):
            pass

        assert captured == GOLDEN_PARAMS

    def test_build_request_params_golden(self):
        provider = make_native()
        params = provider._build_request_params(
            GOLDEN_MESSAGES,
            model="claude-sonnet-4-6",
            temperature=0.2,
            max_tokens=512,
            tools=GOLDEN_TOOLS,
        )
        assert params == GOLDEN_PARAMS


# ---------------------------------------------------------------------------
# Resolver gating
# ---------------------------------------------------------------------------


class TestResolverAnthropic:
    def test_resolver_returns_sandhi_variant_when_enabled(self, monkeypatch):
        monkeypatch.setattr(st, "_sg", SimpleNamespace())
        st.set_sandhi_transport_providers(["anthropic"])
        resolved = st.resolve_transport_class("anthropic", AnthropicProvider, {})
        assert resolved is st.SandhiAnthropicProvider
        assert issubclass(resolved, AnthropicProvider)

    def test_oauth_stays_native_even_when_enabled(self, monkeypatch):
        """The binding sends x-api-key only — OAuth-mode providers must stay native."""
        monkeypatch.setattr(st, "_sg", SimpleNamespace())
        st.set_sandhi_transport_providers(["anthropic"])
        resolved = st.resolve_transport_class(
            "anthropic", AnthropicProvider, {"auth_mode": "oauth"}
        )
        assert resolved is AnthropicProvider

    def test_resolver_off_by_default(self):
        assert st.resolve_transport_class("anthropic", AnthropicProvider, {}) is AnthropicProvider


# ---------------------------------------------------------------------------
# Non-streaming wire seam through the (stubbed) binding
# ---------------------------------------------------------------------------


class TestSandhiAnthropicChat:
    async def test_chat_via_sandhi_parses_identically(self, monkeypatch):
        calls: list = []
        result = {"status": 200, "body": json.dumps(CORPUS), "usage": {}}
        monkeypatch.setattr(st, "_sg", SimpleNamespace(complete=stub_complete(result, calls=calls)))
        provider = make_sandhi()
        native_create = AsyncMock()
        provider.client = SimpleNamespace(messages=SimpleNamespace(create=native_create))

        response = await provider.chat(
            [Message(role="user", content="hi")], model="claude-sonnet-4-6"
        )

        assert response.content == "Hello, world"
        assert response.stop_reason == "end_turn"
        assert response.usage == {
            "prompt_tokens": 1024,
            "completion_tokens": 256,
            "total_tokens": 1280,
            "cache_creation_input_tokens": 2048,
            "cache_read_input_tokens": 4096,
        }
        native_create.assert_not_awaited()

        args, kwargs = calls[0]
        slug, model, base_url, api_key, body_json, session = args
        assert slug == "anthropic"
        assert model == "claude-sonnet-4-6"
        assert base_url == "https://api.anthropic.com"
        assert api_key == "test-key"
        assert session is None
        assert kwargs["max_retries"] == 0, "victor's resilience is the sole retry owner"
        assert kwargs["timeout_secs"] > 0
        # The forwarded body is exactly the params _build_request_params produced.
        expected = provider._build_request_params(
            [Message(role="user", content="hi")],
            model="claude-sonnet-4-6",
            temperature=0.7,
            max_tokens=4096,
            tools=None,
        )
        assert json.loads(body_json) == expected

    async def test_explicit_base_url_forwarded(self, monkeypatch):
        calls: list = []
        result = {"status": 200, "body": json.dumps(CORPUS), "usage": {}}
        monkeypatch.setattr(st, "_sg", SimpleNamespace(complete=stub_complete(result, calls=calls)))
        provider = make_sandhi(base_url="http://127.0.0.1:9999")
        await provider.chat([Message(role="user", content="hi")], model="claude-sonnet-4-6")
        assert calls[0][0][2] == "http://127.0.0.1:9999"

    @pytest.mark.parametrize(
        "raised,expected",
        [
            (RuntimeError("sandhi transport: rate limited (429)"), ProviderRateLimitError),
            (RuntimeError("sandhi transport: auth failed (401/403)"), ProviderAuthError),
        ],
    )
    async def test_error_mapping(self, monkeypatch, raised, expected):
        monkeypatch.setattr(st, "_sg", SimpleNamespace(complete=stub_complete(exc=raised)))
        provider = make_sandhi()
        with pytest.raises(expected):
            await provider.chat([Message(role="user", content="hi")], model="claude-sonnet-4-6")
        assert provider._sandhi_demoted is False, "upstream-semantic errors must not demote"

    async def test_binding_internal_error_demotes_once_and_falls_back_native(self, monkeypatch):
        calls: list = []
        monkeypatch.setattr(
            st,
            "_sg",
            SimpleNamespace(complete=stub_complete(exc=TypeError("bad FFI shape"), calls=calls)),
        )
        provider = make_sandhi()
        provider.client = SimpleNamespace(
            messages=SimpleNamespace(create=AsyncMock(return_value=corpus_message()))
        )

        r1 = await provider.chat([Message(role="user", content="hi")], model="claude-sonnet-4-6")
        assert r1.content == "Hello, world", "the demoting call itself must transparently succeed"
        assert provider._sandhi_demoted is True

        r2 = await provider.chat([Message(role="user", content="hi")], model="claude-sonnet-4-6")
        assert r2.content == "Hello, world"
        assert len(calls) == 1, "after demotion the binding must never be touched again"

    async def test_unparseable_body_demotes_and_falls_back_native(self, monkeypatch):
        result = {"status": 200, "body": "not-json", "usage": {}}
        monkeypatch.setattr(st, "_sg", SimpleNamespace(complete=stub_complete(result)))
        provider = make_sandhi()
        provider.client = SimpleNamespace(
            messages=SimpleNamespace(create=AsyncMock(return_value=corpus_message()))
        )
        response = await provider.chat(
            [Message(role="user", content="hi")], model="claude-sonnet-4-6"
        )
        assert response.content == "Hello, world"
        assert provider._sandhi_demoted is True


class TestStreamingStaysNative:
    def test_stream_not_overridden(self):
        """Wave 2a is chat-only: the streaming entrypoint must be the native one."""
        assert st.SandhiAnthropicProvider.stream is AnthropicProvider.stream
        assert "stream" not in st.SandhiAnthropicProvider.__dict__
