"""Contract tests for Victor's direct typed Sandhi provider boundary."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import victor.providers.sandhi_transport as st
from victor.providers.base import (
    Message,
    ProviderConnectionError,
    ProviderRateLimitError,
    ToolDefinition,
)
from victor.providers.deepseek_provider import DeepSeekProvider
from victor.providers.google_provider import GoogleProvider
from victor.providers.llamacpp_provider import LlamaCppProvider
from victor.providers.lmstudio_provider import LMStudioProvider
from victor.providers.moonshot_provider import MoonshotProvider
from victor.providers.ollama_provider import OllamaProvider
from victor.providers.openai_provider import OpenAIProvider
from victor.providers.qwen_provider import QwenProvider
from victor.providers.vllm_provider import VLLMProvider


class FakeTypedProvider:
    def __init__(self, *, complete_error: BaseException | None = None) -> None:
        self.requests: list[dict] = []
        self.complete_error = complete_error

    async def complete_json(self, request_json: str) -> str:
        self.requests.append(json.loads(request_json))
        if self.complete_error:
            raise self.complete_error
        return json.dumps(
            {
                "schema_version": "1",
                "id": "r1",
                "model": "deepseek-chat",
                "output": {
                    "content": "hello",
                    "tool_calls": [{"id": "c1", "name": "lookup", "arguments": '{"q":1}'}],
                },
                "finish_reason": "tool_calls",
                "usage": {
                    "tokens_in": 6,
                    "tokens_out": 5,
                    "cache_creation_tokens": 0,
                    "cache_read_tokens": 4,
                    "completeness": "final",
                    "attempts": 1,
                },
                "extensions": {
                    "openai": {
                        "id": "r1",
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                    }
                },
            }
        )

    def stream_json(self, request_json: str):
        self.requests.append(json.loads(request_json))

        async def events():
            for event in (
                {"event": "response_start", "id": "r2", "model": "deepseek-chat"},
                {"event": "text_delta", "delta": "he"},
                {"event": "reasoning_delta", "delta": "think"},
                {"event": "tool_call_start", "index": 0, "id": "c1", "name": "lookup"},
                {"event": "tool_call_arguments_delta", "index": 0, "delta": '{"q":'},
                {"event": "tool_call_arguments_delta", "index": 0, "delta": "1}"},
                {"event": "tool_call_end", "index": 0},
                {"event": "finish", "reason": "tool_calls"},
                {
                    "event": "usage",
                    "usage": {
                        "tokens_in": 6,
                        "tokens_out": 5,
                        "cache_creation_tokens": 0,
                        "cache_read_tokens": 4,
                    },
                },
            ):
                yield json.dumps(event)

        return events()


class FakeRuntime:
    def __init__(self, handle: FakeTypedProvider) -> None:
        self.handle = handle
        self.calls: list[tuple] = []

    def provider(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.handle


def install_runtime(monkeypatch, handle: FakeTypedProvider | None = None) -> FakeRuntime:
    runtime = FakeRuntime(handle or FakeTypedProvider())
    monkeypatch.setattr(st, "_sg", SimpleNamespace(ProviderRuntime=lambda: runtime))
    return runtime


def make_provider() -> DeepSeekProvider:
    return DeepSeekProvider(api_key="k", base_url="https://api.deepseek.com/v1")


def test_resolver_always_uses_sandhi_for_admitted_provider(monkeypatch):
    install_runtime(monkeypatch)
    assert st.resolve_transport_class("deepseek", DeepSeekProvider, {}) is DeepSeekProvider


@pytest.mark.parametrize(
    ("name", "provider_cls", "expected"),
    (
        ("openai", OpenAIProvider, st.SandhiOpenAIProvider),
        ("google", GoogleProvider, st.SandhiGoogleProvider),
        ("ollama", OllamaProvider, st.SandhiOllamaProvider),
        ("qwen", QwenProvider, QwenProvider),
        ("lmstudio", LMStudioProvider, st.SandhiLMStudioProvider),
        ("vllm", VLLMProvider, st.SandhiVLLMProvider),
        ("llama.cpp", LlamaCppProvider, st.SandhiLlamaCppProvider),
    ),
)
def test_native_families_resolve_to_typed_sandhi_handles(
    name: str, provider_cls: type, expected: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_runtime(monkeypatch)
    assert st.resolve_transport_class(name, provider_cls, {}) is expected


@pytest.mark.asyncio
async def test_catalog_default_is_omitted_so_sandhi_owns_model_endpoint_routing(monkeypatch):
    runtime = FakeRuntime(FakeTypedProvider())
    monkeypatch.setattr(
        st,
        "_sg",
        SimpleNamespace(
            ProviderRuntime=lambda: runtime,
            provider_spec=lambda provider: {
                "slug": "moonshot",
                "base_url": "https://api.moonshot.cn/v1",
            },
        ),
    )
    resolved = st.resolve_transport_class("moonshot", MoonshotProvider, {})
    provider = resolved(api_key="k")
    await provider.chat([Message(role="user", content="hi")], model="kimi-k3")

    args, kwargs = runtime.calls[0]
    assert args[:3] == ("moonshot", "kimi-k3", "k")
    assert kwargs["base_url"] is None


@pytest.mark.asyncio
async def test_openai_oauth_explicitly_selects_responses_and_refreshes_before_handle(monkeypatch):
    runtime = install_runtime(monkeypatch)
    with patch("victor.providers.openai_provider.OAuthTokenManager") as manager_cls:
        manager = MagicMock()
        manager._load_cached.return_value = SimpleNamespace(
            access_token="cached-oauth", is_expired=False
        )
        manager.get_valid_token = AsyncMock(return_value="fresh-oauth")
        manager.get_chatgpt_account_id.return_value = "workspace_123"
        manager_cls.return_value = manager
        provider = st.SandhiOpenAIProvider(auth_mode="oauth")

    await provider.chat(
        [Message(role="developer", content="policy"), Message(role="user", content="hi")],
        model="o3",
        reasoning_effort="high",
    )

    args, kwargs = runtime.calls[0]
    assert args[:3] == ("openai", "o3", "fresh-oauth")
    assert kwargs["protocol"] == "chatgpt_responses"
    assert kwargs["base_url"] == "https://chatgpt.com/backend-api/codex"
    assert json.loads(kwargs["headers_json"])["originator"] == "victor"
    assert json.loads(kwargs["headers_json"])["ChatGPT-Account-ID"] == "workspace_123"
    request = runtime.handle.requests[0]
    assert "temperature" not in request
    assert request["extensions"] == {"openai_responses": {"reasoning": {"effort": "high"}}}


def test_resolver_fails_closed_when_binding_is_missing(monkeypatch):
    monkeypatch.setattr(st, "_sg", None)
    with pytest.raises(ProviderConnectionError):
        st.resolve_transport_class("deepseek", DeepSeekProvider, {})


def test_unknown_non_admitted_provider_is_unchanged(monkeypatch):
    install_runtime(monkeypatch)

    class Other(BaseException):
        pass

    assert st.resolve_transport_class("other", Other, {}) is Other


@pytest.mark.parametrize("name", sorted(st.VICTOR_NATIVE_ONLY_PROVIDER_ALIASES))
def test_explicit_native_only_boundary_is_preserved(name, monkeypatch):
    install_runtime(monkeypatch)

    class NativeOnly:
        pass

    assert st.resolve_transport_class(name, NativeOnly, {}) is NativeOnly


def test_unclassified_victor_owned_provider_fails_closed(monkeypatch):
    install_runtime(monkeypatch)
    provider_cls = type(
        "FutureProvider",
        (),
        {"__module__": "victor.providers.future_provider"},
    )

    with pytest.raises(ProviderConnectionError, match="not classified"):
        st.resolve_transport_class("future", provider_cls, {})


@pytest.mark.asyncio
async def test_complete_consumes_typed_response_and_reuses_handle(monkeypatch):
    runtime = install_runtime(monkeypatch)
    provider = make_provider()
    messages = [Message(role="developer", content="policy"), Message(role="user", content="hi")]
    tools = [ToolDefinition(name="lookup", description="Lookup", parameters={"type": "object"})]

    first = await provider.chat(messages, model="deepseek-chat", tools=tools)
    second = await provider.chat(messages, model="deepseek-chat", tools=tools)

    assert first.content == "hello"
    assert first.tool_calls == [{"id": "c1", "name": "lookup", "arguments": {"q": 1}}]
    assert first.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cache_read_input_tokens": 4,
    }
    assert len(runtime.calls) == 1, "the typed provider handle must be persistent"
    assert len(runtime.handle.requests) == 2
    assert runtime.handle.requests[0]["messages"][0]["role"] == "developer"
    assert runtime.handle.requests[0]["tools"][0]["name"] == "lookup"
    assert second.raw_response == first.raw_response


@pytest.mark.asyncio
async def test_stream_consumes_typed_events_without_sse_round_trip(monkeypatch):
    install_runtime(monkeypatch)
    provider = make_provider()

    chunks = [
        chunk
        async for chunk in provider.stream(
            [Message(role="user", content="hi")], model="deepseek-chat"
        )
    ]

    assert chunks[0].content == "he"
    assert chunks[1].metadata == {"reasoning_content": "think"}
    assert chunks[-1].is_final
    assert chunks[-1].stop_reason == "tool_calls"
    assert chunks[-1].tool_calls == [{"id": "c1", "name": "lookup", "arguments": {"q": 1}}]
    assert chunks[-1].usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cache_read_input_tokens": 4,
    }


@pytest.mark.asyncio
async def test_binding_failure_is_mapped_and_never_replayed(monkeypatch):
    error = RuntimeError(
        json.dumps(
            {
                "code": "rate_limited",
                "message": "slow down",
                "retryable": True,
                "http_status": 429,
            }
        )
    )
    install_runtime(monkeypatch, FakeTypedProvider(complete_error=error))
    provider = make_provider()

    with pytest.raises(ProviderRateLimitError):
        await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")


def test_pilot_and_raw_bridge_symbols_are_gone():
    for obsolete in (
        "SandhiTransportUnavailable",
        "set_sandhi_transport_providers",
        "configure_from_settings",
        "sse_lines",
        "_binding_complete",
    ):
        assert not hasattr(st, obsolete)


def test_non_routine_usage_state_survives_victor_compatibility_mapping():
    assert st._usage_diagnostics(
        {
            "attempts": 3,
            "completeness": "final",
            "outcome": "success",
            "upstream_request_id": "up_1",
        }
    ) == {
        "attempts": 3,
        "completeness": "final",
        "outcome": "success",
        "upstream_request_id": "up_1",
    }
    assert (
        st._usage_diagnostics({"attempts": 1, "completeness": "final", "outcome": "success"})
        is None
    )
