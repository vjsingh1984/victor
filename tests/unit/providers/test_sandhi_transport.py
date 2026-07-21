"""Unit tests for the flag-gated in-process sandhi transport pilot (FEP-0020 Phase 4b).

Default-off is a hard invariant: with the setting empty, provider creation is byte-identical
to native — pinned by class-identity guard tests. All tests here run without the binding by
monkeypatching the module seam ``_sg``.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

import victor.providers.sandhi_transport as st
from victor.providers.deepseek_provider import DeepSeekProvider
from victor.providers.registry import ProviderRegistry


@pytest.fixture(autouse=True)
def _reset_transport_state(monkeypatch):
    """Isolate the enabled-set override, warn-once latch, and env between tests."""
    monkeypatch.delenv("VICTOR_SANDHI_TRANSPORT_PROVIDERS", raising=False)
    st.set_sandhi_transport_providers(None)
    st._warned_binding_missing = False
    yield
    st.set_sandhi_transport_providers(None)
    st._warned_binding_missing = False


class TestSetting:
    def test_sandhi_transport_setting_defaults_empty(self):
        from victor.config.settings import Settings

        settings = Settings()
        assert settings.sandhi_transport_providers == []

    def test_setting_accepts_comma_separated_env(self, monkeypatch):
        from victor.config.settings import Settings

        monkeypatch.setenv("VICTOR_SANDHI_TRANSPORT_PROVIDERS", "deepseek, xai")
        settings = Settings()
        assert settings.sandhi_transport_providers == ["deepseek", "xai"]


class TestResolver:
    def test_resolver_returns_native_when_setting_off(self):
        resolved = st.resolve_transport_class("deepseek", DeepSeekProvider, {})
        assert resolved is DeepSeekProvider

    def test_resolver_returns_sandhi_variant_when_enabled(self, monkeypatch):
        monkeypatch.setattr(st, "_sg", SimpleNamespace())  # binding "present"
        st.set_sandhi_transport_providers(["deepseek"])
        resolved = st.resolve_transport_class("deepseek", DeepSeekProvider, {})
        assert resolved is st.SandhiDeepSeekProvider
        assert issubclass(resolved, DeepSeekProvider)

    def test_resolver_env_enables(self, monkeypatch):
        monkeypatch.setattr(st, "_sg", SimpleNamespace())
        monkeypatch.setenv("VICTOR_SANDHI_TRANSPORT_PROVIDERS", "deepseek,xai")
        resolved = st.resolve_transport_class("deepseek", DeepSeekProvider, {})
        assert resolved is st.SandhiDeepSeekProvider

    def test_resolver_returns_native_and_warns_once_when_binding_missing(self, monkeypatch, caplog):
        monkeypatch.setattr(st, "_sg", None)
        st.set_sandhi_transport_providers(["deepseek"])
        with caplog.at_level(logging.WARNING):
            r1 = st.resolve_transport_class("deepseek", DeepSeekProvider, {})
            r2 = st.resolve_transport_class("deepseek", DeepSeekProvider, {})
        assert r1 is DeepSeekProvider and r2 is DeepSeekProvider
        warnings = [r for r in caplog.records if "sandhi" in r.getMessage().lower()]
        assert len(warnings) == 1, "must warn exactly once, not per call"

    def test_resolver_ignores_unknown_provider(self, monkeypatch):
        """A provider with no sandhi variant stays native even when named in the setting."""
        monkeypatch.setattr(st, "_sg", SimpleNamespace())
        st.set_sandhi_transport_providers(["ollama"])

        class FakeOllama:  # not in _SANDHI_VARIANTS
            pass

        assert st.resolve_transport_class("ollama", FakeOllama, {}) is FakeOllama

    def test_resolver_never_raises(self, monkeypatch):
        monkeypatch.setattr(st, "_sg", SimpleNamespace())

        def boom():
            raise RuntimeError("enabled-set exploded")

        monkeypatch.setattr(st, "_enabled_providers", boom)
        assert st.resolve_transport_class("deepseek", DeepSeekProvider, {}) is DeepSeekProvider


class TestRegistryGuard:
    def test_registry_create_returns_native_class_when_transport_off(self):
        provider = ProviderRegistry.create("deepseek", api_key="k", base_url="http://x/v1")
        assert type(provider) is DeepSeekProvider

    def test_registry_create_returns_sandhi_variant_when_enabled(self, monkeypatch):
        monkeypatch.setattr(st, "_sg", SimpleNamespace())
        st.set_sandhi_transport_providers(["deepseek"])
        provider = ProviderRegistry.create("deepseek", api_key="k", base_url="http://x/v1")
        assert type(provider) is st.SandhiDeepSeekProvider
        assert isinstance(provider, DeepSeekProvider)

    def test_alias_grok_enables_xai_variant(self, monkeypatch):
        from victor.providers.xai_provider import XAIProvider

        monkeypatch.setattr(st, "_sg", SimpleNamespace())
        st.set_sandhi_transport_providers(["grok"])
        resolved = st.resolve_transport_class("grok", XAIProvider, {})
        assert resolved is st.SandhiXAIProvider


# ---------------------------------------------------------------------------
# T2/T3: mixin wire paths against a stubbed binding (no sandhi-gateway needed)
# ---------------------------------------------------------------------------

import asyncio
import json as _json

import httpx
import respx

from victor.providers.base import (
    Message,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

OK_BODY = {
    "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def make_sandhi_provider() -> st.SandhiDeepSeekProvider:
    return st.SandhiDeepSeekProvider(api_key="k", base_url="https://api.deepseek.com/v1")


def stub_complete(result=None, exc: BaseException | None = None, calls: list | None = None):
    async def _complete(*args, **kwargs):
        if calls is not None:
            calls.append((args, kwargs))
        if exc is not None:
            raise exc
        return result

    return _complete


class FakeByteIter:
    """Mimics the binding's ByteStreamIter: yields dicts, optionally raising mid-way."""

    def __init__(self, chunks: list, exc: BaseException | None = None):
        self._items = list(chunks)
        self._exc = exc

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._items:
            return self._items.pop(0)
        if self._exc is not None:
            exc, self._exc = self._exc, None
            raise exc
        raise StopAsyncIteration


class TestMixinChatPath:
    async def test_chat_via_sandhi_parses_identically_to_native(self, monkeypatch):
        calls: list = []
        payload_result = {
            "status": 200,
            "body": _json.dumps(OK_BODY),
            "usage": {"tokens_in": 10, "tokens_out": 5},
        }
        monkeypatch.setattr(
            st, "_sg", SimpleNamespace(complete=stub_complete(payload_result, calls=calls))
        )
        provider = make_sandhi_provider()
        response = await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")
        assert response.content == "hello"
        assert response.usage is not None and response.usage["total_tokens"] == 15

        args, kwargs = calls[0]
        slug, model, base_url, api_key, body_json, session = args
        assert slug == "deepseek"
        assert model == "deepseek-chat"
        assert base_url == "https://api.deepseek.com/v1"
        assert api_key == "k"
        assert _json.loads(body_json)["model"] == "deepseek-chat"
        assert session is None
        assert kwargs["max_retries"] == 0, "victor's resilience is the sole retry owner"
        assert kwargs["timeout_secs"] > 0

    @pytest.mark.parametrize(
        "raised,expected",
        [
            (RuntimeError("sandhi transport: rate limited (429)"), ProviderRateLimitError),
            (RuntimeError("sandhi transport: auth failed (401/403)"), ProviderAuthError),
            (RuntimeError("sandhi transport: upstream status 502"), ProviderError),
            (
                RuntimeError("sandhi transport: transport error: connection refused"),
                ProviderConnectionError,
            ),
            (RuntimeError("sandhi transport: timed out after 30s"), ProviderTimeoutError),
            (TimeoutError("sandhi transport: timed out after 30s"), ProviderTimeoutError),
            (
                RuntimeError("sandhi transport: circuit open (upstream failing)"),
                ProviderConnectionError,
            ),
        ],
    )
    async def test_error_mapping(self, monkeypatch, raised, expected):
        monkeypatch.setattr(st, "_sg", SimpleNamespace(complete=stub_complete(exc=raised)))
        provider = make_sandhi_provider()
        with pytest.raises(expected):
            await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")
        assert provider._sandhi_demoted is False, "upstream-semantic errors must not demote"

    @respx.mock
    async def test_binding_internal_error_demotes_once_and_falls_back_native(self, monkeypatch):
        respx.post("https://api.deepseek.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=OK_BODY)
        )
        calls: list = []
        monkeypatch.setattr(
            st,
            "_sg",
            SimpleNamespace(complete=stub_complete(exc=TypeError("bad FFI shape"), calls=calls)),
        )
        provider = make_sandhi_provider()
        r1 = await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")
        assert r1.content == "hello", "the demoting call itself must transparently succeed"
        assert provider._sandhi_demoted is True
        r2 = await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")
        assert r2.content == "hello"
        assert len(calls) == 1, "after demotion the binding must never be touched again"

    async def test_cancelled_error_propagates_untouched(self, monkeypatch):
        monkeypatch.setattr(
            st, "_sg", SimpleNamespace(complete=stub_complete(exc=asyncio.CancelledError()))
        )
        provider = make_sandhi_provider()
        with pytest.raises(asyncio.CancelledError):
            await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")
        assert provider._sandhi_demoted is False


class TestSseLines:
    async def test_sse_lines_chunk_boundary_torture(self):
        # é (c3 a9) split across chunks; CRLF; many events per chunk; unterminated tail
        raw = 'data: {"a":"hé"}\r\ndata: two\n\ndata: three\ndata: tail'.encode()
        pieces = [raw[i : i + 3] for i in range(0, len(raw), 3)]  # pathological 3-byte chunks

        async def items():
            for piece in pieces:
                yield {"data": piece, "usage": None}

        lines = [line async for line in st.sse_lines(items())]
        assert lines == ['data: {"a":"hé"}', "data: two", "", "data: three", "data: tail"]


SSE_FIXTURE = (
    'data: {"choices":[{"delta":{"content":"he"}}]}\n\n'
    'data: {"choices":[{"delta":{"content":"llo"}}]}\n\n'
    "data: [DONE]\n\n"
).encode()


class TestMixinStreamPath:
    async def test_stream_via_sandhi_yields_expected_chunks(self, monkeypatch):
        # Split the SSE bytes at pathological boundaries (7-byte chunks)
        chunks = [
            {"data": SSE_FIXTURE[i : i + 7], "usage": None} for i in range(0, len(SSE_FIXTURE), 7)
        ]
        monkeypatch.setattr(st, "_sg", SimpleNamespace(stream=lambda *a, **k: FakeByteIter(chunks)))
        provider = make_sandhi_provider()
        out = [
            c
            async for c in provider.stream(
                [Message(role="user", content="hi")], model="deepseek-chat"
            )
        ]
        assert "".join(c.content for c in out) == "hello"
        assert out[-1].is_final

    @respx.mock
    async def test_stream_failure_before_first_chunk_demotes_and_falls_back(self, monkeypatch):
        sse_native = 'data: {"choices":[{"delta":{"content":"native"}}]}\n\ndata: [DONE]\n\n'
        respx.post("https://api.deepseek.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, content=sse_native, headers={"content-type": "text/event-stream"}
            )
        )
        failing_iter = FakeByteIter([], exc=TypeError("FFI channel broke"))
        monkeypatch.setattr(st, "_sg", SimpleNamespace(stream=lambda *a, **k: failing_iter))
        provider = make_sandhi_provider()
        out = [
            c
            async for c in provider.stream(
                [Message(role="user", content="hi")], model="deepseek-chat"
            )
        ]
        assert "".join(c.content for c in out) == "native"
        assert provider._sandhi_demoted is True

    async def test_stream_failure_after_first_chunk_raises_no_replay(self, monkeypatch):
        # Uses the XAI variant: DeepSeek's native stream() buffers the whole response for
        # DSML tool-call recovery, so no provider-level chunk ever precedes a mid-stream
        # failure there (native behavior, unchanged by the pilot). XAI streams unbuffered.
        first = {"data": b'data: {"choices":[{"delta":{"content":"he"}}]}\n\n', "usage": None}
        failing_iter = FakeByteIter(
            [first], exc=RuntimeError("sandhi stream: transport error: reset")
        )
        monkeypatch.setattr(st, "_sg", SimpleNamespace(stream=lambda *a, **k: failing_iter))
        provider = st.SandhiXAIProvider(api_key="k", base_url="https://api.x.ai/v1")
        received: list = []
        with pytest.raises((ProviderConnectionError, ProviderError)):
            async for c in provider.stream(
                [Message(role="user", content="hi")], model="deepseek-chat"
            ):
                received.append(c)
        assert "".join(c.content for c in received) == "he", "content before failure is kept"
        assert provider._sandhi_demoted is True, "future calls go native"
