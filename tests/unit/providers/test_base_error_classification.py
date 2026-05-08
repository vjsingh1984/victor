import httpx
import pytest

from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    StreamChunk,
)
from victor.providers.resilience import RetryExhaustedError


class _DummyProvider(BaseProvider):
    def __init__(self) -> None:
        self.timeout = 12

    @property
    def name(self) -> str:
        return "dummy"

    async def chat(
        self,
        messages,
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools=None,
        **kwargs,
    ) -> CompletionResponse:
        return CompletionResponse(content="ok")

    async def stream(
        self,
        messages,
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools=None,
        **kwargs,
    ):
        yield StreamChunk(content="ok")

    async def close(self) -> None:
        return None


def test_classify_error_prefers_timeout_over_connection_classification() -> None:
    provider = _DummyProvider()

    result = provider.classify_error(httpx.ReadTimeout("Read timed out"))

    assert isinstance(result, ProviderTimeoutError)
    assert result.provider == "dummy"
    assert "timed out" in str(result).lower()


def test_classify_error_unwraps_retry_exhausted_timeout() -> None:
    provider = _DummyProvider()

    wrapped = RetryExhaustedError(3, httpx.ReadTimeout("Read timed out"))
    result = provider.classify_error(wrapped)

    assert isinstance(result, ProviderTimeoutError)
    assert result.raw_error is not None


def test_classify_error_unwraps_retry_exhausted_rate_limit() -> None:
    provider = _DummyProvider()
    request = httpx.Request("POST", "https://example.com/chat/completions")
    response = httpx.Response(429, request=request, text="Too many requests")
    wrapped = RetryExhaustedError(
        3,
        httpx.HTTPStatusError("429 Too Many Requests", request=request, response=response),
    )

    result = provider.classify_error(wrapped)

    assert isinstance(result, ProviderRateLimitError)
    assert result.status_code == 429


def test_classify_error_keeps_connection_errors_as_connection_errors() -> None:
    provider = _DummyProvider()

    result = provider.classify_error(ConnectionError("Connection refused"))

    assert isinstance(result, ProviderConnectionError)
