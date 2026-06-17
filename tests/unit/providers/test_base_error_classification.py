import httpx
import pytest

from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    ProviderAuthError,
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


class _RetryingDummyProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__(timeout=12, max_retries=1, use_circuit_breaker=False)

    @property
    def name(self) -> str:
        return "retrying-dummy"

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


def test_classify_error_sanitizes_html_auth_response() -> None:
    provider = _DummyProvider()
    request = httpx.Request("POST", "https://example.com/chat/completions")
    response = httpx.Response(
        401,
        request=request,
        text="<html><body><svg>large login page</svg></body></html>",
    )
    error = httpx.HTTPStatusError("401 Unauthorized", request=request, response=response)

    result = provider.classify_error(error)

    assert isinstance(result, ProviderAuthError)
    assert "HTML authentication page" in str(result)
    assert "<html>" not in str(result)


@pytest.mark.asyncio
async def test_execute_with_circuit_breaker_suppresses_follow_on_rate_limited_calls() -> None:
    provider = _RetryingDummyProvider()
    request = httpx.Request("POST", "https://example.com/chat/completions")
    response = httpx.Response(429, request=request, text="Too many requests")
    attempts = 0

    async def always_rate_limited() -> None:
        nonlocal attempts
        attempts += 1
        raise httpx.HTTPStatusError("429 Too Many Requests", request=request, response=response)

    with pytest.raises(ProviderRateLimitError) as first_error:
        await provider._execute_with_circuit_breaker(always_rate_limited)

    assert first_error.value.status_code == 429
    assert attempts == 2

    with pytest.raises(ProviderRateLimitError) as second_error:
        await provider._execute_with_circuit_breaker(always_rate_limited)

    assert second_error.value.retry_after is not None
    assert attempts == 2
