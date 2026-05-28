import pytest

from victor.core.errors import ProviderRateLimitError
from victor.providers.base import BaseProvider, CompletionResponse, Message


class _RateLimitProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "rate-limit-test"

    async def chat(self, messages, *, model: str, **kwargs):
        return await self._execute_with_circuit_breaker(self._call, model=model)

    async def stream(self, messages, *, model: str, **kwargs):
        yield CompletionResponse(content="", model=model)

    async def close(self):
        return None

    async def _call(self, *, model: str):
        raise ProviderRateLimitError("limited", provider=self.name, retry_after=30)


@pytest.mark.asyncio
async def test_rate_limit_suppression_is_shared_across_provider_instances():
    BaseProvider._rate_limit_suppression_by_key.clear()
    first = _RateLimitProvider(max_retries=0, use_circuit_breaker=False)
    second = _RateLimitProvider(max_retries=0, use_circuit_breaker=False)

    with pytest.raises(ProviderRateLimitError):
        await first.chat([Message(role="user", content="hi")], model="same-model")

    with pytest.raises(ProviderRateLimitError) as exc_info:
        await second.chat([Message(role="user", content="hi")], model="same-model")

    assert "temporarily suppressed" in str(exc_info.value)
    assert exc_info.value.retry_after is not None


class _SuccessProvider(_RateLimitProvider):
    async def _call(self, *, model: str):
        return CompletionResponse(content=f"ok:{model}", model=model)


@pytest.mark.asyncio
async def test_rate_limit_suppression_is_model_scoped():
    BaseProvider._rate_limit_suppression_by_key.clear()
    failing = _RateLimitProvider(max_retries=0, use_circuit_breaker=False)
    succeeding = _SuccessProvider(max_retries=0, use_circuit_breaker=False)

    with pytest.raises(ProviderRateLimitError):
        await failing.chat([Message(role="user", content="hi")], model="limited-model")

    response = await succeeding.chat(
        [Message(role="user", content="hi")], model="other-model"
    )

    assert response.content == "ok:other-model"
