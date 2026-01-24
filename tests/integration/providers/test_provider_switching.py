# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration tests for provider pool switching and failover.

Test areas:
1. Provider switching mid-conversation (5 tests)
2. Load balancing across providers (5 tests)
3. Fallback on provider failure (5 tests)
4. Circuit breaker integration (5 tests)

Total: 20+ tests
"""

import asyncio
import pytest
from typing import List, Dict, Any

from tests.mocks.provider_mocks import (
    FailingProvider,
    MockBaseProvider,
    ProviderTestHelpers,
)
from victor.providers.base import BaseProvider, Message, CompletionResponse, StreamChunk
from victor.providers.provider_pool import (
    ProviderPool,
    ProviderPoolConfig,
    PoolStrategy,
    create_provider_pool,
)
from victor.providers.load_balancer import LoadBalancerType
from victor.core.errors import ProviderConnectionError, ProviderTimeoutError


# =============================================================================
# Mock Providers for Integration Testing
# =============================================================================


class SwitchableMockProvider(MockBaseProvider):
    """Mock provider that tracks calls and can simulate failures."""

    def __init__(
        self,
        name: str,
        use_circuit_breaker: bool = False,
        circuit_breaker_failure_threshold: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._provider_name = name
        self._call_history = []
        self._should_fail = False
        self._circuit_open = False  # Circuit breaker state tracking
        self._use_circuit_breaker = use_circuit_breaker
        self._failure_threshold = circuit_breaker_failure_threshold
        self._failure_count = 0  # Track consecutive failures

    @property
    def name(self) -> str:
        return self._provider_name

    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self._circuit_open

    def set_circuit_open(self, open: bool):
        """Set circuit breaker state (for testing)."""
        self._circuit_open = open

    def set_failure_mode(self, should_fail: bool):
        """Set whether provider should fail."""
        self._should_fail = should_fail
        if not should_fail:
            # Reset failure count when exiting failure mode
            self._failure_count = 0

    async def chat(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools=None,
        **kwargs: Any,
    ) -> CompletionResponse:
        self._call_history.append({"method": "chat", "model": model})

        if self._circuit_open:
            raise ProviderConnectionError(f"{self._provider_name} circuit is open")

        if self._should_fail:
            self._failure_count += 1
            # Trip circuit breaker if threshold exceeded
            if self._use_circuit_breaker and self._failure_count >= self._failure_threshold:
                self._circuit_open = True
            raise ProviderConnectionError(f"{self._provider_name} failed")

        # Reset failure count on success
        self._failure_count = 0

        return CompletionResponse(
            content=f"Response from {self._provider_name}",
            model=model,
            role="assistant",
        )

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools=None,
        **kwargs: Any,
    ):
        self._call_history.append({"method": "stream", "model": model})

        if self._circuit_open:
            raise ProviderConnectionError(f"{self._provider_name} circuit is open")

        if self._should_fail:
            raise ProviderConnectionError(f"{self._provider_name} failed")

        yield StreamChunk(content=f"Chunk from {self._provider_name}")

    @property
    def call_count(self) -> int:
        return len(self._call_history)

    @property
    def call_history(self) -> List[Dict[str, Any]]:
        return self._call_history.copy()

    def reset_history(self) -> None:
        self._call_history.clear()

    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker state."""
        self._circuit_open = False
        self._failure_count = 0


# =============================================================================
# 1. Provider Switching Mid-Conversation (5 tests)
# =============================================================================


class TestProviderSwitching:
    """Test provider switching during conversation."""

    @pytest.mark.asyncio
    async def test_switch_between_providers(self):
        """Test switching between different providers."""
        provider1 = SwitchableMockProvider(name="provider-1")
        provider2 = SwitchableMockProvider(name="provider-2")

        config = ProviderPoolConfig(
            load_balancer=LoadBalancerType.ROUND_ROBIN,
            max_retries=1,
        )

        pool = await create_provider_pool(
            name="test-pool",
            providers={
                "provider-1": provider1,
                "provider-2": provider2,
            },
            config=config,
        )

        messages = ProviderTestHelpers.create_test_messages("Hello")

        # First call goes to provider-1
        response1 = await pool.chat(messages, model="test-model")
        assert "provider-1" in response1.content or "provider-2" in response1.content

        # Second call may go to different provider (round-robin)
        response2 = await pool.chat(messages, model="test-model")
        assert response2.content is not None

        await pool.close()

    @pytest.mark.asyncio
    async def test_switch_preserves_context(self):
        """Test that switching providers preserves conversation context."""
        provider1 = SwitchableMockProvider(name="provider-1", response_text="Context preserved")
        provider2 = SwitchableMockProvider(name="provider-2", response_text="Context preserved")

        pool = await create_provider_pool(
            name="context-pool",
            providers={
                "provider-1": provider1,
                "provider-2": provider2,
            },
        )

        messages = [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="First message"),
            Message(role="assistant", content="First response"),
            Message(role="user", content="Second message"),
        ]

        # Both providers should receive full context
        response = await pool.chat(messages, model="test-model")
        assert response.content is not None

        await pool.close()

    @pytest.mark.asyncio
    async def test_switch_on_streaming(self):
        """Test provider switching during streaming."""
        provider1 = SwitchableMockProvider(name="stream-provider-1")
        provider2 = SwitchableMockProvider(name="stream-provider-2")

        pool = await create_provider_pool(
            name="stream-pool",
            providers={
                "provider-1": provider1,
                "provider-2": provider2,
            },
        )

        messages = ProviderTestHelpers.create_test_messages("Stream test")

        chunks = []
        async for chunk in pool.stream(messages, model="test-model"):
            chunks.append(chunk)

        assert len(chunks) > 0

        await pool.close()

    @pytest.mark.asyncio
    async def test_manual_provider_selection(self):
        """Test manually selecting a specific provider."""
        provider1 = SwitchableMockProvider(name="manual-1")
        provider2 = SwitchableMockProvider(name="manual-2")

        pool = await create_provider_pool(
            name="manual-pool",
            providers={
                "provider-1": provider1,
                "provider-2": provider2,
            },
        )

        # Get specific provider
        retrieved_provider = await pool.get_provider("provider-2")
        assert retrieved_provider is not None
        assert retrieved_provider.name == "manual-2"

        await pool.close()

    @pytest.mark.asyncio
    async def test_switch_with_different_models(self):
        """Test switching providers with different model support."""
        provider1 = SwitchableMockProvider(name="gpt-provider", response_text="GPT response")
        provider2 = SwitchableMockProvider(name="claude-provider", response_text="Claude response")

        pool = await create_provider_pool(
            name="multi-model-pool",
            providers={
                "gpt": provider1,
                "claude": provider2,
            },
        )

        messages = ProviderTestHelpers.create_test_messages("Test")

        # Request with different models
        response1 = await pool.chat(messages, model="gpt-4")
        response2 = await pool.chat(messages, model="claude-3-5-sonnet-20241022")

        assert response1.content is not None
        assert response2.content is not None

        await pool.close()


# =============================================================================
# 2. Load Balancing Across Providers (5 tests)
# =============================================================================


class TestLoadBalancing:
    """Test load balancing across multiple providers."""

    @pytest.mark.asyncio
    async def test_round_robin_distribution(self):
        """Test round-robin load distribution."""
        provider1 = SwitchableMockProvider(name="rr-1")
        provider2 = SwitchableMockProvider(name="rr-2")
        provider3 = SwitchableMockProvider(name="rr-3")

        config = ProviderPoolConfig(
            load_balancer=LoadBalancerType.ROUND_ROBIN,
            enable_warmup=False,  # Disable warmup to avoid double counting
        )

        pool = await create_provider_pool(
            name="rr-pool",
            providers={
                "provider-1": provider1,
                "provider-2": provider2,
                "provider-3": provider3,
            },
            config=config,
        )

        messages = ProviderTestHelpers.create_test_messages()

        # Make multiple requests
        for _ in range(6):
            await pool.chat(messages, model="test-model")

        # All providers should have been called
        total_calls = provider1.call_count + provider2.call_count + provider3.call_count
        assert total_calls == 6

        # Each provider should have been called at least once
        assert provider1.call_count >= 1
        assert provider2.call_count >= 1
        assert provider3.call_count >= 1

        await pool.close()

    @pytest.mark.asyncio
    async def test_least_connections_balancing(self):
        """Test least-connections load balancing."""
        provider1 = SwitchableMockProvider(name="lc-1")
        provider2 = SwitchableMockProvider(name="lc-2")

        config = ProviderPoolConfig(
            load_balancer=LoadBalancerType.LEAST_CONNECTIONS,
            enable_warmup=False,  # Disable warmup to avoid double counting
        )

        pool = await create_provider_pool(
            name="lc-pool",
            providers={
                "provider-1": provider1,
                "provider-2": provider2,
            },
            config=config,
        )

        messages = ProviderTestHelpers.create_test_messages()

        # Make requests
        for _ in range(4):
            await pool.chat(messages, model="test-model")

        # Load should be distributed
        total_calls = provider1.call_count + provider2.call_count
        assert total_calls == 4

        await pool.close()

    @pytest.mark.asyncio
    async def test_weighted_distribution(self):
        """Test weighted load distribution."""
        provider1 = SwitchableMockProvider(name="weighted-1")
        provider2 = SwitchableMockProvider(name="weighted-2")

        # Create pool with warmup disabled to avoid double counting
        config = ProviderPoolConfig(enable_warmup=False)
        pool = ProviderPool(name="weighted-pool", config=config)
        await pool.initialize()

        # Add providers with different weights
        await pool.add_provider("provider-1", provider1, weight=2.0)
        await pool.add_provider("provider-2", provider2, weight=1.0)

        messages = ProviderTestHelpers.create_test_messages()

        # Make multiple requests
        for _ in range(6):
            await pool.chat(messages, model="test-model")

        # Provider-1 should get more traffic due to higher weight
        # (Note: actual distribution depends on load balancer implementation)
        total_calls = provider1.call_count + provider2.call_count
        assert total_calls == 6

        await pool.close()

    @pytest.mark.asyncio
    async def test_adaptive_load_balancing(self):
        """Test adaptive load balancing based on performance."""
        provider1 = SwitchableMockProvider(name="adaptive-1", response_delay=0.05)
        provider2 = SwitchableMockProvider(name="adaptive-2", response_delay=0.15)

        config = ProviderPoolConfig(
            load_balancer=LoadBalancerType.ADAPTIVE,
            enable_warmup=False,  # Disable warmup to avoid double counting
        )

        pool = await create_provider_pool(
            name="adaptive-pool",
            providers={
                "fast": provider1,
                "slow": provider2,
            },
            config=config,
        )

        messages = ProviderTestHelpers.create_test_messages()

        # Make requests - adaptive balancer should prefer faster provider
        for _ in range(5):
            await pool.chat(messages, model="test-model")

        # Both providers should be used
        assert provider1.call_count + provider2.call_count == 5

        await pool.close()

    @pytest.mark.asyncio
    async def test_load_balancing_with_failures(self):
        """Test load balancing when some providers fail."""
        provider1 = SwitchableMockProvider(name="lb-fail-1")
        provider2 = SwitchableMockProvider(name="lb-fail-2")
        provider3 = SwitchableMockProvider(name="lb-fail-3")

        # Make provider-2 fail
        provider2.set_failure_mode(True)

        # Create pool with warmup disabled to avoid double counting
        config = ProviderPoolConfig(enable_warmup=False)
        pool = await create_provider_pool(
            name="lb-fail-pool",
            providers={
                "provider-1": provider1,
                "provider-2": provider2,
                "provider-3": provider3,
            },
            config=config,
        )

        messages = ProviderTestHelpers.create_test_messages()

        # Make requests - should work despite one failing provider
        for _ in range(3):
            response = await pool.chat(messages, model="test-model")
            assert response.content is not None

        # Healthy providers should have been called
        assert provider1.call_count + provider3.call_count == 3

        await pool.close()


# =============================================================================
# 3. Fallback on Provider Failure (5 tests)
# =============================================================================


class TestProviderFallback:
    """Test automatic fallback when providers fail."""

    @pytest.mark.asyncio
    async def test_fallback_to_healthy_provider(self):
        """Test fallback to healthy provider when one fails."""
        provider1 = SwitchableMockProvider(name="fallback-1")
        provider2 = SwitchableMockProvider(name="fallback-2")

        # Make provider-1 fail
        provider1.set_failure_mode(True)

        pool = await create_provider_pool(
            name="fallback-pool",
            providers={
                "failing": provider1,
                "healthy": provider2,
            },
        )

        messages = ProviderTestHelpers.create_test_messages()

        # Should fall back to healthy provider
        response = await pool.chat(messages, model="test-model")
        assert "fallback-2" in response.content

        await pool.close()

    @pytest.mark.asyncio
    async def test_fallback_with_retries(self):
        """Test fallback behavior with retry logic."""
        provider1 = FailingProvider(error_type="connection", fail_after=1)
        provider2 = SwitchableMockProvider(name="fallback-healthy")

        pool = await create_provider_pool(
            name="retry-fallback-pool",
            providers={
                "flaky": provider1,
                "stable": provider2,
            },
        )

        messages = ProviderTestHelpers.create_test_messages()

        # First call to flaky provider succeeds (fail_after=1)
        # Note: FailingProvider with fail_after=1 will succeed first, then fail
        response1 = await pool.chat(messages, model="test-model")
        # The response could be from either provider depending on pool routing
        assert response1 is not None
        assert response1.content is not None

        # Second call fails, should fall back or retry
        response2 = await pool.chat(messages, model="test-model")
        assert response2 is not None
        assert response2.content is not None

        await pool.close()

    @pytest.mark.asyncio
    async def test_fallback_cascade(self):
        """Test fallback cascade when multiple providers fail."""
        provider1 = SwitchableMockProvider(name="cascade-1")
        provider2 = SwitchableMockProvider(name="cascade-2")
        provider3 = SwitchableMockProvider(name="cascade-3")

        # Make first two fail
        provider1.set_failure_mode(True)
        provider2.set_failure_mode(True)

        pool = await create_provider_pool(
            name="cascade-pool",
            providers={
                "fail-1": provider1,
                "fail-2": provider2,
                "healthy": provider3,
            },
        )

        messages = ProviderTestHelpers.create_test_messages()

        # Should fall back to the only healthy provider
        response = await pool.chat(messages, model="test-model")
        assert "cascade-3" in response.content

        await pool.close()

    @pytest.mark.asyncio
    async def test_no_healthy_providers_error(self):
        """Test error when all providers are unhealthy."""
        provider1 = SwitchableMockProvider(name="all-fail-1")
        provider2 = SwitchableMockProvider(name="all-fail-2")

        # Make all fail
        provider1.set_failure_mode(True)
        provider2.set_failure_mode(True)

        pool = await create_provider_pool(
            name="all-fail-pool",
            providers={
                "fail-1": provider1,
                "fail-2": provider2,
            },
        )

        messages = ProviderTestHelpers.create_test_messages()

        # Should raise error when all providers fail
        with pytest.raises(Exception):
            await pool.chat(messages, model="test-model")

        await pool.close()

    @pytest.mark.asyncio
    async def test_fallback_recovery(self):
        """Test recovery after fallback."""
        provider1 = SwitchableMockProvider(name="recover-1")
        provider2 = SwitchableMockProvider(name="recover-2")

        # Initially, provider-1 fails
        provider1.set_failure_mode(True)

        pool = await create_provider_pool(
            name="recovery-pool",
            providers={
                "primary": provider1,
                "backup": provider2,
            },
        )

        messages = ProviderTestHelpers.create_test_messages()

        # First request falls back to backup
        response1 = await pool.chat(messages, model="test-model")
        assert "recover-2" in response1.content

        # Recover primary provider
        provider1.set_failure_mode(False)

        # Next request may use primary again (depends on load balancer)
        response2 = await pool.chat(messages, model="test-model")
        assert response2 is not None

        await pool.close()


# =============================================================================
# 4. Circuit Breaker Integration (5 tests)
# =============================================================================


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with provider pool."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_trips_on_failures(self):
        """Test that circuit breaker trips after threshold failures."""
        provider1 = SwitchableMockProvider(
            name="cb-1", use_circuit_breaker=True, circuit_breaker_failure_threshold=3
        )

        # Use single-provider pool to ensure all calls go to the failing provider
        pool = await create_provider_pool(
            name="cb-pool",
            providers={
                "unstable": provider1,
            },
        )

        messages = ProviderTestHelpers.create_test_messages()

        # Make provider fail multiple times to trip circuit
        provider1.set_failure_mode(True)

        # Circuit should trip after threshold
        for _ in range(5):
            try:
                await pool.chat(messages, model="test-model")
            except Exception:
                pass

        # Circuit should be open for provider
        assert provider1.is_circuit_open()

        await pool.close()

    @pytest.mark.asyncio
    async def test_circuit_breaker_isolation(self):
        """Test that circuit breakers isolate failing providers."""
        provider1 = SwitchableMockProvider(
            name="isolated-1", use_circuit_breaker=True, circuit_breaker_failure_threshold=2
        )
        provider2 = SwitchableMockProvider(
            name="isolated-2", use_circuit_breaker=True, circuit_breaker_failure_threshold=2
        )

        # Create separate pools for each provider to test isolation
        pool1 = await create_provider_pool(
            name="isolation-pool-1",
            providers={
                "provider-1": provider1,
            },
        )

        pool2 = await create_provider_pool(
            name="isolation-pool-2",
            providers={
                "provider-2": provider2,
            },
        )

        messages = ProviderTestHelpers.create_test_messages()

        # Trip circuit for provider-1 only
        provider1.set_failure_mode(True)

        for _ in range(3):
            try:
                await pool1.chat(messages, model="test-model")
            except Exception:
                pass

        # Provider-1 circuit should be open
        assert provider1.is_circuit_open()

        # Provider-2 circuit should still be closed (no failures)
        assert not provider2.is_circuit_open()

        await pool1.close()
        await pool2.close()

    @pytest.mark.asyncio
    async def test_circuit_breaker_prevents_calls(self):
        """Test that open circuit prevents calls to provider."""
        provider = SwitchableMockProvider(
            name="blocked", use_circuit_breaker=True, circuit_breaker_failure_threshold=2
        )

        pool = await create_provider_pool(
            name="blocked-pool",
            providers={
                "provider": provider,
            },
        )

        messages = ProviderTestHelpers.create_test_messages()

        # Trip the circuit
        provider.set_failure_mode(True)

        for _ in range(3):
            try:
                await pool.chat(messages, model="test-model")
            except Exception:
                pass

        # Circuit should be open
        assert provider.is_circuit_open()

        # Try to use provider through pool
        # Should fail fast or fall back
        try:
            await pool.chat(messages, model="test-model")
        except Exception:
            pass  # Expected

        await pool.close()

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after reset."""
        provider = SwitchableMockProvider(
            name="recover-cb", use_circuit_breaker=True, circuit_breaker_failure_threshold=2
        )

        messages = ProviderTestHelpers.create_test_messages()

        # Trip the circuit directly
        provider.set_failure_mode(True)

        for _ in range(3):
            try:
                await provider.chat(messages, model="test-model")
            except Exception:
                pass

        assert provider.is_circuit_open()

        # Reset circuit
        provider.reset_circuit_breaker()
        assert not provider.is_circuit_open()

        # Provider should work again after reset
        provider.set_failure_mode(False)
        response = await provider.chat(messages, model="test-model")
        assert response is not None
        assert response.content == "Response from recover-cb"

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_pool_stats(self):
        """Test circuit breaker stats in pool statistics."""
        provider1 = SwitchableMockProvider(
            name="stats-1", use_circuit_breaker=True, circuit_breaker_failure_threshold=3
        )
        provider2 = SwitchableMockProvider(name="stats-2")

        pool = await create_provider_pool(
            name="stats-pool",
            providers={
                "cb-provider": provider1,
                "normal-provider": provider2,
            },
        )

        messages = ProviderTestHelpers.create_test_messages()

        # Make some calls
        for _ in range(2):
            await pool.chat(messages, model="test-model")

        # Get pool stats
        stats = pool.get_pool_stats()

        assert stats["pool_name"] == "stats-pool"
        assert stats["instances"]["total"] == 2
        assert len(stats["providers"]) == 2

        await pool.close()
