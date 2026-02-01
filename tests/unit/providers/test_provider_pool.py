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

"""Unit tests for provider pool."""

import asyncio
import pytest

from victor.providers.provider_pool import (
    ProviderPool,
    ProviderPoolConfig,
    create_provider_pool,
)
from victor.providers.base import BaseProvider, Message, CompletionResponse, StreamChunk
from victor.providers.load_balancer import LoadBalancerType


# Mock provider for testing
class MockProvider(BaseProvider):
    """Mock provider for testing."""

    def __init__(self, name: str = "mock-provider"):
        super().__init__()
        self._name = name
        self.chat_call_count = 0
        self.chat_latency_ms = 100
        self.should_fail = False

    @property
    def name(self) -> str:
        return self._name

    async def chat(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list | None = None,
        **kwargs,
    ) -> CompletionResponse:
        """Mock chat implementation."""
        self.chat_call_count += 1

        if self.should_fail:
            raise RuntimeError(f"Provider {self._name} failed")

        # Simulate latency
        await asyncio.sleep(self.chat_latency_ms / 1000.0)

        return CompletionResponse(
            content=f"Response from {self._name}",
            model=model,
            role="assistant",
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list | None = None,
        **kwargs,
    ):
        """Mock stream implementation."""
        if self.should_fail:
            raise RuntimeError(f"Provider {self._name} failed")

        yield StreamChunk(content=f"Chunk from {self._name}")
        yield StreamChunk(content=" ", is_final=False)
        yield StreamChunk(content="done", is_final=True)

    async def close(self) -> None:
        """Mock close implementation."""
        pass


@pytest.fixture
def mock_providers():
    """Create mock providers."""
    return {
        "provider-1": MockProvider("provider-1"),
        "provider-2": MockProvider("provider-2"),
        "provider-3": MockProvider("provider-3"),
    }


@pytest.fixture
def pool_config():
    """Create pool config."""
    return ProviderPoolConfig(
        pool_size=3,
        load_balancer=LoadBalancerType.ROUND_ROBIN,
        enable_warmup=False,  # Disable for faster tests
        max_retries=2,
    )


class TestProviderPool:
    """Tests for ProviderPool."""

    @pytest.mark.asyncio
    async def test_initialize(self, pool_config) -> None:
        """Test pool initialization."""
        pool = ProviderPool(name="test-pool", config=pool_config)
        assert not pool._initialized

        await pool.initialize()
        assert pool._initialized
        assert pool._load_balancer is not None
        assert pool._health_registry is not None

    @pytest.mark.asyncio
    async def test_add_provider(self, pool_config) -> None:
        """Test adding provider to pool."""
        pool = ProviderPool(name="test-pool", config=pool_config)
        await pool.initialize()

        provider = MockProvider("test-provider")
        await pool.add_provider("provider-1", provider)

        # Check provider was added
        retrieved = await pool.get_provider("provider-1")
        assert retrieved is provider

        # Check instance was created
        stats = pool.get_pool_stats()
        assert stats["instances"]["total"] == 1

    @pytest.mark.asyncio
    async def test_add_multiple_providers(self, pool_config, mock_providers) -> None:
        """Test adding multiple providers."""
        pool = ProviderPool(name="test-pool", config=pool_config)
        await pool.initialize()

        for provider_id, provider in mock_providers.items():
            await pool.add_provider(provider_id, provider)

        stats = pool.get_pool_stats()
        assert stats["instances"]["total"] == 3
        assert stats["instances"]["healthy"] == 3

    @pytest.mark.asyncio
    async def test_remove_provider(self, pool_config) -> None:
        """Test removing provider from pool."""
        pool = ProviderPool(name="test-pool", config=pool_config)
        await pool.initialize()

        provider = MockProvider("test-provider")
        await pool.add_provider("provider-1", provider)

        # Add a connection
        instance = pool._instances["provider-1"]
        instance.acquire_connection()

        await pool.remove_provider("provider-1")

        # Should be removed
        assert await pool.get_provider("provider-1") is None
        stats = pool.get_pool_stats()
        assert stats["instances"]["total"] == 0

    @pytest.mark.asyncio
    async def test_chat_through_pool(self, pool_config, mock_providers) -> None:
        """Test sending chat request through pool."""
        pool = ProviderPool(name="test-pool", config=pool_config)
        await pool.initialize()

        for provider_id, provider in mock_providers.items():
            await pool.add_provider(provider_id, provider)

        messages = [Message(role="user", content="Hello")]
        response = await pool.chat(messages, model="test-model")

        assert response.content.startswith("Response from")
        assert "provider-" in response.content

    @pytest.mark.asyncio
    async def test_chat_with_retry(self, pool_config, mock_providers) -> None:
        """Test chat with retry on failure."""
        pool = ProviderPool(name="test-pool", config=pool_config)
        await pool.initialize()

        for provider_id, provider in mock_providers.items():
            await pool.add_provider(provider_id, provider)

        # Make first provider fail
        mock_providers["provider-1"].should_fail = True

        messages = [Message(role="user", content="Hello")]
        response = await pool.chat(messages, model="test-model")

        # Should succeed with fallback
        assert response.content.startswith("Response from")

    @pytest.mark.asyncio
    async def test_stream_through_pool(self, pool_config, mock_providers) -> None:
        """Test streaming through pool."""
        pool = ProviderPool(name="test-pool", config=pool_config)
        await pool.initialize()

        for provider_id, provider in mock_providers.items():
            await pool.add_provider(provider_id, provider)

        messages = [Message(role="user", content="Hello")]
        chunks = []

        async for chunk in pool.stream(messages, model="test-model"):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert any("Chunk from" in chunk.content for chunk in chunks)

    @pytest.mark.asyncio
    async def test_select_provider(self, pool_config, mock_providers) -> None:
        """Test provider selection."""
        pool = ProviderPool(name="test-pool", config=pool_config)
        await pool.initialize()

        for provider_id, provider in mock_providers.items():
            await pool.add_provider(provider_id, provider)

        selected = await pool.select_provider()
        assert selected is not None
        assert selected.provider_id in mock_providers

    @pytest.mark.asyncio
    async def test_round_robin_distribution(self, pool_config, mock_providers) -> None:
        """Test round-robin load distribution."""
        pool = ProviderPool(name="test-pool", config=pool_config)
        await pool.initialize()

        for provider_id, provider in mock_providers.items():
            await pool.add_provider(provider_id, provider)

        messages = [Message(role="user", content="Hello")]

        # Make multiple requests
        selected_providers = []
        for _ in range(6):
            instance = await pool.select_provider()
            selected_providers.append(instance.provider_id)

        # Should cycle through providers
        assert selected_providers == [
            "provider-1",
            "provider-2",
            "provider-3",
            "provider-1",
            "provider-2",
            "provider-3",
        ]

    @pytest.mark.asyncio
    async def test_connection_tracking(self, pool_config, mock_providers) -> None:
        """Test active connection tracking."""
        pool = ProviderPool(name="test-pool", config=pool_config)
        await pool.initialize()

        for provider_id, provider in mock_providers.items():
            await pool.add_provider(provider_id, provider)

        # Start a chat (acquires connection)
        messages = [Message(role="user", content="Hello")]

        # Create task but don't await yet
        task = asyncio.create_task(pool.chat(messages, model="test-model"))

        # Wait a bit for connection to be acquired
        await asyncio.sleep(0.01)

        # Check active connections
        stats = pool.get_pool_stats()
        assert stats["connections"]["active"] >= 1

        # Wait for completion
        await task

        # Connection should be released
        stats = pool.get_pool_stats()
        assert stats["connections"]["active"] == 0

    @pytest.mark.asyncio
    async def test_pool_stats(self, pool_config, mock_providers) -> None:
        """Test getting pool statistics."""
        pool = ProviderPool(name="test-pool", config=pool_config)
        await pool.initialize()

        for provider_id, provider in mock_providers.items():
            await pool.add_provider(provider_id, provider, weight=2.0)

        stats = pool.get_pool_stats()

        assert stats["pool_name"] == "test-pool"
        assert stats["instances"]["total"] == 3
        assert stats["instances"]["healthy"] == 3
        assert stats["config"]["load_balancer"] == "round_robin"
        assert len(stats["providers"]) == 3

        # Check provider stats
        provider_stats = stats["providers"][0]
        assert "provider_id" in provider_stats
        assert "weight" in provider_stats
        assert provider_stats["weight"] == 2.0

    @pytest.mark.asyncio
    async def test_close_pool(self, pool_config, mock_providers) -> None:
        """Test closing pool."""
        pool = ProviderPool(name="test-pool", config=pool_config)
        await pool.initialize()

        for provider_id, provider in mock_providers.items():
            await pool.add_provider(provider_id, provider)

        await pool.close()

        # Should be cleaned up
        assert not pool._initialized
        assert len(pool._instances) == 0


class TestProviderPoolConfig:
    """Tests for ProviderPoolConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = ProviderPoolConfig()

        assert config.pool_size == 5
        assert config.min_instances == 1
        assert config.load_balancer == LoadBalancerType.ADAPTIVE
        assert config.enable_warmup is True
        assert config.max_retries == 3

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = ProviderPoolConfig(
            pool_size=10,
            min_instances=2,
            load_balancer=LoadBalancerType.LEAST_CONNECTIONS,
            enable_warmup=False,
            max_retries=5,
        )

        assert config.pool_size == 10
        assert config.min_instances == 2
        assert config.load_balancer == LoadBalancerType.LEAST_CONNECTIONS
        assert config.enable_warmup is False
        assert config.max_retries == 5


class TestCreateProviderPool:
    """Tests for create_provider_pool factory."""

    @pytest.mark.asyncio
    async def test_create_pool(self, mock_providers) -> None:
        """Test creating pool with factory."""
        config = ProviderPoolConfig(enable_warmup=False)

        pool = await create_provider_pool(
            name="test-pool",
            providers=mock_providers,
            config=config,
        )

        assert pool._initialized
        stats = pool.get_pool_stats()
        assert stats["instances"]["total"] == 3

    @pytest.mark.asyncio
    async def test_create_pool_with_warmup(self, mock_providers) -> None:
        """Test creating pool with warmup."""
        config = ProviderPoolConfig(enable_warmup=True, warmup_concurrency=2)

        # Reduce latency for faster warmup
        for provider in mock_providers.values():
            provider.chat_latency_ms = 10

        pool = await create_provider_pool(
            name="test-pool",
            providers=mock_providers,
            config=config,
        )

        assert pool._initialized
        # All providers should have been warmed up (chat called at least once)
        for provider in mock_providers.values():
            assert provider.chat_call_count >= 1


class TestPoolFailover:
    """Tests for pool failover behavior."""

    @pytest.mark.asyncio
    async def test_all_providers_fail(self, pool_config, mock_providers) -> None:
        """Test when all providers fail."""
        pool = ProviderPool(name="test-pool", config=pool_config)
        await pool.initialize()

        for provider_id, provider in mock_providers.items():
            await pool.add_provider(provider_id, provider)
            provider.should_fail = True

        messages = [Message(role="user", content="Hello")]

        with pytest.raises(RuntimeError, match="All providers.*failed"):
            await pool.chat(messages, model="test-model")

    @pytest.mark.asyncio
    async def test_unhealthy_provider_excluded(self, pool_config, mock_providers) -> None:
        """Test that unhealthy providers are excluded."""
        pool = ProviderPool(name="test-pool", config=pool_config)
        await pool.initialize()

        for provider_id, provider in mock_providers.items():
            await pool.add_provider(provider_id, provider)

        # Mark first as unhealthy
        from victor.providers.health_monitor import HealthStatus

        pool._instances["provider-1"].health_monitor.set_status(HealthStatus.UNHEALTHY)

        selected = await pool.select_provider()
        assert selected.provider_id != "provider-1"

    @pytest.mark.asyncio
    async def test_no_healthy_providers(self, pool_config, mock_providers) -> None:
        """Test when no healthy providers available."""
        pool = ProviderPool(name="test-pool", config=pool_config)
        await pool.initialize()

        for provider_id, provider in mock_providers.items():
            await pool.add_provider(provider_id, provider)

        # Mark all as unhealthy
        from victor.providers.health_monitor import HealthStatus

        for instance in pool._instances.values():
            instance.health_monitor.set_status(HealthStatus.UNHEALTHY)

        selected = await pool.select_provider()
        assert selected is None
