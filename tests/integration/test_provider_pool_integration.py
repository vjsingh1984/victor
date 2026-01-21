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

"""Integration tests for provider pool system.

Tests the full integration of provider pooling with:
- Settings configuration
- DI container registration
- Orchestrator factory
- CLI flags
"""

import asyncio
import pytest
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from victor.config.settings import Settings
from victor.core.container import ServiceContainer, ServiceLifetime
from victor.agent.service_provider import OrchestratorServiceProvider
from victor.providers.provider_pool import (
    ProviderPool,
    ProviderPoolConfig,
    PoolStrategy,
)
from victor.providers.load_balancer import LoadBalancerType
from victor.providers.health_monitor import HealthCheckConfig


@pytest.fixture
def pool_settings() -> Settings:
    """Create settings with provider pool enabled."""
    return Settings(
        enable_provider_pool=True,
        pool_size=3,
        pool_load_balancer="adaptive",
        pool_enable_warmup=True,
        pool_warmup_concurrency=3,
        pool_health_check_interval=30,
        pool_max_retries=3,
        pool_min_instances=1,
    )


@pytest.fixture
def single_provider_settings() -> Settings:
    """Create settings with provider pool disabled."""
    return Settings(
        enable_provider_pool=False,
        pool_size=3,
    )


class TestProviderPoolSettings:
    """Test provider pool settings configuration."""

    def test_pool_disabled_by_default(self):
        """Test that provider pool is disabled by default."""
        settings = Settings()
        assert settings.enable_provider_pool is False

    def test_pool_settings_validation(self):
        """Test that pool settings are properly validated."""
        settings = Settings(
            enable_provider_pool=True,
            pool_size=5,
            pool_load_balancer="adaptive",
            pool_health_check_interval=60,
        )

        assert settings.enable_provider_pool is True
        assert settings.pool_size == 5
        assert settings.pool_load_balancer == "adaptive"
        assert settings.pool_health_check_interval == 60

    def test_pool_load_balancer_validation(self):
        """Test that pool load balancer strategy is validated."""
        # Valid strategies
        for strategy in ["round_robin", "least_connections", "adaptive", "random"]:
            settings = Settings(pool_load_balancer=strategy)
            assert settings.pool_load_balancer == strategy

        # Invalid strategy
        with pytest.raises(ValueError):
            Settings(pool_load_balancer="invalid_strategy")

    def test_pool_size_bounds(self):
        """Test that pool size is bounded correctly."""
        # Valid sizes
        settings = Settings(pool_size=5)
        assert settings.pool_size == 5

        # Out of bounds (should raise validation error)
        with pytest.raises(ValueError):
            Settings(pool_size=0)

        with pytest.raises(ValueError):
            Settings(pool_size=11)


class TestProviderPoolServiceRegistration:
    """Test provider pool service registration in DI container."""

    def test_pool_services_registered_when_enabled(self, pool_settings):
        """Test that pool services are registered when enabled."""
        container = ServiceContainer()
        provider = OrchestratorServiceProvider(pool_settings)
        provider.register_singleton_services(container)

        # Check that health registry is registered
        from victor.providers.health_monitor import ProviderHealthRegistry
        health_registry = container.get_service(ProviderHealthRegistry)
        assert health_registry is not None

        # Check that load balancer factory is available via service provider
        factory = provider.get_load_balancer_factory()
        assert factory is not None

    def test_pool_services_not_registered_when_disabled(self, single_provider_settings):
        """Test that pool services are not registered when disabled."""
        container = ServiceContainer()
        provider = OrchestratorServiceProvider(single_provider_settings)
        provider.register_singleton_services(container)

        # Load balancer factory should not be available
        factory = provider.get_load_balancer_factory()
        assert factory is None

    def test_load_balancer_factory_creates_balancers(self, pool_settings):
        """Test that load balancer factory creates correct balancers."""
        container = ServiceContainer()
        provider = OrchestratorServiceProvider(pool_settings)
        provider.register_singleton_services(container)

        factory = provider.get_load_balancer_factory()

        # Test creating different balancer types
        for strategy in ["round_robin", "least_connections", "adaptive", "random"]:
            balancer = factory(strategy, name="test-balancer")
            assert balancer is not None
            assert balancer.name == "test-balancer"


class TestProviderPoolCreation:
    """Test provider pool creation through factory."""

    def _create_mock_provider(self, name: str) -> MagicMock:
        """Create a mock provider for testing."""
        provider = MagicMock()
        provider.name = name
        provider.chat = AsyncMock(return_value=MagicMock(content="Response"))
        provider.close = AsyncMock()

        # Mock stream
        async def mock_stream_gen(*args, **kwargs):
            yield MagicMock(content="Chunk")

        provider.stream = mock_stream_gen

        return provider

    @pytest.mark.asyncio
    async def test_create_pool_with_multiple_providers(self, pool_settings):
        """Test creating a pool with multiple provider instances."""
        from victor.providers.provider_pool import create_provider_pool

        # Create mock providers
        providers = {}
        for i in range(3):
            providers[f"provider-{i}"] = self._create_mock_provider(f"provider-{i}")

        # Create pool config
        pool_config = ProviderPoolConfig(
            pool_size=3,
            load_balancer=LoadBalancerType.ADAPTIVE,
            enable_warmup=False,  # Skip warmup for tests
        )

        # Create pool
        pool = await create_provider_pool(
            name="test-pool",
            providers=providers,
            config=pool_config,
        )

        assert pool is not None
        assert pool.name == "test-pool"
        assert pool.config.load_balancer == LoadBalancerType.ADAPTIVE

        # Check pool stats
        stats = pool.get_pool_stats()
        assert stats["pool_name"] == "test-pool"
        assert stats["instances"]["total"] == 3

        await pool.close()

    @pytest.mark.asyncio
    async def test_pool_selects_provider(self, pool_settings):
        """Test that pool can select a provider."""
        from victor.providers.provider_pool import create_provider_pool

        # Create mock providers
        providers = {
            "provider-0": self._create_mock_provider("provider-0"),
            "provider-1": self._create_mock_provider("provider-1"),
        }

        pool_config = ProviderPoolConfig(
            pool_size=2,
            load_balancer=LoadBalancerType.ROUND_ROBIN,
            enable_warmup=False,
        )

        pool = await create_provider_pool(
            name="test-pool",
            providers=providers,
            config=pool_config,
        )

        # Select provider
        instance = await pool.select_provider()
        assert instance is not None
        assert instance.provider_id in ["provider-0", "provider-1"]

        await pool.close()


class TestOrchestratorFactoryPoolIntegration:
    """Test orchestrator factory integration with provider pool."""

    def _create_mock_provider(self, name: str) -> MagicMock:
        """Create a mock provider for testing."""
        provider = MagicMock()
        provider.name = name
        provider.chat = AsyncMock(return_value=MagicMock(content="Response"))
        provider.close = AsyncMock()

        # Mock stream
        async def mock_stream_gen(*args, **kwargs):
            yield MagicMock(content="Chunk")

        provider.stream = mock_stream_gen

        return provider

    @pytest.mark.asyncio
    async def test_factory_creates_pool_when_enabled(self, pool_settings):
        """Test that factory creates pool when enabled."""
        from victor.agent.orchestrator_factory import OrchestratorFactory

        # Mock multiple base URLs
        pool_settings.lmstudio_base_urls = [
            "http://localhost:1234",
            "http://localhost:1235",
            "http://localhost:1236",
        ]

        # Create factory
        base_provider = self._create_mock_provider("lmstudio")
        factory = OrchestratorFactory(
            settings=pool_settings,
            provider=base_provider,
            model="test-model",
        )

        # Create pool
        provider_or_pool, is_pool = await factory.create_provider_pool_if_enabled(
            base_provider
        )

        # Should create pool since we have multiple URLs
        assert is_pool is True
        assert isinstance(provider_or_pool, ProviderPool)

        # Clean up
        if isinstance(provider_or_pool, ProviderPool):
            await provider_or_pool.close()

    @pytest.mark.asyncio
    async def test_factory_skips_pool_with_single_url(self, pool_settings):
        """Test that factory skips pool with only one URL."""
        from victor.agent.orchestrator_factory import OrchestratorFactory
        from victor.providers.mock import MockProvider

        # Only one base URL
        pool_settings.lmstudio_base_urls = ["http://localhost:1234"]

        # Create factory
        base_provider = self._create_mock_provider("lmstudio")
        factory = OrchestratorFactory(
            settings=pool_settings,
            provider=base_provider,
            model="test-model",
        )

        # Try to create pool
        provider_or_pool, is_pool = await factory.create_provider_pool_if_enabled(
            base_provider
        )

        # Should skip pool since we only have one URL
        assert is_pool is False
        assert provider_or_pool is base_provider

    @pytest.mark.asyncio
    async def test_factory_skips_pool_when_disabled(self, single_provider_settings):
        """Test that factory skips pool when disabled."""
        from victor.agent.orchestrator_factory import OrchestratorFactory
        from victor.providers.mock import MockProvider

        # Create factory with pool disabled
        base_provider = MockProvider(name="lmstudio")
        factory = OrchestratorFactory(
            settings=single_provider_settings,
            provider=base_provider,
            model="test-model",
        )

        # Try to create pool
        provider_or_pool, is_pool = await factory.create_provider_pool_if_enabled(
            base_provider
        )

        # Should skip pool since it's disabled
        assert is_pool is False
        assert provider_or_pool is base_provider


class TestProviderPoolChatIntegration:
    """Test provider pool chat integration."""

    def _create_mock_provider(self, name: str) -> MagicMock:
        """Create a mock provider for testing."""
        provider = MagicMock()
        provider.name = name
        provider.chat = AsyncMock(return_value=MagicMock(content="Response"))
        provider.close = AsyncMock()

        # Mock stream
        async def mock_stream_gen(*args, **kwargs):
            yield MagicMock(content="Chunk")

        provider.stream = mock_stream_gen

        return provider

    @pytest.mark.asyncio
    async def test_pool_chat_completion(self, pool_settings):
        """Test that pool can handle chat completions."""
        from victor.providers.provider_pool import create_provider_pool
        from victor.providers.base import Message

        # Create mock providers
        providers = {
            "provider-0": self._create_mock_provider("provider-0"),
            "provider-1": self._create_mock_provider("provider-1"),
        }

        pool_config = ProviderPoolConfig(
            pool_size=2,
            load_balancer=LoadBalancerType.ROUND_ROBIN,
            enable_warmup=False,
        )

        pool = await create_provider_pool(
            name="test-pool",
            providers=providers,
            config=pool_config,
        )

        # Send chat request
        messages = [Message(role="user", content="Hello")]
        response = await pool.chat(
            messages,
            model="test-model",
            max_tokens=100,
        )

        assert response is not None
        assert hasattr(response, "content")

        await pool.close()

    @pytest.mark.asyncio
    async def test_pool_streaming(self, pool_settings):
        """Test that pool can handle streaming requests."""
        from victor.providers.provider_pool import create_provider_pool
        from victor.providers.base import Message

        # Create mock providers
        providers = {
            "provider-0": self._create_mock_provider("provider-0"),
        }

        pool_config = ProviderPoolConfig(
            pool_size=1,
            load_balancer=LoadBalancerType.ROUND_ROBIN,
            enable_warmup=False,
        )

        pool = await create_provider_pool(
            name="test-pool",
            providers=providers,
            config=pool_config,
        )

        # Stream chat request
        messages = [Message(role="user", content="Hello")]
        chunks = []
        async for chunk in pool.stream(
            messages,
            model="test-model",
            max_tokens=100,
        ):
            chunks.append(chunk)

        assert len(chunks) > 0

        await pool.close()


class TestProviderPoolHealthMonitoring:
    """Test provider pool health monitoring."""

    def _create_mock_provider(self, name: str) -> MagicMock:
        """Create a mock provider for testing."""
        provider = MagicMock()
        provider.name = name
        provider.chat = AsyncMock(return_value=MagicMock(content="Response"))
        provider.close = AsyncMock()

        # Mock stream
        async def mock_stream_gen(*args, **kwargs):
            yield MagicMock(content="Chunk")

        provider.stream = mock_stream_gen

        return provider

    @pytest.mark.asyncio
    async def test_health_tracking(self, pool_settings):
        """Test that pool tracks provider health."""
        from victor.providers.provider_pool import create_provider_pool
        from victor.providers.mock import MockProvider

        providers = {
            "provider-0": MockProvider(name="provider-0"),
        }

        pool_config = ProviderPoolConfig(
            pool_size=1,
            enable_warmup=False,
        )

        pool = await create_provider_pool(
            name="test-pool",
            providers=providers,
            config=pool_config,
        )

        # Get health stats
        health_stats = await pool.health_check()
        assert "provider-0" in health_stats

        await pool.close()

    @pytest.mark.asyncio
    async def test_pool_stats(self, pool_settings):
        """Test that pool provides comprehensive stats."""
        from victor.providers.provider_pool import create_provider_pool

        providers = {
            "provider-0": self._create_mock_provider("provider-0"),
            "provider-1": self._create_mock_provider("provider-1"),
        }

        pool_config = ProviderPoolConfig(
            pool_size=2,
            enable_warmup=False,
        )

        pool = await create_provider_pool(
            name="test-pool",
            providers=providers,
            config=pool_config,
        )

        # Get pool stats
        stats = pool.get_pool_stats()
        assert stats["pool_name"] == "test-pool"
        assert stats["instances"]["total"] == 2
        assert stats["config"]["load_balancer"] == "adaptive"
        assert "providers" in stats

        await pool.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
