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

"""Provider pool with connection management and load balancing.

This module provides a pooling system for LLM providers that:
- Manages multiple provider instances
- Distributes load using configurable strategies
- Monitors health and performance
- Handles automatic failover
- Warms up connections for cold start optimization
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from collections.abc import AsyncIterator

from victor.providers.base import BaseProvider, CompletionResponse, Message, StreamChunk
from victor.providers.circuit_breaker import CircuitBreakerConfig
from victor.providers.health_monitor import (
    HealthCheckConfig,
    ProviderHealthRegistry,
    get_health_registry,
)
from victor.providers.load_balancer import (
    LoadBalancer,
    LoadBalancerType,
    ProviderInstance,
    create_load_balancer,
)

logger = logging.getLogger(__name__)


class PoolStrategy(Enum):
    """Provider pool management strategy."""

    # All providers are active and receive traffic
    ACTIVE_ACTIVE = "active_active"

    # One primary provider, others on standby
    ACTIVE_PASSIVE = "active_passive"

    # Geographic/routed distribution
    GEO_ROUTED = "geo_routed"


@dataclass
class ProviderPoolConfig:
    """Configuration for provider pool.

    Attributes:
        pool_size: Maximum number of provider instances
        min_instances: Minimum number of healthy instances required
        load_balancer: Load balancing strategy
        pool_strategy: Pool management strategy
        enable_warmup: Whether to warm up connections
        warmup_concurrency: Number of concurrent warmup requests
        health_check_config: Health check configuration
        circuit_breaker_config: Circuit breaker configuration
        max_retries: Maximum retry attempts across providers
        retry_delay: Initial delay between retries (seconds)
        enable_preemption: Allow preempting to faster providers
    """

    pool_size: int = 5
    min_instances: int = 1
    load_balancer: LoadBalancerType = LoadBalancerType.ADAPTIVE
    pool_strategy: PoolStrategy = PoolStrategy.ACTIVE_ACTIVE
    enable_warmup: bool = True
    warmup_concurrency: int = 3
    health_check_config: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    circuit_breaker_config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    max_retries: int = 3
    retry_delay: float = 0.5
    enable_preemption: bool = False


class ProviderPool:
    """Pool of LLM provider instances with load balancing and failover.

    Manages multiple provider instances, distributing requests according
    to the configured load balancing strategy. Provides automatic failover,
    health monitoring, and connection warmup.

    Example:
        # Create pool with multiple providers
        pool = ProviderPool(
            name="llm-pool",
            config=ProviderPoolConfig(
                load_balancer=LoadBalancerType.ADAPTIVE,
                pool_size=3
            )
        )

        # Add providers
        await pool.add_provider("anthropic-1", anthropic_provider)
        await pool.add_provider("openai-1", openai_provider)

        # Use pool like a single provider
        response = await pool.chat(messages, model="claude-3-5-sonnet-20241022")

        # Streaming
        async for chunk in pool.stream(messages, model="claude-3-5-sonnet-20241022"):
            print(chunk.content)
    """

    def __init__(
        self,
        name: str,
        config: Optional[ProviderPoolConfig] = None,
    ):
        """Initialize provider pool.

        Args:
            name: Unique pool name
            config: Pool configuration
        """
        self.name = name
        self.config = config or ProviderPoolConfig()
        self._instances: dict[str, ProviderInstance] = {}
        self._load_balancer: Optional[LoadBalancer] = None
        self._health_registry: Optional[ProviderHealthRegistry] = None
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the provider pool.

        Creates health registry and load balancer. Must be called before use.
        """
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return  # type: ignore[unreachable]

            # Get or create health registry
            self._health_registry = await get_health_registry()

            # Create load balancer
            self._load_balancer = create_load_balancer(
                strategy=self.config.load_balancer,
                name=f"{self.name}-lb",
            )

            self._initialized = True
            logger.info(
                f"Initialized provider pool '{self.name}' with {self.config.load_balancer.value} load balancing"
            )

    async def add_provider(
        self,
        provider_id: str,
        provider: BaseProvider,
        weight: float = 1.0,
        enabled: bool = True,
    ) -> None:
        """Add a provider to the pool.

        Args:
            provider_id: Unique identifier for this provider instance
            provider: BaseProvider instance
            weight: Weight for load balancing (higher = more traffic)
            enabled: Whether provider is initially enabled
        """
        if not self._initialized:
            await self.initialize()

        async with self._lock:
            # Check if already exists
            if provider_id in self._instances:
                logger.warning(f"Provider {provider_id} already in pool")
                return

            # Create health monitor
            assert self._health_registry is not None
            health_monitor = await self._health_registry.register(
                provider_id=provider_id,
                config=self.config.health_check_config,
            )

            # Start health checks
            await health_monitor.start_health_checks()

            # Create provider instance
            instance = ProviderInstance(
                provider_id=provider_id,
                provider=provider,
                health_monitor=health_monitor,
                weight=weight,
                enabled=enabled,
            )

            self._instances[provider_id] = instance
            logger.info(f"Added provider {provider_id} to pool '{self.name}'")

            # Warm up if enabled
            if self.config.enable_warmup and enabled:
                await self._warmup_provider(instance)

    async def remove_provider(self, provider_id: str) -> None:
        """Remove a provider from the pool.

        Args:
            provider_id: Provider to remove
        """
        async with self._lock:
            instance = self._instances.pop(provider_id, None)
            if instance:
                # Mark as draining first
                from victor.providers.health_monitor import HealthStatus

                instance.health_monitor.set_status(HealthStatus.DRAINING)

                # Wait for active connections to complete
                timeout = 30
                start = time.time()
                while instance.active_connections > 0:
                    if time.time() - start > timeout:
                        logger.warning(
                            f"Timeout waiting for connections to drain for {provider_id}"
                        )
                        break
                    await asyncio.sleep(0.1)

                # Unregister health monitor
                if self._health_registry:
                    await self._health_registry.unregister(provider_id)

                logger.info(f"Removed provider {provider_id} from pool '{self.name}'")

    async def get_provider(self, provider_id: str) -> Optional[BaseProvider]:
        """Get a specific provider by ID.

        Args:
            provider_id: Provider identifier

        Returns:
            BaseProvider instance or None
        """
        instance = self._instances.get(provider_id)
        return instance.provider if instance else None

    async def select_provider(self) -> Optional[ProviderInstance]:
        """Select best provider using load balancer.

        Args:
            instances: List of provider instances

        Returns:
            Selected ProviderInstance or None
        """
        if not self._initialized:
            await self.initialize()

        instances = list(self._instances.values())
        if not instances:
            return None

        assert self._load_balancer is not None
        return await self._load_balancer.select_with_fallback(instances)

    async def chat(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[list[Any]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Send chat completion request through the pool.

        Automatically selects best provider, handles retries and failover.

        Args:
            messages: Conversation messages
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Tool definitions
            **kwargs: Additional parameters

        Returns:
            CompletionResponse

        Raises:
            ProviderError: If all providers fail
        """
        last_error = None

        for attempt in range(self.config.max_retries):
            instance = await self.select_provider()

            if instance is None:
                logger.error(f"No healthy providers available in pool '{self.name}'")
                raise RuntimeError(f"No healthy providers available in pool '{self.name}'")

            # Acquire connection
            if not instance.acquire_connection():
                logger.warning(f"Provider {instance.provider_id} at capacity, trying next")
                continue

            start_time = time.time()

            try:
                # Call provider
                response = await instance.provider.chat(
                    messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    **kwargs,
                )

                # Record success
                latency_ms = (time.time() - start_time) * 1000
                instance.health_monitor.record_success(latency_ms)

                return response

            except Exception as e:
                # Record failure
                instance.health_monitor.record_failure()
                last_error = e

                logger.warning(
                    f"Provider {instance.provider_id} failed: {e}. "
                    f"Attempt {attempt + 1}/{self.config.max_retries}"
                )

                # Retry with backoff
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2**attempt))

            finally:
                instance.release_connection()

        # All retries exhausted
        if last_error:
            raise RuntimeError(f"All providers in pool '{self.name}' failed") from last_error

        raise RuntimeError(f"Provider pool '{self.name}' exhausted retries")

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[list[Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat completion through the pool.

        Args:
            messages: Conversation messages
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Tool definitions
            **kwargs: Additional parameters

        Yields:
            StreamChunk objects

        Raises:
            ProviderError: If all providers fail
        """
        instance = await self.select_provider()

        if instance is None:
            raise RuntimeError(f"No healthy providers available in pool '{self.name}'")

        if not instance.acquire_connection():
            raise RuntimeError(f"Provider {instance.provider_id} at capacity")

        start_time = time.time()
        success = False

        try:
            async for chunk in instance.provider.stream(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                **kwargs,
            ):
                yield chunk
            success = True

        finally:
            latency_ms = (time.time() - start_time) * 1000
            if success:
                instance.health_monitor.record_success(latency_ms)
            else:
                instance.health_monitor.record_failure()
            instance.release_connection()

    async def _warmup_provider(self, instance: ProviderInstance) -> None:
        """Warm up a provider connection.

        Sends a lightweight request to initialize the connection and
        reduce first-request latency.

        Args:
            instance: Provider instance to warm up
        """
        logger.info(f"Warming up provider {instance.provider_id}")

        try:
            # Send a simple completion request
            await asyncio.wait_for(
                instance.provider.chat(
                    [Message(role="user", content="Hi")],
                    model=instance.provider.name,
                    max_tokens=1,
                ),
                timeout=10.0,
            )

            logger.info(f"Warmed up provider {instance.provider_id}")
        except Exception as e:
            logger.warning(f"Failed to warm up provider {instance.provider_id}: {e}")

    async def warmup_all(self) -> None:
        """Warm up all enabled providers in the pool."""
        if not self.config.enable_warmup:
            return

        instances = [inst for inst in self._instances.values() if inst.enabled]

        # Warm up concurrently
        semaphore = asyncio.Semaphore(self.config.warmup_concurrency)

        async def warmup_with_limit(instance: ProviderInstance) -> None:
            async with semaphore:
                await self._warmup_provider(instance)

        tasks = [warmup_with_limit(inst) for inst in instances]
        await asyncio.gather(*tasks, return_exceptions=True)

    def get_pool_stats(self) -> dict[str, Any]:
        """Get comprehensive pool statistics.

        Returns:
            Dictionary of pool metrics and status
        """
        instances = list(self._instances.values())
        healthy = [inst for inst in instances if inst.is_healthy]
        can_accept_traffic = [inst for inst in instances if inst.can_accept_traffic]

        total_connections = sum(inst.active_connections for inst in instances)
        total_requests = sum(inst.total_connections for inst in instances)

        return {
            "pool_name": self.name,
            "config": {
                "load_balancer": self.config.load_balancer.value,
                "pool_size": self.config.pool_size,
                "min_instances": self.config.min_instances,
                "max_retries": self.config.max_retries,
            },
            "instances": {
                "total": len(instances),
                "healthy": len(healthy),
                "accepting_traffic": len(can_accept_traffic),
            },
            "connections": {
                "active": total_connections,
                "total": total_requests,
            },
            "providers": [inst.get_stats() for inst in instances],
        }

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on all providers.

        Returns:
            Health status for each provider
        """
        if self._health_registry:
            # get_all_stats is not async, don't await it
            return self._health_registry.get_all_stats()
        return {}

    async def close(self) -> None:
        """Close all providers and cleanup resources."""
        async with self._lock:
            # Close all provider instances
            for instance in self._instances.values():
                try:
                    await instance.provider.close()
                except Exception as e:
                    logger.error(f"Error closing provider {instance.provider_id}: {e}")

            # Unregister health monitors
            if self._health_registry:
                for provider_id in list(self._instances.keys()):
                    await self._health_registry.unregister(provider_id)

            self._instances.clear()
            self._initialized = False
            logger.info(f"Closed provider pool '{self.name}'")


async def create_provider_pool(
    name: str,
    providers: dict[str, BaseProvider],
    config: Optional[ProviderPoolConfig] = None,
) -> ProviderPool:
    """Create and initialize a provider pool.

    Convenience function that creates a pool and adds all providers.

    Args:
        name: Pool name
        providers: Dictionary mapping provider_id to BaseProvider
        config: Pool configuration

    Returns:
        Initialized ProviderPool
    """
    pool = ProviderPool(name=name, config=config)
    await pool.initialize()

    # Add all providers
    for provider_id, provider in providers.items():
        await pool.add_provider(provider_id, provider)

    # Warm up if enabled
    if config and config.enable_warmup:
        await pool.warmup_all()

    return pool
