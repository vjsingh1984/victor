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

"""Load balancing strategies for provider pool.

This module provides multiple load balancing strategies for distributing
requests across provider instances:
- RoundRobin: Sequential distribution
- LeastConnections: Route to provider with fewest active requests
- Adaptive: Route based on performance metrics (latency, error rate)
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.providers.health_monitor import HealthMonitor, HealthStatus

logger = logging.getLogger(__name__)


class LoadBalancerType(Enum):
    """Available load balancing strategies."""

    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    ADAPTIVE = "adaptive"
    RANDOM = "random"


@dataclass
class ProviderInstance:
    """Represents a provider instance in the pool.

    Attributes:
        provider_id: Unique identifier
        provider: BaseProvider instance
        health_monitor: HealthMonitor for this instance
        active_connections: Current number of active requests
        total_connections: Total connections handled
        last_used: Timestamp of last request
        weight: Weight for weighted load balancing (higher = more traffic)
        enabled: Whether this instance is enabled
    """

    provider_id: str
    provider: Any  # BaseProvider
    health_monitor: "HealthMonitor"
    active_connections: int = 0
    total_connections: int = 0
    last_used: float = 0.0
    weight: float = 1.0
    enabled: bool = True

    @property
    def is_healthy(self) -> bool:
        """Check if provider is healthy."""
        return self.enabled and self.health_monitor.is_healthy

    @property
    def can_accept_traffic(self) -> bool:
        """Check if provider can accept new traffic."""
        return self.enabled and self.health_monitor.can_accept_traffic

    def acquire_connection(self) -> bool:
        """Attempt to acquire a connection slot.

        Returns:
            True if connection acquired, False if at capacity
        """
        if not self.can_accept_traffic:
            return False

        self.active_connections += 1
        self.total_connections += 1
        self.last_used = time.time()
        return True

    def release_connection(self) -> None:
        """Release a connection slot."""
        if self.active_connections > 0:
            self.active_connections -= 1

    def get_stats(self) -> Dict[str, Any]:
        """Get instance statistics.

        Returns:
            Dictionary of instance metrics
        """
        return {
            "provider_id": self.provider_id,
            "active_connections": self.active_connections,
            "total_connections": self.total_connections,
            "last_used": self.last_used,
            "weight": self.weight,
            "enabled": self.enabled,
            "is_healthy": self.is_healthy,
            "can_accept_traffic": self.can_accept_traffic,
            "health_status": self.health_monitor.status.value,
        }


class LoadBalancer(ABC):
    """Abstract base class for load balancing strategies.

    Load balancers select the best provider instance for each request
    based on their specific strategy.
    """

    def __init__(self, name: str):
        """Initialize load balancer.

        Args:
            name: Load balancer name for logging
        """
        self.name = name
        self._lock = asyncio.Lock()

    @abstractmethod
    async def select_provider(
        self,
        instances: List[ProviderInstance],
    ) -> Optional[ProviderInstance]:
        """Select the best provider instance for a request.

        Args:
            instances: List of available provider instances

        Returns:
            Selected ProviderInstance or None if no healthy instances
        """
        pass

    def _filter_healthy_instances(
        self,
        instances: List[ProviderInstance],
    ) -> List[ProviderInstance]:
        """Filter to only healthy instances that can accept traffic.

        Args:
            instances: List of provider instances

        Returns:
            Filtered list of healthy instances
        """
        return [inst for inst in instances if inst.can_accept_traffic]

    async def select_with_fallback(
        self,
        instances: List[ProviderInstance],
        max_attempts: int = 3,
    ) -> Optional[ProviderInstance]:
        """Select provider with automatic fallback on failure.

        Args:
            instances: List of available provider instances
            max_attempts: Maximum selection attempts

        Returns:
            Selected ProviderInstance or None
        """
        for attempt in range(max_attempts):
            selected = await self.select_provider(instances)
            if selected is not None:
                return selected

            logger.warning(
                f"Load balancer {self.name}: No healthy providers "
                f"(attempt {attempt + 1}/{max_attempts})"
            )
            await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff

        return None


class RoundRobinLoadBalancer(LoadBalancer):
    """Round-robin load balancing strategy.

    Distributes requests sequentially across all healthy providers.
    Simple and predictable, ensuring even distribution.
    """

    def __init__(self, name: str = "round_robin"):
        """Initialize round-robin load balancer.

        Args:
            name: Load balancer name
        """
        super().__init__(name)
        self._current_index = 0

    async def select_provider(
        self,
        instances: List[ProviderInstance],
    ) -> Optional[ProviderInstance]:
        """Select next provider in round-robin fashion.

        Args:
            instances: List of available provider instances

        Returns:
            Selected ProviderInstance or None
        """
        healthy = self._filter_healthy_instances(instances)

        if not healthy:
            return None

        async with self._lock:
            selected = healthy[self._current_index % len(healthy)]
            self._current_index += 1
            return selected


class LeastConnectionsLoadBalancer(LoadBalancer):
    """Least-connections load balancing strategy.

    Routes requests to the provider with the fewest active connections.
    Helps prevent any single provider from becoming overloaded.
    """

    def __init__(self, name: str = "least_connections"):
        """Initialize least-connections load balancer.

        Args:
            name: Load balancer name
        """
        super().__init__(name)

    async def select_provider(
        self,
        instances: List[ProviderInstance],
    ) -> Optional[ProviderInstance]:
        """Select provider with fewest active connections.

        Args:
            instances: List of available provider instances

        Returns:
            Selected ProviderInstance or None
        """
        healthy = self._filter_healthy_instances(instances)

        if not healthy:
            return None

        # Sort by active connections, then by weight
        sorted_instances = sorted(
            healthy,
            key=lambda inst: (inst.active_connections / inst.weight, -inst.weight),
        )

        return sorted_instances[0]


class AdaptiveLoadBalancer(LoadBalancer):
    """Adaptive load balancing based on performance metrics.

    Selects providers based on a combination of:
    - Current latency (prefer lower latency)
    - Error rate (prefer lower error rate)
    - Active connections (prefer fewer connections)
    - Recent performance trends

    Uses a weighted scoring system to dynamically adapt to changing conditions.
    """

    def __init__(
        self,
        name: str = "adaptive",
        latency_weight: float = 0.4,
        error_rate_weight: float = 0.3,
        connections_weight: float = 0.2,
        trend_weight: float = 0.1,
    ):
        """Initialize adaptive load balancer.

        Args:
            name: Load balancer name
            latency_weight: Weight for latency in scoring (0-1)
            error_rate_weight: Weight for error rate in scoring (0-1)
            connections_weight: Weight for connections in scoring (0-1)
            trend_weight: Weight for performance trend in scoring (0-1)
        """
        super().__init__(name)
        self.latency_weight = latency_weight
        self.error_rate_weight = error_rate_weight
        self.connections_weight = connections_weight
        self.trend_weight = trend_weight

    async def select_provider(
        self,
        instances: List[ProviderInstance],
    ) -> Optional[ProviderInstance]:
        """Select provider based on adaptive scoring.

        Args:
            instances: List of available provider instances

        Returns:
            Selected ProviderInstance or None
        """
        healthy = self._filter_healthy_instances(instances)

        if not healthy:
            return None

        # Score each provider
        scored_instances = []
        for instance in healthy:
            score = await self._calculate_score(instance)
            scored_instances.append((score, instance))

        # Select highest score
        scored_instances.sort(key=lambda x: x[0], reverse=True)
        return scored_instances[0][1]

    async def _calculate_score(self, instance: ProviderInstance) -> float:
        """Calculate adaptive score for a provider instance.

        Higher score = better candidate for routing.

        Score components:
        - Latency score: Lower latency = higher score
        - Error rate score: Lower error rate = higher score
        - Connections score: Fewer connections = higher score
        - Trend score: Improving performance = higher score

        Args:
            instance: Provider instance to score

        Returns:
            Score between 0 and 1
        """
        metrics = instance.health_monitor.metrics

        # Latency score (0-1)
        latency_score = self._calculate_latency_score(metrics.avg_latency_ms)

        # Error rate score (0-1)
        error_score = 1.0 - metrics.error_rate

        # Connections score (0-1)
        # Assume max reasonable connections is 50
        connections_score = max(0, 1.0 - (instance.active_connections / 50.0))

        # Trend score based on recent performance
        trend_score = self._calculate_trend_score(metrics)

        # Weighted combination
        total_score = (
            (latency_score * self.latency_weight)
            + (error_score * self.error_rate_weight)
            + (connections_score * self.connections_weight)
            + (trend_score * self.trend_weight)
        )

        # Apply weight multiplier
        return total_score * instance.weight

    def _calculate_latency_score(self, avg_latency_ms: float) -> float:
        """Calculate latency score.

        Args:
            avg_latency_ms: Average latency in milliseconds

        Returns:
            Score between 0 and 1 (higher is better)
        """
        # Assume latency <= 100ms is excellent, >= 5000ms is poor
        if avg_latency_ms <= 100:
            return 1.0
        if avg_latency_ms >= 5000:
            return 0.0
        # Linear interpolation
        return 1.0 - ((avg_latency_ms - 100) / 4900)

    def _calculate_trend_score(self, metrics: Any) -> float:
        """Calculate trend score based on recent performance.

        Args:
            metrics: HealthMetrics instance

        Returns:
            Score between 0 and 1 (higher is better)
        """
        # Factor in consecutive successes/failures
        if metrics.consecutive_successes > 5:
            return 1.0
        if metrics.consecutive_failures > 2:
            return 0.0

        # Neutral score if not enough data
        return 0.5


class RandomLoadBalancer(LoadBalancer):
    """Random load balancing strategy.

    Selects a random healthy provider. Useful for testing or when
    all providers are equally capable.
    """

    def __init__(self, name: str = "random"):
        """Initialize random load balancer.

        Args:
            name: Load balancer name
        """
        super().__init__(name)

    async def select_provider(
        self,
        instances: List[ProviderInstance],
    ) -> Optional[ProviderInstance]:
        """Select a random healthy provider.

        Args:
            instances: List of available provider instances

        Returns:
            Selected ProviderInstance or None
        """
        import random

        healthy = self._filter_healthy_instances(instances)

        if not healthy:
            return None

        return random.choice(healthy)


def create_load_balancer(
    strategy: LoadBalancerType,
    name: Optional[str] = None,
    **kwargs: Any,
) -> LoadBalancer:
    """Factory function to create load balancer.

    Args:
        strategy: Load balancing strategy
        name: Optional custom name (defaults to strategy name)
        **kwargs: Additional arguments for specific balancers

    Returns:
        LoadBalancer instance

    Raises:
        ValueError: If strategy is unknown
    """
    # Validate strategy type first
    if not isinstance(strategy, LoadBalancerType):
        raise ValueError(f"Unknown load balancer strategy: {strategy}")

    balancer_name = name or strategy.value

    if strategy == LoadBalancerType.ROUND_ROBIN:
        return RoundRobinLoadBalancer(name=balancer_name)
    elif strategy == LoadBalancerType.LEAST_CONNECTIONS:
        return LeastConnectionsLoadBalancer(name=balancer_name)
    elif strategy == LoadBalancerType.ADAPTIVE:
        return AdaptiveLoadBalancer(name=balancer_name, **kwargs)
    elif strategy == LoadBalancerType.RANDOM:
        return RandomLoadBalancer(name=balancer_name)
    else:
        raise ValueError(f"Unknown load balancer strategy: {strategy}")
