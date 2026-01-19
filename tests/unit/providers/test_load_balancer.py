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

"""Unit tests for load balancing strategies."""

import asyncio
import pytest
from unittest.mock import Mock

from victor.providers.load_balancer import (
    AdaptiveLoadBalancer,
    LeastConnectionsLoadBalancer,
    LoadBalancerType,
    ProviderInstance,
    RandomLoadBalancer,
    RoundRobinLoadBalancer,
    create_load_balancer,
)
from victor.providers.health_monitor import HealthMonitor, HealthStatus


@pytest.fixture
def mock_provider():
    """Create mock provider."""
    provider = Mock()
    provider.name = "test-provider"
    return provider


@pytest.fixture
def health_monitor():
    """Create health monitor."""
    return HealthMonitor(provider_id="test")


@pytest.fixture
def provider_instance(mock_provider, health_monitor):
    """Create provider instance."""
    return ProviderInstance(
        provider_id="test-1",
        provider=mock_provider,
        health_monitor=health_monitor,
    )


@pytest.fixture
def multiple_instances(mock_provider):
    """Create multiple provider instances."""
    instances = []
    for i in range(3):
        monitor = HealthMonitor(provider_id=f"test-{i}")
        instances.append(
            ProviderInstance(
                provider_id=f"test-{i}",
                provider=mock_provider,
                health_monitor=monitor,
            )
        )
    return instances


class TestProviderInstance:
    """Tests for ProviderInstance."""

    def test_initial_state(self, provider_instance) -> None:
        """Test initial instance state."""
        assert provider_instance.provider_id == "test-1"
        assert provider_instance.active_connections == 0
        assert provider_instance.is_healthy
        assert provider_instance.can_accept_traffic

    def test_acquire_connection(self, provider_instance) -> None:
        """Test acquiring connection."""
        assert provider_instance.acquire_connection()
        assert provider_instance.active_connections == 1

    def test_acquire_connection_unhealthy(self, provider_instance) -> None:
        """Test acquiring connection when unhealthy."""
        provider_instance.health_monitor.set_status(HealthStatus.UNHEALTHY)
        assert not provider_instance.acquire_connection()
        assert provider_instance.active_connections == 0

    def test_release_connection(self, provider_instance) -> None:
        """Test releasing connection."""
        provider_instance.acquire_connection()
        provider_instance.release_connection()
        assert provider_instance.active_connections == 0

    def test_disabled_instance(self, provider_instance) -> None:
        """Test disabled instance."""
        provider_instance.enabled = False
        assert not provider_instance.can_accept_traffic
        assert not provider_instance.acquire_connection()

    def test_get_stats(self, provider_instance) -> None:
        """Test getting instance statistics."""
        provider_instance.acquire_connection()
        stats = provider_instance.get_stats()

        assert stats["provider_id"] == "test-1"
        assert stats["active_connections"] == 1
        assert stats["is_healthy"] is True


class TestRoundRobinLoadBalancer:
    """Tests for RoundRobinLoadBalancer."""

    @pytest.mark.asyncio
    async def test_round_robin_selection(self, multiple_instances) -> None:
        """Test round-robin provider selection."""
        balancer = RoundRobinLoadBalancer()

        selected_1 = await balancer.select_provider(multiple_instances)
        selected_2 = await balancer.select_provider(multiple_instances)
        selected_3 = await balancer.select_provider(multiple_instances)
        selected_4 = await balancer.select_provider(multiple_instances)

        # Should cycle through instances
        assert selected_1.provider_id == "test-0"
        assert selected_2.provider_id == "test-1"
        assert selected_3.provider_id == "test-2"
        assert selected_4.provider_id == "test-0"  # Back to start

    @pytest.mark.asyncio
    async def test_filters_unhealthy(self, multiple_instances) -> None:
        """Test filtering unhealthy instances."""
        balancer = RoundRobinLoadBalancer()

        # Mark one as unhealthy
        multiple_instances[1].health_monitor.set_status(HealthStatus.UNHEALTHY)

        selected = await balancer.select_provider(multiple_instances)
        assert selected.provider_id != "test-1"

    @pytest.mark.asyncio
    async def test_no_healthy_instances(self, multiple_instances) -> None:
        """Test when no healthy instances available."""
        balancer = RoundRobinLoadBalancer()

        # Mark all as unhealthy
        for instance in multiple_instances:
            instance.health_monitor.set_status(HealthStatus.UNHEALTHY)

        selected = await balancer.select_provider(multiple_instances)
        assert selected is None


class TestLeastConnectionsLoadBalancer:
    """Tests for LeastConnectionsLoadBalancer."""

    @pytest.mark.asyncio
    async def test_selects_least_loaded(self, multiple_instances) -> None:
        """Test selecting instance with fewest connections."""
        balancer = LeastConnectionsLoadBalancer()

        # Add connections to first instance
        multiple_instances[0].acquire_connection()
        multiple_instances[0].acquire_connection()
        multiple_instances[1].acquire_connection()

        selected = await balancer.select_provider(multiple_instances)
        assert selected.provider_id == "test-2"  # No connections

    @pytest.mark.asyncio
    async def test_respects_weight(self, multiple_instances) -> None:
        """Test weight consideration in selection."""
        balancer = LeastConnectionsLoadBalancer()

        # Give first instance higher weight
        multiple_instances[0].weight = 2.0
        multiple_instances[1].weight = 1.0
        multiple_instances[2].weight = 1.0

        # Add same connections to all
        for instance in multiple_instances:
            instance.acquire_connection()

        selected = await balancer.select_provider(multiple_instances)
        # First should be selected due to higher weight (connections/weight is lower)
        assert selected.provider_id == "test-0"

    @pytest.mark.asyncio
    async def test_filters_unhealthy(self, multiple_instances) -> None:
        """Test filtering unhealthy instances."""
        balancer = LeastConnectionsLoadBalancer()

        # Mark middle instance as unhealthy
        multiple_instances[1].health_monitor.set_status(HealthStatus.UNHEALTHY)

        # Add connections to first
        multiple_instances[0].acquire_connection()

        selected = await balancer.select_provider(multiple_instances)
        assert selected.provider_id == "test-2"


class TestAdaptiveLoadBalancer:
    """Tests for AdaptiveLoadBalancer."""

    @pytest.mark.asyncio
    async def test_prefers_low_latency(self, multiple_instances) -> None:
        """Test preferring low latency providers."""
        balancer = AdaptiveLoadBalancer()

        # Record different latencies
        multiple_instances[0].health_monitor.record_success(100.0)
        multiple_instances[1].health_monitor.record_success(1000.0)
        multiple_instances[2].health_monitor.record_success(500.0)

        selected = await balancer.select_provider(multiple_instances)
        assert selected.provider_id == "test-0"  # Lowest latency

    @pytest.mark.asyncio
    async def test_prefers_low_error_rate(self, multiple_instances) -> None:
        """Test preferring low error rate providers."""
        balancer = AdaptiveLoadBalancer()

        # Set up similar latency but different error rates
        for i in range(10):
            multiple_instances[0].health_monitor.record_success(200.0)
            multiple_instances[1].health_monitor.record_success(200.0)
            multiple_instances[2].health_monitor.record_success(200.0)

        # Add failures to first
        multiple_instances[0].health_monitor.record_failure()
        multiple_instances[0].health_monitor.record_failure()

        selected = await balancer.select_provider(multiple_instances)
        # Should prefer instances with no errors
        assert selected.provider_id in ["test-1", "test-2"]

    @pytest.mark.asyncio
    async def test_considers_active_connections(self, multiple_instances) -> None:
        """Test considering active connections in scoring."""
        balancer = AdaptiveLoadBalancer()

        # Similar performance
        for instance in multiple_instances:
            instance.health_monitor.record_success(200.0)

        # Add connections to first two
        multiple_instances[0].acquire_connection()
        multiple_instances[0].acquire_connection()
        multiple_instances[1].acquire_connection()

        selected = await balancer.select_provider(multiple_instances)
        # Should prefer instance with fewest connections
        assert selected.provider_id == "test-2"

    @pytest.mark.asyncio
    async def test_custom_weights(self, multiple_instances) -> None:
        """Test custom scoring weights."""
        balancer = AdaptiveLoadBalancer(
            latency_weight=0.8,
            error_rate_weight=0.1,
            connections_weight=0.1,
        )

        # High latency but no errors
        for i in range(10):
            multiple_instances[0].health_monitor.record_success(1000.0)

        # Low latency but some errors
        for i in range(8):
            multiple_instances[1].health_monitor.record_success(100.0)
        for i in range(2):
            multiple_instances[1].health_monitor.record_failure()

        # Third instance with moderate latency and no errors
        for i in range(10):
            multiple_instances[2].health_monitor.record_success(500.0)

        # With high latency weight, should prefer moderate latency and no errors
        selected = await balancer.select_provider(multiple_instances)
        assert selected.provider_id == "test-2"


class TestRandomLoadBalancer:
    """Tests for RandomLoadBalancer."""

    @pytest.mark.asyncio
    async def test_random_selection(self, multiple_instances) -> None:
        """Test random provider selection."""
        balancer = RandomLoadBalancer()

        selected = await balancer.select_provider(multiple_instances)
        assert selected is not None
        assert selected.provider_id in ["test-0", "test-1", "test-2"]

    @pytest.mark.asyncio
    async def test_distribution(self, multiple_instances) -> None:
        """Test distribution over multiple selections."""
        balancer = RandomLoadBalancer()

        selections = {}
        for _ in range(100):
            selected = await balancer.select_provider(multiple_instances)
            selections[selected.provider_id] = selections.get(selected.provider_id, 0) + 1

        # All instances should be selected
        assert len(selections) == 3
        # Each should be selected roughly 1/3 of the time (allowing variance)
        for count in selections.values():
            assert 20 <= count <= 50  # Allow 20-50% variance


class TestLoadBalancerFactory:
    """Tests for load balancer factory."""

    def test_create_round_robin(self) -> None:
        """Test creating round-robin balancer."""
        balancer = create_load_balancer(LoadBalancerType.ROUND_ROBIN)
        assert isinstance(balancer, RoundRobinLoadBalancer)

    def test_create_least_connections(self) -> None:
        """Test creating least-connections balancer."""
        balancer = create_load_balancer(LoadBalancerType.LEAST_CONNECTIONS)
        assert isinstance(balancer, LeastConnectionsLoadBalancer)

    def test_create_adaptive(self) -> None:
        """Test creating adaptive balancer."""
        balancer = create_load_balancer(LoadBalancerType.ADAPTIVE)
        assert isinstance(balancer, AdaptiveLoadBalancer)

    def test_create_random(self) -> None:
        """Test creating random balancer."""
        balancer = create_load_balancer(LoadBalancerType.RANDOM)
        assert isinstance(balancer, RandomLoadBalancer)

    def test_custom_name(self) -> None:
        """Test creating balancer with custom name."""
        balancer = create_load_balancer(
            LoadBalancerType.ROUND_ROBIN,
            name="custom-lb",
        )
        assert balancer.name == "custom-lb"

    def test_unknown_strategy(self) -> None:
        """Test creating balancer with unknown strategy."""
        with pytest.raises(ValueError):
            create_load_balancer("unknown")  # type: ignore
