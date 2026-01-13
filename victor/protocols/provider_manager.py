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

"""Provider manager protocol for dependency inversion.

This module defines the IProviderManager protocol that enables
dependency injection for LLM provider management, following the
Dependency Inversion Principle (DIP).

Design Principles:
    - DIP: High-level modules depend on this protocol, not concrete implementations
    - OCP: New provider types can be added without modifying existing code
    - SRP: Protocol contains only provider lifecycle methods

Usage:
    class ProviderManager(IProviderManager):
        async def get_provider(self, provider_name: str) -> IProvider:
            return self._providers[provider_name]

        async def switch_provider(self, from_provider: str, to_provider: str, model: Optional[str]) -> SwitchResult:
            # Switch provider with validation
            ...

        async def check_health(self, provider_name: str) -> HealthStatus:
            # Health check implementation
            ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@dataclass
class SwitchResult:
    """Result from provider switch operation.

    Attributes:
        success: Whether the switch succeeded
        from_provider: Previous provider name
        to_provider: New provider name
        model: Model selected for new provider
        error_message: Error message if switch failed
        metadata: Additional metadata about the switch
    """

    success: bool
    from_provider: str
    to_provider: str
    model: str | None = None
    error_message: str | None = None
    metadata: Dict[str, Any] | None = None


@dataclass
class HealthStatus:
    """Health status of a provider.

    Attributes:
        provider_name: Name of the provider
        healthy: Whether provider is healthy
        latency_ms: Current latency in milliseconds
        error_message: Error message if unhealthy
        last_check: Timestamp of last health check
        metadata: Additional health metadata
    """

    provider_name: str
    healthy: bool
    latency_ms: float | None = None
    error_message: str | None = None
    last_check: str | None = None
    metadata: Dict[str, Any] | None = None


@runtime_checkable
class IProviderManager(Protocol):
    """Protocol for LLM provider lifecycle management.

    Implementations manage provider instances, handle switching
    between providers, and monitor provider health.

    Responsibilities:
    - Provider instance management
    - Provider switching with validation
    - Health checking
    - Capability discovery
    """

    async def get_provider(self, provider_name: str) -> "IProvider":
        """Get provider instance by name.

        Args:
            provider_name: Name of the provider (e.g., 'anthropic', 'openai')

        Returns:
            Provider instance

        Raises:
            ProviderNotFoundError: If provider not registered
            ProviderError: If provider initialization fails

        Example:
            provider = await manager.get_provider("anthropic")
        """
        ...

    async def switch_provider(
        self,
        from_provider: str,
        to_provider: str,
        model: Optional[str] = None,
    ) -> SwitchResult:
        """Switch from one provider to another.

        Validates the switch, initializes the new provider, and
        handles any necessary cleanup or state migration.

        Args:
            from_provider: Current provider name
            to_provider: Target provider name
            model: Model to use with new provider (None = default)

        Returns:
            SwitchResult with success status and metadata

        Example:
            result = await manager.switch_provider(
                from_provider="anthropic",
                to_provider="openai",
                model="gpt-4-turbo"
            )
        """
        ...

    async def check_health(self, provider_name: str) -> HealthStatus:
        """Check health of a provider.

        Performs a health check by making a test request
        or pinging the provider's API endpoint.

        Args:
            provider_name: Name of the provider to check

        Returns:
            HealthStatus with health information

        Example:
            status = await manager.check_health("anthropic")
            if status.healthy:
                print(f"Latency: {status.latency_ms}ms")
        """
        ...


__all__ = ["IProviderManager", "SwitchResult", "HealthStatus"]
