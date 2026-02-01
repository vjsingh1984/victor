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

"""Provider-related protocols.

This module contains protocols related to LLM provider management, health monitoring,
and provider switching. These protocols define contracts for:

- Provider health monitoring and lifecycle management
- Provider switching and model selection
- Tool adapter coordination
- Provider event emission
- Provider classification

Usage:
    from victor.protocols.agent_providers import (
        IProviderHealthMonitor,
        IProviderSwitcher,
        IToolAdapterCoordinator,
    )
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


# =============================================================================
# Provider Management Refinements
# =============================================================================


@runtime_checkable
class IProviderHealthMonitor(Protocol):
    """Protocol for provider health monitoring.

    Defines interface for monitoring provider health and triggering fallbacks.
    Separated from IProviderSwitcher to follow ISP.
    """

    async def check_health(self, provider: Any) -> bool:
        """Check if provider is healthy.

        Args:
            provider: Provider instance to check

        Returns:
            True if provider is healthy, False otherwise
        """
        ...

    async def start_health_checks(
        self,
        interval: Optional[float] = None,
        provider: Optional[Any] = None,
        provider_name: Optional[str] = None,
    ) -> None:
        """Start periodic health checks.

        Args:
            interval: Interval between health checks in seconds
            provider: Provider to monitor (optional)
            provider_name: Provider name (optional)
        """
        ...

    async def stop_health_checks(self) -> None:
        """Stop health checks."""
        ...

    def is_monitoring(self) -> bool:
        """Check if health monitoring is currently active.

        Returns:
            True if monitoring is active, False otherwise
        """
        ...


@runtime_checkable
class IProviderSwitcher(Protocol):
    """Protocol for provider switching operations.

    Defines interface for switching between providers and models.
    Separated from IProviderHealthMonitor to follow ISP.
    """

    def get_current_provider(self) -> Optional[Any]:
        """Get current provider instance.

        Returns:
            Current provider or None if not configured
        """
        ...

    def get_current_model(self) -> str:
        """Get current model name.

        Returns:
            Current model name or empty string if not configured
        """
        ...

    def get_current_state(self) -> Optional[Any]:
        """Get current switcher state.

        Returns:
            Current state or None if not configured
        """
        ...

    def set_initial_state(
        self,
        provider: Any,
        provider_name: str,
        model: str,
    ) -> None:
        """Set initial provider state (used during initialization).

        Args:
            provider: Provider instance
            provider_name: Provider name
            model: Model name
        """
        ...

    async def switch_provider(
        self,
        provider_name: str,
        model: str,
        reason: str = "manual",
        settings: Optional[Any] = None,
        **provider_kwargs: Any,
    ) -> bool:
        """Switch to a different provider/model.

        Args:
            provider_name: Name of provider to switch to
            model: Model identifier
            reason: Reason for switch (default "manual")
            settings: Optional settings for provider configuration
            **provider_kwargs: Additional provider arguments

        Returns:
            True if switch succeeded, False otherwise
        """
        ...

    async def switch_model(self, model: str, reason: str = "manual") -> bool:
        """Switch to a different model on current provider.

        Args:
            model: Model identifier
            reason: Reason for the switch

        Returns:
            True if switch succeeded, False otherwise
        """
        ...

    def get_switch_history(self) -> list[dict[str, Any]]:
        """Get history of provider switches.

        Returns:
            List of switch event dictionaries
        """
        ...


@runtime_checkable
class IToolAdapterCoordinator(Protocol):
    """Protocol for tool adapter coordination.

    Defines interface for initializing and managing tool adapters.
    Separated to allow independent testing and mocking.
    """

    def initialize_adapter(self) -> Any:
        """Initialize tool adapter for current provider.

        Returns:
            ToolCallingCapabilities instance

        Raises:
            ValueError: If no provider is configured
        """
        ...

    def get_capabilities(self) -> Any:
        """Get tool calling capabilities.

        Returns:
            ToolCallingCapabilities instance

        Raises:
            ValueError: If adapter not initialized
        """
        ...

    def get_adapter(self) -> Any:
        """Get current tool adapter instance.

        Returns:
            Tool adapter instance

        Raises:
            ValueError: If adapter not initialized
        """
        ...

    def is_initialized(self) -> bool:
        """Check if adapter has been initialized.

        Returns:
            True if adapter is initialized, False otherwise
        """
        ...


@runtime_checkable
class IProviderEventEmitter(Protocol):
    """Protocol for provider-related events.

    Defines interface for emitting and handling provider events.
    Separated to support different event implementations.
    """

    def emit_switch_event(self, event: dict[str, Any]) -> None:
        """Emit provider switch event.

        Args:
            event: Event dictionary with switch details
        """
        ...

    def on_switch(self, callback: Any) -> None:
        """Register callback for provider switches.

        Args:
            callback: Callable to invoke on switch
        """
        ...


@runtime_checkable
class IProviderClassificationStrategy(Protocol):
    """Protocol for provider classification.

    Defines interface for classifying providers by type.
    Supports Open/Closed Principle via strategy pattern.
    """

    def is_cloud_provider(self, provider_name: str) -> bool:
        """Check if provider is cloud-based.

        Args:
            provider_name: Name of the provider

        Returns:
            True if cloud provider, False otherwise
        """
        ...

    def is_local_provider(self, provider_name: str) -> bool:
        """Check if provider is local.

        Args:
            provider_name: Name of the provider

        Returns:
            True if local provider, False otherwise
        """
        ...

    def get_provider_type(self, provider_name: str) -> str:
        """Get provider type category.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider type ("cloud", "local", "unknown")
        """
        ...


@runtime_checkable
class ProviderRegistryProtocol(Protocol):
    """Protocol for provider registry.

    Manages available LLM providers.
    """

    def register(self, name: str, provider_class: Any) -> None:
        """Register a provider.

        Args:
            name: Provider name
            provider_class: Provider class
        """
        ...

    def get(self, name: str) -> Any:
        """Get a provider class by name.

        Args:
            name: Provider name

        Returns:
            Provider class
        """
        ...

    def list_providers(self) -> list[str]:
        """Get list of registered provider names.

        Returns:
            List of provider names
        """
        ...


__all__ = [
    # Provider management refinements
    "IProviderHealthMonitor",
    "IProviderSwitcher",
    "IToolAdapterCoordinator",
    "IProviderEventEmitter",
    "IProviderClassificationStrategy",
    # Provider registry
    "ProviderRegistryProtocol",
]
