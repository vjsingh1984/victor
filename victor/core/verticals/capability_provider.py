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

"""Capability provider protocol and registry for DI-based capability management.

This module defines the protocol-based architecture for capability providers,
enabling SOLID-compliant dependency injection for vertical capabilities.

Design Principles:
- ISP: Narrow, focused capability protocols
- DIP: Depend on capability abstractions, not concretions
- OCP: Open for extension via registration
- SRP: Each provider handles one capability type
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from victor.core.container import ServiceContainer

logger = logging.getLogger(__name__)


# =============================================================================
# Capability Provider Protocol (ISP + DIP)
# =============================================================================


@runtime_checkable
class ICapabilityProvider(Protocol):
    """Protocol for capability providers.

    This narrow protocol follows the Interface Segregation Principle (ISP)
    by defining only the essential methods for a capability provider.

    All capability providers must implement this protocol for DI compatibility.
    """

    @property
    def name(self) -> str:
        """Get the capability name."""
        ...

    def get_instance(self) -> Any:
        """Get or create the capability instance."""
        ...

    def reset(self) -> None:
        """Reset the capability instance (for testing)."""
        ...


@runtime_checkable
class IConfigurableCapability(Protocol):
    """Protocol for capabilities that accept configuration.

    This extends ISP by allowing capabilities to be configured
    without forcing all capabilities to implement configuration.
    """

    def configure(self, **kwargs: Any) -> None:
        """Configure the capability with parameters."""
        ...


# =============================================================================
# Base Provider Class (SRP + Template Method Pattern)
# =============================================================================


class BaseCapabilityProvider(ABC):
    """Base class for capability providers.

    Implements the Template Method pattern for consistent provider behavior
    while allowing subclasses to customize instance creation.

    Subclasses only need to implement _create_instance().

    Design Principles:
    - SRP: Handles only provider lifecycle management
    - OCP: Extend by subclassing, not modifying
    - Template Method: Defines algorithm skeleton, defers steps to subclasses
    """

    def __init__(
        self,
        name: str,
        container: Optional["ServiceContainer"] = None,
    ) -> None:
        """Initialize the capability provider.

        Args:
            name: Unique capability name
            container: Optional DI container for service resolution
        """
        self._name = name
        self._container = container
        self._instance: Optional[Any] = None
        logger.debug(f"CapabilityProvider initialized: {name}")

    @property
    def name(self) -> str:
        """Get the capability name."""
        return self._name

    def get_instance(self) -> Any:
        """Get or create the capability instance.

        Implements lazy initialization pattern.

        Returns:
            Capability instance
        """
        if self._instance is None:
            self._instance = self._create_instance()
            logger.debug(f"Created capability instance: {self._name}")
        return self._instance

    def reset(self) -> None:
        """Reset the capability instance.

        Used primarily for testing to ensure clean state.
        """
        self._instance = None
        logger.debug(f"Reset capability instance: {self._name}")

    @abstractmethod
    def _create_instance(self) -> Any:
        """Create the capability instance.

        Subclasses must implement this method to define how
        the capability is instantiated.

        Returns:
            New capability instance
        """
        ...


# =============================================================================
# Capability Registry (OCP + DIP)
# =============================================================================


class CapabilityProviderRegistry:
    """Registry for capability providers.

    Manages dynamic registration and retrieval of capability providers.
    Follows Open/Closed Principle by allowing registration of new
    providers without modifying existing code.

    Design Principles:
    - SRP: Manages provider registration only
    - OCP: Open for extension via registration
    - DIP: Works with ICapabilityProvider abstraction
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._providers: dict[str, ICapabilityProvider] = {}

    def register(self, provider: ICapabilityProvider) -> None:
        """Register a capability provider.

        Args:
            provider: Provider to register

        Raises:
            ValueError: If provider name already registered
        """
        name = provider.name
        if name in self._providers:
            raise ValueError(f"Capability '{name}' already registered")
        self._providers[name] = provider
        logger.debug(f"Registered capability provider: {name}")

    def unregister(self, name: str) -> bool:
        """Unregister a capability provider.

        Args:
            name: Provider name to unregister

        Returns:
            True if unregistered, False if not found
        """
        if name in self._providers:
            del self._providers[name]
            logger.debug(f"Unregistered capability provider: {name}")
            return True
        return False

    def get_provider(self, name: str) -> Optional[ICapabilityProvider]:
        """Get a registered capability provider.

        Args:
            name: Provider name

        Returns:
            Provider instance or None if not found
        """
        return self._providers.get(name)

    def has_provider(self, name: str) -> bool:
        """Check if a provider is registered.

        Args:
            name: Provider name

        Returns:
            True if registered
        """
        return name in self._providers

    def list_providers(self) -> list[str]:
        """List all registered provider names.

        Returns:
            List of provider names
        """
        return list(self._providers.keys())

    def clear(self) -> None:
        """Clear all registered providers.

        Used primarily for testing.
        """
        self._providers.clear()
        logger.debug("Cleared all capability providers")

    def reset_all(self) -> None:
        """Reset all provider instances.

        Calls reset() on all registered providers.
        Used primarily for testing.
        """
        for provider in self._providers.values():
            provider.reset()
        logger.debug("Reset all capability provider instances")


# =============================================================================
# Global Registry Instance
# =============================================================================

_global_registry: Optional[CapabilityProviderRegistry] = None


def get_capability_registry() -> CapabilityProviderRegistry:
    """Get the global capability provider registry.

    Returns:
        Global CapabilityProviderRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = CapabilityProviderRegistry()
    return _global_registry


def reset_global_registry() -> None:
    """Reset the global registry.

    Used primarily for testing to ensure clean state.
    """
    global _global_registry
    _global_registry = None
    logger.debug("Reset global capability registry")


__all__ = [
    # Protocols
    "ICapabilityProvider",
    "IConfigurableCapability",
    # Base class
    "BaseCapabilityProvider",
    # Registry
    "CapabilityProviderRegistry",
    "get_capability_registry",
    "reset_global_registry",
]
