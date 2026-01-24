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

"""Capability injector for verticals.

Phase 1.4: Auto-inject FileOperationsCapability via DI.
Extended: Multi-capability support with registry pattern.

This module provides a capability injector that manages shared capabilities
for verticals, eliminating the need for each vertical to instantiate
capabilities independently.

Design Patterns (SOLID-compliant):
- SRP: CapabilityInjector manages injection, CapabilityProviderRegistry manages registration
- OCP: Open for extension via registry, not modification
- LSP: All capabilities implement ICapabilityProvider protocol
- ISP: Narrow capability protocols (ICapabilityProvider, IConfigurableCapability)
- DIP: Depend on protocol abstractions, not concrete implementations

Usage:
    # Get capability by name (new way - recommended)
    injector = get_capability_injector()
    file_ops = injector.get_capability("file_operations")
    web_ops = injector.get_capability("web_operations")

    # Or use typed methods (backward compatible)
    file_ops = injector.get_file_operations_capability()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from victor.core.verticals.capability_provider import (
    BaseCapabilityProvider,
    get_capability_registry,
    ICapabilityProvider,
)

if TYPE_CHECKING:
    from victor.core.container import ServiceContainer
    from victor.framework.capabilities import FileOperationsCapability

logger = logging.getLogger(__name__)

# Global singleton instance
_injector_instance: Optional["CapabilityInjector"] = None


# =============================================================================
# Built-in Capability Providers
# =============================================================================


class _FileOperationsProvider(BaseCapabilityProvider):
    """Provider for FileOperationsCapability."""

    def __init__(
        self,
        container: Optional["ServiceContainer"] = None,
    ) -> None:
        super().__init__("file_operations", container)

    def _create_instance(self) -> "FileOperationsCapability":
        from victor.framework.capabilities import FileOperationsCapability

        return FileOperationsCapability()


class _WebOperationsProvider(BaseCapabilityProvider):
    """Provider for WebOperationsCapability."""

    def __init__(
        self,
        container: Optional["ServiceContainer"] = None,
    ) -> None:
        super().__init__("web_operations", container)

    def _create_instance(self) -> Any:
        from victor.framework.capabilities import WebOperationsCapability

        return WebOperationsCapability()


class _GitOperationsProvider(BaseCapabilityProvider):
    """Provider for GitOperationsCapability."""

    def __init__(
        self,
        container: Optional["ServiceContainer"] = None,
    ) -> None:
        super().__init__("git_operations", container)

    def _create_instance(self) -> Any:
        from victor.framework.capabilities import GitOperationsCapability

        return GitOperationsCapability()


class _TestOperationsProvider(BaseCapabilityProvider):
    """Provider for TestOperationsCapability."""

    def __init__(
        self,
        container: Optional["ServiceContainer"] = None,
    ) -> None:
        super().__init__("test_operations", container)

    def _create_instance(self) -> Any:
        from victor.framework.capabilities import TestOperationsCapability

        return TestOperationsCapability()


class CapabilityInjector:
    """Injector for shared vertical capabilities.

    This class manages singleton instances of capabilities that are
    shared across multiple verticals using a registry pattern.

    Design Principles:
    - SRP: Manages capability injection and retrieval
    - OCP: Open for extension via registry registration
    - DIP: Works with ICapabilityProvider protocol
    - LSP: All capabilities are substitutable via protocol

    Usage:
        # Get capability by name (recommended)
        injector = get_capability_injector()
        file_ops = injector.get_capability("file_operations")
        web_ops = injector.get_capability("web_operations")

        # Or use typed methods (backward compatible)
        file_ops = injector.get_file_operations_capability()

        # Register custom capability
        injector.register_provider(MyCustomProvider())

        # Via DI container
        injector = container.get(CapabilityInjector)
    """

    # Built-in capability names
    CAPABILITY_FILE_OPERATIONS = "file_operations"
    CAPABILITY_WEB_OPERATIONS = "web_operations"
    CAPABILITY_GIT_OPERATIONS = "git_operations"
    CAPABILITY_TEST_OPERATIONS = "test_operations"

    def __init__(
        self,
        container: Optional["ServiceContainer"] = None,
        *,
        file_operations: Optional["FileOperationsCapability"] = None,
        auto_register_builtins: bool = True,
    ) -> None:
        """Initialize the capability injector.

        Args:
            container: Optional DI container for service resolution
            file_operations: Optional custom FileOperationsCapability instance (for testing)
            auto_register_builtins: Whether to auto-register built-in capabilities
        """
        self._container = container
        self._registry = get_capability_registry()

        # For backward compatibility, support direct injection
        self._file_ops_override: Optional["FileOperationsCapability"] = file_operations

        # Auto-register built-in capabilities
        if auto_register_builtins and not self._registry.has_provider(
            self.CAPABILITY_FILE_OPERATIONS
        ):
            self._register_builtin_capabilities()

        logger.debug("CapabilityInjector initialized")

    def _register_builtin_capabilities(self) -> None:
        """Register built-in capability providers.

        Registers standard capabilities that are commonly used across verticals.
        """
        builtins = [
            _FileOperationsProvider(self._container),
            # Note: Other capabilities are optional and may not exist
            # _WebOperationsProvider(self._container),
            # _GitOperationsProvider(self._container),
            # _TestOperationsProvider(self._container),
        ]

        for provider in builtins:
            try:
                self._registry.register(provider)
            except ValueError:
                # Already registered, skip
                pass

    def get_capability(self, name: str, default: Any = None) -> Any:
        """Get a capability by name.

        This is the recommended way to get capabilities as it supports
        dynamic capability lookup.

        Args:
            name: Capability name (e.g., "file_operations", "web_operations")
            default: Default value to return if capability not found

        Returns:
            Capability instance or default if not found

        Example:
            injector = get_capability_injector()
            file_ops = injector.get_capability("file_operations")
        """
        # Handle override for file_operations (backward compat)
        if name == self.CAPABILITY_FILE_OPERATIONS and self._file_ops_override is not None:
            return self._file_ops_override

        provider = self._registry.get_provider(name)
        if provider is None:
            logger.warning(f"Capability '{name}' not found, returning default")
            return default

        return provider.get_instance()

    def get_file_operations_capability(self) -> "FileOperationsCapability":
        """Get the FileOperationsCapability instance.

        Returns a singleton instance of FileOperationsCapability.
        Creates the instance on first call if not already provided.

        Returns:
            FileOperationsCapability instance

        Note:
            This method is kept for backward compatibility.
            New code should use get_capability("file_operations").
        """
        # Use override if provided (for testing)
        if self._file_ops_override is not None:
            return self._file_ops_override

        instance = self.get_capability(self.CAPABILITY_FILE_OPERATIONS)
        if instance is None:
            # Fallback to direct creation if registry not initialized
            from victor.framework.capabilities import FileOperationsCapability

            instance = FileOperationsCapability()
            logger.warning("FileOperationsCapability created directly (registry unavailable)")

        return instance

    def register_provider(self, provider: ICapabilityProvider) -> None:
        """Register a custom capability provider.

        Allows registration of custom capabilities beyond the built-in ones.

        Args:
            provider: Provider to register

        Example:
            from victor.core.verticals.capability_provider import BaseCapabilityProvider

            class MyCapabilityProvider(BaseCapabilityProvider):
                def _create_instance(self):
                    return MyCapability()

            injector.register_provider(MyCapabilityProvider())
        """
        self._registry.register(provider)
        logger.debug(f"Registered custom capability provider: {provider.name}")

    def has_capability(self, name: str) -> bool:
        """Check if a capability is available.

        Args:
            name: Capability name

        Returns:
            True if capability is registered
        """
        return self._registry.has_provider(name)

    def list_capabilities(self) -> list[str]:
        """List all registered capability names.

        Returns:
            List of capability names
        """
        return self._registry.list_providers()

    def reset(self) -> None:
        """Reset all capability instances.

        Clears all cached capability instances. Used primarily for testing
        to ensure clean state between tests.
        """
        self._registry.reset_all()
        logger.debug("Reset all capability instances")

    @classmethod
    def reset_global(cls) -> None:
        """Reset the global singleton instance.

        Used for testing to ensure clean state between tests.
        """
        global _injector_instance
        _injector_instance = None
        logger.debug("CapabilityInjector global singleton reset")


def get_capability_injector() -> CapabilityInjector:
    """Get the global capability injector singleton.

    Returns:
        Global CapabilityInjector instance
    """
    global _injector_instance
    if _injector_instance is None:
        _injector_instance = CapabilityInjector()
    return _injector_instance


def create_capability_injector(
    container: "ServiceContainer",
) -> CapabilityInjector:
    """Factory function for creating CapabilityInjector via DI.

    This function is used for DI container registration.

    Args:
        container: DI container for service resolution

    Returns:
        CapabilityInjector instance
    """
    return CapabilityInjector(container, auto_register_builtins=False)


__all__ = [
    "CapabilityInjector",
    "get_capability_injector",
    "create_capability_injector",
    # Provider classes (for custom capability registration)
    "_FileOperationsProvider",
    "_WebOperationsProvider",
    "_GitOperationsProvider",
    "_TestOperationsProvider",
]
