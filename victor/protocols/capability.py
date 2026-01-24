# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0

"""Capability container protocol for ISP compliance.

This protocol defines the minimal interface for capability containers,
eliminating hasattr() checks and enabling type-safe capability access.

SOLID Compliance:
- ISP (Interface Segregation): Minimal interface with only capability-related methods
- DIP (Dependency Inversion): Depend on protocol, not concrete implementations
"""

from typing import Protocol, runtime_checkable, Any, Optional


@runtime_checkable
class CapabilityContainerProtocol(Protocol):
    """Protocol for objects that can manage capabilities.

    This protocol replaces hasattr() checks with type-safe protocol
    conformance, enabling better static analysis and type checking.

    Example:
        ```python
        @runtime_checkable
        class MyCapabilities(CapabilityContainerProtocol, Protocol):
            def has_capability(self, capability_name: str) -> bool:
                return capability_name in self._capabilities

            def get_capability(self, name: str) -> Optional[Any]:
                return self._capabilities.get(name)
        ```

    Usage:
        ```python
        container: CapabilityContainerProtocol = get_capability_container()
        if isinstance(container, CapabilityContainerProtocol):
            if container.has_capability("code_analysis"):
                capability = container.get_capability("code_analysis")
        ```
    """

    def has_capability(self, capability_name: str) -> bool:
        """Check if a capability is available.

        Args:
            capability_name: Name of the capability to check

        Returns:
            True if the capability is available, False otherwise

        Raises:
            TypeError: If capability_name is not a string
        """
        ...

    def get_capability(self, name: str) -> Optional[Any]:
        """Get a capability by name.

        Args:
            name: Name of the capability to retrieve

        Returns:
            The capability object if found, None otherwise

        Raises:
            TypeError: If name is not a string
        """
        ...


def get_capability_registry() -> Optional[CapabilityContainerProtocol]:
    """Get the global capability registry instance.

    This function provides access to the global capability registry
    that implements CapabilityContainerProtocol.

    Returns:
        The global capability registry, or None if not initialized

    Example:
        ```python
        registry = get_capability_registry()
        if registry and registry.has_capability("my_capability"):
            capability = registry.get_capability("my_capability")
        ```
    """
    # Lazy import to avoid circular dependencies
    try:
        from victor.framework.capabilities import CapabilityRegistryMixin

        # Try to get the global registry instance
        # This will be implemented when integrating with the existing capability system
        return None  # Placeholder - will be implemented in Phase 2
    except ImportError:
        return None
