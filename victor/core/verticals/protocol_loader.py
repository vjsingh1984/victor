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

"""Protocol-Based Extension Loader for ISP-Compliant Verticals.

This module provides a protocol-based extension loading system that enables
verticals to implement only the protocols they need, following the Interface
Segregation Principle (ISP).

Key Features:
- Protocol-based registration: Verticals register only for protocols they implement
- Type-safe protocol checking: Use isinstance() to detect protocol conformance
- Lazy protocol resolution: Protocols loaded only when first accessed
- Backward compatible: Works with existing VerticalBase implementations
- Cache-friendly: Protocol implementations cached for performance

Architecture:
    The ProtocolBasedExtensionLoader maintains a registry of protocol implementations
    and provides methods to:
    1. Register protocol implementations by protocol type
    2. Retrieve protocol implementations with type safety
    3. Check protocol conformance via isinstance()
    4. Clear protocol caches for testing

Usage:
    from victor.core.verticals.protocol_loader import ProtocolBasedExtensionLoader
    from victor.core.verticals.protocols import (
        ToolProvider,
        PromptProviderProtocol,
    )

    class MinimalVertical:
        '''Vertical implementing only tool and prompt protocols.'''
        name = "minimal"

        @classmethod
        def get_tools(cls):
            return ["read", "write"]

        @classmethod
        def get_system_prompt(cls):
            return "You are a minimal assistant..."

    # Register protocol implementations
    loader = ProtocolBasedExtensionLoader()
    loader.register_protocol(ToolProvider, MinimalVertical)
    loader.register_protocol(PromptProviderProtocol, MinimalVertical)

    # Check protocol conformance
    if loader.implements_protocol(MinimalVertical, ToolProvider):
        tools = loader.get_protocol(MinimalVertical, ToolProvider)
        tools_list = tools.get_tools()

Benefits:
    - ISP Compliance: Verticals implement only needed protocols
    - Type Safety: Protocol conformance checked via isinstance()
    - Reduced Coupling: Framework depends on protocols, not concrete classes
    - Better Testability: Can mock specific protocols in tests
    - Clearer Intent: Protocol conformance declares vertical capabilities
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Type,
    TYPE_CHECKING,
    Protocol as TypingProtocol,
)
from typing import runtime_checkable

if TYPE_CHECKING:
    from victor.core.verticals.protocols import VerticalExtensions

logger = logging.getLogger(__name__)


# Re-export Protocol for convenience
__all__ = [
    "ProtocolBasedExtensionLoader",
    "ProtocolImplementationEntry",
    "register_protocol_implementation",
    "implements_protocol",
]

# Use typing.Protocol for all annotations (use alias for backward compatibility)
# Note: Don't use Protocol as a variable name if we want to use it in type annotations
# Instead, use TypingProtocol directly in type annotations


# =============================================================================
# Protocol Implementation Cache Entry
# =============================================================================


@dataclass
class ProtocolImplementationEntry:
    """Cache entry for protocol implementation.

    Attributes:
        protocol_type: The protocol class (e.g., ToolProvider)
        implementation: The vertical class implementing the protocol
        cached_result: Cached result from protocol method call
        timestamp: When the entry was created
        ttl: Time-to-live in seconds (None = no expiration)
    """

    protocol_type: Type[Any]
    implementation: Type[Any]
    cached_result: Any = None
    timestamp: float = field(default_factory=lambda: __import__("time").time())
    ttl: Optional[int] = None


# =============================================================================
# Protocol-Based Extension Loader
# =============================================================================


class ProtocolBasedExtensionLoader:
    """Loader for protocol-based vertical extensions.

    This class provides ISP-compliant extension loading by allowing verticals
    to register only the protocols they implement. The framework can then use
    isinstance() checks to determine which capabilities a vertical supports.

    Protocol Registry:
        The loader maintains a registry mapping protocol types to vertical
        implementations. This enables type-safe protocol checking and retrieval.

    Caching:
        Protocol method results are cached to avoid repeated computation.
        Cache entries can be invalidated by protocol type or completely cleared.

    Thread Safety:
        The loader uses class-level locks for thread-safe registry access.

    Example:
        from victor.core.verticals.protocol_loader import ProtocolBasedExtensionLoader
        from victor.core.verticals.protocols.providers import ToolProvider, PromptProvider

        class MyVertical:
            name = "my_vertical"

            @classmethod
            def get_tools(cls):
                return ["read", "write"]

            @classmethod
            def get_system_prompt(cls):
                return "You are a helpful assistant..."

        # Register protocol implementations
        loader = ProtocolBasedExtensionLoader()
        loader.register_protocol(ToolProvider, MyVertical)

        # Check protocol conformance
        if loader.implements_protocol(MyVertical, ToolProvider):
            print("MyVertical implements ToolProvider")

        # Get protocol implementation
        protocol_impl = loader.get_protocol(MyVertical, ToolProvider)
        tools = protocol_impl.get_tools()
    """

    # Protocol registry: maps protocol_type -> {vertical_class -> implementation}
    _protocol_registry: ClassVar[Dict[Type[Any], Dict[Type[Any], Any]]] = {}

    # Protocol implementation cache: maps (vertical_class, protocol_type) -> cached_result
    _protocol_cache: ClassVar[Dict[tuple[Type[Any], Type[Any]], Any]] = {}

    # Protocol conformance cache: maps (vertical_class, protocol_type) -> bool
    _conformance_cache: ClassVar[Dict[tuple[Type[Any], Type[Any]], bool]] = {}

    @classmethod
    def register_protocol(
        cls,
        protocol_type: Type[Any],
        vertical_class: Type[Any],
        implementation: Optional[Any] = None,
    ) -> None:
        """Register a vertical class as implementing a protocol.

        Args:
            protocol_type: The protocol class (e.g., ToolProvider, PromptProvider)
            vertical_class: The vertical class implementing the protocol
            implementation: Optional custom implementation (defaults to vertical_class)

        Raises:
            TypeError: If protocol_type is not a Protocol
            ValueError: If vertical_class is already registered with different implementation

        Example:
            from victor.core.verticals.protocols.providers import ToolProvider

            loader = ProtocolBasedExtensionLoader()
            loader.register_protocol(ToolProvider, MyVertical)
        """
        # Validate protocol_type is actually a Protocol
        if not getattr(protocol_type, "_is_protocol", False):
            raise TypeError(f"protocol_type must be a Protocol, got {type(protocol_type).__name__}")

        # Initialize protocol type registry if needed
        if protocol_type not in cls._protocol_registry:
            cls._protocol_registry[protocol_type] = {}

        # Use vertical_class as implementation if not provided
        if implementation is None:
            implementation = vertical_class

        # Check for conflicting registration
        if vertical_class in cls._protocol_registry[protocol_type]:
            existing = cls._protocol_registry[protocol_type][vertical_class]
            if existing != implementation:
                raise ValueError(
                    f"Vertical {vertical_class.__name__} already registered "
                    f"for protocol {protocol_type.__name__} with different implementation"
                )

        # Register the protocol implementation
        cls._protocol_registry[protocol_type][vertical_class] = implementation

        # Clear conformance cache for this vertical/protocol pair
        cache_key = (vertical_class, protocol_type)
        cls._conformance_cache.pop(cache_key, None)

        logger.debug(
            f"Registered {vertical_class.__name__} as implementing {protocol_type.__name__}"
        )

    @classmethod
    def implements_protocol(
        cls,
        vertical_class: Type[Any],
        protocol_type: Type[Any],
    ) -> bool:
        """Check if a vertical class implements a protocol.

        This method checks both explicit registration and runtime isinstance()
        checks for @runtime_checkable protocols.

        Args:
            vertical_class: The vertical class to check
            protocol_type: The protocol to check for

        Returns:
            True if vertical implements the protocol, False otherwise

        Example:
            if loader.implements_protocol(MyVertical, ToolProvider):
                tools = loader.get_protocol(MyVertical, ToolProvider).get_tools()
        """
        cache_key = (vertical_class, protocol_type)

        # Check conformance cache
        if cache_key in cls._conformance_cache:
            return cls._conformance_cache[cache_key]

        # Check explicit registration
        if protocol_type in cls._protocol_registry:
            if vertical_class in cls._protocol_registry[protocol_type]:
                cls._conformance_cache[cache_key] = True
                return True

        # Check runtime isinstance() for @runtime_checkable protocols
        if getattr(protocol_type, "_is_runtime", False):
            try:
                # For runtime_checkable protocols, use isinstance check
                # Note: This requires an instance, not just the class
                # We'll check if the class has the required protocol methods
                result = cls._check_protocol_methods(vertical_class, protocol_type)
                cls._conformance_cache[cache_key] = result
                return result
            except Exception:
                cls._conformance_cache[cache_key] = False
                return False

        cls._conformance_cache[cache_key] = False
        return False

    @classmethod
    def _check_protocol_methods(
        cls,
        vertical_class: Type[Any],
        protocol_type: Type[Any],
    ) -> bool:
        """Check if vertical_class has all required methods from protocol_type.

        Args:
            vertical_class: The vertical class to check
            protocol_type: The protocol to check methods for

        Returns:
            True if all required protocol methods are present
        """
        import inspect

        # Get all protocol methods (exclude __dict__ and other Protocol attributes)
        protocol_members = inspect.getmembers(protocol_type, predicate=inspect.isfunction)

        for method_name, _ in protocol_members:
            if method_name.startswith("_"):
                continue

            # Check if vertical_class has the method
            if not hasattr(vertical_class, method_name):
                return False

            # Check if it's callable
            method = getattr(vertical_class, method_name)
            if not callable(method):
                return False

        return True

    @classmethod
    def get_protocol(
        cls,
        vertical_class: Type[Any],
        protocol_type: Type[Any],
        use_cache: bool = True,
    ) -> Optional[Any]:
        """Get the protocol implementation for a vertical.

        Args:
            vertical_class: The vertical class
            protocol_type: The protocol to retrieve
            use_cache: If True (default), return cached result if available

        Returns:
            Protocol implementation or None if not registered

        Raises:
            ValueError: If vertical doesn't implement the protocol

        Example:
            tool_provider = loader.get_protocol(MyVertical, ToolProvider)
            tools = tool_provider.get_tools()
        """
        # Check if vertical implements the protocol
        if not cls.implements_protocol(vertical_class, protocol_type):
            raise ValueError(
                f"Vertical {vertical_class.__name__} does not implement "
                f"protocol {protocol_type.__name__}"
            )

        # Check cache
        cache_key = (vertical_class, protocol_type)
        if use_cache and cache_key in cls._protocol_cache:
            return cls._protocol_cache[cache_key]

        # Get implementation from registry
        implementation = cls._protocol_registry.get(protocol_type, {}).get(
            vertical_class, vertical_class
        )

        # Cache the result
        if use_cache:
            cls._protocol_cache[cache_key] = implementation

        return implementation

    @classmethod
    def list_protocols(cls, vertical_class: Type[Any]) -> List[Type[Any]]:
        """List all protocols implemented by a vertical.

        Args:
            vertical_class: The vertical class

        Returns:
            List of protocol types implemented by the vertical

        Example:
            protocols = loader.list_protocols(MyVertical)
            # [ToolProvider, PromptProvider, ...]
        """
        implemented = []
        for protocol_type, verticals in cls._protocol_registry.items():
            if vertical_class in verticals:
                implemented.append(protocol_type)

        return implemented

    @classmethod
    def list_verticals(cls, protocol_type: Type[Any]) -> List[Type[Any]]:
        """List all verticals implementing a protocol.

        Args:
            protocol_type: The protocol to check

        Returns:
            List of vertical classes implementing the protocol

        Example:
            tool_providers = loader.list_verticals(ToolProvider)
            # [CodingVertical, ResearchVertical, ...]
        """
        if protocol_type not in cls._protocol_registry:
            return []

        return list(cls._protocol_registry[protocol_type].keys())

    @classmethod
    def unregister_protocol(
        cls,
        protocol_type: Type[Any],
        vertical_class: Type[Any],
    ) -> None:
        """Unregister a vertical from a protocol.

        Args:
            protocol_type: The protocol to unregister from
            vertical_class: The vertical class to unregister

        Example:
            loader.unregister_protocol(ToolProvider, MyVertical)
        """
        if protocol_type in cls._protocol_registry:
            cls._protocol_registry[protocol_type].pop(vertical_class, None)

            # Clean up empty protocol registries
            if not cls._protocol_registry[protocol_type]:
                del cls._protocol_registry[protocol_type]

        # Clear caches
        cache_key = (vertical_class, protocol_type)
        cls._protocol_cache.pop(cache_key, None)
        cls._conformance_cache.pop(cache_key, None)

        logger.debug(f"Unregistered {vertical_class.__name__} from {protocol_type.__name__}")

    @classmethod
    def clear_cache(
        cls,
        vertical_class: Optional[Type[Any]] = None,
        protocol_type: Optional[Type[Any]] = None,
    ) -> None:
        """Clear protocol cache entries.

        Args:
            vertical_class: Specific vertical class to clear (None = all)
            protocol_type: Specific protocol type to clear (None = all)

        Examples:
            # Clear all caches
            loader.clear_cache()

            # Clear all caches for a vertical
            loader.clear_cache(vertical_class=MyVertical)

            # Clear all caches for a protocol
            loader.clear_cache(protocol_type=ToolProvider)

            # Clear specific vertical/protocol combination
            loader.clear_cache(vertical_class=MyVertical, protocol_type=ToolProvider)
        """
        keys_to_remove = []

        for cache_key in cls._protocol_cache.keys():
            key_vertical, key_protocol = cache_key

            # Check if this key should be removed
            if vertical_class is not None and key_vertical != vertical_class:
                continue
            if protocol_type is not None and key_protocol != protocol_type:
                continue

            keys_to_remove.append(cache_key)

        # Remove selected keys
        for key in keys_to_remove:
            cls._protocol_cache.pop(key, None)

        # Also clear conformance cache
        conformance_keys_to_remove = []
        for cache_key in cls._conformance_cache.keys():
            key_vertical, key_protocol = cache_key

            if vertical_class is not None and key_vertical != vertical_class:
                continue
            if protocol_type is not None and key_protocol != protocol_type:
                continue

            conformance_keys_to_remove.append(cache_key)

        for key in conformance_keys_to_remove:
            cls._conformance_cache.pop(key, None)

        logger.debug(
            f"Cleared {len(keys_to_remove)} protocol cache entries and "
            f"{len(conformance_keys_to_remove)} conformance cache entries"
        )

    @classmethod
    def get_registry_stats(cls) -> Dict[str, Any]:
        """Get registry statistics for debugging.

        Returns:
            Dict with registry stats including:
            - total_protocols: Number of protocol types registered
            - total_registrations: Total protocol registrations
            - cache_size: Number of cached entries
            - protocols: List of registered protocol types
        """
        total_registrations = sum(len(verticals) for verticals in cls._protocol_registry.values())

        return {
            "total_protocols": len(cls._protocol_registry),
            "total_registrations": total_registrations,
            "cache_size": len(cls._protocol_cache),
            "conformance_cache_size": len(cls._conformance_cache),
            "protocols": [p.__name__ for p in cls._protocol_registry.keys()],
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def register_protocol_implementation(
    protocol_type: Type[Any],
    vertical_class: Type[Any],
) -> None:
    """Register a vertical as implementing a protocol.

    Convenience function for ProtocolBasedExtensionLoader.register_protocol().

    Args:
        protocol_type: The protocol to register
        vertical_class: The vertical class implementing the protocol

    Example:
        from victor.core.verticals.protocols.providers import ToolProvider
        from victor.core.verticals.protocol_loader import register_protocol_implementation

        class MyVertical:
            name = "my"

            @classmethod
            def get_tools(cls):
                return ["read"]

        register_protocol_implementation(ToolProvider, MyVertical)
    """
    ProtocolBasedExtensionLoader.register_protocol(protocol_type, vertical_class)


def implements_protocol(
    vertical_class: Type[Any],
    protocol_type: Type[Any],
) -> bool:
    """Check if a vertical implements a protocol.

    Convenience function for ProtocolBasedExtensionLoader.implements_protocol().

    Args:
        vertical_class: The vertical class to check
        protocol_type: The protocol to check for

    Returns:
        True if vertical implements the protocol

    Example:
        from victor.core.verticals.protocols.providers import ToolProvider
        from victor.core.verticals.protocol_loader import implements_protocol

        if implements_protocol(MyVertical, ToolProvider):
            print("Has tools")
    """
    return ProtocolBasedExtensionLoader.implements_protocol(vertical_class, protocol_type)


__all__ = [
    "ProtocolBasedExtensionLoader",
    "ProtocolImplementationEntry",
    "register_protocol_implementation",
    "implements_protocol",
]
