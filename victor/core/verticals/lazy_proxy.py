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

"""Type-safe lazy loading proxy for LSP compliance.

This module provides a generic LazyProxy[T] that maintains isinstance()
compatibility with VerticalBase while providing lazy loading benefits.

Phase 4 LSP Compliance:
This implementation follows the Liskov Substitution Principle by ensuring
that LazyProxy instances can be used anywhere VerticalBase instances are
expected, with isinstance() checks working correctly.

Design Philosophy:
- LazyProxy inherits from VerticalBase (LSP compliance)
- Delegates all attribute access to loaded instance
- Thread-safe lazy initialization with double-checked locking
- Type-safe with generic LazyProxy[T] parameter
- Minimal overhead (~5-10ms on first access)

OLD Pattern (violates LSP - isinstance() fails):
    class LazyVerticalProxy:
        # Doesn't inherit from VerticalBase
        # isinstance(proxy, VerticalBase) returns False

NEW Pattern (LSP compliant):
    class LazyProxy(VerticalBase):
        # Inherits from VerticalBase
        # isinstance(proxy, VerticalBase) returns True
        # Delegates all attribute access transparently

Usage:
    from victor.core.verticals.lazy_proxy import LazyProxy

    proxy = LazyProxy[MockVertical](
        vertical_name="mock_vertical",
        loader=lambda: MockVertical()
    )

    # Works with isinstance() checks
    assert isinstance(proxy, VerticalBase)  # True

    # Lazy loads on first access
    tools = proxy.get_tools()  # Triggers lazy load
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Type,
    TypeVar,
)

# Import VerticalBase at runtime (needed for inheritance)
from victor.core.verticals.base import VerticalBase

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T", bound="VerticalBase")


# =============================================================================
# Lazy Proxy Type Enum
# =============================================================================


class LazyProxyType(str, Enum):
    """When to load the vertical."""

    LAZY = "lazy"
    """Load on first attribute access (default)."""

    ON_DEMAND = "on_demand"
    """Load immediately when proxy is created."""

    EAGER = "eager"
    """Alias for ON_DEMAND."""


# =============================================================================
# Type-Safe Lazy Proxy
# =============================================================================


class LazyProxy(VerticalBase, Generic[T]):  # type: ignore[misc]
    """Type-safe lazy-loading proxy for VerticalBase subclasses.

    This proxy maintains LSP compliance by inheriting from VerticalBase,
    ensuring isinstance() checks work correctly while providing lazy loading.

    Thread Safety:
        Uses double-checked locking pattern:
        1. Check if loaded (no lock) - fast path
        2. Acquire lock
        3. Check if loaded again (race condition check)
        4. Load if not loaded
        5. Release lock

    Type Safety:
        Generic LazyProxy[T] preserves type information for type checkers.
        Type stubs enable mypy strict mode compliance.

    Attributes:
        vertical_name: Name of the vertical
        proxy_type: When to load (LAZY, ON_DEMAND, EAGER)
        _loader: Callable that loads the vertical
        _loaded: Whether the vertical has been loaded
        _instance: Cached loaded vertical instance
        _load_lock: Thread lock for safe lazy loading
        _loading: Flag to prevent recursive loading
    """

    # Don't define class attributes - use instance attributes and __getattr__

    def __init__(
        self,
        vertical_name: str,
        loader: Callable[[], Type[T]],
        proxy_type: LazyProxyType = LazyProxyType.LAZY,
    ) -> None:
        """Initialize the lazy proxy.

        Args:
            vertical_name: Name of the vertical to load
            loader: Callable that loads and returns the vertical class
            proxy_type: When to load (LAZY, ON_DEMAND, EAGER)
        """
        # Set instance attributes
        self.name = vertical_name

        self._vertical_name = vertical_name
        self._loader = loader
        self._loaded = False
        self._instance: Optional[Type[T]] = None
        self._load_lock = threading.RLock()  # Use RLock for reentrant locking
        self._loading = False
        self.proxy_type = proxy_type

        # Auto-detect proxy type from environment if not explicitly set
        if proxy_type == LazyProxyType.LAZY:
            lazy_loading = os.getenv("VICTOR_LAZY_LOADING", "true").lower() == "true"
            if not lazy_loading:
                self.proxy_type = LazyProxyType.ON_DEMAND

        # Load immediately if ON_DEMAND or EAGER
        if self.proxy_type in (LazyProxyType.ON_DEMAND, LazyProxyType.EAGER):
            self.load()

    @property
    def vertical_name(self) -> str:
        """Get the vertical name.

        Returns:
            Vertical name
        """
        return self._vertical_name

    @property
    def display_name(self) -> str:
        """Get the display name from the loaded vertical.

        Returns:
            Display name from loaded vertical
        """
        # Ensure loaded
        if not self._loaded:
            self.load()

        if self._instance and hasattr(self._instance, 'display_name'):
            return str(self._instance.display_name)
        return f"{self._vertical_name.title()} (Lazy Proxy)"

    @property
    def description(self) -> str:
        """Get the description from the loaded vertical.

        Returns:
            Description from loaded vertical
        """
        # Ensure loaded
        if not self._loaded:
            self.load()

        if self._instance and hasattr(self._instance, 'description'):
            return str(self._instance.description)
        return f"Lazy loading proxy for {self._vertical_name}"

    def load(self) -> Type[T]:
        """Load vertical on first access with thread-safe lazy initialization.

        Uses double-checked locking pattern for thread safety:
        1. Fast path: check if already loaded (no lock)
        2. Slow path: acquire lock and load if needed

        Returns:
            Loaded vertical class

        Raises:
            RuntimeError: If recursive loading is detected
        """
        # Fast path: already loaded
        if self._loaded:
            instance = self._instance
            if instance is not None:
                return instance
            raise RuntimeError(f"Vertical '{self._vertical_name}' marked as loaded but instance is None")

        # Slow path: need to load
        with self._load_lock:
            # Double-check: another thread may have loaded it
            if self._loaded:
                instance = self._instance  # type: ignore[unreachable]
                if instance is not None:
                    return instance
                raise RuntimeError(
                    f"Vertical '{self._vertical_name}' marked as loaded but instance is None"
                )

            # Check for recursive loading
            if self._loading:
                raise RuntimeError(
                    f"Recursive loading detected for vertical '{self._vertical_name}'"
                )

            try:
                self._loading = True
                logger.debug(f"Lazy loading vertical: {self._vertical_name}")
                self._instance = self._loader()
                self._loaded = True

                # Note: display_name and description are now properties that
                # delegate to the loaded instance, so no need to update them here

                logger.debug(f"Successfully loaded vertical: {self._vertical_name}")
                return self._instance
            except Exception as e:
                logger.error(f"Failed to load vertical '{self._vertical_name}': {e}")
                raise
            finally:
                self._loading = False

    def unload(self) -> None:
        """Unload vertical to free memory.

        This clears the cached instance, allowing it to be garbage collected.
        The next call to load() will reload the vertical.
        """
        with self._load_lock:
            self._instance = None
            self._loaded = False
            logger.debug(f"Unloaded vertical: {self._vertical_name}")

    def is_loaded(self) -> bool:
        """Check if vertical is currently loaded.

        Returns:
            True if vertical has been loaded
        """
        return self._loaded

    def get_tools(self) -> list[Any]:
        """Get tools from the loaded vertical.

        Returns:
            List of tools provided by the vertical
        """
        vertical = self.load()
        return list(vertical.get_tools())

    def get_system_prompt(self) -> str:
        """Get system prompt from the loaded vertical.

        Returns:
            System prompt string
        """
        vertical = self.load()
        return str(vertical.get_system_prompt())

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the loaded vertical.

        This is called when an attribute is accessed on the proxy.
        It ensures the vertical is loaded, then forwards the access.

        Args:
            name: Attribute name

        Returns:
            Attribute value from loaded vertical
        """
        # Ensure loaded
        vertical = self.load()
        return getattr(vertical, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Proxy callable interface to the loaded vertical.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of calling the vertical
        """
        vertical = self.load()
        return vertical(*args, **kwargs)

    def __repr__(self) -> str:
        """String representation of the proxy.

        Returns:
            String representation
        """
        if self._loaded:
            return f"<LazyProxy({self._vertical_name}) loaded>"
        return f"<LazyProxy({self._vertical_name}) unloaded>"


# =============================================================================
# Proxy Factory
# =============================================================================


class LazyProxyFactory:
    """Factory for creating and caching LazyProxy instances.

    This factory provides a centralized way to create lazy proxies with
    caching to avoid duplicate proxies for the same vertical.
    """

    def __init__(self) -> None:
        """Initialize the factory."""
        self._proxies: Dict[str, LazyProxy[Any]] = {}

    def create_proxy(
        self,
        vertical_name: str,
        loader: Callable[[], Type[T]],
        vertical_class: Type[T],
        proxy_type: LazyProxyType = LazyProxyType.LAZY,
    ) -> LazyProxy[Any]:
        """Create or retrieve cached LazyProxy for a vertical.

        Args:
            vertical_name: Name of the vertical
            loader: Callable that loads the vertical
            vertical_class: Type of the vertical (for type checking)
            proxy_type: When to load (LAZY, ON_DEMAND, EAGER)

        Returns:
            LazyProxy instance (cached if already created)
        """
        # Check cache
        if vertical_name in self._proxies:
            return self._proxies[vertical_name]

        # Create new proxy
        proxy = LazyProxy(
            vertical_name=vertical_name,
            loader=loader,
            proxy_type=proxy_type,
        )

        # Cache it
        self._proxies[vertical_name] = proxy
        return proxy

    def clear_cache(self) -> None:
        """Clear the proxy cache."""
        self._proxies.clear()
        logger.debug("Cleared LazyProxy factory cache")


# =============================================================================
# Factory Functions
# =============================================================================


_proxy_factory: Optional[LazyProxyFactory] = None


def get_lazy_proxy_factory() -> LazyProxyFactory:
    """Get or create singleton LazyProxyFactory instance.

    Returns:
        LazyProxyFactory singleton instance
    """
    global _proxy_factory
    if _proxy_factory is None:
        _proxy_factory = LazyProxyFactory()
    return _proxy_factory


def clear_proxy_factory_cache() -> None:
    """Clear the singleton factory cache (mainly for testing)."""
    global _proxy_factory
    _proxy_factory = None


def clear_proxy_cache() -> None:
    """Clear all cached proxies.

    This is a convenience function that clears the factory cache.
    """
    factory = get_lazy_proxy_factory()
    factory.clear_cache()


__all__ = [
    "LazyProxyType",
    "LazyProxy",
    "LazyProxyFactory",
    "get_lazy_proxy_factory",
    "clear_proxy_factory_cache",
    "clear_proxy_cache",
]
