"""Type stubs for lazy_proxy.py (mypy strict mode compliance).

This stub file provides precise type information for mypy strict mode,
ensuring that LazyProxy[T] is properly understood as a generic type.
"""

from __future__ import annotations

from typing import TypeVar, Generic, Callable, Optional, Type, Any, Dict, List
from enum import Enum

# Import VerticalBase for type checking
from victor.core.verticals.base import VerticalBase

# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T", bound=VerticalBase)

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

class LazyProxy(VerticalBase, Generic[T]):
    """Type-safe lazy-loading proxy for VerticalBase subclasses.

    This proxy maintains LSP compliance by inheriting from VerticalBase,
    ensuring isinstance() checks work correctly while providing lazy loading.

    Type Parameters:
        T: The vertical type being proxied (must be subclass of VerticalBase)

    Attributes:
        name: The vertical name
        vertical_name: Name of the vertical (read-only property)
        display_name: Display name from loaded vertical (triggers lazy load)
        description: Description from loaded vertical (triggers lazy load)
        proxy_type: When to load (LAZY, ON_DEMAND, EAGER)
    """

    # Instance attributes (defined in __init__)
    # Note: 'name' is inherited as class var from VerticalMetadataProvider, overridden here as instance var
    _vertical_name: str
    _loader: Callable[[], Type[T]]
    _loaded: bool
    _instance: Optional[Type[T]]
    _load_lock: Any  # threading.RLock
    _loading: bool
    proxy_type: LazyProxyType

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
        ...

    @property
    def vertical_name(self) -> str:
        """Get the vertical name.

        Returns:
            Vertical name
        """
        ...

    @property
    def display_name(self) -> str:
        """Get the display name from the loaded vertical.

        Triggers lazy load if not already loaded.

        Returns:
            Display name from loaded vertical
        """
        ...

    @property
    def description(self) -> str:  # type: ignore[override]
        """Get the description from the loaded vertical.

        Triggers lazy load if not already loaded.

        Returns:
            Description from loaded vertical
        """
        ...

    def load(self) -> Type[T]:
        """Load vertical on first access with thread-safe lazy initialization.

        Returns:
            Loaded vertical class

        Raises:
            RuntimeError: If recursive loading is detected
        """
        ...

    def unload(self) -> None:
        """Unload vertical to free memory.

        This clears the cached instance, allowing it to be garbage collected.
        The next call to load() will reload the vertical.
        """
        ...

    def is_loaded(self) -> bool:
        """Check if vertical is currently loaded.

        Returns:
            True if vertical has been loaded
        """
        ...

    def get_tools(self) -> List[str]:  # type: ignore[override]
        """Get tools from the loaded vertical.

        Triggers lazy load if not already loaded.

        Returns:
            List of tools provided by the vertical
        """
        ...

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get system prompt from the loaded vertical.

        Triggers lazy load if not already loaded.

        Returns:
            System prompt string
        """
        ...

    def __getattr__(self, name: str) -> Any: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def __repr__(self) -> str: ...

# =============================================================================
# Proxy Factory
# =============================================================================

class LazyProxyFactory:
    """Factory for creating and caching LazyProxy instances."""

    def __init__(self) -> None:
        """Initialize the factory."""
        ...

    def create_proxy(
        self,
        vertical_name: str,
        loader: Callable[[], Type[T]],
        vertical_class: Type[T],
        proxy_type: LazyProxyType = LazyProxyType.LAZY,
    ) -> LazyProxy[T]:
        """Create or retrieve cached LazyProxy for a vertical.

        Args:
            vertical_name: Name of the vertical
            loader: Callable that loads the vertical
            vertical_class: Type of the vertical (for type checking)
            proxy_type: When to load (LAZY, ON_DEMAND, EAGER)

        Returns:
            LazyProxy instance (cached if already created)
        """
        ...

    def clear_cache(self) -> None:
        """Clear the proxy cache."""
        ...

# =============================================================================
# Factory Functions
# =============================================================================

def get_lazy_proxy_factory() -> LazyProxyFactory:
    """Get or create singleton LazyProxyFactory instance.

    Returns:
        LazyProxyFactory singleton instance
    """
    ...

def clear_proxy_factory_cache() -> None:
    """Clear the singleton factory cache (mainly for testing)."""
    ...

def clear_proxy_cache() -> None:
    """Clear all cached proxies.

    This is a convenience function that clears the factory cache.
    """
    ...

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "LazyProxyType",
    "LazyProxy",
    "LazyProxyFactory",
    "get_lazy_proxy_factory",
    "clear_proxy_factory_cache",
    "clear_proxy_cache",
]
