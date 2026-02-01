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

"""Handler Registry for explicit compute handler registration.

This module provides a centralized registry for workflow compute handlers,
replacing the previous import-side-effect registration pattern with an
explicit, testable, and traceable registration mechanism.

Example:
    registry = get_handler_registry()
    registry.register("code_validation", handler, vertical="coding")

    # Get handler
    handler = registry.get("code_validation")

    # List handlers by vertical
    coding_handlers = registry.list_by_vertical("coding")

    # Sync with executor's global handlers
    registry.sync_with_executor()

    # Auto-discover handlers from a vertical
    count = registry.discover_from_vertical("coding")
"""

import importlib
import logging
from dataclasses import dataclass
from typing import Any, Optional
from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass
class HandlerEntry:
    """Metadata for a registered handler.

    Attributes:
        name: Handler name (e.g., "code_validation")
        handler: The handler instance or callable
        vertical: Optional vertical that owns this handler
        description: Optional human-readable description
    """

    name: str
    handler: Any
    vertical: Optional[str] = None
    description: Optional[str] = None


class HandlerRegistry:
    """Registry for compute handlers with explicit registration.

    Provides a centralized, singleton registry for workflow compute handlers.
    Supports vertical namespacing, handler discovery, and replacement.

    Thread Safety:
        This implementation is NOT thread-safe. For concurrent access,
        use external synchronization or a thread-safe variant.
    """

    _instance: Optional["HandlerRegistry"] = None

    def __init__(self) -> None:
        """Initialize empty handler registry."""
        self._handlers: dict[str, HandlerEntry] = {}

    @classmethod
    def get_instance(cls) -> "HandlerRegistry":
        """Get singleton instance of the registry.

        Returns:
            The global HandlerRegistry instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    def register(
        self,
        name: str,
        handler: Any,
        vertical: Optional[str] = None,
        description: Optional[str] = None,
        replace: bool = False,
    ) -> None:
        """Register a handler.

        Args:
            name: Handler name (should be unique)
            handler: Handler instance or callable
            vertical: Optional vertical namespace
            description: Optional description
            replace: If True, replace existing handler

        Raises:
            ValueError: If handler already exists and replace is False
        """
        if name in self._handlers and not replace:
            raise ValueError(
                f"Handler '{name}' already registered. " f"Use replace=True to override."
            )

        self._handlers[name] = HandlerEntry(
            name=name,
            handler=handler,
            vertical=vertical,
            description=description,
        )
        logger.debug(f"Registered handler: {name} (vertical={vertical})")

    def unregister(self, name: str) -> bool:
        """Remove a handler from the registry.

        Args:
            name: Handler name to remove

        Returns:
            True if handler was removed, False if not found
        """
        if name in self._handlers:
            del self._handlers[name]
            logger.debug(f"Unregistered handler: {name}")
            return True
        return False

    def get(self, name: str) -> Optional[Any]:
        """Get handler by name.

        Args:
            name: Handler name

        Returns:
            Handler instance or None if not found
        """
        entry = self._handlers.get(name)
        return entry.handler if entry else None

    def get_entry(self, name: str) -> Optional[HandlerEntry]:
        """Get full handler entry with metadata.

        Args:
            name: Handler name

        Returns:
            HandlerEntry or None if not found
        """
        return self._handlers.get(name)

    def has(self, name: str) -> bool:
        """Check if handler exists.

        Args:
            name: Handler name

        Returns:
            True if handler is registered
        """
        return name in self._handlers

    def list_handlers(self) -> list[str]:
        """List all registered handler names.

        Returns:
            List of handler names
        """
        return list(self._handlers.keys())

    def list_entries(self) -> list[HandlerEntry]:
        """List all handler entries with metadata.

        Returns:
            List of HandlerEntry objects
        """
        return list(self._handlers.values())

    def list_by_vertical(self, vertical: str) -> list[str]:
        """List handlers for a specific vertical.

        Args:
            vertical: Vertical name

        Returns:
            List of handler names for that vertical
        """
        return [name for name, entry in self._handlers.items() if entry.vertical == vertical]

    def clear(self) -> None:
        """Clear all handlers from the registry."""
        self._handlers.clear()
        logger.debug("Cleared handler registry")

    def register_from_vertical(
        self,
        vertical_name: str,
        handlers: dict[str, Any],
        replace: bool = False,
    ) -> int:
        """Bulk register handlers from a vertical.

        Args:
            vertical_name: Name of the vertical
            handlers: Dict of name -> handler mappings
            replace: If True, replace existing handlers

        Returns:
            Number of handlers registered
        """
        count = 0
        for name, handler in handlers.items():
            self.register(
                name=name,
                handler=handler,
                vertical=vertical_name,
                replace=replace,
            )
            count += 1
        return count

    def sync_with_executor(
        self,
        *,
        direction: str = "bidirectional",
        replace: bool = False,
    ) -> tuple[int, int]:
        """Bridge to workflows/executor.py global handler dict.

        Synchronizes handlers between this registry and the executor's
        global _compute_handlers dictionary. This enables handlers
        registered via either mechanism to be available everywhere.

        Args:
            direction: Sync direction:
                - "to_executor": Push registry handlers to executor
                - "from_executor": Pull executor handlers to registry
                - "bidirectional": Both directions (default)
            replace: If True, replace existing handlers during sync

        Returns:
            Tuple of (handlers_pushed, handlers_pulled)

        Example:
            registry = get_handler_registry()
            registry.register("my_handler", handler, vertical="coding")

            # Sync to make handler available in executor
            pushed, pulled = registry.sync_with_executor()
        """
        from victor.workflows.executor import (
            _compute_handlers,
            register_compute_handler,
            get_compute_handler,
        )

        pushed = 0
        pulled = 0

        # Push registry handlers to executor
        if direction in ("to_executor", "bidirectional"):
            for name, entry in self._handlers.items():
                existing = get_compute_handler(name)
                if existing is None or replace:
                    register_compute_handler(name, entry.handler)
                    pushed += 1
                    logger.debug(f"Pushed handler to executor: {name}")

        # Pull executor handlers to registry
        if direction in ("from_executor", "bidirectional"):
            for name, handler in _compute_handlers.items():
                if not self.has(name) or replace:
                    # We don't know the vertical for executor handlers
                    self.register(name, handler, vertical=None, replace=replace)
                    pulled += 1
                    logger.debug(f"Pulled handler from executor: {name}")

        logger.debug(f"Sync complete: pushed={pushed}, pulled={pulled}")
        return (pushed, pulled)

    def discover_from_vertical(
        self,
        vertical_name: str,
        *,
        replace: bool = False,
        sync_to_executor: bool = True,
    ) -> int:
        """Auto-discover and register handlers from a vertical module.

        Imports and reloads victor.{vertical_name}.handlers to trigger
        @handler_decorator auto-registration, then returns count.

        Args:
            vertical_name: Name of the vertical (e.g., "coding", "research")
            replace: If True, replace existing handlers before discovery
            sync_to_executor: If True, sync handlers to executor

        Returns:
            Number of handlers registered for this vertical

        Raises:
            ImportError: If the handlers module cannot be imported

        Example:
            registry = get_handler_registry()

            # Discover and register handlers from coding vertical
            count = registry.discover_from_vertical("coding")
            print(f"Registered {count} handlers")
        """
        from victor.core.verticals.naming import get_vertical_module_name

        module_name = get_vertical_module_name(vertical_name)
        module_path = f"victor.{module_name}.handlers"

        # Clear existing handlers for this vertical if replace=True
        if replace:
            to_remove = [
                name for name, entry in self._handlers.items() if entry.vertical == vertical_name
            ]
            for name in to_remove:
                del self._handlers[name]

        # Get singleton registry to check if handlers already exist
        singleton = HandlerRegistry.get_instance()

        # Import and reload module to trigger @handler_decorator auto-registration
        # We temporarily enable replace mode in singleton to avoid duplicate errors
        try:
            module = importlib.import_module(module_path)

            # Clear handlers for this vertical from singleton to avoid duplicate errors on reload
            to_remove_singleton = [
                name
                for name, entry in singleton._handlers.items()
                if entry.vertical == vertical_name
            ]
            for name in to_remove_singleton:
                del singleton._handlers[name]

            # Force reload to re-trigger decorators even if module was cached
            importlib.reload(module)
        except ImportError as e:
            logger.warning(f"Could not import handlers from {module_path}: {e}")
            raise

        # Copy handlers from singleton to this instance if different
        if self is not singleton:
            for name, entry in singleton._handlers.items():
                if entry.vertical == vertical_name and name not in self._handlers:
                    self._handlers[name] = entry

        # Count handlers registered for this vertical
        count = sum(1 for entry in self._handlers.values() if entry.vertical == vertical_name)

        # Optionally sync to executor
        if sync_to_executor and count > 0:
            self.sync_with_executor(direction="to_executor", replace=replace)

        logger.info(f"Discovered {count} handlers from {module_path}")
        return count

    def list_verticals(self) -> list[str]:
        """List all verticals with registered handlers.

        Returns:
            List of unique vertical names
        """
        verticals = set()
        for entry in self._handlers.values():
            if entry.vertical:
                verticals.add(entry.vertical)
        return list(verticals)


# Module-level convenience functions
def get_handler_registry() -> HandlerRegistry:
    """Get the global handler registry singleton.

    Returns:
        HandlerRegistry instance
    """
    return HandlerRegistry.get_instance()


def register_handler(
    name: str,
    handler: Any,
    vertical: Optional[str] = None,
    description: Optional[str] = None,
    replace: bool = False,
) -> None:
    """Register a handler with the global registry.

    Convenience function for `get_handler_registry().register(...)`.

    Args:
        name: Handler name
        handler: Handler instance
        vertical: Optional vertical namespace
        description: Optional description
        replace: If True, replace existing
    """
    get_handler_registry().register(
        name=name,
        handler=handler,
        vertical=vertical,
        description=description,
        replace=replace,
    )


def get_handler(name: str) -> Optional[Any]:
    """Get a handler from the global registry.

    Convenience function for `get_handler_registry().get(...)`.

    Args:
        name: Handler name

    Returns:
        Handler instance or None
    """
    return get_handler_registry().get(name)


def sync_handlers_with_executor(
    *,
    direction: str = "bidirectional",
    replace: bool = False,
) -> tuple[int, int]:
    """Sync global handler registry with executor.

    Convenience function for `get_handler_registry().sync_with_executor(...)`.

    Args:
        direction: Sync direction ("to_executor", "from_executor", "bidirectional")
        replace: If True, replace existing handlers

    Returns:
        Tuple of (handlers_pushed, handlers_pulled)
    """
    return get_handler_registry().sync_with_executor(direction=direction, replace=replace)


def discover_handlers_from_vertical(
    vertical_name: str,
    *,
    replace: bool = False,
    sync_to_executor: bool = True,
) -> int:
    """Discover and register handlers from a vertical.

    Convenience function for `get_handler_registry().discover_from_vertical(...)`.

    Args:
        vertical_name: Name of the vertical
        replace: If True, replace existing handlers
        sync_to_executor: If True, sync to executor after discovery

    Returns:
        Number of handlers registered
    """
    return get_handler_registry().discover_from_vertical(
        vertical_name=vertical_name,
        replace=replace,
        sync_to_executor=sync_to_executor,
    )


# =============================================================================
# Phase 1.3: @handler_decorator Class Decorator
# =============================================================================

# Known vertical names for auto-detection
KNOWN_VERTICALS = {"coding", "research", "devops", "dataanalysis", "rag", "benchmark"}


def get_vertical_from_module(module_name: str) -> Optional[str]:
    """Extract vertical name from module path.

    Attempts to detect the vertical name from a module path like
    'victor.coding.handlers' -> 'coding'.

    Args:
        module_name: Full module path (e.g., 'victor.coding.handlers')

    Returns:
        Vertical name if detected, None otherwise

    Example:
        >>> get_vertical_from_module('victor.coding.handlers')
        'coding'
        >>> get_vertical_from_module('victor.framework.handlers')
        None
    """
    if not module_name or not module_name.startswith("victor."):
        return None

    parts = module_name.split(".")
    if len(parts) < 3:
        return None

    # Second part should be the vertical name
    potential_vertical = parts[1]

    if potential_vertical in KNOWN_VERTICALS:
        return potential_vertical

    return None


def handler_decorator(
    name: str,
    *,
    vertical: Optional[str] = None,
    description: Optional[str] = None,
    replace: bool = False,
) -> Callable[[type[Any]], type[Any]]:
    """Class decorator for automatic handler registration.

    Phase 1.3: Provides decorator-based auto-registration for handlers,
    replacing the HANDLERS dict pattern in verticals.

    Usage:
        @handler_decorator("code_validation", vertical="coding")
        @dataclass
        class CodeValidationHandler(BaseHandler):
            async def execute(self, node, context, tool_registry) -> Tuple[Any, int]:
                ...

    Args:
        name: Handler name for registration (e.g., "code_validation")
        vertical: Optional vertical name. If not provided, attempts to
                  auto-detect from the module path.
        description: Optional handler description
        replace: If True, replace existing handler with same name

    Returns:
        Decorator function that registers the handler and returns the class

    Example:
        # With explicit vertical
        @handler_decorator("my_handler", vertical="coding")
        class MyHandler:
            pass

        # Auto-detect vertical from module path
        # In victor/coding/handlers.py:
        @handler_decorator("code_validation")  # vertical="coding" inferred
        class CodeValidationHandler:
            pass
    """

    def decorator(cls: type[Any]) -> type[Any]:
        # Determine vertical (explicit or auto-detect)
        final_vertical = vertical
        if final_vertical is None:
            # Try to auto-detect from module
            module_name = cls.__module__
            final_vertical = get_vertical_from_module(module_name)

        # Create handler instance
        try:
            handler_instance = cls()
        except Exception as e:
            logger.warning(f"Could not instantiate handler {name}: {e}. Registering class instead.")
            handler_instance = cls

        # Register with global registry
        get_handler_registry().register(
            name=name,
            handler=handler_instance,
            vertical=final_vertical,
            description=description,
            replace=replace,
        )

        logger.debug(f"Decorated handler registered: {name} (vertical={final_vertical})")

        # Return class unchanged
        return cls

    return decorator


__all__ = [
    "HandlerEntry",
    "HandlerRegistry",
    "get_handler_registry",
    "register_handler",
    "get_handler",
    "sync_handlers_with_executor",
    "discover_handlers_from_vertical",
    # Phase 1.3: Class decorator
    "handler_decorator",
    "get_vertical_from_module",
    "KNOWN_VERTICALS",
]
