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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
        self._handlers: Dict[str, HandlerEntry] = {}

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

    def list_handlers(self) -> List[str]:
        """List all registered handler names.

        Returns:
            List of handler names
        """
        return list(self._handlers.keys())

    def list_entries(self) -> List[HandlerEntry]:
        """List all handler entries with metadata.

        Returns:
            List of HandlerEntry objects
        """
        return list(self._handlers.values())

    def list_by_vertical(self, vertical: str) -> List[str]:
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
        handlers: Dict[str, Any],
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
    ) -> Tuple[int, int]:
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

        Attempts to import victor.{vertical_name}.handlers and register
        all handlers from its HANDLERS dictionary.

        Args:
            vertical_name: Name of the vertical (e.g., "coding", "research")
            replace: If True, replace existing handlers
            sync_to_executor: If True, sync new handlers to executor

        Returns:
            Number of handlers registered

        Raises:
            ImportError: If the handlers module cannot be imported

        Example:
            registry = get_handler_registry()

            # Discover and register handlers from coding vertical
            count = registry.discover_from_vertical("coding")
            print(f"Registered {count} handlers")
        """
        module_path = f"victor.{vertical_name}.handlers"

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            logger.warning(f"Could not import handlers from {module_path}: {e}")
            raise

        handlers_dict = getattr(module, "HANDLERS", None)
        if handlers_dict is None:
            logger.warning(f"No HANDLERS dict found in {module_path}")
            return 0

        count = self.register_from_vertical(
            vertical_name=vertical_name,
            handlers=handlers_dict,
            replace=replace,
        )

        # Optionally sync to executor
        if sync_to_executor and count > 0:
            self.sync_with_executor(direction="to_executor", replace=replace)

        logger.info(f"Discovered {count} handlers from {module_path}")
        return count

    def list_verticals(self) -> List[str]:
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
) -> Tuple[int, int]:
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


__all__ = [
    "HandlerEntry",
    "HandlerRegistry",
    "get_handler_registry",
    "register_handler",
    "get_handler",
    "sync_handlers_with_executor",
    "discover_handlers_from_vertical",
]
