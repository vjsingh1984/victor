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

"""Base registry classes for DRY singleton registry pattern.

This module provides base classes for implementing singleton registries,
eliminating code duplication across the codebase. Multiple registry classes
(ProgressiveToolsRegistry, SharedToolRegistry, ToolPluginRegistry, etc.)
share the same singleton pattern - this consolidates that logic.

Design Pattern: Template Method + Singleton
==========================================
- SingletonRegistry provides the singleton infrastructure
- Subclasses implement domain-specific registration logic
- Thread-safe with double-checked locking

Usage:
    from victor.core.registry_base import SingletonRegistry

    class MyRegistry(SingletonRegistry["MyRegistry"]):
        def __init__(self):
            super().__init__()
            self._items = {}

        def register(self, name: str, item: Any) -> None:
            self._items[name] = item

        def get(self, name: str) -> Optional[Any]:
            return self._items.get(name)

    # Usage
    registry = MyRegistry.get_instance()
    registry.register("foo", some_item)

    # For testing
    MyRegistry.reset_instance()
"""

from __future__ import annotations

import logging
import threading
from abc import ABC
from typing import Any, ClassVar, Dict, Generic, List, Optional, Set, TypeVar, cast

logger = logging.getLogger(__name__)

# Type variable for self-referential generic typing
T = TypeVar("T", bound="SingletonRegistry[Any]")


class SingletonRegistry(ABC, Generic[T]):
    """Base class for thread-safe singleton registries.

    This class provides the common singleton pattern infrastructure used by
    many registry classes throughout Victor. It eliminates code duplication
    and ensures consistent thread-safe behavior.

    Features:
    - Thread-safe singleton with double-checked locking
    - Optional protection against direct instantiation
    - Consistent reset_instance() for test isolation
    - Generic typing for proper subclass type hints

    Thread Safety:
        All singleton operations are protected by a class-level lock.
        Subclasses should use their own locks for data access if needed.

    Attributes:
        _instance: The singleton instance (class variable)
        _lock: Threading lock for thread-safe singleton access (class variable)
        _instantiation_allowed: Flag to control direct instantiation (class variable)
    """

    # Class-level singleton state (overridden by each subclass)
    _instance: ClassVar[Optional[T]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _instantiation_allowed: ClassVar[bool] = True  # Set False for strict singletons

    def __init__(self) -> None:
        """Initialize the registry.

        Subclasses should call super().__init__() and initialize their data.

        Raises:
            RuntimeError: If _instantiation_allowed is False and called directly
        """
        cls = type(self)
        if not cls._instantiation_allowed and cls._instance is not None:
            raise RuntimeError(
                f"{cls.__name__} is a singleton. Use {cls.__name__}.get_instance() "
                "to get the singleton instance."
            )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize subclass with its own singleton state.

        Each subclass gets its own _instance and _lock to maintain
        separate singleton instances per class hierarchy.
        """
        super().__init_subclass__(**kwargs)
        cls._instance = None
        cls._lock = threading.Lock()
        cls._instantiation_allowed = True

    @classmethod
    def get_instance(cls: type[T]) -> T:
        """Get the singleton instance of this registry.

        This method is thread-safe and ensures only one instance is created
        even when called concurrently from multiple threads.

        Returns:
            The singleton registry instance
        """
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking pattern
                if cls._instance is None:
                    was_allowed = cls._instantiation_allowed
                    cls._instantiation_allowed = True
                    try:
                        instance = cls()
                        cls._instance = instance
                        logger.debug(f"{cls.__name__} singleton instance created")
                    finally:
                        cls._instantiation_allowed = was_allowed
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance.

        This method is primarily intended for test isolation. It allows
        tests to start with a fresh registry instance.

        Thread-safe but should only be called during test setup/teardown.
        """
        with cls._lock:
            cls._instance = None
            logger.debug(f"{cls.__name__} singleton instance reset")

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if the singleton instance exists.

        Returns:
            True if the singleton instance has been created
        """
        return cls._instance is not None


class ItemRegistry(SingletonRegistry[T], Generic[T]):
    """Base class for registries that store items by name.

    Extends SingletonRegistry with common item storage patterns used by
    tool registries, plugin registries, etc.

    Type Parameters:
        T: The registry subclass type
        ItemT: The type of items stored in the registry (implicit via methods)
    """

    def __init__(self) -> None:
        """Initialize the item registry with empty storage."""
        super().__init__()
        self._items: Dict[str, Any] = {}
        self._items_lock: threading.RLock = threading.RLock()

    def register(self, name: str, item: Any) -> None:
        """Register an item by name.

        Args:
            name: Unique name for the item
            item: The item to register
        """
        with self._items_lock:
            self._items[name] = item
            logger.debug(f"{type(self).__name__}: Registered '{name}'")

    def unregister(self, name: str) -> bool:
        """Unregister an item by name.

        Args:
            name: Name of the item to remove

        Returns:
            True if item was found and removed
        """
        with self._items_lock:
            if name in self._items:
                del self._items[name]
                logger.debug(f"{type(self).__name__}: Unregistered '{name}'")
                return True
            return False

    def get(self, name: str) -> Optional[Any]:
        """Get an item by name.

        Args:
            name: Name of the item to retrieve

        Returns:
            The item or None if not found
        """
        with self._items_lock:
            return self._items.get(name)

    def contains(self, name: str) -> bool:
        """Check if an item is registered.

        Args:
            name: Name to check

        Returns:
            True if item exists
        """
        with self._items_lock:
            return name in self._items

    def list_names(self) -> List[str]:
        """Get list of all registered item names.

        Returns:
            List of registered names
        """
        with self._items_lock:
            return list(self._items.keys())

    def list_items(self) -> List[Any]:
        """Get list of all registered items.

        Returns:
            List of registered items
        """
        with self._items_lock:
            return list(self._items.values())

    def count(self) -> int:
        """Get count of registered items.

        Returns:
            Number of registered items
        """
        with self._items_lock:
            return len(self._items)

    def clear(self) -> int:
        """Clear all registered items.

        Returns:
            Number of items that were cleared
        """
        with self._items_lock:
            count = len(self._items)
            self._items.clear()
            logger.debug(f"{type(self).__name__}: Cleared {count} items")
            return count


__all__ = [
    "SingletonRegistry",
    "ItemRegistry",
]
