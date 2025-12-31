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

"""Generic base registry implementation."""

from typing import Dict, Generic, Iterator, List, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class BaseRegistry(Generic[K, V]):
    """Generic registry base class for all registry implementations.

    Provides a consistent interface for registering, retrieving, and managing
    items by key. This class can be extended to create specialized registries
    for tools, providers, or any other type of managed object.

    Type Parameters:
        K: The key type (typically str)
        V: The value type (e.g., BaseTool, BaseProvider)

    Example:
        >>> class ToolRegistry(BaseRegistry[str, BaseTool]):
        ...     pass
        >>> registry = ToolRegistry()
        >>> registry.register("my_tool", my_tool_instance)
        >>> registry.get("my_tool")
        <BaseTool instance>

    Attributes:
        _items: Internal dictionary storing registered items
    """

    def __init__(self) -> None:
        """Initialize the registry with an empty items dictionary."""
        self._items: Dict[K, V] = {}

    def register(self, key: K, value: V) -> None:
        """Register an item with the given key.

        Args:
            key: The unique identifier for the item
            value: The item to register

        Note:
            If an item with the same key already exists, it will be overwritten.
        """
        self._items[key] = value

    def get(self, key: K) -> Optional[V]:
        """Get an item by key.

        Args:
            key: The unique identifier for the item

        Returns:
            The registered item, or None if not found
        """
        return self._items.get(key)

    def list_all(self) -> List[K]:
        """List all registered keys.

        Returns:
            A list of all keys in the registry
        """
        return list(self._items.keys())

    def unregister(self, key: K) -> bool:
        """Unregister an item by key.

        Args:
            key: The unique identifier for the item to remove

        Returns:
            True if the item was found and removed, False otherwise
        """
        if key in self._items:
            del self._items[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all items from the registry."""
        self._items.clear()

    def __contains__(self, key: K) -> bool:
        """Check if a key exists in the registry.

        Args:
            key: The key to check

        Returns:
            True if the key exists, False otherwise
        """
        return key in self._items

    def __len__(self) -> int:
        """Get the number of items in the registry.

        Returns:
            The count of registered items
        """
        return len(self._items)

    def __iter__(self) -> Iterator[K]:
        """Iterate over registered keys.

        Returns:
            An iterator over the registry keys
        """
        return iter(self._items)

    def values(self) -> List[V]:
        """Get all registered values.

        Returns:
            A list of all registered values
        """
        return list(self._items.values())

    def items(self) -> List[tuple[K, V]]:
        """Get all key-value pairs.

        Returns:
            A list of (key, value) tuples
        """
        return list(self._items.items())
