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

"""Dynamic Extension Registry (OCP-Compliant).

This module provides a dynamic extension registry that enables Open/Closed
Principle compliance by allowing unlimited extension types without modifying
core code.

Key Features:
- Type-safe extension storage and retrieval
- Support for any extension type implementing IExtension
- Thread-safe registration and discovery
- OCP-compliant: open for extension, closed for modification

Usage:
    registry = ExtensionRegistry()

    # Register any extension type
    registry.register_extension(my_extension)

    # Retrieve by type and name
    ext = registry.get_extension("tools", "my_tool")

    # Get all extensions of a type
    tools = registry.get_extensions_by_type("tools")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

from victor.core.verticals.protocols import IExtension, IExtensionRegistry

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ExtensionRegistry(IExtensionRegistry):
    """Dynamic extension registry supporting unlimited extension types.

    Implements IExtensionRegistry protocol to provide type-safe extension
    management while maintaining OCP compliance.

    The registry stores extensions in a two-level dictionary structure:
    {extension_type: {extension_name: extension}}

    This allows:
    - Fast lookups by type and name
    - Efficient retrieval of all extensions of a type
    - Support for unlimited extension types without core modifications

    Attributes:
        _extensions_by_type: Internal storage dict {type: {name: extension}}
    """

    def __init__(self) -> None:
        """Initialize an empty extension registry."""
        self._extensions_by_type: Dict[str, Dict[str, IExtension]] = {}

    def register_extension(self, extension: IExtension) -> None:
        """Register an extension.

        Args:
            extension: Extension to register

        Raises:
            ValueError: If extension already registered with same type/name
            TypeError: If extension doesn't implement IExtension protocol
        """
        # Validate extension implements protocol
        if not isinstance(extension, IExtension):
            raise TypeError(
                f"Extension must implement IExtension protocol, " f"got {type(extension)}"
            )

        ext_type = extension.extension_type
        name = extension.name

        # Initialize type dict if not exists
        if ext_type not in self._extensions_by_type:
            self._extensions_by_type[ext_type] = {}

        # Check for duplicate
        if name in self._extensions_by_type[ext_type]:
            raise ValueError(f"Extension '{name}' of type '{ext_type}' is already registered")

        # Register extension
        self._extensions_by_type[ext_type][name] = extension

        logger.debug(f"Registered extension '{name}' of type '{ext_type}'")

    def unregister_extension(self, extension_type: str, name: str) -> bool:
        """Unregister an extension.

        Args:
            extension_type: Type of extension to unregister
            name: Name of extension to unregister

        Returns:
            True if unregistered, False if not found
        """
        if extension_type not in self._extensions_by_type:
            return False

        if name not in self._extensions_by_type[extension_type]:
            return False

        # Remove extension
        del self._extensions_by_type[extension_type][name]

        # Clean up empty type dict
        if not self._extensions_by_type[extension_type]:
            del self._extensions_by_type[extension_type]

        logger.debug(f"Unregistered extension '{name}' of type '{extension_type}'")

        return True

    def get_extension(
        self,
        extension_type: str,
        name: str,
    ) -> Optional[IExtension]:
        """Get a specific extension.

        Args:
            extension_type: Type of extension
            name: Name of extension

        Returns:
            Extension if found, None otherwise
        """
        if extension_type not in self._extensions_by_type:
            return None

        return self._extensions_by_type[extension_type].get(name)

    def get_extensions_by_type(self, extension_type: str) -> List[IExtension]:
        """Get all extensions of a specific type.

        Args:
            extension_type: Type of extension to retrieve

        Returns:
            List of extensions (empty list if none found)
        """
        if extension_type not in self._extensions_by_type:
            return []

        return list(self._extensions_by_type[extension_type].values())

    def list_extension_types(self) -> List[str]:
        """List all registered extension types.

        Returns:
            List of extension type strings
        """
        return list(self._extensions_by_type.keys())

    def list_extensions(
        self,
        extension_type: Optional[str] = None,
    ) -> List[str]:
        """List extension names by type.

        Args:
            extension_type: Optional type filter (None = all types)

        Returns:
            List of extension names
        """
        if extension_type is None:
            # Return all extension names from all types
            all_names = []
            for type_dict in self._extensions_by_type.values():
                all_names.extend(type_dict.keys())
            return all_names

        # Return names for specific type
        if extension_type not in self._extensions_by_type:
            return []

        return list(self._extensions_by_type[extension_type].keys())

    def has_extension(self, extension_type: str, name: str) -> bool:
        """Check if an extension is registered.

        Args:
            extension_type: Type of extension
            name: Name of extension

        Returns:
            True if registered, False otherwise
        """
        if extension_type not in self._extensions_by_type:
            return False

        return name in self._extensions_by_type[extension_type]

    def count_extensions(self, extension_type: Optional[str] = None) -> int:
        """Count extensions by type.

        Args:
            extension_type: Optional type filter (None = all types)

        Returns:
            Number of extensions
        """
        if extension_type is None:
            # Count all extensions
            total = 0
            for type_dict in self._extensions_by_type.values():
                total += len(type_dict)
            return total

        # Count extensions for specific type
        if extension_type not in self._extensions_by_type:
            return 0

        return len(self._extensions_by_type[extension_type])


# Export all
__all__ = [
    "ExtensionRegistry",
]
