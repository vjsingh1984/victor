# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0

"""Extension provider protocol for ISP compliance.

This protocol defines the minimal interface for extension providers,
enabling type-safe extension access without hasattr() checks.
"""

from typing import Protocol, runtime_checkable, List, Callable, Any


@runtime_checkable
class ExtensionProviderProtocol(Protocol):
    """Protocol for objects that provide extensions.

    This protocol replaces hasattr() checks for extension-related methods,
    enabling type-safe extension access.

    Example:
        ```python
        @runtime_checkable
        class MyExtensionProvider(ExtensionProviderProtocol, Protocol):
            def get_extensions(self) -> List[Callable[..., Any]]:
                return self._extensions

            def register_extension(self, extension: Callable[..., Any]) -> None:
                self._extensions.append(extension)
        ```
    """

    def get_extensions(self) -> List[Callable[..., Any]]:
        """Get all available extensions.

        Returns:
            List of extension callables
        """
        ...

    def register_extension(self, extension: Callable[..., Any]) -> None:
        """Register a new extension.

        Args:
            extension: Extension callable to register
        """
        ...
