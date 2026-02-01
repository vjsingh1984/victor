# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0

"""Tiered config provider protocol for ISP compliance.

This protocol defines the minimal interface for tiered configuration,
replacing duck typing with type-safe protocol conformance.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class TieredConfigProviderProtocol(Protocol):
    """Protocol for tiered tool configuration.

    This protocol replaces hasattr() checks for tiered config objects,
    enabling type-safe access to mandatory and core tool sets.

    Example:
        ```python
        @runtime_checkable
        class MyTieredConfig(TieredConfigProviderProtocol, Protocol):
            @property
            def mandatory(self) -> Set[str]:
                return self._mandatory_tools

            @property
            def core(self) -> Set[str]:
                return self._core_tools
        ```
    """

    @property
    def mandatory(self) -> set[str]:
        """Get the set of mandatory tools.

        Returns:
            Set of mandatory tool names that must be available
        """
        ...

    @property
    def core(self) -> set[str]:
        """Get the set of core tools.

        Returns:
            Set of core tool names for standard operations
        """
        ...
