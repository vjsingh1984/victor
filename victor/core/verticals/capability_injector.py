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

"""Capability injector for verticals.

Phase 1.4: Auto-inject FileOperationsCapability via DI.

This module provides a capability injector that manages shared capabilities
for verticals, eliminating the need for each vertical to instantiate
FileOperationsCapability independently.

Design Pattern:
- Singleton pattern for global injector instance
- Factory pattern for capability creation
- DI-compatible for container registration
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from victor.core.container import ServiceContainer
    from victor.framework.capabilities import FileOperationsCapability

logger = logging.getLogger(__name__)

# Global singleton instance
_injector_instance: Optional["CapabilityInjector"] = None


class CapabilityInjector:
    """Injector for shared vertical capabilities.

    This class manages singleton instances of capabilities that are
    shared across multiple verticals. It supports both DI injection
    and direct instantiation for backward compatibility.

    Usage:
        # Via global function (recommended)
        injector = get_capability_injector()
        file_ops = injector.get_file_operations_capability()

        # Via DI container
        injector = container.get(CapabilityInjector)
        file_ops = injector.get_file_operations_capability()

        # With custom capability (for testing)
        injector = CapabilityInjector(container, file_operations=custom_cap)

    Attributes:
        _container: Optional DI container reference
        _file_ops: Cached FileOperationsCapability instance
    """

    def __init__(
        self,
        container: Optional["ServiceContainer"] = None,
        *,
        file_operations: Optional["FileOperationsCapability"] = None,
    ) -> None:
        """Initialize the capability injector.

        Args:
            container: Optional DI container for service resolution
            file_operations: Optional custom FileOperationsCapability instance
        """
        self._container = container
        self._file_ops: Optional["FileOperationsCapability"] = file_operations
        logger.debug("CapabilityInjector initialized")

    def get_file_operations_capability(self) -> "FileOperationsCapability":
        """Get the FileOperationsCapability instance.

        Returns a singleton instance of FileOperationsCapability.
        Creates the instance on first call if not already provided.

        Returns:
            FileOperationsCapability instance
        """
        if self._file_ops is None:
            from victor.framework.capabilities import FileOperationsCapability

            self._file_ops = FileOperationsCapability()
            logger.debug("Created FileOperationsCapability instance")

        return self._file_ops

    @classmethod
    def reset(cls) -> None:
        """Reset the global singleton instance.

        Used for testing to ensure clean state between tests.
        """
        global _injector_instance
        _injector_instance = None
        logger.debug("CapabilityInjector singleton reset")


def get_capability_injector() -> CapabilityInjector:
    """Get the global capability injector singleton.

    Returns:
        CapabilityInjector instance
    """
    global _injector_instance
    if _injector_instance is None:
        _injector_instance = CapabilityInjector()
    return _injector_instance


def create_capability_injector(
    container: "ServiceContainer",
) -> CapabilityInjector:
    """Factory function for creating CapabilityInjector via DI.

    This function is used for DI container registration.

    Args:
        container: DI container for service resolution

    Returns:
        CapabilityInjector instance
    """
    return CapabilityInjector(container)


__all__ = [
    "CapabilityInjector",
    "get_capability_injector",
    "create_capability_injector",
]
