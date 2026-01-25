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

"""Migration helpers for adopting CapabilityInjector in verticals.

This module provides helper functions to make it easy for verticals to
adopt the new DI-based capability injection system.

Design Principles:
- SRP: Each helper function has a single responsibility
- OCP: Migration path without modifying existing code
- LSP: Migrated code is substitutable for original
- ISP: Narrow migration interface
- DIP: Depend on capability abstractions

Migration Pattern:
    # OLD (before migration):
    from victor.framework.capabilities import FileOperationsCapability

    class MyVertical(VerticalBase):
        def __init__(self):
            self._file_ops = FileOperationsCapability()

    # NEW (after migration):
    from victor.core.verticals import get_capability_injector

    class MyVertical(VerticalBase):
        def __init__(self):
            injector = get_capability_injector()
            self._file_ops = injector.get_capability("file_operations")
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Type, TypeVar, cast

if TYPE_CHECKING:
    from victor.core.verticals.capability_injector import CapabilityInjector

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Migration Decorators
# =============================================================================


def deprecated_direct_instantiation(
    capability_name: str,
    migration_guide: str = "",
) -> Callable[[type[T]], type[T]]:
    """Decorator to deprecate direct capability instantiation.

    Issues a deprecation warning when a capability is instantiated directly
    instead of using the CapabilityInjector.

    Args:
        capability_name: Name of the capability (e.g., "file_operations")
        migration_guide: Optional migration guide string

    Returns:
        Decorated class

    Example:
        @deprecated_direct_instantiation(
            "file_operations",
            "Use get_capability_injector().get_capability('file_operations')"
        )
        class FileOperationsCapability:
            pass
    """

    def decorator(cls: type[T]) -> type[T]:
        original_init = cls.__init__

        def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
            # Issue deprecation warning
            message = (
                f"Direct instantiation of {capability_name} is deprecated. "
                f"Use CapabilityInjector instead: get_capability_injector().get_capability('{capability_name}')"
            )
            if migration_guide:
                message += f"\n  Migration: {migration_guide}"
            warnings.warn(message, DeprecationWarning, stacklevel=2)

            # Call original init
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init  # type: ignore[method-assign]
        return cls

    return decorator


def migrate_capability_property(
    old_property_name: str,
    capability_name: str,
) -> property:
    """Create a migrating property that uses CapabilityInjector.

    Creates a property that automatically migrates from direct property
    access to CapabilityInjector-based access.

    Args:
        old_property_name: Name of the old property (for deprecation message)
        capability_name: Name of the capability in the injector

    Returns:
        Property descriptor

    Example:
        class MyVertical(VerticalBase):
            # Old way (direct instantiation):
            # _file_ops = FileOperationsCapability()

            # New way (migrated property):
            @migrate_capability_property("_file_ops", "file_operations")
            def file_ops(self):
                pass
    """

    def getter(self: Any) -> Any:
        from victor.core.verticals.capability_injector import get_capability_injector

        # Issue deprecation warning for old property access
        if hasattr(self, f"_{old_property_name}"):
            warnings.warn(
                f"Property '{old_property_name}' is deprecated. "
                f"Use CapabilityInjector: get_capability_injector().get_capability('{capability_name}')",
                DeprecationWarning,
                stacklevel=2,
            )
            return getattr(self, f"_{old_property_name}")

        # Use injector
        injector = get_capability_injector()
        return injector.get_capability(capability_name)

    return property(getter)


# =============================================================================
# Migration Helper Functions
# =============================================================================


def migrate_to_injector(
    vertical_class: Type[Any],
    capability_mappings: Dict[str, str],
    injector: Optional["CapabilityInjector"] = None,
) -> None:
    """Migrate a vertical class to use CapabilityInjector.

    This function modifies a vertical class to use CapabilityInjector
    instead of direct capability instantiation.

    Args:
        vertical_class: The vertical class to migrate
        capability_mappings: Mapping of old attribute names to capability names
            Example: {"_file_ops": "file_operations", "_web_ops": "web_operations"}
        injector: Optional injector instance (uses global if None)

    Example:
        # Before:
        class MyVertical(VerticalBase):
            def __init__(self):
                self._file_ops = FileOperationsCapability()
                self._web_ops = WebOperationsCapability()

        # After:
        migrate_to_injector(
            MyVertical,
            {"_file_ops": "file_operations", "_web_ops": "web_operations"}
        )
    """
    from victor.core.verticals.capability_injector import get_capability_injector

    if injector is None:
        injector = get_capability_injector()

    for old_attr, capability_name in capability_mappings.items():
        # Check if old attribute exists
        if hasattr(vertical_class, old_attr):
            logger.info(
                f"Migrating {vertical_class.__name__}.{old_attr} to use CapabilityInjector('{capability_name}')"
            )

            # Create property that uses injector
            def make_property(cap_name: str) -> property:
                def getter(self_obj: Any) -> Any:
                    return injector.get_capability(cap_name)

                return property(getter)

            # Replace attribute with property
            setattr(vertical_class, old_attr, make_property(capability_name))


def get_capability_or_create(
    capability_name: str,
    factory: Callable[[], T],
    injector: Optional["CapabilityInjector"] = None,
) -> T:
    """Get a capability from injector or create with factory if not found.

    This helper function provides a migration path for capabilities that
    aren't yet registered in the injector but will be in the future.

    Args:
        capability_name: Name of the capability
        factory: Factory function to create capability if not found
        injector: Optional injector instance (uses global if None)

    Returns:
        Capability instance

    Example:
        # Migration pattern for gradual adoption:
        def get_file_operations_capability():
            return get_capability_or_create(
                "file_operations",
                FileOperationsCapability,
            )
    """
    from victor.core.verticals.capability_injector import get_capability_injector

    if injector is None:
        injector = get_capability_injector()

    capability = injector.get_capability(capability_name)
    if capability is None:
        logger.debug(f"Capability '{capability_name}' not in injector, creating with factory")
        created = factory()
        # Type ignore needed because factory() returns Any but we need T
        return created  # type: ignore[no-any-return]
    # Type ignore: get_capability returns Any but we need T
    return capability  # type: ignore[no-any-return, misc]


# =============================================================================
# Migration Status Utilities
# =============================================================================


def check_migration_status(vertical_class: Type[Any]) -> dict[str, bool]:
    """Check migration status of a vertical class.

    Analyzes a vertical class to determine which capabilities have been
    migrated to CapabilityInjector and which are still using direct
    instantiation.

    Args:
        vertical_class: The vertical class to check (can be class or instance)

    Returns:
        Dictionary mapping capability names to migration status (True = migrated)

    Example:
        status = check_migration_status(MyVertical)
        # Returns: {"file_operations": True, "web_operations": False}
    """
    from victor.core.verticals.capability_injector import get_capability_injector

    status = {}

    # Check for common capability attributes
    common_capabilities = {
        "_file_ops": "file_operations",
        "_web_ops": "web_operations",
        "_git_ops": "git_operations",
        "_test_ops": "test_operations",
    }

    for old_attr, capability_name in common_capabilities.items():
        # Check if class has the attribute (could be class attribute or instance attribute)
        if hasattr(vertical_class, old_attr):
            # Get the attribute value
            try:
                attr_value = getattr(vertical_class, old_attr)
            except AttributeError:
                continue

            # Check if it's a property (likely migrated) or has value (not migrated)
            is_property = isinstance(attr_value, property)

            # If we can't tell (e.g., it's an instance), assume not migrated
            if not is_property and not isinstance(attr_value, type):
                # It's an instance attribute, check if it's from injector
                is_property = False

            status[capability_name] = is_property

    return status


def print_migration_report(vertical_class: Type[Any]) -> None:
    """Print a migration report for a vertical class.

    Args:
        vertical_class: The vertical class to analyze

    Example:
        print_migration_report(MyVertical)
        # Output:
        # Migration Report for MyVertical
        # ====================================
        # ✅ file_operations: Migrated
        # ❌ web_operations: Not migrated (direct instantiation)
        # ❌ git_operations: Not found
    """
    status = check_migration_status(vertical_class)

    print(f"\nMigration Report for {vertical_class.__name__}")
    print("=" * 50)

    if not status:
        print("No capabilities detected")
        return

    for capability_name, migrated in sorted(status.items()):
        if migrated:
            print(f"✅ {capability_name}: Migrated")
        else:
            print(f"❌ {capability_name}: Not migrated (direct instantiation)")

    print()


__all__ = [
    # Decorators
    "deprecated_direct_instantiation",
    "migrate_capability_property",
    # Migration helpers
    "migrate_to_injector",
    "get_capability_or_create",
    # Status utilities
    "check_migration_status",
    "print_migration_report",
]
