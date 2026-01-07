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

"""Event Category Registry - Extensible event category system for Victor.

This module provides a registry for custom event categories, allowing
verticals and plugins to define their own event types beyond the built-in
EventCategory enum.

Design Pattern:
    Singleton with thread-safe registration.

Example:
    from victor.observability.event_registry import EventCategoryRegistry
from victor.core.events import Event, ObservabilityBus, get_observability_bus

    registry = EventCategoryRegistry.get_instance()

    # Register a custom category
    registry.register(
        name="security_audit",
        description="Security-related audit events",
        registered_by="victor.security",
    )

    # Check if category exists
    if registry.has_category("security_audit"):
        # Use the custom category
        pass
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Set


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CustomEventCategory:
    """Represents a custom event category registered by a plugin or vertical.

    Attributes:
        name: Unique identifier for the category.
        description: Human-readable description of what events this category covers.
        registered_by: Module or component that registered this category.
        registered_at: Timestamp when the category was registered.
    """

    name: str
    description: str
    registered_by: str
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Validate category attributes."""
        if not self.name:
            raise ValueError("Category name cannot be empty")
        if not self.name.isidentifier() or not self.name.islower():
            raise ValueError(
                f"Category name '{self.name}' must be a valid lowercase identifier "
                "(e.g., 'security_audit', 'plugin_events')"
            )


class EventCategoryRegistry:
    """Singleton registry for custom event categories.

    This registry allows verticals and plugins to define custom event
    categories beyond the built-in EventCategory enum. Custom categories
    can be used with EventCategory.CUSTOM events that include a
    'custom_category' field in their data.

    Thread-safe for concurrent registration and lookup.

    Example:
        registry = EventCategoryRegistry.get_instance()

        # Register custom category
        registry.register(
            name="security_audit",
            description="Security audit trail events",
            registered_by="victor.security",
        )

        # List all categories (built-in + custom)
        all_cats = registry.list_all()
        # {'tool', 'state', 'model', ..., 'security_audit'}
    """

    _instance: Optional["EventCategoryRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "EventCategoryRegistry":
        """Create or return singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the registry."""
        if getattr(self, "_initialized", False):
            return

        self._categories: Dict[str, CustomEventCategory] = {}
        self._instance_lock = threading.Lock()
        self._initialized = True

    @classmethod
    def get_instance(cls) -> "EventCategoryRegistry":
        """Get the singleton EventCategoryRegistry instance.

        Returns:
            EventCategoryRegistry singleton.
        """
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing).

        Warning: Only use in tests!
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance._categories = {}

    def register(
        self,
        name: str,
        description: str,
        registered_by: str,
    ) -> CustomEventCategory:
        """Register a custom event category.

        Args:
            name: Unique identifier for the category (lowercase, valid Python identifier).
            description: Human-readable description of the category.
            registered_by: Module or component registering this category.

        Returns:
            The registered CustomEventCategory.

        Raises:
            ValueError: If name is invalid or already registered by a different source.

        Example:
            category = registry.register(
                name="ml_pipeline",
                description="ML pipeline execution events",
                registered_by="victor.dataanalysis",
            )
        """
        # Check if it conflicts with built-in categories
        builtin_names = {cat.value for cat in EventCategory}
        if name in builtin_names:
            raise ValueError(
                f"Category '{name}' conflicts with built-in EventCategory. "
                f"Use EventCategory.{name.upper()} instead."
            )

        category = CustomEventCategory(
            name=name,
            description=description,
            registered_by=registered_by,
        )

        with self._instance_lock:
            if name in self._categories:
                existing = self._categories[name]
                if existing.registered_by != registered_by:
                    raise ValueError(
                        f"Category '{name}' already registered by '{existing.registered_by}'. "
                        f"Cannot re-register from '{registered_by}'."
                    )
                # Same registrant - idempotent, return existing
                logger.debug(f"Category '{name}' already registered by '{registered_by}'")
                return existing

            self._categories[name] = category
            logger.info(f"Registered custom event category '{name}' by '{registered_by}'")

        return category

    def unregister(self, name: str, registered_by: str) -> bool:
        """Unregister a custom event category.

        Only the original registrant can unregister a category.

        Args:
            name: Category name to unregister.
            registered_by: Module that originally registered the category.

        Returns:
            True if category was unregistered, False if not found.

        Raises:
            ValueError: If trying to unregister a category registered by someone else.
        """
        with self._instance_lock:
            if name not in self._categories:
                return False

            existing = self._categories[name]
            if existing.registered_by != registered_by:
                raise ValueError(
                    f"Category '{name}' was registered by '{existing.registered_by}', "
                    f"cannot unregister from '{registered_by}'."
                )

            del self._categories[name]
            logger.info(f"Unregistered custom event category '{name}'")
            return True

    def has_category(self, name: str) -> bool:
        """Check if a category exists (built-in or custom).

        Args:
            name: Category name to check.

        Returns:
            True if category exists.
        """
        # Check built-in categories
        builtin_names = {cat.value for cat in EventCategory}
        if name in builtin_names:
            return True

        # Check custom categories
        with self._instance_lock:
            return name in self._categories

    def get_category(self, name: str) -> Optional[CustomEventCategory]:
        """Get a custom category by name.

        Args:
            name: Category name to retrieve.

        Returns:
            CustomEventCategory if found, None otherwise.
            Note: Returns None for built-in categories (use EventCategory enum).
        """
        with self._instance_lock:
            return self._categories.get(name)

    def list_custom(self) -> Set[str]:
        """List all registered custom category names.

        Returns:
            Set of custom category names.
        """
        with self._instance_lock:
            return set(self._categories.keys())

    def list_all(self) -> Set[str]:
        """List all category names (built-in + custom).

        Returns:
            Set of all category names.
        """
        builtin_names = {cat.value for cat in EventCategory}
        with self._instance_lock:
            return builtin_names | set(self._categories.keys())

    def get_all_custom(self) -> Dict[str, CustomEventCategory]:
        """Get all custom categories as a dictionary.

        Returns:
            Dictionary mapping names to CustomEventCategory objects.
        """
        with self._instance_lock:
            return dict(self._categories)

    def count(self) -> int:
        """Get count of custom categories.

        Returns:
            Number of registered custom categories.
        """
        with self._instance_lock:
            return len(self._categories)
