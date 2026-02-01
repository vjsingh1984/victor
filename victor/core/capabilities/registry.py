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

"""Capability registry for centralized management (Phase 5.1).

This module provides a thread-safe singleton registry for capability management:
- Register capabilities from Python code
- Discover capabilities from entry points
- Load capabilities from YAML files
- Auto-generate handlers for configure/get operations

Design Principles:
- Thread-safe: All operations are protected by locks
- Singleton: Single global registry for consistent access
- Open/Closed: External capabilities via entry points
- YAML-first: Prefer YAML definitions over Python code

Example:
    from victor.core.capabilities.registry import CapabilityRegistry
    from victor.core.capabilities.types import CapabilityDefinition, CapabilityType

    # Get singleton instance
    registry = CapabilityRegistry.get_instance()

    # Register capability
    definition = CapabilityDefinition(
        name="git_safety",
        capability_type=CapabilityType.SAFETY,
        description="Git safety rules",
    )
    registry.register(definition)

    # Discover from entry points
    registry.discover_from_entry_points("victor.capabilities")

    # Load from YAML
    registry.load_from_yaml(Path("capabilities.yaml"))

    # Get handler
    handler = registry.get_handler("git_safety")
    handler.configure(context, {"block_force_push": True})
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import yaml

from victor.core.capabilities.types import CapabilityDefinition, CapabilityType
from victor.core.capabilities.handler import CapabilityHandler

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Python 3.9+ compatible entry_points import
if sys.version_info >= (3, 10):
    from importlib.metadata import entry_points
else:
    from importlib.metadata import entry_points


def validate_capability_yaml(data: dict[str, Any]) -> list[str]:
    """Validate YAML capability data against schema.

    Args:
        data: YAML data dictionary

    Returns:
        List of validation error messages (empty if valid)
    """
    errors: list[str] = []

    if "capabilities" not in data:
        # Empty file is valid (no capabilities defined)
        return errors

    capabilities = data.get("capabilities", [])
    if not isinstance(capabilities, list):
        errors.append("'capabilities' must be a list")
        return errors

    valid_types = {t.value for t in CapabilityType}

    for i, cap in enumerate(capabilities):
        prefix = f"capabilities[{i}]"

        if not isinstance(cap, dict):
            errors.append(f"{prefix}: must be a dictionary")
            continue

        # Required fields
        if "name" not in cap:
            errors.append(f"{prefix}: missing required field 'name'")
        elif not isinstance(cap["name"], str) or not cap["name"].strip():
            errors.append(f"{prefix}: 'name' must be a non-empty string")

        if "capability_type" not in cap:
            errors.append(f"{prefix}: missing required field 'capability_type'")
        elif cap["capability_type"] not in valid_types:
            errors.append(
                f"{prefix}: invalid capability_type '{cap['capability_type']}'. "
                f"Valid types: {sorted(valid_types)}"
            )

        # Optional field validation
        if "default_config" in cap and not isinstance(cap["default_config"], dict):
            errors.append(f"{prefix}: 'default_config' must be a dictionary")

        if "tags" in cap and not isinstance(cap["tags"], list):
            errors.append(f"{prefix}: 'tags' must be a list")

        if "dependencies" in cap and not isinstance(cap["dependencies"], list):
            errors.append(f"{prefix}: 'dependencies' must be a list")

    return errors


class CapabilityRegistry:
    """Thread-safe singleton registry for capability management.

    Provides centralized registration, discovery, and handler generation
    for capabilities across all verticals.

    Thread Safety:
        All public methods are protected by a reentrant lock.

    Example:
        registry = CapabilityRegistry.get_instance()
        registry.register(definition)

        # Get handler for configure/get operations
        handler = registry.get_handler("capability_name")
    """

    _instance: Optional["CapabilityRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize capability registry.

        Note: Use get_instance() instead of direct instantiation.
        """
        self._definitions: dict[str, CapabilityDefinition] = {}
        self._handlers: dict[str, CapabilityHandler] = {}
        self._op_lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "CapabilityRegistry":
        """Get singleton registry instance.

        Returns:
            CapabilityRegistry singleton
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing).

        Warning: Only use in tests to reset state.
        """
        with cls._lock:
            cls._instance = None

    def register(
        self,
        definition: CapabilityDefinition,
        replace: bool = False,
    ) -> None:
        """Register a capability definition.

        Args:
            definition: Capability definition to register
            replace: If True, replace existing definition

        Raises:
            ValueError: If capability already registered and replace=False
        """
        with self._op_lock:
            if definition.name in self._definitions and not replace:
                raise ValueError(
                    f"Capability '{definition.name}' is already registered. "
                    "Use replace=True to overwrite."
                )

            self._definitions[definition.name] = definition

            # Generate handler
            self._handlers[definition.name] = CapabilityHandler(definition)

            logger.debug(f"Registered capability '{definition.name}'")

    def unregister(self, name: str) -> bool:
        """Unregister a capability.

        Args:
            name: Capability name

        Returns:
            True if capability was unregistered, False if not found
        """
        with self._op_lock:
            if name in self._definitions:
                del self._definitions[name]
                del self._handlers[name]
                logger.debug(f"Unregistered capability '{name}'")
                return True
            return False

    def get(self, name: str) -> Optional[CapabilityDefinition]:
        """Get capability definition by name.

        Args:
            name: Capability name

        Returns:
            CapabilityDefinition or None if not found
        """
        with self._op_lock:
            return self._definitions.get(name)

    def get_handler(self, name: str) -> Optional[CapabilityHandler]:
        """Get auto-generated handler for capability.

        Args:
            name: Capability name

        Returns:
            CapabilityHandler or None if not found
        """
        with self._op_lock:
            return self._handlers.get(name)

    def list_all(self) -> list[str]:
        """List all registered capability names.

        Returns:
            List of capability names
        """
        with self._op_lock:
            return list(self._definitions.keys())

    def list_by_type(self, capability_type: CapabilityType) -> list[str]:
        """List capabilities by type.

        Args:
            capability_type: Type to filter by

        Returns:
            List of matching capability names
        """
        with self._op_lock:
            return [
                name
                for name, definition in self._definitions.items()
                if definition.capability_type == capability_type
            ]

    def list_by_tag(self, tag: str) -> list[str]:
        """List capabilities by tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of matching capability names
        """
        with self._op_lock:
            return [
                name for name, definition in self._definitions.items() if tag in definition.tags
            ]

    def discover_from_entry_points(self, group: str = "victor.capabilities") -> int:
        """Discover capabilities from entry points.

        Entry points can provide:
        - CapabilityDefinition instances directly
        - Classes with get_definition() classmethod
        - Callables returning CapabilityDefinition

        Args:
            group: Entry point group name

        Returns:
            Number of capabilities discovered
        """
        count = 0

        try:
            eps = entry_points()
            if hasattr(eps, "select"):
                # Python 3.10+ / importlib_metadata >= 3.6
                selected = eps.select(group=group)
            else:
                # Python 3.9 fallback
                selected = eps.get(group, [])  # type: ignore[arg-type]

            for ep in selected:
                try:
                    loaded = ep.load()
                    definition = self._resolve_definition(loaded, ep.name)

                    if definition:
                        self.register(definition, replace=True)
                        count += 1
                        logger.info(f"Discovered capability '{definition.name}' from entry point")
                except Exception as e:
                    logger.warning(
                        f"Failed to load entry point '{ep.name}': {e}",
                        exc_info=True,
                    )

        except Exception as e:
            logger.error(f"Error discovering entry points: {e}", exc_info=True)

        return count

    def _resolve_definition(self, loaded: Any, name: str) -> Optional[CapabilityDefinition]:
        """Resolve loaded entry point to CapabilityDefinition.

        Args:
            loaded: Loaded entry point object
            name: Entry point name (fallback for definition name)

        Returns:
            CapabilityDefinition or None if cannot be resolved
        """
        # Direct CapabilityDefinition
        if isinstance(loaded, CapabilityDefinition):
            return loaded

        # Class with get_definition() classmethod
        if isinstance(loaded, type) and hasattr(loaded, "get_definition"):
            method = loaded.get_definition
            if callable(method):
                result = method()
                if isinstance(result, CapabilityDefinition):
                    return result

        # Callable returning CapabilityDefinition
        if callable(loaded):
            try:
                result = loaded()
                if isinstance(result, CapabilityDefinition):
                    return result
            except Exception:
                pass

        logger.warning(f"Could not resolve entry point '{name}' to CapabilityDefinition")
        return None

    def load_from_yaml(self, path: Path) -> int:
        """Load capabilities from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Number of capabilities loaded

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If YAML is invalid
        """
        if not path.exists():
            raise FileNotFoundError(f"Capability file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        # Validate YAML schema
        errors = validate_capability_yaml(data)
        if errors:
            raise ValueError(f"Invalid capability YAML: {'; '.join(errors)}")

        count = 0
        for cap_data in data.get("capabilities", []):
            try:
                definition = CapabilityDefinition.from_yaml_dict(cap_data)
                self.register(definition, replace=True)
                count += 1
            except Exception as e:
                logger.warning(
                    f"Failed to load capability from YAML: {e}",
                    exc_info=True,
                )

        logger.info(f"Loaded {count} capabilities from {path}")
        return count

    def load_from_yaml_string(self, yaml_content: str) -> int:
        """Load capabilities from YAML string.

        Args:
            yaml_content: YAML content string

        Returns:
            Number of capabilities loaded
        """
        data = yaml.safe_load(yaml_content) or {}

        errors = validate_capability_yaml(data)
        if errors:
            raise ValueError(f"Invalid capability YAML: {'; '.join(errors)}")

        count = 0
        for cap_data in data.get("capabilities", []):
            try:
                definition = CapabilityDefinition.from_yaml_dict(cap_data)
                self.register(definition, replace=True)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to load capability: {e}")

        return count

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dictionary with registry stats
        """
        with self._op_lock:
            type_counts: dict[str, int] = {}
            for definition in self._definitions.values():
                type_name = definition.capability_type.value
                type_counts[type_name] = type_counts.get(type_name, 0) + 1

            return {
                "total_count": len(self._definitions),
                "type_counts": type_counts,
                "capabilities": self.list_all(),
            }


__all__ = [
    "CapabilityRegistry",
    "validate_capability_yaml",
]
