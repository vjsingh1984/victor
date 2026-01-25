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

"""Escape Hatch Registry for YAML workflow conditions and transforms.

Provides a centralized registry for escape hatch functions used in YAML
workflows. Escape hatches are Python functions that implement complex
conditions or transforms that cannot be expressed in YAML.

Example:
    from victor.framework.escape_hatch_registry import (
        EscapeHatchRegistry,
        condition,
        transform,
    )

    @condition("tests_passing", vertical="coding")
    def tests_passing(ctx: Dict[str, Any]) -> str:
        if ctx.get("test_results", {}).get("failed", 0) > 0:
            return "failing"
        return "passing"

    @transform("merge_results", vertical="coding")
    def merge_results(ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {"merged": ctx.get("a", []) + ctx.get("b", [])}

    # In YAML workflow:
    - id: check_tests
      type: condition
      condition: "tests_passing"
      branches:
        "passing": deploy
        "failing": fix_code
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, TypeVar, cast

logger = logging.getLogger(__name__)

# Type variable for decorator return types
F = TypeVar("F", bound=Callable[..., Any])


class ConditionFunction(Protocol):
    """Protocol for condition functions used in YAML workflow condition nodes.

    Condition functions evaluate workflow context and return a string
    that determines which branch to take in the workflow.

    Example:
        def my_condition(ctx: Dict[str, Any]) -> str:
            if ctx.get("score", 0) > 0.8:
                return "high"
            return "low"
    """

    def __call__(self, ctx: Dict[str, Any]) -> str:
        """Evaluate the condition.

        Args:
            ctx: Workflow context dictionary containing execution state

        Returns:
            String identifying which branch to take
        """
        ...


class TransformFunction(Protocol):
    """Protocol for transform functions used in YAML workflow transform nodes.

    Transform functions modify workflow context by returning updated values
    that are merged into the context.

    Example:
        def my_transform(ctx: Dict[str, Any]) -> Dict[str, Any]:
            return {"computed": ctx.get("a", 0) + ctx.get("b", 0)}
    """

    def __call__(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the transform.

        Args:
            ctx: Workflow context dictionary containing execution state

        Returns:
            Dictionary of values to merge into context
        """
        ...


class EscapeHatchRegistry:
    """Singleton registry for escape hatch conditions and transforms.

    Provides centralized registration and lookup of escape hatch functions
    organized by vertical. Supports both explicit registration and
    decorator-based registration.

    Thread Safety:
        This implementation is NOT thread-safe. For concurrent access,
        use external synchronization.

    Example:
        registry = EscapeHatchRegistry.get_instance()

        # Register functions
        registry.register_condition("my_cond", my_func, vertical="coding")
        registry.register_transform("my_trans", trans_func, vertical="coding")

        # Lookup functions
        conditions, transforms = registry.get_registry_for_vertical("coding")
        fn = conditions.get("my_cond")
    """

    _instance: Optional["EscapeHatchRegistry"] = None

    # Class-level storage to persist registrations across instance resets
    # This ensures escape hatches survive singleton resets during tests
    _class_conditions: Dict[str, Dict[str, ConditionFunction]] = {}
    _class_transforms: Dict[str, Dict[str, TransformFunction]] = {}
    _class_global_conditions: Dict[str, ConditionFunction] = {}
    _class_global_transforms: Dict[str, TransformFunction] = {}

    def __init__(self) -> None:
        """Initialize escape hatch registry with persisted registrations."""
        # Copy from class-level storage (persists across instance resets)
        self._conditions = {k: dict(v) for k, v in self._class_conditions.items()}
        self._transforms = {k: dict(v) for k, v in self._class_transforms.items()}
        self._global_conditions = dict(self._class_global_conditions)
        self._global_transforms = dict(self._class_global_transforms)

    @classmethod
    def get_instance(cls) -> "EscapeHatchRegistry":
        """Get singleton instance of the registry.

        Returns:
            The global EscapeHatchRegistry instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    def register_condition(
        self,
        name: str,
        fn: ConditionFunction,
        vertical: Optional[str] = None,
        *,
        replace: bool = False,
    ) -> None:
        """Register a condition function.

        Args:
            name: Condition name (referenced in YAML)
            fn: Condition function
            vertical: Optional vertical namespace
            replace: If True, replace existing condition

        Raises:
            ValueError: If condition already exists and replace is False
        """
        if vertical:
            if vertical not in self._conditions:
                self._conditions[vertical] = {}
            if vertical not in self._class_conditions:
                self._class_conditions[vertical] = {}
            registry = self._conditions[vertical]
            class_registry = self._class_conditions[vertical]
        else:
            registry = self._global_conditions
            class_registry = self._class_global_conditions

        if name in registry and not replace:
            raise ValueError(
                f"Condition '{name}' already registered"
                + (f" for vertical '{vertical}'" if vertical else "")
                + ". Use replace=True to override."
            )

        registry[name] = fn
        class_registry[name] = fn  # Also store at class level
        logger.debug(f"Registered condition: {name} (vertical={vertical})")

    def register_transform(
        self,
        name: str,
        fn: TransformFunction,
        vertical: Optional[str] = None,
        *,
        replace: bool = False,
    ) -> None:
        """Register a transform function.

        Args:
            name: Transform name (referenced in YAML)
            fn: Transform function
            vertical: Optional vertical namespace
            replace: If True, replace existing transform

        Raises:
            ValueError: If transform already exists and replace is False
        """
        if vertical:
            if vertical not in self._transforms:
                self._transforms[vertical] = {}
            if vertical not in self._class_transforms:
                self._class_transforms[vertical] = {}
            registry = self._transforms[vertical]
            class_registry = self._class_transforms[vertical]
        else:
            registry = self._global_transforms
            class_registry = self._class_global_transforms

        if name in registry and not replace:
            raise ValueError(
                f"Transform '{name}' already registered"
                + (f" for vertical '{vertical}'" if vertical else "")
                + ". Use replace=True to override."
            )

        registry[name] = fn
        class_registry[name] = fn  # Also store at class level
        logger.debug(f"Registered transform: {name} (vertical={vertical})")

    def register_from_vertical(
        self,
        vertical: str,
        conditions: Optional[Dict[str, ConditionFunction]] = None,
        transforms: Optional[Dict[str, TransformFunction]] = None,
        *,
        replace: bool = False,
    ) -> Tuple[int, int]:
        """Bulk register conditions and transforms from a vertical.

        Convenience method for registering all escape hatches from a
        vertical's CONDITIONS and TRANSFORMS dictionaries.

        Args:
            vertical: Vertical name
            conditions: Dict of condition name -> function
            transforms: Dict of transform name -> function
            replace: If True, replace existing entries

        Returns:
            Tuple of (conditions_registered, transforms_registered)

        Example:
            # In your vertical's escape_hatches.py module:
            # CONDITIONS = {"my_condition": my_condition_func}
            # TRANSFORMS = {"my_transform": my_transform_func}
            #
            # Then register:
            registry.register_from_vertical(
                "my_vertical",
                conditions=CONDITIONS,
                transforms=TRANSFORMS,
            )
        """
        cond_count = 0
        trans_count = 0

        if conditions:
            for name, fn in conditions.items():
                # Explicitly cast to avoid protocol confusion
                self.register_condition(name, fn, vertical=vertical, replace=replace)
                cond_count += 1

        if transforms:
            for name, fn in transforms.items():
                # Explicitly cast to avoid protocol confusion
                self.register_transform(name, fn, vertical=vertical, replace=replace)
                trans_count += 1

        logger.debug(
            f"Registered {cond_count} conditions and {trans_count} transforms "
            f"for vertical '{vertical}'"
        )

        return (cond_count, trans_count)

    def get_registry_for_vertical(
        self,
        vertical: str,
        *,
        include_global: bool = True,
    ) -> Tuple[Dict[str, ConditionFunction], Dict[str, TransformFunction]]:
        """Get all escape hatches for a vertical.

        Returns merged dictionaries containing both vertical-specific
        and global escape hatches (if include_global is True).

        Args:
            vertical: Vertical name
            include_global: If True, include global escape hatches

        Returns:
            Tuple of (conditions_dict, transforms_dict)
        """
        conditions: Dict[str, ConditionFunction] = {}
        transforms: Dict[str, TransformFunction] = {}

        # Add global first (can be overridden by vertical-specific)
        if include_global:
            conditions.update(self._global_conditions)
            transforms.update(self._global_transforms)

        # Add vertical-specific
        if vertical in self._conditions:
            conditions.update(self._conditions[vertical])
        if vertical in self._transforms:
            transforms.update(self._transforms[vertical])

        return (conditions, transforms)

    def get_condition(
        self,
        name: str,
        vertical: Optional[str] = None,
    ) -> Optional[ConditionFunction]:
        """Get a condition function by name.

        Looks up in vertical namespace first, then global.

        Args:
            name: Condition name
            vertical: Optional vertical to search first

        Returns:
            Condition function or None if not found
        """
        # Check vertical-specific first
        if vertical and vertical in self._conditions:
            if name in self._conditions[vertical]:
                return self._conditions[vertical][name]

        # Fall back to global
        return self._global_conditions.get(name)

    def get_transform(
        self,
        name: str,
        vertical: Optional[str] = None,
    ) -> Optional[TransformFunction]:
        """Get a transform function by name.

        Looks up in vertical namespace first, then global.

        Args:
            name: Transform name
            vertical: Optional vertical to search first

        Returns:
            Transform function or None if not found
        """
        # Check vertical-specific first
        if vertical and vertical in self._transforms:
            if name in self._transforms[vertical]:
                return self._transforms[vertical][name]

        # Fall back to global
        return self._global_transforms.get(name)

    def list_conditions(
        self,
        vertical: Optional[str] = None,
    ) -> List[str]:
        """List all condition names.

        Args:
            vertical: If provided, list only conditions for that vertical

        Returns:
            List of condition names
        """
        if vertical:
            return list(self._conditions.get(vertical, {}).keys())
        # Return all conditions
        names = set(self._global_conditions.keys())
        for v_conditions in self._conditions.values():
            names.update(v_conditions.keys())
        return list(names)

    def list_transforms(
        self,
        vertical: Optional[str] = None,
    ) -> List[str]:
        """List all transform names.

        Args:
            vertical: If provided, list only transforms for that vertical

        Returns:
            List of transform names
        """
        if vertical:
            return list(self._transforms.get(vertical, {}).keys())
        # Return all transforms
        names = set(self._global_transforms.keys())
        for v_transforms in self._transforms.values():
            names.update(v_transforms.keys())
        return list(names)

    def list_verticals(self) -> List[str]:
        """List all verticals with registered escape hatches.

        Returns:
            List of vertical names
        """
        verticals = set(self._conditions.keys())
        verticals.update(self._transforms.keys())
        return list(verticals)

    def clear(self, vertical: Optional[str] = None) -> None:
        """Clear registered escape hatches.

        Args:
            vertical: If provided, clear only that vertical. Otherwise clear all.
        """
        if vertical:
            self._conditions.pop(vertical, None)
            self._transforms.pop(vertical, None)
            logger.debug(f"Cleared escape hatches for vertical: {vertical}")
        else:
            self._conditions.clear()
            self._transforms.clear()
            self._global_conditions.clear()
            self._global_transforms.clear()
            logger.debug("Cleared all escape hatches")

    def discover_from_vertical(
        self,
        vertical_name: str,
        *,
        replace: bool = False,
    ) -> Tuple[int, int]:
        """Auto-discover and register escape hatches from a vertical module.

        Attempts to import victor.{vertical_name}.escape_hatches and register
        its CONDITIONS and TRANSFORMS dictionaries.

        Args:
            vertical_name: Name of the vertical (e.g., "coding", "research")
            replace: If True, replace existing entries

        Returns:
            Tuple of (conditions_registered, transforms_registered)

        Raises:
            ImportError: If the escape_hatches module cannot be imported
        """
        from victor.core.verticals.naming import get_vertical_module_name

        module_name = get_vertical_module_name(vertical_name)
        module_path = f"victor.{module_name}.escape_hatches"

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            logger.warning(f"Could not import escape hatches from {module_path}: {e}")
            raise

        conditions = getattr(module, "CONDITIONS", {})
        transforms = getattr(module, "TRANSFORMS", {})

        return self.register_from_vertical(
            vertical=vertical_name,
            conditions=conditions,
            transforms=transforms,
            replace=replace,
        )

    def discover_from_all_verticals(
        self,
        *,
        replace: bool = False,
    ) -> Tuple[int, int]:
        """Auto-discover and register escape hatches from all verticals.

        Uses VerticalDiscovery to find all verticals and load their escape
        hatches. This is OCP-compliant as it doesn't hardcode vertical names.

        Args:
            replace: If True, replace existing entries

        Returns:
            Tuple of (total_conditions_registered, total_transforms_registered)

        Example:
            registry = EscapeHatchRegistry.get_instance()
            cond_count, trans_count = registry.discover_from_all_verticals()
            print(f"Registered {cond_count} conditions and {trans_count} transforms")
        """
        from victor.framework.discovery import VerticalDiscovery

        total_conditions = 0
        total_transforms = 0

        try:
            # Discover escape hatches from all verticals
            escape_hatches = VerticalDiscovery.discover_escape_hatches()

            # Register each vertical's escape hatches
            for vertical_name, hatch_dict in escape_hatches.items():
                conditions = hatch_dict.get("conditions", {})
                transforms = hatch_dict.get("transforms", {})

                if conditions or transforms:
                    cond_count, trans_count = self.register_from_vertical(
                        vertical=vertical_name,
                        conditions=conditions,
                        transforms=transforms,
                        replace=replace,
                    )
                    total_conditions += cond_count
                    total_transforms += trans_count

            logger.debug(
                f"Discovered and registered {total_conditions} conditions and "
                f"{total_transforms} transforms from {len(escape_hatches)} verticals"
            )

        except Exception as e:
            logger.error(f"Error during escape hatch discovery: {e}")

        return (total_conditions, total_transforms)


# Module-level convenience functions


def get_escape_hatch_registry() -> EscapeHatchRegistry:
    """Get the global escape hatch registry singleton.

    Returns:
        EscapeHatchRegistry instance
    """
    return EscapeHatchRegistry.get_instance()


def condition(
    name: str,
    vertical: Optional[str] = None,
    *,
    replace: bool = False,
) -> Callable[[F], F]:
    """Decorator to register a condition function.

    Args:
        name: Condition name (referenced in YAML)
        vertical: Optional vertical namespace
        replace: If True, replace existing condition

    Returns:
        Decorator function

    Example:
        @condition("tests_passing", vertical="coding")
        def tests_passing(ctx: Dict[str, Any]) -> str:
            if ctx.get("failures", 0) > 0:
                return "failing"
            return "passing"
    """

    def decorator(fn: F) -> F:
        registry = get_escape_hatch_registry()
        registry.register_condition(name, fn, vertical=vertical, replace=replace)
        return fn

    return decorator


def transform(
    name: str,
    vertical: Optional[str] = None,
    *,
    replace: bool = False,
) -> Callable[[F], F]:
    """Decorator to register a transform function.

    Args:
        name: Transform name (referenced in YAML)
        vertical: Optional vertical namespace
        replace: If True, replace existing transform

    Returns:
        Decorator function

    Example:
        @transform("merge_results", vertical="research")
        def merge_results(ctx: Dict[str, Any]) -> Dict[str, Any]:
            return {"merged": ctx.get("a", []) + ctx.get("b", [])}
    """

    def decorator(fn: F) -> F:
        registry = get_escape_hatch_registry()
        registry.register_transform(name, fn, vertical=vertical, replace=replace)
        return fn

    return decorator


__all__ = [
    # Protocols
    "ConditionFunction",
    "TransformFunction",
    # Registry
    "EscapeHatchRegistry",
    "get_escape_hatch_registry",
    # Decorators
    "condition",
    "transform",
]
