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

"""Decorator-based vertical registration system.

Provides @register_vertical decorator for declarative vertical registration
with automatic metadata extraction and validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, List, Optional, Set, Type

if TYPE_CHECKING:
    from victor_sdk.verticals.manifest import ExtensionManifest, ExtensionType

logger = logging.getLogger(__name__)


@dataclass
class ExtensionDependency:
    """Dependency on another extension or vertical.

    Attributes:
        extension_name: Name of the extension or vertical this depends on
        min_version: Optional PEP 440 version specifier (e.g., ">=1.0.0")
        optional: If True, dependency is not required for activation
    """

    extension_name: str
    min_version: Optional[str] = None
    optional: bool = False

    def __post_init__(self) -> None:
        """Validate dependency parameters."""
        if not self.extension_name:
            raise ValueError("extension_name cannot be empty")


def register_vertical(
    name: str,
    *,
    version: str = "1.0.0",
    api_version: int = 1,
    min_framework_version: Optional[str] = None,
    requires: Optional[Set["ExtensionType"]] = None,
    provides: Optional[Set["ExtensionType"]] = None,
    extension_dependencies: Optional[List[ExtensionDependency]] = None,
    canonicalize_tool_names: bool = True,
    tool_dependency_strategy: str = "auto",
    strict_mode: bool = False,
    load_priority: int = 0,
    plugin_namespace: str = "default",
    requires_features: Optional[Set[str]] = None,
    excludes_features: Optional[Set[str]] = None,
    lazy_load: bool = True,
) -> Callable[[Type], Type]:
    """Decorator to register a vertical with rich metadata.

    This decorator replaces manual vertical registration with a declarative
    pattern that attaches metadata directly to the class. The metadata is
    used by the framework for capability negotiation, dependency resolution,
    and configuration.

    Args:
        name: Vertical identifier (e.g., "coding", "devops", "research")
        version: Vertical version string following semver (default: "1.0.0")
        api_version: Manifest API version this vertical targets (default: 1)
        min_framework_version: Minimum victor-ai version required (PEP 440)
        requires: Extension types required from framework
        provides: Extension types this vertical provides
        extension_dependencies: List of extensions/verticals this depends on
        canonicalize_tool_names: Whether to normalize tool names (default: True)
        tool_dependency_strategy: How to load tool dependencies ("auto", "entry_point", "factory", "none")
        strict_mode: If True, all extension load failures raise exceptions
        load_priority: Higher values load first in dependency resolution (default: 0)
        plugin_namespace: Namespace for plugin isolation (default: "default")
        requires_features: Framework features required (e.g., {"async_tools"})
        excludes_features: Framework features that are incompatible
        lazy_load: If True, extensions loaded on first access (default: True)

    Returns:
        Decorator function that modifies and returns the class

    Raises:
        ValueError: If name is empty or parameters are invalid

    Examples:
        Basic usage:
        >>> @register_vertical(name="security")
        ... class SecurityAssistant(VerticalBase):
        ...     pass

        With full metadata:
        >>> @register_vertical(
        ...     name="security",
        ...     version="1.2.0",
        ...     provides={ExtensionType.SAFETY, ExtensionType.TOOLS},
        ...     extension_dependencies=[
        ...         ExtensionDependency("coding", min_version=">=1.0.0")
        ...     ],
        ...     canonicalize_tool_names=False,
        ... )
        ... class SecurityAssistant(VerticalBase):
        ...     pass

        With feature requirements:
        >>> @register_vertical(
        ...     name="ai_ml",
        ...     requires_features={"async_tools", "langchain"},
        ...     min_framework_version=">=0.6.0",
        ... )
        ... class AIMLAssistant(VerticalBase):
        ...     pass
    """
    if not name:
        raise ValueError("Vertical name cannot be empty")

    # Import ExtensionType and ExtensionManifest here to avoid circular imports
    try:
        from victor_sdk.verticals.manifest import ExtensionManifest, ExtensionType
    except ImportError:
        # Fallback for when victor-sdk is not available
        logger.warning(
            "victor_sdk not available - creating minimal manifest. "
            "Full metadata requires victor-sdk package."
        )
        ExtensionType = None  # type: ignore
        ExtensionManifest = None  # type: ignore

    def decorator(cls: Type) -> Type:
        """Decorator function that attaches manifest and registers vertical.

        Args:
            cls: The vertical class to decorate

        Returns:
            The modified class with manifest attached
        """
        # Attach manifest to class
        if ExtensionManifest is not None:
            manifest = ExtensionManifest(
                api_version=api_version,
                name=name,
                version=version,
                min_framework_version=min_framework_version,
                provides=provides or set(),
                requires=requires or set(),
                extension_dependencies=extension_dependencies or [],
                canonicalize_tool_names=canonicalize_tool_names,
                tool_dependency_strategy=tool_dependency_strategy,
                strict_mode=strict_mode,
                load_priority=load_priority,
                plugin_namespace=plugin_namespace,
                requires_features=requires_features or set(),
                excludes_features=excludes_features or set(),
                lazy_load=lazy_load,
            )
            cls._victor_manifest = manifest  # type: ignore
        else:
            # Fallback: create a simple dict manifest
            cls._victor_manifest = {  # type: ignore
                "name": name,
                "version": version,
                "api_version": api_version,
                "min_framework_version": min_framework_version,
            }

        # Set name attribute if not already defined
        if not hasattr(cls, "name") or not cls.name:
            cls.name = name  # type: ignore

        # Set version attribute if not already defined
        if not hasattr(cls, "version"):
            cls.version = version  # type: ignore

        # Register with VerticalRegistry if available
        try:
            from victor.core.verticals.base import VerticalRegistry

            VerticalRegistry.register(cls)
            logger.debug(f"Registered vertical '{name}' ({cls.__name__})")
        except ImportError:
            logger.debug(f"VerticalRegistry not available - skipping registration for '{name}'")

        # Register behavior configuration if manifest is available
        if ExtensionManifest is not None:
            try:
                from victor.core.verticals.config_registry import (
                    VerticalBehaviorConfigRegistry,
                )

                # Create and register behavior config from manifest
                behavior_config = VerticalBehaviorConfigRegistry.from_manifest(manifest)
                VerticalBehaviorConfigRegistry.register(name, behavior_config)
                logger.debug(f"Registered behavior configuration for '{name}'")
            except ImportError:
                logger.debug(
                    f"VerticalBehaviorConfigRegistry not available - "
                    f"skipping behavior config registration for '{name}'"
                )

        return cls

    return decorator


def get_vertical_manifest(vertical_class: Type) -> Optional[ExtensionManifest]:
    """Get the manifest from a vertical class.

    Args:
        vertical_class: The vertical class to get manifest from

    Returns:
        ExtensionManifest if available, None otherwise

    Examples:
        >>> manifest = get_vertical_manifest(CodingAssistant)
        >>> if manifest:
        ...     print(f"Vertical: {manifest.name} v{manifest.version}")
    """
    if not hasattr(vertical_class, "_victor_manifest"):
        return None

    manifest = vertical_class._victor_manifest  # type: ignore

    # Convert dict manifest to ExtensionManifest if needed
    if isinstance(manifest, dict):
        try:
            from victor_sdk.verticals.manifest import ExtensionManifest

            return ExtensionManifest(**manifest)
        except (ImportError, TypeError):
            return None

    return manifest
