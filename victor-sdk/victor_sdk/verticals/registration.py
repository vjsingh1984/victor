"""Runtime-free registration helpers for external vertical packages.

CONSOLIDATION: plugin-vertical unification — see memory plugin_vertical_consolidation.md
The ``@register_vertical`` decorator is the SDK's canonical annotation for an
external package's assistant class. The corresponding runtime registration at
the plugin seam is ``PluginContext.register_vertical``. A future SDK minor
release adds ``@register_plugin`` / ``PluginBase`` aliases (S4 in the plan);
until then, ``vertical`` is the public noun at the SDK layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Set, Type

from victor_sdk.verticals.manifest import ExtensionManifest, ExtensionType


@dataclass
class ExtensionDependency:
    """Dependency on another extension or vertical package."""

    extension_name: str
    min_version: Optional[str] = None
    optional: bool = False

    def __post_init__(self) -> None:
        if not self.extension_name:
            raise ValueError("extension_name cannot be empty")


def register_vertical(
    name: str,
    *,
    version: str = "1.0.0",
    api_version: int = 1,
    min_framework_version: Optional[str] = None,
    requires: Optional[Set[ExtensionType]] = None,
    provides: Optional[Set[ExtensionType]] = None,
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
    """Attach manifest metadata to a vertical class without runtime registration."""

    if not name:
        raise ValueError("Vertical name cannot be empty")

    def decorator(cls: Type) -> Type:
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
        cls._victor_manifest = manifest  # type: ignore[attr-defined]

        if not getattr(cls, "name", None):
            cls.name = name  # type: ignore[attr-defined]
        if not getattr(cls, "version", None):
            cls.version = version  # type: ignore[attr-defined]

        return cls

    return decorator


def get_vertical_manifest(vertical_class: Type) -> Optional[ExtensionManifest]:
    """Return the attached SDK manifest for a decorated vertical class."""

    manifest = getattr(vertical_class, "_victor_manifest", None)
    if isinstance(manifest, ExtensionManifest):
        return manifest
    return None


__all__ = [
    "ExtensionDependency",
    "get_vertical_manifest",
    "register_vertical",
]
