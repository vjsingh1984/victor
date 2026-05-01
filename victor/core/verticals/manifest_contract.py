"""Helpers for normalizing vertical manifests and runtime metadata.

This module centralizes compatibility handling for legacy vertical metadata.
It ensures the core runtime always works with a normalized ExtensionManifest,
even when a vertical was registered through older paths.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum
import importlib.util
import logging
from pathlib import Path
from typing import Any, Optional, Type

from victor_sdk.verticals.manifest import (
    ExtensionDependency,
    ExtensionManifest,
    ExtensionType,
)
from victor_sdk.verticals.protocols.base import VerticalBase as SdkVerticalBase

logger = logging.getLogger(__name__)

_VERTICAL_RUNTIME_PROVENANCE_ATTR = "_victor_runtime_provenance"
_VERTICAL_PACKAGE_MANIFEST = "victor-vertical.toml"


class VerticalRuntimeProvenance(str, Enum):
    """Runtime origin metadata for conflict resolution and diagnostics."""

    BUILTIN = "builtin"
    CONTRIB = "contrib"
    EXTERNAL = "external"


def infer_vertical_runtime_provenance(vertical_class: Type[Any]) -> VerticalRuntimeProvenance:
    """Infer runtime provenance from module layout as a compatibility fallback."""

    module_path = getattr(vertical_class, "__module__", "")
    if "verticals.contrib" in module_path:
        return VerticalRuntimeProvenance.CONTRIB
    if module_path.startswith("victor."):
        return VerticalRuntimeProvenance.BUILTIN
    return VerticalRuntimeProvenance.EXTERNAL


def set_vertical_runtime_provenance(
    vertical_class: Type[Any],
    provenance: VerticalRuntimeProvenance | str,
) -> VerticalRuntimeProvenance:
    """Persist explicit runtime provenance on a vertical class."""

    normalized = _coerce_vertical_runtime_provenance(provenance)
    if normalized is None:
        raise ValueError(f"Unsupported vertical runtime provenance: {provenance!r}")
    setattr(vertical_class, _VERTICAL_RUNTIME_PROVENANCE_ATTR, normalized.value)
    return normalized


def get_vertical_runtime_provenance(vertical_class: Type[Any]) -> VerticalRuntimeProvenance:
    """Return explicit runtime provenance, inferring and caching when absent."""

    candidate = getattr(vertical_class, _VERTICAL_RUNTIME_PROVENANCE_ATTR, None)
    normalized = _coerce_vertical_runtime_provenance(candidate)
    if normalized is not None:
        return normalized

    inferred = infer_vertical_runtime_provenance(vertical_class)
    setattr(vertical_class, _VERTICAL_RUNTIME_PROVENANCE_ATTR, inferred.value)
    return inferred


def _coerce_vertical_runtime_provenance(
    candidate: Any,
) -> Optional[VerticalRuntimeProvenance]:
    """Normalize arbitrary provenance values into the runtime enum."""

    if isinstance(candidate, VerticalRuntimeProvenance):
        return candidate
    if isinstance(candidate, str):
        try:
            return VerticalRuntimeProvenance(candidate)
        except ValueError:
            return None
    return None


def infer_plugin_namespace(vertical_class: Type[Any]) -> str:
    """Infer a default plugin namespace from the class module path."""

    provenance = get_vertical_runtime_provenance(vertical_class)
    if provenance is VerticalRuntimeProvenance.CONTRIB:
        return "contrib"
    if provenance is VerticalRuntimeProvenance.BUILTIN:
        return "default"
    return "external"


def get_or_create_vertical_manifest(
    vertical_class: Type[Any],
) -> Optional[ExtensionManifest]:
    """Return a normalized manifest for *vertical_class*.

    Preference order:
    1. Explicit ``_victor_manifest`` attached by decorators/registration
    2. Overridden ``get_manifest()``
    3. Package sidecar ``victor-vertical.toml``
    4. Synthesized manifest from structural class metadata
    """

    candidate = getattr(vertical_class, "_victor_manifest", None)

    if isinstance(candidate, ExtensionManifest):
        manifest = candidate
    elif isinstance(candidate, Mapping):
        manifest = _manifest_from_mapping(vertical_class, candidate)
    elif _has_custom_manifest(vertical_class):
        try:
            custom_manifest = vertical_class.get_manifest()
        except Exception as exc:
            logger.debug(
                "Failed to build custom manifest for '%s': %s",
                getattr(vertical_class, "__name__", "<unknown>"),
                exc,
            )
            return None

        if isinstance(custom_manifest, ExtensionManifest):
            manifest = custom_manifest
        elif isinstance(custom_manifest, Mapping):
            manifest = _manifest_from_mapping(vertical_class, custom_manifest)
        else:
            logger.debug(
                "Unsupported manifest type for '%s': %s",
                getattr(vertical_class, "__name__", "<unknown>"),
                type(custom_manifest).__name__,
            )
            return None
    else:
        manifest = load_vertical_package_manifest_for_module(
            getattr(vertical_class, "__module__", "")
        )
        if manifest is None:
            manifest = _synthesize_manifest(vertical_class)

    if manifest is None:
        return None

    _apply_manifest_defaults(vertical_class, manifest)
    vertical_class._victor_manifest = manifest  # type: ignore[attr-defined]
    if not getattr(vertical_class, "name", None):
        vertical_class.name = manifest.name  # type: ignore[attr-defined]
    if not getattr(vertical_class, "version", None):
        vertical_class.version = manifest.version  # type: ignore[attr-defined]
    return manifest


def get_vertical_runtime_metadata(vertical_class: Type[Any]) -> dict[str, str]:
    """Return normalized runtime metadata for a vertical class."""

    manifest = get_or_create_vertical_manifest(vertical_class)
    if manifest is None:
        provenance = get_vertical_runtime_provenance(vertical_class)
        return {
            "vertical_name": getattr(
                vertical_class, "name", getattr(vertical_class, "__name__", "")
            ),
            "vertical_manifest_version": getattr(vertical_class, "version", "1.0.0"),
            "vertical_plugin_namespace": infer_plugin_namespace(vertical_class),
            "vertical_runtime_provenance": provenance.value,
        }

    provenance = get_vertical_runtime_provenance(vertical_class)
    return {
        "vertical_name": manifest.name,
        "vertical_manifest_version": manifest.version,
        "vertical_plugin_namespace": manifest.plugin_namespace
        or infer_plugin_namespace(vertical_class),
        "vertical_runtime_provenance": provenance.value,
    }


def _has_custom_manifest(vertical_class: Type[Any]) -> bool:
    """Return True if the class overrides the SDK manifest hook."""

    return "get_manifest" in getattr(vertical_class, "__dict__", {})


def load_vertical_package_manifest_for_module(module_name: str) -> Optional[ExtensionManifest]:
    """Load a sidecar package manifest for *module_name* without importing it.

    The canonical package-level metadata file is ``victor-vertical.toml``.
    This helper resolves the owning package path from import metadata, loads
    the sidecar if present, and normalizes it into an ``ExtensionManifest``.
    """

    manifest_path = _find_vertical_package_manifest_path(module_name)
    if manifest_path is None:
        return None

    try:
        from victor.core.verticals.package_schema import VerticalPackageMetadata

        metadata = VerticalPackageMetadata.from_toml(manifest_path)
    except Exception as exc:
        logger.debug(
            "Failed to load package manifest for module '%s' from %s: %s",
            module_name,
            manifest_path,
            exc,
        )
        return None

    provides: set[ExtensionType] = set()
    class_spec = getattr(metadata, "class_spec", None)
    if class_spec is not None:
        if getattr(class_spec, "provides_tools", None):
            provides.add(ExtensionType.TOOLS)
        if getattr(class_spec, "provides_workflows", None):
            provides.add(ExtensionType.WORKFLOWS)
        if getattr(class_spec, "provides_capabilities", None):
            provides.add(ExtensionType.CAPABILITIES)

    dependencies = getattr(getattr(metadata, "dependencies", None), "verticals", []) or []
    manifest = ExtensionManifest(
        name=str(metadata.name),
        version=str(metadata.version),
        min_framework_version=str(metadata.requires_victor),
        provides=provides,
        extension_dependencies=[
            ExtensionDependency(extension_name=str(dependency_name))
            for dependency_name in dependencies
            if dependency_name
        ],
        # Let runtime defaults derive the namespace from the actual class provenance.
        plugin_namespace="",
    )
    logger.debug("Loaded package manifest for module '%s' from %s", module_name, manifest_path)
    return manifest


def _find_vertical_package_manifest_path(module_name: str) -> Optional[Path]:
    """Return the first sidecar manifest path associated with *module_name*."""

    for candidate in _iter_vertical_package_manifest_candidates(module_name):
        if candidate.is_file():
            return candidate
    return None


def _iter_vertical_package_manifest_candidates(module_name: str) -> Iterable[Path]:
    """Yield candidate sidecar manifest paths for *module_name*."""

    if not module_name:
        return ()

    package_name = module_name.split(".", 1)[0]
    if not package_name:
        return ()

    try:
        spec = importlib.util.find_spec(package_name)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        logger.debug("Unable to resolve package spec for module '%s': %s", module_name, exc)
        return ()

    if spec is None:
        return ()

    candidates: list[Path] = []
    search_locations = list(spec.submodule_search_locations or [])
    for location in search_locations:
        candidates.append(Path(location) / _VERTICAL_PACKAGE_MANIFEST)

    origin = getattr(spec, "origin", None)
    if isinstance(origin, str) and origin not in {"built-in", "frozen"}:
        candidates.append(Path(origin).resolve().parent / _VERTICAL_PACKAGE_MANIFEST)

    return tuple(dict.fromkeys(candidates))


def _manifest_from_mapping(
    vertical_class: Type[Any],
    manifest_mapping: Mapping[str, Any],
) -> Optional[ExtensionManifest]:
    """Convert a legacy manifest mapping into an ExtensionManifest."""

    data = dict(manifest_mapping)
    if "framework_version_requirement" in data and "min_framework_version" not in data:
        data["min_framework_version"] = data.pop("framework_version_requirement")

    if "provides" in data:
        data["provides"] = _coerce_extension_types(data.get("provides"))
    if "requires" in data:
        data["requires"] = _coerce_extension_types(data.get("requires"))
    if "extension_dependencies" in data:
        data["extension_dependencies"] = _coerce_dependencies(data["extension_dependencies"])

    try:
        manifest = ExtensionManifest(**data)
    except Exception as exc:
        logger.debug(
            "Failed to normalize manifest mapping for '%s': %s",
            getattr(vertical_class, "__name__", "<unknown>"),
            exc,
        )
        return None

    _apply_manifest_defaults(vertical_class, manifest)
    return manifest


def _coerce_extension_types(values: Any) -> set[ExtensionType]:
    """Normalize manifest extension type values into ExtensionType objects."""

    normalized: set[ExtensionType] = set()
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
        return normalized

    for value in values:
        if isinstance(value, ExtensionType):
            normalized.add(value)
            continue
        if isinstance(value, str):
            try:
                normalized.add(ExtensionType(value))
            except ValueError:
                logger.debug("Ignoring unknown extension type '%s' in legacy manifest", value)
    return normalized


def _coerce_dependencies(values: Any) -> list[ExtensionDependency]:
    """Normalize legacy dependency declarations into SDK dependency objects."""

    normalized: list[ExtensionDependency] = []
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
        return normalized

    for value in values:
        if isinstance(value, ExtensionDependency):
            normalized.append(value)
            continue
        if isinstance(value, Mapping):
            try:
                normalized.append(
                    ExtensionDependency(
                        extension_name=str(value.get("extension_name", "")),
                        min_version=value.get("min_version"),
                        optional=bool(value.get("optional", False)),
                    )
                )
            except Exception:
                logger.debug("Ignoring invalid legacy dependency: %r", value)
    return normalized


def _synthesize_manifest(vertical_class: Type[Any]) -> ExtensionManifest:
    """Synthesize a minimal manifest from class structure."""

    name = getattr(vertical_class, "name", None)
    if not name:
        getter = getattr(vertical_class, "get_name", None)
        if callable(getter):
            name = getter()
    if not name:
        name = getattr(vertical_class, "__name__", "unknown").lower()

    provides: set[ExtensionType] = {ExtensionType.TOOLS}
    method_to_type = {
        "get_middleware": ExtensionType.MIDDLEWARE,
        "get_safety_extension": ExtensionType.SAFETY,
        "get_workflow_spec": ExtensionType.WORKFLOWS,
        "get_team_declarations": ExtensionType.TEAMS,
        "get_mode_config": ExtensionType.MODE_CONFIG,
        "get_rl_config": ExtensionType.RL_CONFIG,
        "get_enrichment_strategy": ExtensionType.ENRICHMENT,
        "get_capability_requirements": ExtensionType.CAPABILITIES,
        "get_service_provider": ExtensionType.SERVICE_PROVIDER,
    }
    for method_name, extension_type in method_to_type.items():
        method = getattr(vertical_class, method_name, None)
        base_method = getattr(SdkVerticalBase, method_name, None)
        if method is not None and base_method is not None and method is not base_method:
            provides.add(extension_type)

    api_version = getattr(vertical_class, "VERTICAL_API_VERSION", None)
    if api_version is None:
        api_version = 1

    manifest = ExtensionManifest(
        api_version=int(api_version),
        name=str(name),
        version=str(getattr(vertical_class, "version", "1.0.0") or "1.0.0"),
        provides=provides,
    )
    _apply_manifest_defaults(vertical_class, manifest)
    return manifest


def _apply_manifest_defaults(vertical_class: Type[Any], manifest: ExtensionManifest) -> None:
    """Fill required defaults on a manifest in-place."""

    if not manifest.name:
        manifest.name = str(
            getattr(vertical_class, "name", None)
            or getattr(vertical_class, "__name__", "unknown").lower()
        )
    if not manifest.version:
        manifest.version = str(getattr(vertical_class, "version", "1.0.0") or "1.0.0")
    if not getattr(manifest, "api_version", None):
        api_version = getattr(vertical_class, "VERTICAL_API_VERSION", None)
        manifest.api_version = int(1 if api_version is None else api_version)
    if not getattr(manifest, "plugin_namespace", None):
        manifest.plugin_namespace = infer_plugin_namespace(vertical_class)
    # Persist once so the registry/loader stop depending on module-path parsing.
    get_vertical_runtime_provenance(vertical_class)
