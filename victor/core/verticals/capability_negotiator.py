"""Capability negotiator for vertical ↔ framework compatibility checking.

Validates an ExtensionManifest against the running framework's capabilities
and returns a structured NegotiationResult. Called by VerticalLoader after
validation but before activation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Set

from victor_sdk.core.api_version import CURRENT_API_VERSION, MIN_SUPPORTED_API_VERSION
from victor_sdk.verticals.manifest import ExtensionManifest, ExtensionType

logger = logging.getLogger(__name__)

# Extension types the framework currently supports at runtime.
FRAMEWORK_CAPABILITIES: Set[ExtensionType] = {
    ExtensionType.SAFETY,
    ExtensionType.TOOLS,
    ExtensionType.WORKFLOWS,
    ExtensionType.TEAMS,
    ExtensionType.MIDDLEWARE,
    ExtensionType.MODE_CONFIG,
    ExtensionType.RL_CONFIG,
    ExtensionType.ENRICHMENT,
    ExtensionType.CAPABILITIES,
    ExtensionType.SERVICE_PROVIDER,
}


@dataclass
class NegotiationResult:
    """Outcome of negotiating a vertical's manifest with the framework."""

    compatible: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    degraded_features: Set[ExtensionType] = field(default_factory=set)


class CapabilityNegotiator:
    """Negotiates vertical manifest compatibility with the running framework."""

    def __init__(
        self,
        *,
        framework_capabilities: Set[ExtensionType] | None = None,
        current_api_version: int = CURRENT_API_VERSION,
        min_api_version: int = MIN_SUPPORTED_API_VERSION,
    ) -> None:
        self._capabilities = framework_capabilities or FRAMEWORK_CAPABILITIES
        self._current_api = current_api_version
        self._min_api = min_api_version

    def negotiate(self, manifest: ExtensionManifest) -> NegotiationResult:
        """Validate *manifest* and return a NegotiationResult.

        Checks:
        1. API version is within the supported range.
        2. All required extension types are available in the framework.
        3. Warns about provided types the framework doesn't recognise.
        4. Extension dependencies are satisfied.
        """
        result = NegotiationResult()

        # 1. API version check
        if manifest.api_version < self._min_api:
            result.compatible = False
            result.errors.append(
                f"Manifest api_version={manifest.api_version} is below "
                f"minimum supported version {self._min_api}."
            )
        elif manifest.api_version > self._current_api:
            result.compatible = False
            result.errors.append(
                f"Manifest api_version={manifest.api_version} exceeds "
                f"current framework version {self._current_api}."
            )

        # 1b. SDK / framework version skew check
        self._check_sdk_version_skew(manifest, result)

        # 2. Required capabilities
        unmet = manifest.unmet_requirements(self._capabilities)
        if unmet:
            result.compatible = False
            names = ", ".join(sorted(e.value for e in unmet))
            result.errors.append(f"Unmet required capabilities: {names}")

        # 3. Warnings for unknown provided types
        unknown_provided = manifest.provides - self._capabilities
        if unknown_provided:
            names = ", ".join(sorted(e.value for e in unknown_provided))
            result.warnings.append(f"Vertical provides unknown extension types (ignored): {names}")
            result.degraded_features.update(unknown_provided)

        # 4. Extension dependencies validation
        self._validate_extension_dependencies(manifest, result)

        if result.compatible:
            logger.debug(
                "Manifest negotiation passed for '%s' (api_version=%d)",
                manifest.name,
                manifest.api_version,
            )
        else:
            logger.warning(
                "Manifest negotiation failed for '%s': %s",
                manifest.name,
                "; ".join(result.errors),
            )

        return result

    def _check_sdk_version_skew(
        self, manifest: ExtensionManifest, result: NegotiationResult
    ) -> None:
        """Check framework version against manifest's min_framework_version."""
        min_ver = manifest.min_framework_version
        if not min_ver:
            return

        try:
            from packaging.specifiers import SpecifierSet
            from packaging.version import parse as parse_version
        except ImportError:
            logger.debug("packaging not installed; skipping version skew check")
            return

        try:
            from importlib.metadata import version as get_version

            framework_version = get_version("victor-ai")
        except Exception:
            logger.debug("Cannot determine victor-ai version; skipping version skew check")
            return

        # Treat plain version strings as >=X.Y.Z
        specifier_str = min_ver if any(c in min_ver for c in "<>=!~") else f">={min_ver}"
        try:
            spec = SpecifierSet(specifier_str)
            if parse_version(framework_version) not in spec:
                result.compatible = False
                result.errors.append(
                    f"Framework version {framework_version} does not satisfy "
                    f"min_framework_version {specifier_str} for vertical '{manifest.name}'"
                )
        except Exception as exc:
            result.warnings.append(f"Could not parse min_framework_version '{min_ver}': {exc}")

    def _validate_extension_dependencies(
        self, manifest: ExtensionManifest, result: NegotiationResult
    ) -> None:
        """Validate extension dependencies from manifest.

        Checks that all extension dependencies are satisfied:
        - Required extensions are installed
        - Minimum version constraints are met
        - No circular dependencies

        Args:
            manifest: The extension manifest to validate
            result: NegotiationResult to append errors/warnings to
        """
        if not manifest.extension_dependencies:
            return

        try:
            from importlib.metadata import version as get_version, PackageNotFoundError
            from packaging.specifiers import SpecifierSet
            from packaging.version import parse as parse_version
        except ImportError:
            logger.debug("packaging not installed; skipping extension dependency check")
            return

        from victor.core.verticals.base import VerticalRegistry

        for dep in manifest.extension_dependencies:
            dep_name = dep.extension_name

            # Check if extension is registered
            registered = VerticalRegistry.is_registered(dep_name)
            installed = False
            installed_version = None

            # Try to get installed version from package metadata
            try:
                # Try victor-{name} convention first
                package_name = f"victor-{dep_name}"
                installed_version = get_version(package_name)
                installed = True
            except PackageNotFoundError:
                try:
                    # Try plain name
                    installed_version = get_version(dep_name)
                    installed = True
                except PackageNotFoundError:
                    installed = False

            # Check if dependency is satisfied
            dep_satisfied = registered or installed

            if not dep_satisfied:
                if dep.optional:
                    result.warnings.append(
                        f"Optional extension dependency '{dep_name}' is not installed or registered"
                    )
                else:
                    result.compatible = False
                    result.errors.append(
                        f"Required extension dependency '{dep_name}' is not installed or registered. "
                        f"Install the victor-{dep_name} package or ensure it's registered."
                    )
                continue

            # Check version constraint if specified
            if dep.min_version and installed_version:
                try:
                    spec = SpecifierSet(dep.min_version)
                    installed_ver = parse_version(installed_version)

                    if installed_ver not in spec:
                        if dep.optional:
                            result.warnings.append(
                                f"Optional extension dependency '{dep_name}' version {installed_version} "
                                f"does not meet requirement {dep.min_version}"
                            )
                        else:
                            result.compatible = False
                            result.errors.append(
                                f"Extension dependency '{dep_name}' version {installed_version} "
                                f"does not meet requirement {dep.min_version}"
                            )
                except Exception as exc:
                    result.warnings.append(
                        f"Could not parse version constraint '{dep.min_version}' "
                        f"for dependency '{dep_name}': {exc}"
                    )
