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
            result.warnings.append(
                f"Vertical provides unknown extension types (ignored): {names}"
            )
            result.degraded_features.update(unknown_provided)

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
