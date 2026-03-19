"""Extension manifest for vertical capability declaration.

Provides a structured manifest that verticals use to declare their capabilities,
API version requirements, and dependencies. The framework uses manifests for
capability negotiation during vertical activation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Set


class ExtensionType(str, Enum):
    """Types of extensions a vertical can provide or require."""

    SAFETY = "safety"
    TOOLS = "tool_dependencies"
    WORKFLOWS = "workflows"
    TEAMS = "teams"
    MIDDLEWARE = "middleware"
    MODE_CONFIG = "mode_config"
    RL_CONFIG = "rl_config"
    ENRICHMENT = "enrichment"
    API_ROUTER = "api_router"
    CAPABILITIES = "capabilities"
    SERVICE_PROVIDER = "service_provider"


@dataclass
class ExtensionManifest:
    """Manifest declaring a vertical's capabilities and requirements.

    Used by the framework's capability negotiator to validate compatibility
    and determine which features to activate for a vertical.

    Attributes:
        api_version: The manifest API version this vertical targets.
        name: Vertical identifier (should match get_name()).
        version: Vertical version string (semver).
        min_framework_version: Minimum victor-ai version required, or None for any.
        provides: Set of extension types this vertical provides.
        requires: Set of extension types this vertical requires from the framework.
    """

    api_version: int = 1
    name: str = ""
    version: str = "0.0.0"
    min_framework_version: Optional[str] = None
    sdk_version: Optional[str] = None
    provides: Set[ExtensionType] = field(default_factory=set)
    requires: Set[ExtensionType] = field(default_factory=set)

    def is_provider(self, ext_type: ExtensionType) -> bool:
        """Check if this manifest declares the given extension type."""
        return ext_type in self.provides

    def has_requirement(self, ext_type: ExtensionType) -> bool:
        """Check if this manifest requires the given extension type."""
        return ext_type in self.requires

    def unmet_requirements(self, available: Set[ExtensionType]) -> Set[ExtensionType]:
        """Return required extension types not present in ``available``."""
        return self.requires - available
