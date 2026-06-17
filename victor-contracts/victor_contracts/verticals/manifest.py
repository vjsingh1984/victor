"""Extension manifest for vertical capability declaration.

Provides a structured manifest that verticals use to declare their capabilities,
API version requirements, and dependencies. The framework uses manifests for
capability negotiation during vertical activation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


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
class ExtensionDependency:
    """Dependency on another extension or vertical.

    This enables verticals to declare dependencies on other verticals
    or specific extensions, enabling proper load ordering and validation.

    Attributes:
        extension_name: Name of the extension or vertical this depends on
        min_version: Optional PEP 440 version specifier (e.g., ">=1.0.0", "~=1.0")
        optional: If True, dependency is not required for activation
    """

    extension_name: str
    min_version: Optional[str] = None
    optional: bool = False

    def __post_init__(self) -> None:
        """Validate dependency parameters."""
        if not self.extension_name:
            raise ValueError("extension_name cannot be empty")

    def __hash__(self) -> int:
        """Make ExtensionDependency hashable for sets."""
        return hash((self.extension_name, self.min_version, self.optional))


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
        sdk_version: Optional SDK version this vertical was built against.
        provides: Set of extension types this vertical provides.
        requires: Set of extension types this vertical requires from the framework.
        extension_dependencies: List of extensions/verticals this depends on.
        canonicalize_tool_names: Whether to normalize tool names (default: True).
        tool_dependency_strategy: How to load tool dependencies.
        strict_mode: If True, all extension load failures raise exceptions.
        load_priority: Higher values load first in dependency resolution.
        plugin_namespace: Namespace for plugin isolation.
        requires_features: Framework features required (e.g., {"async_tools"}).
        excludes_features: Framework features that are incompatible.
        lazy_load: If True, extensions loaded on first access.
    """

    api_version: int = 1
    name: str = ""
    version: str = "0.0.0"
    min_framework_version: Optional[str] = None
    sdk_version: Optional[str] = None
    provides: Set[ExtensionType] = field(default_factory=set)
    requires: Set[ExtensionType] = field(default_factory=set)

    # NEW: Extension dependencies
    extension_dependencies: List[ExtensionDependency] = field(default_factory=list)

    # NEW: Configuration options
    canonicalize_tool_names: bool = True
    tool_dependency_strategy: str = "auto"  # "auto", "entry_point", "factory", "none"
    strict_mode: bool = False
    load_priority: int = 0

    # NEW: Plugin namespace
    plugin_namespace: str = "default"

    # NEW: Feature requirements
    requires_features: Set[str] = field(default_factory=set)
    excludes_features: Set[str] = field(default_factory=set)

    # NEW: Performance hints
    lazy_load: bool = True

    def is_provider(self, ext_type: ExtensionType) -> bool:
        """Check if this manifest declares the given extension type."""
        return ext_type in self.provides

    def has_requirement(self, ext_type: ExtensionType) -> bool:
        """Check if this manifest requires the given extension type."""
        return ext_type in self.requires

    def unmet_requirements(self, available: Set[ExtensionType]) -> Set[ExtensionType]:
        """Return required extension types not present in ``available``."""
        return self.requires - available

    def get_extension_dependencies(self) -> Set[str]:
        """Get the set of required extension dependencies (excluding optional)."""
        return {dep.extension_name for dep in self.extension_dependencies if not dep.optional}

    def has_extension_dependency(self, extension_name: str) -> bool:
        """Check if this vertical depends on a specific extension."""
        return any(dep.extension_name == extension_name for dep in self.extension_dependencies)

    def __post_init__(self) -> None:
        """Validate manifest parameters after initialization."""
        # Validate tool_dependency_strategy
        valid_strategies = {"auto", "entry_point", "factory", "none"}
        if self.tool_dependency_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid tool_dependency_strategy: {self.tool_dependency_strategy}. "
                f"Must be one of {valid_strategies}"
            )

        # Validate load_priority is non-negative
        if self.load_priority < 0:
            raise ValueError(f"load_priority must be non-negative, got {self.load_priority}")

        # Log if both requires and excludes have the same feature
        overlap = self.requires_features & self.excludes_features
        if overlap:
            logger.warning(
                f"Vertical '{self.name}' has conflicting feature requirements: "
                f"features in both requires and excludes: {overlap}"
            )
