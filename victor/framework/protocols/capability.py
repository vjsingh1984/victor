"""Capability discovery and versioning protocols.

These protocols enable explicit capability checking instead of
hasattr/getattr duck-typing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Protocol, runtime_checkable


class CapabilityType(str, Enum):
    """Types of orchestrator capabilities.

    Categorizes capabilities for discovery and documentation.
    """

    TOOL = "tool"
    """Tool-related capabilities (enable, disable, budget)."""

    PROMPT = "prompt"
    """Prompt building capabilities (system prompt, hints)."""

    MODE = "mode"
    """Mode/configuration capabilities (adaptive mode, budgets)."""

    SAFETY = "safety"
    """Safety-related capabilities (patterns, middleware)."""

    RL = "rl"
    """Reinforcement learning capabilities (hooks, learners)."""

    TEAM = "team"
    """Team/multi-agent capabilities (specs, coordination)."""

    WORKFLOW = "workflow"
    """Workflow capabilities (sequences, dependencies)."""

    VERTICAL = "vertical"
    """Vertical integration capabilities (context, extensions)."""


@dataclass
class OrchestratorCapability:
    """Explicit capability declaration for orchestrator features.

    Replaces hasattr/getattr duck-typing with explicit contracts.
    Each capability declares how to interact with it.

    Versioning:
        Capabilities support semantic versioning for backward compatibility.
        Version format: "MAJOR.MINOR" (e.g., "1.0", "2.1")
        - MAJOR: Breaking changes (incompatible signature changes)
        - MINOR: Backward-compatible additions

        When invoking a capability, callers can specify minimum required version.
        Default version is "1.0" for backward compatibility.

    Attributes:
        name: Unique capability identifier
        capability_type: Category of capability
        version: Semantic version string (default "1.0")
        setter: Method name to set/configure (if settable)
        getter: Method name to get current value (if gettable)
        attribute: Direct attribute name (if attribute access)
        description: Human-readable description
        required: Whether this capability is mandatory
        deprecated: Whether this capability is deprecated
        deprecated_message: Message explaining deprecation and migration path
    """

    name: str
    capability_type: CapabilityType
    version: str = "1.0"
    setter: Optional[str] = None
    getter: Optional[str] = None
    attribute: Optional[str] = None
    description: str = ""
    required: bool = False
    deprecated: bool = False
    deprecated_message: str = ""

    def __post_init__(self) -> None:
        """Validate capability declaration."""
        # Validate access method
        if not any([self.setter, self.getter, self.attribute]):
            raise ValueError(
                f"Capability '{self.name}' must specify at least one of: "
                "setter, getter, or attribute"
            )
        # Validate version format
        if not self._is_valid_version(self.version):
            raise ValueError(
                f"Capability '{self.name}' has invalid version '{self.version}'. "
                "Expected format: 'MAJOR.MINOR' or 'MAJOR.MINOR.PATCH' (e.g., '1.0', '2.1', '1.2.3')"
            )

    @staticmethod
    def _is_valid_version(version: str) -> bool:
        """Validate version string format.

        Args:
            version: Version string to validate

        Returns:
            True if version is valid MAJOR.MINOR or MAJOR.MINOR.PATCH format
        """
        try:
            parts = version.split(".")
            if len(parts) not in (2, 3):
                return False
            # Validate all parts are non-negative integers
            return all(int(part) >= 0 for part in parts)
        except (ValueError, AttributeError):
            return False

    def is_compatible_with(self, required_version: str) -> bool:
        """Check if this capability's version is compatible with required version.

        A capability is compatible if its version >= required version.
        Comparison is done semantically (1.10 > 1.9).

        Args:
            required_version: Minimum required version

        Returns:
            True if this capability meets the version requirement
        """
        try:
            req_parts = required_version.split(".")
            cap_parts = self.version.split(".")
            req_major, req_minor = int(req_parts[0]), int(req_parts[1])
            cap_major, cap_minor = int(cap_parts[0]), int(cap_parts[1])

            # Major version must match or be greater
            if cap_major > req_major:
                return True
            if cap_major < req_major:
                return False
            # Same major, check minor
            return cap_minor >= req_minor
        except (ValueError, IndexError, AttributeError):
            return False


@runtime_checkable
class CapabilityRegistryProtocol(Protocol):
    """Protocol for capability discovery and invocation.

    Enables explicit capability checking instead of hasattr duck-typing.
    Implementations should register all capabilities at initialization.

    Versioning Support:
        All methods support optional version requirements:
        - has_capability(name, min_version="1.0") - check version compatibility
        - invoke_capability(name, *args, min_version="1.0") - version-safe invoke

    Example:
        # Instead of:
        if hasattr(orch, "set_enabled_tools") and callable(orch.set_enabled_tools):
            orch.set_enabled_tools(tools)

        # Use:
        if orch.has_capability("enabled_tools"):
            orch.invoke_capability("enabled_tools", tools)

        # With version checking:
        if orch.has_capability("enabled_tools", min_version="1.1"):
            orch.invoke_capability("enabled_tools", tools, min_version="1.1")
    """

    def get_capabilities(self) -> Dict[str, OrchestratorCapability]:
        """Get all registered capabilities.

        Returns:
            Dict mapping capability names to their declarations
        """
        ...

    def has_capability(
        self,
        name: str,
        min_version: Optional[str] = None,
    ) -> bool:
        """Check if a capability is available and meets version requirements.

        Args:
            name: Capability name to check
            min_version: Minimum required version (default: None = any version)

        Returns:
            True if capability is registered, functional, and meets version requirement
        """
        ...

    def get_capability(self, name: str) -> Optional[OrchestratorCapability]:
        """Get a specific capability declaration.

        Args:
            name: Capability name

        Returns:
            Capability declaration or None if not found
        """
        ...

    def get_capability_version(self, name: str) -> Optional[str]:
        """Get the version of a registered capability.

        Args:
            name: Capability name

        Returns:
            Version string or None if capability not found
        """
        ...

    def invoke_capability(
        self,
        name: str,
        *args: Any,
        min_version: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke a capability's setter method with optional version check.

        Args:
            name: Capability name
            *args: Positional arguments for setter
            min_version: Minimum required version (default: None = no check)
            **kwargs: Keyword arguments for setter

        Returns:
            Result from setter method

        Raises:
            KeyError: If capability not found
            TypeError: If capability has no setter
            IncompatibleVersionError: If capability version is incompatible
        """
        ...

    def get_capability_value(self, name: str) -> Any:
        """Get a capability's current value via getter or attribute.

        Args:
            name: Capability name

        Returns:
            Current value

        Raises:
            KeyError: If capability not found
            TypeError: If capability has no getter/attribute
        """
        ...

    def get_capabilities_by_type(
        self, capability_type: CapabilityType
    ) -> Dict[str, OrchestratorCapability]:
        """Get all capabilities of a specific type.

        Args:
            capability_type: Type to filter by

        Returns:
            Dict of matching capabilities
        """
        ...
