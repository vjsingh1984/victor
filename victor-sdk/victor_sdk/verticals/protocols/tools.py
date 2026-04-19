"""Tool-related protocol definitions.

These protocols define how verticals provide and configure tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set, runtime_checkable

from victor_sdk.core.types import StageDefinition, Tier, TieredToolConfig


@runtime_checkable
class ToolProvider(Protocol):
    """Protocol for providing tool configurations.

    Any class that implements get_tools() can be used as a tool provider.
    """

    def get_tools(self) -> List[str]:
        """Return list of tool names for this vertical.

        Returns:
            List of tool names that should be available
        """
        ...


@runtime_checkable
class ToolSelectionStrategy(Protocol):
    """Protocol for stage-aware tool selection.

    This protocol enables verticals to optimize tool selection based on
    the current workflow stage and task type.
    """

    def get_tools_for_stage(self, stage: str, task_type: str) -> List[str]:
        """Return optimized tools for given stage and task type.

        Args:
            stage: Current workflow stage (e.g., "planning", "execution")
            task_type: Type of task (e.g., "code_generation", "debugging")

        Returns:
            List of tool names optimized for this stage and task
        """
        ...

    def get_stage_definitions(self) -> Dict[str, StageDefinition]:
        """Return stage definitions for this vertical.

        Returns:
            Dictionary mapping stage names to StageDefinition objects
        """
        ...


@runtime_checkable
class TieredToolConfigProvider(Protocol):
    """Protocol for providing tiered tool configurations.

    This protocol enables progressive enhancement where more tools
    are available at higher capability tiers.
    """

    def get_tiered_config(self) -> TieredToolConfig:
        """Return tiered tool configuration.

        Returns:
            TieredToolConfig with tool lists for each tier
        """
        ...

    def get_available_tiers(self) -> List[Tier]:
        """Return list of tiers that have tools configured.

        Returns:
            List of Tier enums that have at least one tool
        """
        ...


@dataclass(frozen=True)
class ToolDependency:
    """Dependency relationship between tools exposed through the SDK."""

    tool_name: str
    depends_on: Set[str] = field(default_factory=set)
    enables: Set[str] = field(default_factory=set)
    weight: float = 1.0


@runtime_checkable
class ToolDependencyProviderProtocol(Protocol):
    """Protocol for verticals providing tool dependency metadata."""

    def get_dependencies(self) -> List[ToolDependency]:
        """Return tool dependency definitions for this vertical."""
        ...

    def get_tool_sequences(self) -> List[List[str]]:
        """Return common tool-call sequences for this vertical."""
        return []


# =============================================================================
# Promoted Data Types (zero dependency on victor-ai)
# =============================================================================


@dataclass(frozen=True)
class ToolResultData:
    """SDK-compatible tool execution result.

    Mirrors victor.tools.base.ToolResult as a plain dataclass
    (no Pydantic dependency). Use for type hints in external verticals.
    """

    success: bool
    output: Any = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class MessageData:
    """SDK-compatible LLM message.

    Mirrors victor.providers.base.Message as a plain dataclass.
    """

    role: str
    content: str
    name: Optional[str] = None


# =============================================================================
# Promoted Protocols (structural typing for external verticals)
# =============================================================================


@runtime_checkable
class ToolRegistryProtocol(Protocol):
    """Protocol for tool registry access.

    External verticals should type-hint against this protocol instead
    of importing the concrete ToolRegistry from victor.tools.registry.
    """

    def register(self, name: str, tool: Any) -> None: ...

    def get(self, name: str) -> Any: ...

    def list_tools(self) -> List[str]: ...

    def is_enabled(self, name: str) -> bool: ...


@runtime_checkable
class ProviderRegistryProtocol(Protocol):
    """Protocol for LLM provider registry access.

    External verticals should type-hint against this protocol instead
    of importing ProviderRegistry from victor.providers.registry.
    """

    def get_provider(self, name: str) -> Any: ...

    def list_providers(self) -> List[str]: ...


__all__ = [
    "ToolDependency",
    "ToolDependencyProviderProtocol",
    "ToolProvider",
    "ToolSelectionStrategy",
    "TieredToolConfigProvider",
    # Promoted data types
    "ToolResultData",
    "MessageData",
    # Promoted protocols
    "ToolRegistryProtocol",
    "ProviderRegistryProtocol",
]
