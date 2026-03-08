"""Core type definitions for Victor SDK.

These types define the data structures used across vertical configurations
and framework interactions without depending on any runtime implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class Tier(str, Enum):
    """Vertical capability tier for progressive enhancement."""

    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"


@dataclass(frozen=True)
class StageDefinition:
    """Definition of a workflow stage with tool configuration.

    Attributes:
        name: Stage identifier (e.g., "planning", "execution")
        description: Human-readable stage description
        required_tools: Tools that MUST be available in this stage
        optional_tools: Tools that MAY be used if available
        allow_custom_tools: Whether user can add custom tools in this stage
    """

    name: str
    description: str
    required_tools: List[str] = field(default_factory=list)
    optional_tools: List[str] = field(default_factory=list)
    allow_custom_tools: bool = True

    def get_effective_tools(self, available_tools: List[str]) -> List[str]:
        """Get effective tool list for this stage.

        Args:
            available_tools: Tools currently available in the system

        Returns:
            List of tool names that should be used in this stage
        """
        effective = set(self.required_tools)

        # Add optional tools that are available
        effective.update(tool for tool in self.optional_tools if tool in available_tools)

        return sorted(effective)


@dataclass(frozen=True)
class TieredToolConfig:
    """Tool configuration with tier-based progressive enhancement.

    Attributes:
        basic_tools: Tools available at BASIC tier (minimal functionality)
        standard_tools: Additional tools at STANDARD tier
        advanced_tools: Additional tools at ADVANCED tier (full functionality)
    """

    basic_tools: List[str] = field(default_factory=list)
    standard_tools: List[str] = field(default_factory=list)
    advanced_tools: List[str] = field(default_factory=list)

    def get_tools_for_tier(self, tier: Union[Tier, str]) -> List[str]:
        """Get tools available at a specific tier.

        Args:
            tier: The capability tier

        Returns:
            List of tool names available at that tier
        """
        tier = Tier(tier) if isinstance(tier, str) else tier
        tools = []

        if tier == Tier.BASIC:
            tools = self.basic_tools.copy()
        elif tier == Tier.STANDARD:
            tools = self.basic_tools + self.standard_tools
        elif tier == Tier.ADVANCED:
            tools = self.basic_tools + self.standard_tools + self.advanced_tools

        return tools

    def get_max_tier_for_tools(self, available_tools: List[str]) -> Tier:
        """Determine max tier based on available tools.

        Args:
            available_tools: Tools that are currently available

        Returns:
            The highest tier that can be supported
        """
        available_set = set(available_tools)

        if available_set.issuperset(set(self.basic_tools + self.standard_tools + self.advanced_tools)):
            return Tier.ADVANCED
        elif available_set.issuperset(set(self.basic_tools + self.standard_tools)):
            return Tier.STANDARD
        else:
            return Tier.BASIC


@dataclass(frozen=True)
class ToolSet:
    """A set of tools with metadata.

    Attributes:
        names: List of tool names
        description: Human-readable description of this tool set
        tier: Capability tier for this tool set
    """

    names: List[str]
    description: str = ""
    tier: Tier = Tier.STANDARD

    def __contains__(self, tool_name: str) -> bool:
        """Check if a tool is in this set."""
        return tool_name in self.names

    def __len__(self) -> int:
        """Return number of tools in this set."""
        return len(self.names)

    def __iter__(self):
        """Iterate over tool names."""
        return iter(self.names)


@dataclass
class VerticalConfig:
    """Configuration for a vertical.

    This is the main configuration object that verticals provide to the framework.
    It contains all necessary information for the framework to create and configure
    an agent for this vertical.

    Attributes:
        name: Vertical identifier
        description: Human-readable description
        tools: Tool names or ToolSet for this vertical
        system_prompt: System prompt for the agent
        stages: Stage definitions for multi-stage workflows
        tier: Capability tier for this vertical
        metadata: Additional metadata as key-value pairs
        extensions: Extension configurations (protocols, capabilities, etc.)
    """

    name: str
    description: str
    tools: Union[List[str], ToolSet]
    system_prompt: str
    stages: Dict[str, StageDefinition] = field(default_factory=dict)
    tier: Tier = Tier.STANDARD
    metadata: Dict[str, Any] = field(default_factory=dict)
    extensions: Dict[str, Any] = field(default_factory=dict)

    def get_tool_names(self) -> List[str]:
        """Get list of tool names from config."""
        if isinstance(self.tools, ToolSet):
            return self.tools.names
        return self.tools

    def get_stage_names(self) -> List[str]:
        """Get list of stage names."""
        return list(self.stages.keys())

    def with_metadata(self, **kwargs) -> VerticalConfig:
        """Return a new config with additional metadata."""
        new_metadata = {**self.metadata, **kwargs}
        return VerticalConfig(
            name=self.name,
            description=self.description,
            tools=self.tools,
            system_prompt=self.system_prompt,
            stages=self.stages,
            tier=self.tier,
            metadata=new_metadata,
            extensions=self.extensions,
        )

    def with_extension(self, key: str, value: Any) -> VerticalConfig:
        """Return a new config with an extension added."""
        new_extensions = {**self.extensions, key: value}
        return VerticalConfig(
            name=self.name,
            description=self.description,
            tools=self.tools,
            system_prompt=self.system_prompt,
            stages=self.stages,
            tier=self.tier,
            metadata=self.metadata,
            extensions=new_extensions,
        )
