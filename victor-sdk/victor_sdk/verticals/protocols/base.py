"""Base class for vertical implementations.

This module defines the abstract base class that all verticals should inherit from.
It contains NO runtime logic - only abstract method definitions and default
implementations that raise NotImplementedError.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from victor_sdk.core.types import VerticalConfig, StageDefinition, Tier, ToolSet
from victor_sdk.core.exceptions import VerticalConfigurationError


class VerticalBase(ABC):
    """Abstract base class for domain-specific assistants (verticals).

    This is the ONLY base class external verticals need to inherit from.
    Contains NO runtime logic - only abstract method definitions.

    External verticals can implement this class with ZERO runtime dependencies:
    ```python
    from victor_sdk.verticals.protocols.base import VerticalBase

    class MyVertical(VerticalBase):
        @classmethod
        def get_name(cls) -> str:
            return "my-vertical"

        @classmethod
        def get_description(cls) -> str:
            return "My custom vertical"

        @classmethod
        def get_tools(cls) -> List[str]:
            return ["read", "write", "search"]

        @classmethod
        def get_system_prompt(cls) -> str:
            return "You are a helpful assistant."
    ```

    The victor-ai framework provides a concrete implementation of this class
    that adds all runtime logic while maintaining backward compatibility.
    """

    # Class attributes that subclasses SHOULD override
    name: str
    description: str

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Return vertical identifier.

        This should be a unique, lowercase identifier with no spaces.
        Examples: "coding", "research", "devops"

        Returns:
            Vertical name/identifier
        """
        ...

    @classmethod
    @abstractmethod
    def get_description(cls) -> str:
        """Return human-readable description.

        This should be a short, one-line description of what this vertical does.

        Returns:
            Vertical description
        """
        ...

    @classmethod
    @abstractmethod
    def get_tools(cls) -> List[str]:
        """Return list of tool names for this vertical.

        The tool names should match the tool names registered in the framework.
        Common tool names include: "read", "write", "search", "shell", "git",
        "web_search", "database", "docker", etc.

        Returns:
            List of tool names
        """
        ...

    @classmethod
    @abstractmethod
    def get_system_prompt(cls) -> str:
        """Return system prompt text for this vertical.

        The system prompt defines the behavior and personality of the agent.
        It should be specific to the vertical's domain.

        Returns:
            System prompt text
        """
        ...

    @classmethod
    def get_config(cls) -> VerticalConfig:
        """Generate vertical configuration.

        This is a template method that assembles configuration from various
        subclass methods. Subclasses can override specific methods to customize
        the configuration without overriding this method.

        Note: In the SDK version, this creates a simple config. The victor-ai
        framework overrides this to provide full configuration with ToolSet objects.

        Returns:
            VerticalConfig object with all necessary configuration
        """
        return VerticalConfig(
            name=cls.get_name(),
            description=cls.get_description(),
            tools=cls.get_tools(),  # Use tools list directly in SDK version
            system_prompt=cls.get_system_prompt(),
            stages=cls.get_stages(),
            tier=cls.get_tier(),
            metadata=cls.get_metadata(),
        )

    @classmethod
    def get_stages(cls) -> Dict[str, StageDefinition]:
        """Return stage definitions for multi-stage workflows.

        Default implementation provides basic 3-stage workflow.
        Subclasses can override to provide custom stages.

        Returns:
            Dictionary mapping stage names to StageDefinition objects
        """
        return {
            "planning": StageDefinition(
                name="planning",
                description="Plan the approach before execution",
                required_tools=[],
                optional_tools=["search", "read"],
            ),
            "execution": StageDefinition(
                name="execution",
                description="Execute the planned approach",
                required_tools=["read", "write"],
                optional_tools=["shell", "git"],
            ),
            "verification": StageDefinition(
                name="verification",
                description="Verify the results",
                required_tools=[],
                optional_tools=["test", "shell"],
            ),
        }

    @classmethod
    def get_tier(cls) -> Tier:
        """Return capability tier for this vertical.

        Returns:
            Capability tier (basic, standard, advanced)
        """
        return Tier.STANDARD

    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Return additional metadata about this vertical.

        Subclasses can override to provide custom metadata.

        Returns:
            Dictionary of metadata key-value pairs
        """
        return {}

    @classmethod
    def _get_toolset(cls) -> ToolSet:
        """Convert tool names to ToolSet (implementation in core).

        This method is implemented in victor-ai to provide the actual
        ToolSet object with metadata.

        Raises:
            NotImplementedError: Always raised in SDK (implemented in core)
        """
        raise NotImplementedError(
            "_get_toolset is implemented in victor-ai framework. "
            "Use get_tools() to get tool names as a list."
        )
