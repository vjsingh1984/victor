"""Base class for vertical implementations.

This module defines the abstract base class that all verticals should inherit from.
It contains NO runtime logic - only abstract method definitions and default
implementations that raise NotImplementedError.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from victor_sdk.core.types import (
    CapabilityRequirementLike,
    PromptMetadata,
    TeamMetadata,
    WorkflowMetadata,
    TeamDefinitionLike,
    ToolRequirementLike,
    VerticalConfig,
    VerticalDefinition,
    StageDefinition,
    Tier,
    ToolSet,
    normalize_capability_requirements,
    normalize_prompt_metadata,
    normalize_prompt_templates,
    normalize_task_type_hints,
    normalize_team_metadata,
    normalize_team_definitions,
    normalize_tool_requirements,
    normalize_workflow_metadata,
)
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
    version: str = "1.0.0"

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
        return cls.get_definition().to_config()

    @classmethod
    def get_definition(cls) -> VerticalDefinition:
        """Return the serializable definition-layer contract for this vertical."""
        vertical_name = getattr(cls, "name", cls.__name__)

        try:
            vertical_name = cls.get_name()
            return VerticalDefinition(
                name=vertical_name,
                description=cls.get_description(),
                version=cls.get_version(),
                tools=[
                    requirement.tool_name
                    for requirement in normalize_tool_requirements(
                        cls.get_tool_requirements()
                    )
                ],
                tool_requirements=normalize_tool_requirements(cls.get_tool_requirements()),
                capability_requirements=normalize_capability_requirements(
                    cls.get_capability_requirements()
                ),
                system_prompt=cls.get_system_prompt(),
                prompt_metadata=cls.get_prompt_metadata(),
                stages=cls.get_stages(),
                team_metadata=cls.get_team_metadata(),
                workflow_metadata=cls.get_workflow_metadata(),
                tier=cls.get_tier(),
                metadata=cls.get_metadata(),
            )
        except VerticalConfigurationError as exc:
            if exc.vertical_name is not None:
                raise
            raise VerticalConfigurationError(
                exc.message,
                vertical_name=vertical_name,
                details=exc.details,
            ) from exc
        except Exception as exc:
            raise VerticalConfigurationError(
                "Invalid vertical definition generated from protocol hooks.",
                vertical_name=vertical_name,
                details={"error": str(exc)},
            ) from exc

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
    def get_version(cls) -> str:
        """Return the version for this vertical definition."""

        return getattr(cls, "version", "1.0.0")

    @classmethod
    def get_tool_requirements(cls) -> List[ToolRequirementLike]:
        """Return required tools for this vertical definition."""

        return cls.get_tools()

    @classmethod
    def get_capability_requirements(cls) -> List[CapabilityRequirementLike]:
        """Return required runtime capabilities for this vertical definition."""

        return []

    @classmethod
    def get_prompt_templates(cls) -> Dict[str, Any]:
        """Return task-specific prompt templates for this vertical."""

        return {}

    @classmethod
    def get_task_type_hints(cls) -> Dict[str, Any]:
        """Return task-type hints for this vertical."""

        return {}

    @classmethod
    def get_prompt_metadata(cls) -> PromptMetadata:
        """Return serializable prompt metadata for this vertical."""

        return PromptMetadata(
            templates=normalize_prompt_templates(cls.get_prompt_templates()),
            task_type_hints=normalize_task_type_hints(cls.get_task_type_hints()),
        )

    @classmethod
    def get_team_declarations(cls) -> Dict[str, TeamDefinitionLike]:
        """Return declarative team definitions for this vertical."""

        return {}

    @classmethod
    def get_default_team(cls) -> Optional[str]:
        """Return the default declarative team for this vertical."""

        return None

    @classmethod
    def get_team_metadata(cls) -> TeamMetadata:
        """Return serializable team metadata for this vertical."""

        return normalize_team_metadata(
            {
                "teams": normalize_team_definitions(cls.get_team_declarations()),
                "default_team": cls.get_default_team(),
            }
        )

    @classmethod
    def get_initial_stage(cls) -> Optional[str]:
        """Return the initial stage name for this vertical workflow."""

        stages = cls.get_stages()
        return next(iter(stages.keys()), None)

    @classmethod
    def get_workflow_spec(cls) -> Dict[str, Any]:
        """Return serializable workflow metadata for this vertical."""

        return {"stage_order": list(cls.get_stages().keys())}

    @classmethod
    def get_provider_hints(cls) -> Dict[str, Any]:
        """Return provider selection hints for this vertical."""

        return {}

    @classmethod
    def get_evaluation_criteria(cls) -> List[str]:
        """Return evaluation criteria for this vertical."""

        return []

    @classmethod
    def get_workflow_metadata(cls) -> WorkflowMetadata:
        """Return serializable workflow metadata for this vertical."""

        return WorkflowMetadata(
            initial_stage=cls.get_initial_stage(),
            workflow_spec=cls.get_workflow_spec(),
            provider_hints=cls.get_provider_hints(),
            evaluation_criteria=cls.get_evaluation_criteria(),
        )

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
