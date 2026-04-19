"""Unit tests for the SDK vertical definition contract."""

import pytest

from victor_sdk import (
    CapabilityIds,
    CapabilityRequirement,
    CURRENT_DEFINITION_VERSION,
    StageDefinition,
    ToolRequirement,
    VerticalConfig,
    VerticalDefinition,
    WorkflowMetadata,
    is_supported_definition_version,
    validate_definition_version,
)
from victor_sdk.core.types import Tier
from victor_sdk.core.exceptions import VerticalConfigurationError
from victor_sdk.verticals.protocols.base import VerticalBase


def test_vertical_definition_to_dict_and_to_config_round_trip() -> None:
    """VerticalDefinition should serialize cleanly and bridge to VerticalConfig."""

    definition = VerticalDefinition(
        name="coding",
        description="Coding assistant",
        version="2.1.0",
        definition_version="1.0",
        tools=["read", "write", "git"],
        tool_requirements=[
            ToolRequirement(tool_name="read", purpose="inspect files"),
            ToolRequirement(tool_name="write", purpose="modify files"),
            ToolRequirement(
                tool_name="git", required=False, purpose="optional vcs workflows"
            ),
        ],
        capability_requirements=[
            CapabilityRequirement(
                capability_id=CapabilityIds.FILE_OPS,
                purpose="workspace edits",
            )
        ],
        system_prompt="You are a coding assistant.",
        prompt_metadata={
            "templates": {"analysis": "Analyze the repository."},
            "task_type_hints": {
                "analysis": {
                    "hint": "Start with reading the relevant files.",
                    "tool_budget": 12,
                    "priority_tools": ["read", "grep"],
                }
            },
        },
        stages={
            "planning": StageDefinition(
                name="planning",
                description="Plan first",
                optional_tools=["read"],
            )
        },
        workflow_metadata=WorkflowMetadata(
            initial_stage="planning",
            workflow_spec={"stage_order": ["planning"]},
            provider_hints={"preferred_providers": ["anthropic"]},
            evaluation_criteria=["accuracy"],
        ),
        tier=Tier.ADVANCED,
        metadata={"domain": "software"},
        extensions={"prompt_style": "strict"},
    )

    payload = definition.to_dict()
    config = definition.to_config()
    round_trip = VerticalDefinition.from_config(config)

    assert payload["name"] == "coding"
    assert payload["tool_requirements"][2]["tool_name"] == "git"
    assert (
        payload["capability_requirements"][0]["capability_id"] == CapabilityIds.FILE_OPS
    )
    assert payload["prompt_metadata"]["templates"][0]["task_type"] == "analysis"
    assert payload["stages"]["planning"]["description"] == "Plan first"
    assert payload["workflow_metadata"]["initial_stage"] == "planning"
    assert isinstance(config, VerticalConfig)
    assert config.metadata["vertical_version"] == "2.1.0"
    assert config.extensions["tool_requirements"][2].tool_name == "git"
    assert (
        config.extensions["capability_requirements"][0].capability_id
        == CapabilityIds.FILE_OPS
    )
    assert (
        config.extensions["prompt_metadata"]["templates"][0]["task_type"] == "analysis"
    )
    assert config.extensions["workflow_metadata"]["initial_stage"] == "planning"
    assert round_trip.name == definition.name
    assert round_trip.tool_requirements[2].tool_name == "git"
    assert round_trip.capability_requirements[0].capability_id == CapabilityIds.FILE_OPS
    assert round_trip.prompt_metadata.templates[0].task_type == "analysis"
    assert round_trip.workflow_metadata.initial_stage == "planning"
    assert round_trip.metadata["domain"] == "software"

    reconstructed = VerticalDefinition.from_dict(payload)
    assert reconstructed.tool_requirements[2].tool_name == "git"
    assert reconstructed.prompt_metadata.task_type_hints[0].priority_tools == [
        "read",
        "grep",
    ]
    assert reconstructed.workflow_metadata.initial_stage == "planning"


def test_sdk_vertical_base_exposes_get_definition() -> None:
    """Default SDK vertical base should generate a definition from class hooks."""

    class ExampleVertical(VerticalBase):
        version = "3.0.0"

        @classmethod
        def get_name(cls) -> str:
            return "example"

        @classmethod
        def get_description(cls) -> str:
            return "Example vertical"

        @classmethod
        def get_tools(cls) -> list[str]:
            return ["read", "write"]

        @classmethod
        def get_system_prompt(cls) -> str:
            return "Example prompt"

        @classmethod
        def get_capability_requirements(cls) -> list[CapabilityRequirement]:
            return [
                CapabilityRequirement(
                    capability_id=CapabilityIds.FILE_OPS,
                    purpose="workspace access",
                )
            ]

        @classmethod
        def get_prompt_templates(cls) -> dict[str, str]:
            return {"analysis": "Analyze the workspace."}

        @classmethod
        def get_task_type_hints(cls) -> dict[str, dict[str, object]]:
            return {
                "analysis": {
                    "hint": "Start with repo inspection.",
                    "tool_budget": 8,
                    "priority_tools": ["read", "ls"],
                }
            }

        @classmethod
        def get_provider_hints(cls) -> dict[str, object]:
            return {"preferred_providers": ["anthropic"]}

        @classmethod
        def get_evaluation_criteria(cls) -> list[str]:
            return ["accuracy", "coverage"]

    definition = ExampleVertical.get_definition()
    config = ExampleVertical.get_config()

    assert definition.name == "example"
    assert definition.version == "3.0.0"
    assert definition.tool_requirements[0].tool_name == "read"
    assert definition.capability_requirements[0].capability_id == CapabilityIds.FILE_OPS
    assert definition.prompt_metadata.templates[0].task_type == "analysis"
    assert definition.prompt_metadata.task_type_hints[0].tool_budget == 8
    assert definition.workflow_metadata.provider_hints["preferred_providers"] == [
        "anthropic"
    ]
    assert definition.workflow_metadata.evaluation_criteria == ["accuracy", "coverage"]
    assert config.metadata["vertical_version"] == "3.0.0"
    assert config.extensions["tool_requirements"][0].tool_name == "read"
    assert (
        config.extensions["capability_requirements"][0].capability_id
        == CapabilityIds.FILE_OPS
    )
    assert (
        config.extensions["prompt_metadata"]["templates"][0]["task_type"] == "analysis"
    )
    assert config.extensions["workflow_metadata"]["provider_hints"][
        "preferred_providers"
    ] == ["anthropic"]


def test_sdk_vertical_base_allows_explicit_tool_requirements() -> None:
    """Structured tool requirements should drive definition tool names."""

    class ExplicitRequirementVertical(VerticalBase):
        @classmethod
        def get_name(cls) -> str:
            return "explicit"

        @classmethod
        def get_description(cls) -> str:
            return "Explicit requirement vertical"

        @classmethod
        def get_tools(cls) -> list[str]:
            return ["read", "write"]

        @classmethod
        def get_tool_requirements(cls) -> list[ToolRequirement]:
            return [
                ToolRequirement(tool_name="read", purpose="required"),
                ToolRequirement(
                    tool_name="shell", required=False, purpose="optional execution"
                ),
            ]

        @classmethod
        def get_system_prompt(cls) -> str:
            return "Explicit prompt"

    definition = ExplicitRequirementVertical.get_definition()
    assert definition.tools == ["read", "shell"]
    assert definition.tool_requirements[1].required is False


def test_vertical_definition_normalizes_serialized_stage_payloads() -> None:
    """Serialized stage metadata should normalize during construction."""

    definition = VerticalDefinition(
        name="staged",
        description="Stage normalization",
        tools=["read"],
        system_prompt="Normalize stage payloads",
        stages={
            "planning": {
                "description": "Plan before execution",
                "required_tools": ["read"],
            }
        },
        workflow_metadata={
            "initial_stage": "planning",
            "workflow_spec": {"stage_order": ["planning"]},
        },
    )

    assert definition.stages["planning"].name == "planning"
    assert definition.stages["planning"].required_tools == ["read"]
    assert definition.workflow_metadata.initial_stage == "planning"


def test_definition_version_validation_helpers() -> None:
    """Definition version helpers should enforce SDK compatibility."""

    validate_definition_version(CURRENT_DEFINITION_VERSION)

    assert is_supported_definition_version("1.0") is True
    assert is_supported_definition_version("1.1") is False
    assert is_supported_definition_version("invalid") is False

    with pytest.raises(VerticalConfigurationError):
        validate_definition_version("1.1")

    with pytest.raises(VerticalConfigurationError):
        validate_definition_version("invalid")


def test_vertical_definition_rejects_invalid_workflow_stage_references() -> None:
    """Workflow metadata should only reference declared stages."""

    with pytest.raises(VerticalConfigurationError, match="initial_stage"):
        VerticalDefinition(
            name="broken",
            description="Broken workflow",
            tools=["read"],
            system_prompt="Broken workflow",
            stages={
                "planning": StageDefinition(
                    name="planning",
                    description="Plan first",
                )
            },
            workflow_metadata={"initial_stage": "execution"},
        )

    with pytest.raises(VerticalConfigurationError, match="stage_order"):
        VerticalDefinition(
            name="broken-order",
            description="Broken stage order",
            tools=["read"],
            system_prompt="Broken order",
            stages={
                "planning": StageDefinition(
                    name="planning",
                    description="Plan first",
                )
            },
            workflow_metadata={
                "workflow_spec": {"stage_order": ["planning", "execution"]}
            },
        )


def test_vertical_definition_rejects_tool_requirement_mismatches() -> None:
    """The flat tool list must stay aligned with the typed tool requirements."""

    with pytest.raises(VerticalConfigurationError, match="tools must match"):
        VerticalDefinition(
            name="mismatch",
            description="Mismatched tools",
            tools=["read", "write"],
            tool_requirements=[ToolRequirement(tool_name="read")],
            system_prompt="Mismatch",
        )
