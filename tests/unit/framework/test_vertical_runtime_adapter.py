"""Tests for host-owned vertical runtime translation."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from victor.framework.vertical_runtime_adapter import VerticalRuntimeAdapter
from victor_sdk.core.types import VerticalDefinition


class DefinitionVertical:
    """Definition-first vertical fixture."""

    name = "definition_vertical"

    @classmethod
    def get_definition(cls) -> VerticalDefinition:
        return VerticalDefinition(
            name="definition_vertical",
            description="Definition-first vertical",
            version="2.0.0",
            tools=["read", "write"],
            tool_requirements=[
                {"tool_name": "read", "purpose": "inspect files"},
                {"tool_name": "write", "purpose": "persist changes"},
            ],
            capability_requirements=[
                {"capability_id": "file_ops", "purpose": "workspace access"},
            ],
            system_prompt="Translate from definition.",
            workflow_metadata={
                "provider_hints": {"preferred_providers": ["anthropic"]},
                "evaluation_criteria": ["accuracy"],
            },
        )


class LegacyConfigVertical:
    """Legacy config-only vertical fixture."""

    name = "legacy_vertical"
    description = "Legacy vertical"
    version = "3.0.0"

    @classmethod
    def get_config(cls):
        from victor.core.verticals.base import VerticalConfig
        from victor.framework.tools import ToolSet

        return VerticalConfig(
            tools=ToolSet.from_tools(["search"]),
            system_prompt="Legacy path.",
            provider_hints={"preferred_providers": ["openai"]},
            evaluation_criteria=["coverage"],
            metadata={
                "vertical_version": "3.0.0",
                "description": "Legacy vertical",
            },
        )


def test_build_runtime_binding_from_definition():
    """The adapter should translate SDK definitions into runtime config."""

    binding = VerticalRuntimeAdapter.build_runtime_binding(DefinitionVertical)

    assert binding.definition.name == "definition_vertical"
    assert binding.runtime_config.tools.tools == {"read", "write"}
    assert binding.runtime_config.system_prompt == "Translate from definition."
    assert binding.runtime_config.provider_hints == {"preferred_providers": ["anthropic"]}
    assert binding.runtime_config.evaluation_criteria == ["accuracy"]
    assert binding.runtime_config.metadata["vertical_version"] == "2.0.0"
    assert binding.runtime_config.metadata["definition_version"] == "1.0"
    assert binding.runtime_config.metadata["tool_requirements"][0]["tool_name"] == "read"
    assert binding.runtime_config.metadata["capability_requirements"][0]["capability_id"] == "file_ops"


def test_resolve_definition_falls_back_to_legacy_config():
    """Legacy config-only verticals should still resolve through the adapter."""

    definition = VerticalRuntimeAdapter.resolve_definition(LegacyConfigVertical)

    assert definition.name == "legacy_vertical"
    assert definition.description == "Legacy vertical"
    assert definition.version == "3.0.0"
    assert definition.tools == ["search"]
    assert definition.system_prompt == "Legacy path."
    assert definition.workflow_metadata.provider_hints == {"preferred_providers": ["openai"]}
    assert definition.workflow_metadata.evaluation_criteria == ["coverage"]


@pytest.mark.asyncio
async def test_create_agent_routes_through_framework_agent():
    """Agent creation should remain host-owned after translation."""

    with patch(
        "victor.framework.agent.Agent.create",
        new=AsyncMock(return_value="agent-instance"),
    ) as mock_create:
        agent = await VerticalRuntimeAdapter.create_agent(
            DefinitionVertical,
            provider="openai",
            model="gpt-4.1",
            thinking=True,
        )

    assert agent == "agent-instance"
    mock_create.assert_awaited_once()
    assert mock_create.await_args.kwargs["provider"] == "openai"
    assert mock_create.await_args.kwargs["model"] == "gpt-4.1"
    assert mock_create.await_args.kwargs["vertical"] is DefinitionVertical
    assert mock_create.await_args.kwargs["thinking"] is True
    assert mock_create.await_args.kwargs["tools"].tools == {"read", "write"}
