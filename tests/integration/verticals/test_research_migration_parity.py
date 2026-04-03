# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration parity checks for the Research vertical migration."""

from __future__ import annotations

import pytest

from victor.config.settings import Settings
from victor.core.bootstrap import bootstrap_container
from victor.core.container import ServiceContainer, set_container
from victor.core.verticals.protocols import VerticalExtensions
from victor.core.verticals.vertical_loader import get_vertical_loader
from victor.framework.vertical_runtime_adapter import VerticalRuntimeAdapter
from victor.verticals.contrib.research import ResearchAssistant as RuntimeResearchAssistant
from victor.verticals.contrib.research.assistant import (
    ResearchAssistant as ResearchAssistantDefinition,
)
from victor.verticals.contrib.research.mode_config import ResearchModeConfigProvider
from victor.verticals.contrib.research.prompts import ResearchPromptContributor
from victor.verticals.contrib.research.rl import ResearchRLConfig
from victor.verticals.contrib.research.safety import ResearchSafetyExtension
from victor.verticals.contrib.research.teams import ResearchTeamSpecProvider
from victor.verticals.contrib.research.workflows import ResearchWorkflowProvider
from victor_sdk import CapabilityIds


@pytest.fixture(autouse=True)
def reset_research_vertical_state() -> None:
    """Reset shared loader/container state between parity checks."""

    loader = get_vertical_loader()
    loader.reset()
    clear_config_cache = getattr(ResearchAssistantDefinition, "clear_config_cache", None)
    if callable(clear_config_cache):
        clear_config_cache()
    set_container(ServiceContainer())

    yield

    loader.reset()
    if callable(clear_config_cache):
        clear_config_cache()
    set_container(ServiceContainer())


def test_research_migration_discovery_parity_preserves_loader_and_binding_contracts() -> None:
    """Discovery should still resolve Research through the normal loader/runtime path."""

    loader = get_vertical_loader()
    vertical = loader.load("research")
    binding = VerticalRuntimeAdapter.build_runtime_binding(vertical)
    definition = binding.definition

    assert definition.name == "research"
    assert vertical.__name__ == RuntimeResearchAssistant.__name__
    assert loader.active_vertical_name == "research"
    assert set(loader.get_tools()) == set(ResearchAssistantDefinition.get_tools())
    assert loader.get_system_prompt() == definition.system_prompt
    assert set(binding.runtime_config.tools.tools) == set(definition.get_tool_names())
    assert set(binding.runtime_config.stages) == set(definition.stages)
    assert [
        item["capability_id"] for item in binding.runtime_config.metadata["capability_requirements"]
    ] == [
        CapabilityIds.FILE_OPS,
        CapabilityIds.WEB_ACCESS,
        CapabilityIds.SOURCE_VERIFICATION,
        CapabilityIds.VALIDATION,
    ]


def test_research_migration_activation_parity_registers_runtime_extensions() -> None:
    """Bootstrap activation should still expose Research runtime integrations."""

    container = bootstrap_container(settings=Settings(), vertical="research")
    extensions = container.get_optional(VerticalExtensions)

    assert extensions is not None
    assert isinstance(extensions.mode_config_provider, ResearchModeConfigProvider)
    assert isinstance(extensions.workflow_provider, ResearchWorkflowProvider)
    assert isinstance(extensions.rl_config_provider, ResearchRLConfig)
    assert isinstance(extensions.team_spec_provider, ResearchTeamSpecProvider)
    assert any(
        isinstance(contributor, ResearchPromptContributor)
        for contributor in extensions.prompt_contributors
    )
    assert any(
        isinstance(extension, ResearchSafetyExtension) for extension in extensions.safety_extensions
    )
    assert "deep" in extensions.get_all_mode_configs()
    assert "fact_check" in extensions.get_all_task_hints()


def test_research_migration_behavior_parity_uses_shared_runtime_helper_defaults() -> None:
    """Behavior should stay intact after runtime helpers leave the package root."""

    workflow_provider = RuntimeResearchAssistant.get_workflow_provider()
    rl_config = RuntimeResearchAssistant.get_rl_config_provider()
    team_spec_provider = RuntimeResearchAssistant.get_team_spec_provider()

    assert isinstance(RuntimeResearchAssistant.get_prompt_contributor(), ResearchPromptContributor)
    assert isinstance(
        RuntimeResearchAssistant.get_mode_config_provider(), ResearchModeConfigProvider
    )
    assert isinstance(RuntimeResearchAssistant.get_safety_extension(), ResearchSafetyExtension)
    assert isinstance(workflow_provider, ResearchWorkflowProvider)
    assert isinstance(rl_config, ResearchRLConfig)
    assert isinstance(team_spec_provider, ResearchTeamSpecProvider)
    assert "literature_review" in workflow_provider.get_workflow_names()
    assert "fact_check" in rl_config.task_type_mappings
    assert "deep_research_team" in team_spec_provider.list_team_types()
