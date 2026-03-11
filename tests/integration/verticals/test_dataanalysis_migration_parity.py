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

"""Integration parity checks for the Data Analysis vertical migration."""

from __future__ import annotations

import pytest

from victor.config.settings import Settings
from victor.core.bootstrap import bootstrap_container
from victor.core.container import ServiceContainer, set_container
from victor.core.verticals.protocols import VerticalExtensions
from victor.core.verticals.vertical_loader import get_vertical_loader
from victor.framework.vertical_runtime_adapter import VerticalRuntimeAdapter
from victor.verticals.contrib.dataanalysis import (
    DataAnalysisAssistant as RuntimeDataAnalysisAssistant,
)
from victor.verticals.contrib.dataanalysis.assistant import (
    DataAnalysisAssistant as DataAnalysisAssistantDefinition,
)
from victor.verticals.contrib.dataanalysis.mode_config import DataAnalysisModeConfigProvider
from victor.verticals.contrib.dataanalysis.prompts import DataAnalysisPromptContributor
from victor.verticals.contrib.dataanalysis.rl import DataAnalysisRLConfig
from victor.verticals.contrib.dataanalysis.safety import DataAnalysisSafetyExtension
from victor.verticals.contrib.dataanalysis.teams import DataAnalysisTeamSpecProvider
from victor.verticals.contrib.dataanalysis.workflows import DataAnalysisWorkflowProvider
from victor_sdk import CapabilityIds


@pytest.fixture(autouse=True)
def reset_dataanalysis_vertical_state() -> None:
    """Reset shared loader/container state between parity checks."""

    loader = get_vertical_loader()
    loader.reset()
    clear_config_cache = getattr(DataAnalysisAssistantDefinition, "clear_config_cache", None)
    if callable(clear_config_cache):
        clear_config_cache()
    set_container(ServiceContainer())

    yield

    loader.reset()
    if callable(clear_config_cache):
        clear_config_cache()
    set_container(ServiceContainer())


def test_dataanalysis_migration_discovery_parity_preserves_loader_and_binding_contracts() -> None:
    """Discovery should still resolve Data Analysis through the normal loader/runtime path."""

    loader = get_vertical_loader()
    vertical = loader.load("dataanalysis")
    binding = VerticalRuntimeAdapter.build_runtime_binding(vertical)
    definition = binding.definition

    assert definition.name == "dataanalysis"
    assert vertical.__name__ == RuntimeDataAnalysisAssistant.__name__
    assert loader.active_vertical_name == "dataanalysis"
    assert set(loader.get_tools()) == set(DataAnalysisAssistantDefinition.get_tools())
    assert loader.get_system_prompt() == definition.system_prompt
    assert set(binding.runtime_config.tools.tools) == set(definition.get_tool_names())
    assert set(binding.runtime_config.stages) == set(definition.stages)
    assert [
        item["capability_id"] for item in binding.runtime_config.metadata["capability_requirements"]
    ] == [
        CapabilityIds.FILE_OPS,
        CapabilityIds.SHELL_ACCESS,
        CapabilityIds.VALIDATION,
        CapabilityIds.WEB_ACCESS,
    ]


def test_dataanalysis_migration_activation_parity_registers_runtime_extensions() -> None:
    """Bootstrap activation should still expose Data Analysis runtime integrations."""

    container = bootstrap_container(settings=Settings(), vertical="dataanalysis")
    extensions = container.get_optional(VerticalExtensions)

    assert extensions is not None
    assert isinstance(extensions.mode_config_provider, DataAnalysisModeConfigProvider)
    assert isinstance(extensions.workflow_provider, DataAnalysisWorkflowProvider)
    assert isinstance(extensions.rl_config_provider, DataAnalysisRLConfig)
    assert isinstance(extensions.team_spec_provider, DataAnalysisTeamSpecProvider)
    assert any(
        isinstance(contributor, DataAnalysisPromptContributor)
        for contributor in extensions.prompt_contributors
    )
    assert any(
        isinstance(extension, DataAnalysisSafetyExtension)
        for extension in extensions.safety_extensions
    )
    assert "research" in extensions.get_all_mode_configs()
    assert "data_analysis" in extensions.get_all_task_hints()


def test_dataanalysis_migration_behavior_parity_uses_shared_runtime_helper_defaults() -> None:
    """Behavior should stay intact after runtime helpers leave the package root."""

    workflow_provider = RuntimeDataAnalysisAssistant.get_workflow_provider()
    rl_config = RuntimeDataAnalysisAssistant.get_rl_config_provider()
    team_spec_provider = RuntimeDataAnalysisAssistant.get_team_spec_provider()

    assert isinstance(RuntimeDataAnalysisAssistant.get_prompt_contributor(), DataAnalysisPromptContributor)
    assert isinstance(RuntimeDataAnalysisAssistant.get_mode_config_provider(), DataAnalysisModeConfigProvider)
    assert isinstance(RuntimeDataAnalysisAssistant.get_safety_extension(), DataAnalysisSafetyExtension)
    assert isinstance(workflow_provider, DataAnalysisWorkflowProvider)
    assert isinstance(rl_config, DataAnalysisRLConfig)
    assert isinstance(team_spec_provider, DataAnalysisTeamSpecProvider)
    assert "eda_pipeline" in workflow_provider.get_workflow_names()
    assert "eda" in rl_config.task_type_mappings
    assert "ml_team" in team_spec_provider.list_team_types()
