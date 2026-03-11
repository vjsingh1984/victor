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

"""Integration parity checks for the DevOps vertical migration."""

from __future__ import annotations

import pytest

from victor.config.settings import Settings
from victor.core.bootstrap import bootstrap_container
from victor.core.container import ServiceContainer, set_container
from victor.core.verticals.protocols import VerticalExtensions
from victor.core.verticals.vertical_loader import get_vertical_loader
from victor.framework.middleware import (
    GitSafetyMiddleware,
    LoggingMiddleware,
    SecretMaskingMiddleware,
)
from victor.framework.vertical_runtime_adapter import VerticalRuntimeAdapter
from victor.verticals.contrib.devops import DevOpsAssistant as RuntimeDevOpsAssistant
from victor.verticals.contrib.devops.assistant import DevOpsAssistant as DevOpsAssistantDefinition
from victor.verticals.contrib.devops.mode_config import DevOpsModeConfigProvider
from victor.verticals.contrib.devops.prompts import DevOpsPromptContributor
from victor.verticals.contrib.devops.rl import DevOpsRLConfig
from victor.verticals.contrib.devops.safety import DevOpsSafetyExtension
from victor.verticals.contrib.devops.teams import DevOpsTeamSpecProvider
from victor.verticals.contrib.devops.workflows import DevOpsWorkflowProvider
from victor_sdk import CapabilityIds


@pytest.fixture(autouse=True)
def reset_devops_vertical_state() -> None:
    """Reset shared loader/container state between parity checks."""

    loader = get_vertical_loader()
    loader.reset()
    clear_config_cache = getattr(DevOpsAssistantDefinition, "clear_config_cache", None)
    if callable(clear_config_cache):
        clear_config_cache()
    set_container(ServiceContainer())

    yield

    loader.reset()
    if callable(clear_config_cache):
        clear_config_cache()
    set_container(ServiceContainer())


def test_devops_migration_discovery_parity_preserves_loader_and_binding_contracts() -> None:
    """Discovery should still resolve DevOps through the normal loader/runtime path."""

    loader = get_vertical_loader()
    vertical = loader.load("devops")
    binding = VerticalRuntimeAdapter.build_runtime_binding(vertical)
    definition = binding.definition

    assert definition.name == "devops"
    assert vertical.__name__ == RuntimeDevOpsAssistant.__name__
    assert loader.active_vertical_name == "devops"
    assert set(loader.get_tools()) == set(DevOpsAssistantDefinition.get_tools())
    assert loader.get_system_prompt() == definition.system_prompt
    assert set(binding.runtime_config.tools.tools) == set(definition.get_tool_names())
    assert set(binding.runtime_config.stages) == set(definition.stages)
    assert [
        item["capability_id"] for item in binding.runtime_config.metadata["capability_requirements"]
    ] == [
        CapabilityIds.FILE_OPS,
        CapabilityIds.SHELL_ACCESS,
        CapabilityIds.GIT,
        CapabilityIds.CONTAINER_RUNTIME,
        CapabilityIds.VALIDATION,
        CapabilityIds.WEB_ACCESS,
    ]


def test_devops_migration_activation_parity_registers_runtime_extensions() -> None:
    """Bootstrap activation should still expose DevOps runtime integrations."""

    container = bootstrap_container(settings=Settings(), vertical="devops")
    extensions = container.get_optional(VerticalExtensions)

    assert extensions is not None
    assert isinstance(extensions.mode_config_provider, DevOpsModeConfigProvider)
    assert isinstance(extensions.workflow_provider, DevOpsWorkflowProvider)
    assert isinstance(extensions.rl_config_provider, DevOpsRLConfig)
    assert isinstance(extensions.team_spec_provider, DevOpsTeamSpecProvider)
    assert any(
        isinstance(contributor, DevOpsPromptContributor)
        for contributor in extensions.prompt_contributors
    )
    assert any(
        isinstance(extension, DevOpsSafetyExtension) for extension in extensions.safety_extensions
    )
    assert any(isinstance(middleware, GitSafetyMiddleware) for middleware in extensions.middleware)
    assert any(
        isinstance(middleware, SecretMaskingMiddleware) for middleware in extensions.middleware
    )
    assert any(isinstance(middleware, LoggingMiddleware) for middleware in extensions.middleware)
    assert "migration" in extensions.get_all_mode_configs()
    assert "terraform" in extensions.get_all_task_hints()


def test_devops_migration_behavior_parity_uses_shared_runtime_helper_defaults() -> None:
    """Behavior should stay intact after runtime helpers leave assistant.py."""

    assert "get_middleware" not in DevOpsAssistantDefinition.__dict__

    middleware = RuntimeDevOpsAssistant.get_middleware()
    workflow_provider = RuntimeDevOpsAssistant.get_workflow_provider()
    rl_config = RuntimeDevOpsAssistant.get_rl_config_provider()
    team_spec_provider = RuntimeDevOpsAssistant.get_team_spec_provider()

    assert len(middleware) == 3
    assert isinstance(middleware[0], GitSafetyMiddleware)
    assert isinstance(middleware[1], SecretMaskingMiddleware)
    assert isinstance(middleware[2], LoggingMiddleware)
    assert isinstance(RuntimeDevOpsAssistant.get_prompt_contributor(), DevOpsPromptContributor)
    assert isinstance(workflow_provider, DevOpsWorkflowProvider)
    assert isinstance(rl_config, DevOpsRLConfig)
    assert isinstance(team_spec_provider, DevOpsTeamSpecProvider)
    assert "container_setup" in workflow_provider.get_workflow_names()
    assert "deployment" in rl_config.task_type_mappings
    assert "container_team" in team_spec_provider.list_team_types()
