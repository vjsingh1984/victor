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

"""Integration parity checks for the coding vertical migration."""

from __future__ import annotations

import pytest

from victor.config.settings import Settings
from victor.core.bootstrap import bootstrap_container
from victor.core.container import ServiceContainer, set_container
from victor.core.verticals.protocols import ModeConfigProviderProtocol, VerticalExtensions
from victor.core.verticals.vertical_loader import get_vertical_loader
from victor.framework.vertical_runtime_adapter import VerticalRuntimeAdapter
from victor.verticals.contrib.coding.assistant import CodingAssistant
from victor.verticals.contrib.coding.composed_chains import CODING_CHAINS
from victor.verticals.contrib.coding.middleware import (
    CodeCorrectionMiddleware,
    GitSafetyMiddleware,
)
from victor.verticals.contrib.coding.prompts import CodingPromptContributor
from victor.verticals.contrib.coding.safety import CodingSafetyExtension
from victor.verticals.contrib.coding.service_provider import (
    CodingMiddlewareProtocol,
    CodingPromptProtocol,
    CodingSafetyProtocol,
    CodingServiceProvider,
)
from victor.verticals.contrib.coding.teams import CODING_PERSONAS
from victor_sdk import CapabilityIds


@pytest.fixture(autouse=True)
def reset_coding_vertical_state() -> None:
    """Reset shared loader/container state between parity checks."""

    loader = get_vertical_loader()
    loader.reset()
    CodingAssistant.clear_config_cache()
    set_container(ServiceContainer())

    yield

    loader.reset()
    CodingAssistant.clear_config_cache()
    set_container(ServiceContainer())


def test_coding_migration_discovery_parity_preserves_loader_and_binding_contracts() -> None:
    """Discovery should still resolve coding through the normal loader/runtime path."""

    loader = get_vertical_loader()
    vertical = loader.load("coding")
    binding = VerticalRuntimeAdapter.build_runtime_binding(vertical)
    definition = vertical.get_definition()

    assert vertical is CodingAssistant
    assert loader.active_vertical_name == "coding"
    assert loader.get_tools() == CodingAssistant.get_tools()
    assert loader.get_system_prompt() == definition.system_prompt
    assert set(binding.runtime_config.tools.tools) == set(definition.get_tool_names())
    assert set(binding.runtime_config.stages) == set(definition.stages)
    assert [
        item["capability_id"]
        for item in binding.runtime_config.metadata["capability_requirements"]
    ] == [
        CapabilityIds.FILE_OPS,
        CapabilityIds.GIT,
        CapabilityIds.LSP,
    ]


def test_coding_migration_activation_parity_registers_di_visible_runtime_extensions() -> None:
    """Bootstrap activation should still expose coding runtime integrations."""

    container = bootstrap_container(settings=Settings(), vertical="coding")
    extensions = container.get_optional(VerticalExtensions)

    assert extensions is not None
    assert isinstance(extensions.service_provider, CodingServiceProvider)
    assert any(
        isinstance(middleware, CodeCorrectionMiddleware)
        for middleware in extensions.middleware
    )
    assert any(
        isinstance(middleware, GitSafetyMiddleware) for middleware in extensions.middleware
    )
    assert any(
        isinstance(contributor, CodingPromptContributor)
        for contributor in extensions.prompt_contributors
    )
    assert any(
        isinstance(extension, CodingSafetyExtension)
        for extension in extensions.safety_extensions
    )
    assert isinstance(
        container.get_optional(CodingMiddlewareProtocol), CodeCorrectionMiddleware
    )
    assert isinstance(container.get_optional(CodingSafetyProtocol), CodingSafetyExtension)
    assert isinstance(container.get_optional(CodingPromptProtocol), CodingPromptContributor)
    assert container.get_optional(ModeConfigProviderProtocol) is not None
    assert "architect" in extensions.get_all_mode_configs()
    assert "edit" in extensions.get_all_task_hints()
    assert len(extensions.get_all_safety_patterns()) > 0


def test_coding_migration_behavior_parity_uses_shared_runtime_helper_defaults() -> None:
    """Behavior should stay intact even though runtime hooks left assistant.py."""

    assert "get_middleware" not in CodingAssistant.__dict__
    assert "get_service_provider" not in CodingAssistant.__dict__
    assert "get_composed_chains" not in CodingAssistant.__dict__
    assert "get_personas" not in CodingAssistant.__dict__

    middleware = CodingAssistant.get_middleware()

    assert len(middleware) == 2
    assert isinstance(middleware[0], CodeCorrectionMiddleware)
    assert isinstance(middleware[1], GitSafetyMiddleware)
    assert isinstance(CodingAssistant.get_service_provider(), CodingServiceProvider)
    assert CodingAssistant.get_composed_chains() == CODING_CHAINS
    assert CodingAssistant.get_personas() == CODING_PERSONAS
