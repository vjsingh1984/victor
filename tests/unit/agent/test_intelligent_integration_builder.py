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

"""Tests for IntelligentIntegrationBuilder."""

from unittest.mock import MagicMock

from victor.agent.builders.intelligent_integration_builder import (
    IntelligentIntegrationBuilder,
)


def test_intelligent_integration_builder_wires_components():
    """IntelligentIntegrationBuilder assigns integration wiring."""
    settings = MagicMock()
    settings.intelligent_pipeline_enabled = False

    factory = MagicMock()
    factory.create_integration_config.return_value = "integration-config"
    factory.setup_subagent_orchestration.return_value = ("subagent", True)

    orchestrator = MagicMock()

    builder = IntelligentIntegrationBuilder(settings=settings, factory=factory)
    components = builder.build(orchestrator)

    assert orchestrator._intelligent_integration is None
    assert orchestrator._intelligent_integration_config == "integration-config"
    assert orchestrator._intelligent_pipeline_enabled is False
    assert orchestrator._subagent_orchestrator == "subagent"
    assert orchestrator._subagent_orchestration_enabled is True
    assert components["intelligent_pipeline_enabled"] is False
