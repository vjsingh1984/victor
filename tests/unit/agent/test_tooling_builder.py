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

"""Tests for ToolingBuilder."""

from unittest.mock import MagicMock, patch

from victor.agent.builders.tooling_builder import ToolingBuilder


def test_tooling_builder_wires_components():
    """ToolingBuilder assigns tooling components."""
    settings = MagicMock()
    factory = MagicMock()
    tools = MagicMock()
    tools.register_before_hook = MagicMock()
    tool_registrar = MagicMock()
    tool_registrar._register_tool_dependencies = MagicMock()
    tool_registrar._load_tool_configurations = MagicMock()
    tool_registrar.set_background_task_callback = MagicMock()

    factory.create_tool_registry.return_value = tools
    factory.create_tool_registrar.return_value = tool_registrar
    factory.initialize_plugin_system.return_value = "plugin-manager"
    factory.create_argument_normalizer.return_value = "argument-normalizer"
    factory.create_middleware_chain.return_value = ("middleware", "correction")
    factory.create_safety_checker.return_value = "safety"
    factory.create_auto_committer.return_value = "auto"
    factory.create_tool_executor.return_value = "tool-executor"
    factory.create_parallel_executor.return_value = "parallel-executor"
    factory.create_response_completer.return_value = "response-completer"
    factory.create_unified_tracker.return_value = "unified-tracker"
    factory.create_tool_selector.return_value = "tool-selector"
    factory.create_tool_access_controller.return_value = "tool-access-controller"
    factory.create_budget_manager.return_value = "budget-manager"

    orchestrator = MagicMock()
    orchestrator.tool_graph = "tool-graph"
    orchestrator.tool_adapter = "tool-adapter"
    orchestrator._mode_coordinator = "mode-coordinator"
    orchestrator._response_coordinator = MagicMock()
    orchestrator._tool_calling_caps_internal = "tool-caps"
    orchestrator.conversation_state = "conversation-state"
    orchestrator.model = "model"
    orchestrator.provider_name = "provider"
    orchestrator.tool_selection = {}
    orchestrator._record_tool_selection = MagicMock()
    orchestrator._create_background_task = MagicMock()
    orchestrator._register_default_tools = MagicMock()
    orchestrator._log_tool_call = MagicMock()
    orchestrator.tool_cache = "tool-cache"
    orchestrator._code_correction_middleware = "correction"
    orchestrator._on_tool_start_callback = MagicMock()
    orchestrator._on_tool_complete_callback = MagicMock()

    with patch(
        "victor.agent.coordinators.config_coordinator.ToolAccessConfigCoordinator",
        return_value="access-config",
    ):
        builder = ToolingBuilder(settings=settings, factory=factory)
        components = builder.build(orchestrator, provider="provider", model="model")

    assert orchestrator.tools == tools
    assert orchestrator.tool_registry == tools
    assert orchestrator.tool_registrar == tool_registrar
    assert orchestrator.plugin_manager == "plugin-manager"
    assert orchestrator.argument_normalizer == "argument-normalizer"
    assert orchestrator._middleware_chain == "middleware"
    assert orchestrator._safety_checker == "safety"
    assert orchestrator._auto_committer == "auto"
    assert orchestrator.tool_executor == "tool-executor"
    assert orchestrator.parallel_executor == "parallel-executor"
    assert orchestrator.response_completer == "response-completer"
    assert orchestrator.unified_tracker == "unified-tracker"
    assert orchestrator.tool_selector == "tool-selector"
    assert orchestrator._tool_access_controller == "tool-access-controller"
    assert orchestrator._tool_access_config_coordinator == "access-config"
    assert orchestrator._budget_manager == "budget-manager"
    assert orchestrator._response_coordinator._tool_adapter == "tool-adapter"
    assert orchestrator._response_coordinator._tool_registry == tools
    assert orchestrator._response_coordinator._shell_variant_resolver == "mode-coordinator"
    assert components["tool_registry"] == tools
