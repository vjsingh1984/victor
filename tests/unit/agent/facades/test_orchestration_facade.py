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

"""Tests for OrchestrationFacade domain facade."""

import pytest
from unittest.mock import MagicMock

from victor.agent.facades.orchestration_facade import OrchestrationFacade
from victor.agent.facades.protocols import OrchestrationFacadeProtocol
from victor.agent.services.chat_compat_telemetry import (
    get_deprecated_chat_shim_telemetry,
    reset_deprecated_chat_shim_telemetry,
)


@pytest.fixture(autouse=True)
def _reset_chat_compat_telemetry():
    reset_deprecated_chat_shim_telemetry()
    yield
    reset_deprecated_chat_shim_telemetry()


class TestOrchestrationFacadeInit:
    """Tests for OrchestrationFacade initialization."""

    def test_init_with_all_components(self):
        """OrchestrationFacade initializes with all components provided."""
        adapter = MagicMock()
        analyzer = MagicMock()

        facade = OrchestrationFacade(
            interaction_runtime=MagicMock(),
            chat_service=MagicMock(),
            chat_stream_runtime=MagicMock(),
            tool_service=MagicMock(),
            session_service=MagicMock(),
            context_service=MagicMock(),
            provider_service=MagicMock(),
            recovery_service=MagicMock(),
            deprecated_chat_coordinator=MagicMock(),
            deprecated_tool_coordinator=MagicMock(),
            deprecated_session_coordinator=MagicMock(),
            turn_executor=MagicMock(),
            deprecated_sync_chat_coordinator=MagicMock(),
            deprecated_streaming_chat_coordinator=MagicMock(),
            deprecated_unified_chat_coordinator=MagicMock(),
            protocol_adapter=adapter,
            streaming_handler=MagicMock(),
            streaming_controller=MagicMock(),
            streaming_coordinator=MagicMock(),
            iteration_coordinator=MagicMock(),
            task_analyzer=analyzer,
            exploration_state_passed=MagicMock(),
            system_prompt_state_passed=MagicMock(),
            safety_state_passed=MagicMock(),
            coordination_state_passed=MagicMock(),
            presentation=MagicMock(),
            vertical_integration_adapter=MagicMock(),
            vertical_context=MagicMock(),
            observability=MagicMock(),
            execution_tracer=MagicMock(),
            tool_call_tracer=MagicMock(),
            intelligent_integration=MagicMock(),
            subagent_orchestrator=MagicMock(),
        )

        assert facade.protocol_adapter is adapter
        assert facade.task_analyzer is analyzer

    def test_init_with_minimal_components(self):
        """OrchestrationFacade initializes with no required components (all optional)."""
        facade = OrchestrationFacade()

        assert facade.interaction_runtime is None
        assert facade.chat_service is None
        assert facade.chat_stream_runtime is None
        assert facade.tool_service is None
        assert facade.session_service is None
        assert facade.context_service is None
        assert facade.provider_service is None
        assert facade.recovery_service is None
        assert facade.chat_coordinator is None
        assert facade.tool_coordinator is None
        assert facade.session_coordinator is None
        assert facade.turn_executor is None
        assert facade.sync_chat_coordinator is None
        assert facade.streaming_chat_coordinator is None
        assert facade.unified_chat_coordinator is None
        assert facade.protocol_adapter is None
        assert facade.streaming_handler is None
        assert facade.streaming_controller is None
        assert facade.streaming_coordinator is None
        assert facade.iteration_coordinator is None
        assert facade.task_analyzer is None
        assert facade.exploration_state_passed is None
        assert facade.system_prompt_state_passed is None
        assert facade.safety_state_passed is None
        assert facade.coordination_state_passed is None
        assert facade.presentation is None
        assert facade.vertical_integration_adapter is None
        assert facade.vertical_context is None
        assert facade.observability is None
        assert facade.execution_tracer is None
        assert facade.tool_call_tracer is None
        assert facade.intelligent_integration is None
        assert facade.subagent_orchestrator is None


class TestOrchestrationFacadeProperties:
    """Tests for OrchestrationFacade property access."""

    @pytest.fixture
    def facade(self):
        """Create an OrchestrationFacade with mock components."""
        return OrchestrationFacade(
            interaction_runtime=MagicMock(name="interaction"),
            chat_service=MagicMock(name="chat_service"),
            chat_stream_runtime=MagicMock(name="chat_stream_runtime"),
            tool_service=MagicMock(name="tool_service"),
            session_service=MagicMock(name="session_service"),
            context_service=MagicMock(name="context_service"),
            provider_service=MagicMock(name="provider_service"),
            recovery_service=MagicMock(name="recovery_service"),
            deprecated_chat_coordinator=MagicMock(name="chat"),
            deprecated_tool_coordinator=MagicMock(name="tool"),
            deprecated_session_coordinator=MagicMock(name="session"),
            turn_executor=MagicMock(name="execution"),
            deprecated_sync_chat_coordinator=MagicMock(name="sync"),
            deprecated_streaming_chat_coordinator=MagicMock(name="streaming_chat"),
            deprecated_unified_chat_coordinator=MagicMock(name="unified"),
            protocol_adapter=MagicMock(name="adapter"),
            streaming_handler=MagicMock(name="handler"),
            streaming_controller=MagicMock(name="controller"),
            streaming_coordinator=MagicMock(name="coordinator"),
            iteration_coordinator=MagicMock(name="iteration"),
            task_analyzer=MagicMock(name="analyzer"),
            exploration_state_passed=MagicMock(name="exploration_state_passed"),
            system_prompt_state_passed=MagicMock(name="system_prompt_state_passed"),
            safety_state_passed=MagicMock(name="safety_state_passed"),
            coordination_state_passed=MagicMock(name="coordination_state_passed"),
            presentation=MagicMock(name="presentation"),
            vertical_integration_adapter=MagicMock(name="vertical_adapter"),
            vertical_context=MagicMock(name="vertical_ctx"),
            observability=MagicMock(name="observability"),
            intelligent_integration=MagicMock(name="intelligent"),
            subagent_orchestrator=MagicMock(name="subagent"),
        )

    def test_protocol_adapter_property(self, facade):
        """ProtocolAdapter property returns the adapter."""
        assert facade.protocol_adapter._mock_name == "adapter"

    def test_chat_service_property(self, facade):
        """ChatService property returns the canonical service."""
        assert facade.chat_service._mock_name == "chat_service"

    def test_tool_service_property(self, facade):
        """ToolService property returns the canonical service."""
        assert facade.tool_service._mock_name == "tool_service"

    def test_chat_stream_runtime_property(self, facade):
        """chat_stream_runtime returns the canonical service-owned runtime."""
        assert facade.chat_stream_runtime._mock_name == "chat_stream_runtime"

    def test_chat_stream_runtime_property_resolves_lazy_getter(self):
        """chat_stream_runtime resolves lazily when only a getter is provided."""
        runtime = MagicMock(name="chat_stream_runtime")
        facade = OrchestrationFacade(get_chat_stream_runtime=lambda: runtime)

        assert facade.chat_stream_runtime is runtime

    def test_exploration_state_passed_property(self, facade):
        """State-passed exploration coordinator should be exposed directly."""
        assert facade.exploration_state_passed._mock_name == "exploration_state_passed"

    def test_system_prompt_state_passed_property(self, facade):
        """State-passed system prompt coordinator should be exposed directly."""
        assert facade.system_prompt_state_passed._mock_name == "system_prompt_state_passed"

    def test_safety_state_passed_property(self, facade):
        """State-passed safety coordinator should be exposed directly."""
        assert facade.safety_state_passed._mock_name == "safety_state_passed"

    def test_coordination_state_passed_property(self, facade):
        """State-passed coordination coordinator should be exposed directly."""
        assert facade.coordination_state_passed._mock_name == "coordination_state_passed"

    def test_chat_coordinator_property_is_deprecated(self, facade):
        """ChatCoordinator property remains available as a deprecated shim."""
        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade.chat_coordinator is deprecated",
        ):
            coordinator = facade.chat_coordinator

        assert coordinator._mock_name == "chat"
        telemetry = get_deprecated_chat_shim_telemetry()
        assert telemetry["orchestration_facade.chat_coordinator.direct_slot"] == 1

    def test_chat_coordinator_property_resolves_lazy_compatibility_getter(self):
        """ChatCoordinator compatibility accessor resolves lazily when needed."""
        chat = MagicMock(name="chat")
        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade\\(get_chat_coordinator=...\\) is deprecated",
        ):
            facade = OrchestrationFacade(get_chat_coordinator=lambda: chat)

        telemetry = get_deprecated_chat_shim_telemetry()
        assert telemetry["orchestration_facade.get_chat_coordinator.deprecated_input"] == 1

        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade.chat_coordinator is deprecated",
        ):
            coordinator = facade.chat_coordinator

        assert coordinator is chat
        telemetry = get_deprecated_chat_shim_telemetry()
        assert telemetry["orchestration_facade.chat_coordinator.lazy_getter"] == 1

    def test_chat_coordinator_property_records_missing_surface_access(self):
        facade = OrchestrationFacade()

        assert facade.chat_coordinator is None

        telemetry = get_deprecated_chat_shim_telemetry()
        assert telemetry["orchestration_facade.chat_coordinator.missing_surface"] == 1

    def test_old_chat_coordinator_kwarg_warns(self):
        """Old chat_coordinator kwarg should warn and remain compatible."""
        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade\\(chat_coordinator=...\\) is deprecated",
        ):
            facade = OrchestrationFacade(chat_coordinator=MagicMock(name="chat"))

        telemetry = get_deprecated_chat_shim_telemetry()
        assert telemetry["orchestration_facade.chat_coordinator.deprecated_input"] == 1

        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade.chat_coordinator is deprecated",
        ):
            assert facade.chat_coordinator is not None

    def test_old_and_new_chat_coordinator_kwargs_conflict(self):
        with pytest.raises(
            TypeError,
            match="Use only one of chat_coordinator or deprecated_chat_coordinator",
        ):
            OrchestrationFacade(
                chat_coordinator=MagicMock(),
                deprecated_chat_coordinator=MagicMock(),
            )

    def test_tool_coordinator_property_is_deprecated(self, facade):
        """ToolCoordinator property remains available as a deprecated shim."""
        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade.tool_coordinator is deprecated",
        ):
            coordinator = facade.tool_coordinator

        assert coordinator._mock_name == "tool"

    def test_tool_coordinator_property_resolves_lazy_compatibility_getter(self):
        """ToolCoordinator compatibility accessor resolves lazily when needed."""
        tool = MagicMock(name="tool")
        facade = OrchestrationFacade(get_tool_coordinator=lambda: tool)

        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade.tool_coordinator is deprecated",
        ):
            coordinator = facade.tool_coordinator

        assert coordinator is tool

    def test_session_coordinator_property_is_deprecated(self, facade):
        """SessionCoordinator property remains available as a deprecated shim."""
        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade.session_coordinator is deprecated",
        ):
            coordinator = facade.session_coordinator

        assert coordinator._mock_name == "session"

    def test_session_coordinator_property_resolves_lazy_compatibility_getter(self):
        """SessionCoordinator compatibility accessor resolves lazily when needed."""
        session = MagicMock(name="session")
        facade = OrchestrationFacade(get_session_coordinator=lambda: session)

        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade.session_coordinator is deprecated",
        ):
            coordinator = facade.session_coordinator

        assert coordinator is session

    def test_old_tool_coordinator_kwarg_warns(self):
        """Old tool_coordinator kwarg should warn and remain compatible."""
        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade\\(tool_coordinator=...\\) is deprecated",
        ):
            facade = OrchestrationFacade(tool_coordinator=MagicMock(name="tool"))

        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade.tool_coordinator is deprecated",
        ):
            assert facade.tool_coordinator is not None

    def test_old_session_coordinator_kwarg_warns(self):
        """Old session_coordinator kwarg should warn and remain compatible."""
        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade\\(session_coordinator=...\\) is deprecated",
        ):
            facade = OrchestrationFacade(session_coordinator=MagicMock(name="session"))

        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade.session_coordinator is deprecated",
        ):
            assert facade.session_coordinator is not None

    def test_old_and_new_tool_coordinator_kwargs_conflict(self):
        with pytest.raises(
            TypeError,
            match="Use only one of tool_coordinator or deprecated_tool_coordinator",
        ):
            OrchestrationFacade(
                tool_coordinator=MagicMock(),
                deprecated_tool_coordinator=MagicMock(),
            )

    def test_old_and_new_session_coordinator_kwargs_conflict(self):
        with pytest.raises(
            TypeError,
            match="Use only one of session_coordinator or deprecated_session_coordinator",
        ):
            OrchestrationFacade(
                session_coordinator=MagicMock(),
                deprecated_session_coordinator=MagicMock(),
            )

    def test_sync_chat_coordinator_property_is_deprecated(self, facade):
        """Sync chat coordinator remains available as a deprecated shim."""
        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade.sync_chat_coordinator is deprecated",
        ):
            coordinator = facade.sync_chat_coordinator

        assert coordinator._mock_name == "sync"
        telemetry = get_deprecated_chat_shim_telemetry()
        assert telemetry["orchestration_facade.sync_chat_coordinator.direct_slot"] == 1

    def test_sync_chat_coordinator_property_resolves_lazy_compatibility_getter(self):
        """Sync chat coordinator compatibility accessor resolves lazily when needed."""
        sync = MagicMock(name="sync")
        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade\\(get_sync_chat_coordinator=...\\) is deprecated",
        ):
            facade = OrchestrationFacade(get_sync_chat_coordinator=lambda: sync)

        telemetry = get_deprecated_chat_shim_telemetry()
        assert telemetry["orchestration_facade.get_sync_chat_coordinator.deprecated_input"] == 1

        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade.sync_chat_coordinator is deprecated",
        ):
            coordinator = facade.sync_chat_coordinator

        assert coordinator is sync
        telemetry = get_deprecated_chat_shim_telemetry()
        assert telemetry["orchestration_facade.sync_chat_coordinator.lazy_getter"] == 1

    def test_streaming_chat_coordinator_property_is_deprecated(self, facade):
        """Streaming chat coordinator remains available as a deprecated shim."""
        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade.streaming_chat_coordinator is deprecated",
        ):
            coordinator = facade.streaming_chat_coordinator

        assert coordinator._mock_name == "streaming_chat"
        telemetry = get_deprecated_chat_shim_telemetry()
        assert telemetry["orchestration_facade.streaming_chat_coordinator.direct_slot"] == 1

    def test_streaming_chat_coordinator_property_resolves_lazy_compatibility_getter(self):
        """Streaming chat coordinator compatibility accessor resolves lazily when needed."""
        streaming = MagicMock(name="streaming_chat")
        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade\\(get_streaming_chat_coordinator=...\\) is deprecated",
        ):
            facade = OrchestrationFacade(get_streaming_chat_coordinator=lambda: streaming)

        telemetry = get_deprecated_chat_shim_telemetry()
        assert (
            telemetry["orchestration_facade.get_streaming_chat_coordinator.deprecated_input"] == 1
        )

        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade.streaming_chat_coordinator is deprecated",
        ):
            coordinator = facade.streaming_chat_coordinator

        assert coordinator is streaming
        telemetry = get_deprecated_chat_shim_telemetry()
        assert telemetry["orchestration_facade.streaming_chat_coordinator.lazy_getter"] == 1

    def test_unified_chat_coordinator_property_is_deprecated(self, facade):
        """Unified chat coordinator remains available as a deprecated shim."""
        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade.unified_chat_coordinator is deprecated",
        ):
            coordinator = facade.unified_chat_coordinator

        assert coordinator._mock_name == "unified"
        telemetry = get_deprecated_chat_shim_telemetry()
        assert telemetry["orchestration_facade.unified_chat_coordinator.direct_slot"] == 1

    def test_unified_chat_coordinator_property_resolves_lazy_compatibility_getter(self):
        """Unified chat coordinator compatibility accessor resolves lazily when needed."""
        unified = MagicMock(name="unified")
        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade\\(get_unified_chat_coordinator=...\\) is deprecated",
        ):
            facade = OrchestrationFacade(get_unified_chat_coordinator=lambda: unified)

        telemetry = get_deprecated_chat_shim_telemetry()
        assert telemetry["orchestration_facade.get_unified_chat_coordinator.deprecated_input"] == 1

        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade.unified_chat_coordinator is deprecated",
        ):
            coordinator = facade.unified_chat_coordinator

        assert coordinator is unified
        telemetry = get_deprecated_chat_shim_telemetry()
        assert telemetry["orchestration_facade.unified_chat_coordinator.lazy_getter"] == 1

    def test_deprecated_chat_compat_properties_record_missing_surface_access(self):
        facade = OrchestrationFacade()

        assert facade.sync_chat_coordinator is None
        assert facade.streaming_chat_coordinator is None
        assert facade.unified_chat_coordinator is None

        telemetry = get_deprecated_chat_shim_telemetry()
        assert telemetry["orchestration_facade.sync_chat_coordinator.missing_surface"] == 1
        assert telemetry["orchestration_facade.streaming_chat_coordinator.missing_surface"] == 1
        assert telemetry["orchestration_facade.unified_chat_coordinator.missing_surface"] == 1

    def test_old_sync_chat_coordinator_kwarg_warns(self):
        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade\\(sync_chat_coordinator=...\\) is deprecated",
        ):
            facade = OrchestrationFacade(sync_chat_coordinator=MagicMock(name="sync"))

        telemetry = get_deprecated_chat_shim_telemetry()
        assert telemetry["orchestration_facade.sync_chat_coordinator.deprecated_input"] == 1

        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade.sync_chat_coordinator is deprecated",
        ):
            assert facade.sync_chat_coordinator is not None

    def test_old_streaming_chat_coordinator_kwarg_warns(self):
        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade\\(streaming_chat_coordinator=...\\) is deprecated",
        ):
            facade = OrchestrationFacade(
                streaming_chat_coordinator=MagicMock(name="streaming_chat")
            )

        telemetry = get_deprecated_chat_shim_telemetry()
        assert telemetry["orchestration_facade.streaming_chat_coordinator.deprecated_input"] == 1

        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade.streaming_chat_coordinator is deprecated",
        ):
            assert facade.streaming_chat_coordinator is not None

    def test_old_unified_chat_coordinator_kwarg_warns(self):
        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade\\(unified_chat_coordinator=...\\) is deprecated",
        ):
            facade = OrchestrationFacade(unified_chat_coordinator=MagicMock(name="unified"))

        telemetry = get_deprecated_chat_shim_telemetry()
        assert telemetry["orchestration_facade.unified_chat_coordinator.deprecated_input"] == 1

        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade.unified_chat_coordinator is deprecated",
        ):
            assert facade.unified_chat_coordinator is not None

    def test_old_and_new_sync_chat_coordinator_kwargs_conflict(self):
        with pytest.raises(
            TypeError,
            match="Use only one of sync_chat_coordinator or deprecated_sync_chat_coordinator",
        ):
            OrchestrationFacade(
                sync_chat_coordinator=MagicMock(),
                deprecated_sync_chat_coordinator=MagicMock(),
            )

    def test_old_and_new_streaming_chat_coordinator_kwargs_conflict(self):
        with pytest.raises(
            TypeError,
            match=(
                "Use only one of streaming_chat_coordinator or "
                "deprecated_streaming_chat_coordinator"
            ),
        ):
            OrchestrationFacade(
                streaming_chat_coordinator=MagicMock(),
                deprecated_streaming_chat_coordinator=MagicMock(),
            )

    def test_old_and_new_unified_chat_coordinator_kwargs_conflict(self):
        with pytest.raises(
            TypeError,
            match="Use only one of unified_chat_coordinator or deprecated_unified_chat_coordinator",
        ):
            OrchestrationFacade(
                unified_chat_coordinator=MagicMock(),
                deprecated_unified_chat_coordinator=MagicMock(),
            )

    def test_protocol_adapter_setter(self, facade):
        """ProtocolAdapter setter updates the adapter."""
        new_adapter = MagicMock(name="new_adapter")
        facade.protocol_adapter = new_adapter
        assert facade.protocol_adapter is new_adapter

    def test_task_analyzer_property(self, facade):
        """TaskAnalyzer property returns the analyzer."""
        assert facade.task_analyzer._mock_name == "analyzer"

    def test_streaming_controller_property(self, facade):
        """StreamingController property returns the controller."""
        assert facade.streaming_controller._mock_name == "controller"

    def test_streaming_handler_property(self, facade):
        """StreamingHandler property returns the handler."""
        assert facade.streaming_handler._mock_name == "handler"

    def test_vertical_context_property(self, facade):
        """VerticalContext property returns the context."""
        assert facade.vertical_context._mock_name == "vertical_ctx"

    def test_observability_property(self, facade):
        """Observability property returns the integration."""
        assert facade.observability._mock_name == "observability"

    def test_observability_setter(self, facade):
        """Observability setter updates the integration."""
        new_obs = MagicMock(name="new_obs")
        facade.observability = new_obs
        assert facade.observability is new_obs

    def test_turn_executor_setter(self, facade):
        """TurnExecutor setter updates the coordinator."""
        new_coord = MagicMock(name="new_coord")
        facade.turn_executor = new_coord
        assert facade.turn_executor is new_coord

    def test_sync_chat_coordinator_setter_warns_and_records_telemetry(self, facade):
        new_sync = MagicMock(name="new_sync")

        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade.sync_chat_coordinator is deprecated",
        ):
            facade.sync_chat_coordinator = new_sync

        assert facade._deprecated_sync_chat_coordinator is new_sync
        telemetry = get_deprecated_chat_shim_telemetry()
        assert telemetry["orchestration_facade.sync_chat_coordinator.setter"] == 1

    def test_streaming_chat_coordinator_setter_warns_and_records_telemetry(self, facade):
        new_streaming = MagicMock(name="new_streaming")

        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade.streaming_chat_coordinator is deprecated",
        ):
            facade.streaming_chat_coordinator = new_streaming

        assert facade._deprecated_streaming_chat_coordinator is new_streaming
        telemetry = get_deprecated_chat_shim_telemetry()
        assert telemetry["orchestration_facade.streaming_chat_coordinator.setter"] == 1

    def test_unified_chat_coordinator_setter_warns_and_records_telemetry(self, facade):
        new_unified = MagicMock(name="new_unified")

        with pytest.warns(
            DeprecationWarning,
            match="OrchestrationFacade.unified_chat_coordinator is deprecated",
        ):
            facade.unified_chat_coordinator = new_unified

        assert facade._deprecated_unified_chat_coordinator is new_unified
        telemetry = get_deprecated_chat_shim_telemetry()
        assert telemetry["orchestration_facade.unified_chat_coordinator.setter"] == 1

    def test_intelligent_integration_setter(self, facade):
        """IntelligentIntegration setter updates the integration."""
        new_int = MagicMock(name="new_int")
        facade.intelligent_integration = new_int
        assert facade.intelligent_integration is new_int

    def test_subagent_orchestrator_setter(self, facade):
        """SubagentOrchestrator setter updates the orchestrator."""
        new_sub = MagicMock(name="new_sub")
        facade.subagent_orchestrator = new_sub
        assert facade.subagent_orchestrator is new_sub

    def test_iteration_coordinator_setter(self, facade):
        """IterationCoordinator setter updates the coordinator."""
        new_iter = MagicMock(name="new_iter")
        facade.iteration_coordinator = new_iter
        assert facade.iteration_coordinator is new_iter


class TestOrchestrationFacadeProtocolConformance:
    """Tests that OrchestrationFacade satisfies OrchestrationFacadeProtocol."""

    def test_satisfies_protocol(self):
        """OrchestrationFacade structurally conforms to OrchestrationFacadeProtocol."""
        facade = OrchestrationFacade()
        assert isinstance(facade, OrchestrationFacadeProtocol)

    def test_protocol_properties_present(self):
        """All protocol-required properties are present on OrchestrationFacade."""
        required = [
            "protocol_adapter",
            "task_analyzer",
            "exploration_state_passed",
            "system_prompt_state_passed",
            "safety_state_passed",
            "coordination_state_passed",
        ]
        facade = OrchestrationFacade()
        for prop in required:
            assert hasattr(facade, prop), f"Missing protocol property: {prop}"
