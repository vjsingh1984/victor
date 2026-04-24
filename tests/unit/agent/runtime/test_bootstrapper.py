"""Tests for AgentRuntimeBootstrapper extraction."""

from unittest.mock import MagicMock, patch

from victor.agent.coordinators.exploration_state_passed import (
    ExplorationStatePassedCoordinator,
)
from victor.agent.coordinators.safety_state_passed import SafetyStatePassedCoordinator
from victor.agent.coordinators.system_prompt_state_passed import (
    SystemPromptStatePassedCoordinator,
)
from victor.agent.runtime.bootstrapper import AgentRuntimeBootstrapper


class TestAgentRuntimeBootstrapper:
    """Verify bootstrapper correctly delegates to orchestrator attributes."""

    def _make_mock_orchestrator(self):
        """Create a mock orchestrator with all attributes needed by facades."""
        orch = MagicMock()
        orch.active_session_id = "test-session-123"
        orch._background_tasks = set()
        return orch

    def test_create_facades_sets_all_eight(self):
        orch = self._make_mock_orchestrator()
        AgentRuntimeBootstrapper.create_facades(orch)

        assert hasattr(orch, "_chat_facade")
        assert hasattr(orch, "_tool_facade")
        assert hasattr(orch, "_provider_facade")
        assert hasattr(orch, "_session_facade")
        assert hasattr(orch, "_metrics_facade")
        assert hasattr(orch, "_resilience_facade")
        assert hasattr(orch, "_workflow_facade")
        assert hasattr(orch, "_orchestration_facade")

    def test_create_facades_uses_explicit_tool_coordinator_compat_getter(self):
        orch = self._make_mock_orchestrator()

        with patch("victor.agent.facades.OrchestrationFacade") as facade_cls:
            AgentRuntimeBootstrapper.create_facades(orch)

        facade_cls.assert_called_once()
        kwargs = facade_cls.call_args.kwargs

        assert kwargs["interaction_runtime"] is orch._interaction_runtime
        assert kwargs["chat_service"] is getattr(orch, "_chat_service", None)
        assert kwargs["get_chat_stream_runtime"] is orch._get_service_streaming_runtime
        assert kwargs["tool_service"] is getattr(orch, "_tool_service", None)
        assert kwargs["session_service"] is getattr(orch, "_session_service", None)
        assert kwargs["context_service"] is getattr(orch, "_context_service", None)
        assert kwargs["provider_service"] is getattr(orch, "_provider_service", None)
        assert kwargs["recovery_service"] is getattr(orch, "_recovery_service", None)
        assert kwargs["get_chat_coordinator"] is orch._get_deprecated_chat_coordinator
        assert kwargs["get_tool_coordinator"] is orch._get_deprecated_tool_coordinator
        assert kwargs["get_session_coordinator"] is orch._get_deprecated_session_coordinator
        assert kwargs["turn_executor"] is orch._turn_executor
        assert kwargs["get_sync_chat_coordinator"] is orch._get_deprecated_sync_chat_coordinator
        assert (
            kwargs["get_streaming_chat_coordinator"]
            is orch._get_deprecated_streaming_chat_coordinator
        )
        assert kwargs["get_unified_chat_coordinator"] is orch._get_deprecated_unified_chat_coordinator
        assert kwargs["protocol_adapter"] is orch._protocol_adapter
        assert kwargs["streaming_handler"] is orch._streaming_handler
        assert kwargs["streaming_controller"] is orch._streaming_controller
        assert kwargs["streaming_coordinator"] is orch._streaming_coordinator
        assert kwargs["iteration_coordinator"] is getattr(orch, "_iteration_coordinator", None)
        assert kwargs["task_analyzer"] is orch._task_analyzer
        assert isinstance(kwargs["exploration_state_passed"], ExplorationStatePassedCoordinator)
        assert isinstance(kwargs["system_prompt_state_passed"], SystemPromptStatePassedCoordinator)
        assert isinstance(kwargs["safety_state_passed"], SafetyStatePassedCoordinator)
        assert kwargs["presentation"] is orch._presentation
        assert kwargs["vertical_integration_adapter"] is orch._vertical_integration_adapter
        assert kwargs["vertical_context"] is orch._vertical_context
        assert kwargs["observability"] is orch._observability
        assert kwargs["execution_tracer"] is getattr(orch, "_execution_tracer", None)
        assert kwargs["tool_call_tracer"] is getattr(orch, "_tool_call_tracer", None)
        assert kwargs["intelligent_integration"] is orch._intelligent_integration
        assert kwargs["subagent_orchestrator"] is orch._subagent_orchestrator

    def test_wire_lifecycle_calls_setters(self):
        orch = self._make_mock_orchestrator()
        AgentRuntimeBootstrapper.wire_lifecycle(orch)

        orch._lifecycle_manager.set_provider.assert_called_once()
        orch._lifecycle_manager.set_code_manager.assert_called_once()
        orch._lifecycle_manager.set_semantic_selector.assert_called_once()
        orch._lifecycle_manager.set_usage_logger.assert_called_once()
        orch._lifecycle_manager.set_background_tasks.assert_called_once_with([])
        orch._lifecycle_manager.set_flush_analytics_callback.assert_called_once()
        orch._lifecycle_manager.set_stop_health_monitoring_callback.assert_called_once()

    def test_setup_session_context(self):
        orch = self._make_mock_orchestrator()
        with patch("victor.core.context.set_session_id") as mock_set:
            AgentRuntimeBootstrapper.setup_session_context(orch)
            mock_set.assert_called_once_with("test-session-123")

    def test_finalize_calls_all_steps(self):
        orch = self._make_mock_orchestrator()
        with (
            patch.object(AgentRuntimeBootstrapper, "create_facades") as mock_facades,
            patch.object(AgentRuntimeBootstrapper, "wire_lifecycle") as mock_lifecycle,
            patch.object(AgentRuntimeBootstrapper, "assert_protocol_conformance") as mock_protocol,
            patch.object(AgentRuntimeBootstrapper, "setup_session_context") as mock_session,
        ):
            AgentRuntimeBootstrapper.finalize(orch)

            mock_facades.assert_called_once_with(orch)
            mock_lifecycle.assert_called_once_with(orch)
            mock_protocol.assert_called_once_with(orch)
            mock_session.assert_called_once_with(orch)

    def test_prepare_components_creates_checkpoint_and_workflow(self):
        orch = self._make_mock_orchestrator()
        # MagicMock doesn't auto-create dunder-named methods
        orch.__init_capability_registry__ = MagicMock()
        settings = MagicMock()
        AgentRuntimeBootstrapper.prepare_components(orch, settings)

        # Verify factory methods were called
        orch._factory.create_checkpoint_manager.assert_called_once()
        orch._factory.create_workflow_optimization_components.assert_called_once()
        orch._factory.wire_component_dependencies.assert_called_once()

        # Verify runtime boundaries were initialized
        orch._initialize_interaction_runtime.assert_called_once()
        orch._initialize_services.assert_called_once()
        orch.__init_capability_registry__.assert_called_once()

        # Verify lazy placeholders are None
        assert orch._mode_workflow_team_coordinator is None
        assert orch._turn_executor is None
        assert orch._deprecated_sync_chat_coordinator is None
        assert orch._deprecated_streaming_chat_coordinator is None
        assert orch._deprecated_unified_chat_coordinator is None
        assert orch._protocol_adapter is None
