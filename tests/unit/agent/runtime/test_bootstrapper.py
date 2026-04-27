"""Tests for AgentRuntimeBootstrapper extraction."""

from unittest.mock import MagicMock, patch, sentinel

from victor.agent.runtime.bootstrapper import AgentRuntimeBootstrapper
from victor.agent.runtime.provider_runtime import LazyRuntimeProxy


class TestAgentRuntimeBootstrapper:
    """Verify bootstrapper correctly delegates to orchestrator attributes."""

    def _make_mock_orchestrator(self):
        """Create a mock orchestrator with all attributes needed by facades."""
        orch = MagicMock()
        orch.active_session_id = "test-session-123"
        orch._background_tasks = set()
        orch._deprecated_sync_chat_coordinator = None
        orch._deprecated_streaming_chat_coordinator = None
        orch._deprecated_unified_chat_coordinator = None
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

    def test_create_facades_lazifies_compatibility_facades(self):
        orch = self._make_mock_orchestrator()

        AgentRuntimeBootstrapper.create_facades(orch)

        assert isinstance(orch._chat_facade, LazyRuntimeProxy)
        assert isinstance(orch._tool_facade, LazyRuntimeProxy)
        assert isinstance(orch._provider_facade, LazyRuntimeProxy)
        assert isinstance(orch._session_facade, LazyRuntimeProxy)
        assert isinstance(orch._metrics_facade, LazyRuntimeProxy)
        assert isinstance(orch._resilience_facade, LazyRuntimeProxy)
        assert isinstance(orch._workflow_facade, LazyRuntimeProxy)
        assert isinstance(orch._orchestration_facade, LazyRuntimeProxy)
        assert orch._chat_facade.initialized is False
        assert orch._tool_facade.initialized is False
        assert orch._provider_facade.initialized is False
        assert orch._session_facade.initialized is False
        assert orch._metrics_facade.initialized is False
        assert orch._resilience_facade.initialized is False
        assert orch._workflow_facade.initialized is False
        assert orch._orchestration_facade.initialized is False

    def test_lazy_facades_materialize_from_current_orchestrator_state(self):
        orch = self._make_mock_orchestrator()
        orch._provider_runtime.provider_coordinator = sentinel.provider_coordinator
        orch._provider_runtime.provider_switch_coordinator = sentinel.provider_switch_coordinator

        AgentRuntimeBootstrapper.create_facades(orch)

        assert orch._chat_facade.conversation_controller is orch._conversation_controller
        assert orch._tool_facade.tool_pipeline is orch._tool_pipeline
        assert orch._provider_facade.provider_manager is orch._provider_manager
        assert orch._provider_facade.provider_coordinator is sentinel.provider_coordinator
        assert (
            orch._provider_facade.provider_switch_coordinator
            is sentinel.provider_switch_coordinator
        )
        assert orch._session_facade.session_ledger is orch._session_ledger
        assert orch._metrics_facade.metrics_runtime is orch._metrics_runtime
        assert orch._resilience_facade.recovery_coordinator is orch._recovery_coordinator
        assert orch._workflow_facade.workflow_registry is orch._workflow_registry
        assert orch._chat_facade.initialized is True
        assert orch._tool_facade.initialized is True
        assert orch._provider_facade.initialized is True
        assert orch._session_facade.initialized is True
        assert orch._metrics_facade.initialized is True
        assert orch._resilience_facade.initialized is True
        assert orch._workflow_facade.initialized is True

    def test_create_facades_provider_facade_derives_compatibility_from_runtime(self):
        orch = self._make_mock_orchestrator()

        with patch("victor.agent.facades.ProviderFacade") as facade_cls:
            AgentRuntimeBootstrapper.create_facades(orch)
            assert (
                orch._provider_facade.provider_manager is facade_cls.return_value.provider_manager
            )

        kwargs = facade_cls.call_args.kwargs
        assert kwargs["provider_runtime"] is orch._provider_runtime
        assert "provider_coordinator" not in kwargs
        assert "provider_switch_coordinator" not in kwargs

    def test_lazy_orchestration_facade_materializes_state_passed_and_runtime_handles(self):
        orch = self._make_mock_orchestrator()
        exploration_state_passed = MagicMock(name="exploration_state_passed")
        system_prompt_state_passed = MagicMock(name="system_prompt_state_passed")
        safety_state_passed = MagicMock(name="safety_state_passed")
        orch._factory.create_exploration_state_passed_coordinator.return_value = (
            exploration_state_passed
        )
        orch._factory.create_system_prompt_state_passed_coordinator.return_value = (
            system_prompt_state_passed
        )
        orch._factory.create_safety_state_passed_coordinator.return_value = safety_state_passed

        with patch("victor.agent.facades.OrchestrationFacade") as facade_cls:
            AgentRuntimeBootstrapper.create_facades(orch)

            facade_cls.assert_not_called()

            assert (
                orch._orchestration_facade.streaming_handler
                is facade_cls.return_value.streaming_handler
            )

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
        assert callable(kwargs["get_chat_coordinator"])
        assert callable(kwargs["get_tool_coordinator"])
        assert callable(kwargs["get_session_coordinator"])
        assert kwargs["turn_executor"] is orch._turn_executor
        assert "get_sync_chat_coordinator" not in kwargs
        assert "get_streaming_chat_coordinator" not in kwargs
        assert "get_unified_chat_coordinator" not in kwargs
        assert kwargs["protocol_adapter"] is orch._protocol_adapter
        assert kwargs["streaming_handler"] is orch._streaming_handler
        assert kwargs["streaming_controller"] is orch._streaming_controller
        assert kwargs["streaming_coordinator"] is orch._streaming_coordinator
        assert kwargs["iteration_coordinator"] is getattr(orch, "_iteration_coordinator", None)
        assert kwargs["task_analyzer"] is orch._task_analyzer
        assert kwargs["exploration_state_passed"] is exploration_state_passed
        assert kwargs["system_prompt_state_passed"] is system_prompt_state_passed
        assert kwargs["safety_state_passed"] is safety_state_passed
        assert kwargs["presentation"] is orch._presentation
        assert kwargs["vertical_integration_adapter"] is orch._vertical_integration_adapter
        assert kwargs["vertical_context"] is orch._vertical_context
        assert kwargs["observability"] is orch._observability
        assert kwargs["execution_tracer"] is getattr(orch, "_execution_tracer", None)
        assert kwargs["tool_call_tracer"] is getattr(orch, "_tool_call_tracer", None)
        assert kwargs["intelligent_integration"] is orch._intelligent_integration
        assert kwargs["subagent_orchestrator"] is orch._subagent_orchestrator
        orch._factory.create_exploration_state_passed_coordinator.assert_called_once_with()
        orch._factory.create_system_prompt_state_passed_coordinator.assert_called_once_with(
            task_analyzer=orch._task_analyzer
        )
        orch._factory.create_safety_state_passed_coordinator.assert_called_once_with()

    def test_create_facades_binds_deprecated_chat_tool_session_getters_to_slots(self):
        orch = self._make_mock_orchestrator()
        orch._deprecated_chat_coordinator = sentinel.chat_coordinator
        orch._deprecated_tool_coordinator = sentinel.tool_coordinator
        orch._deprecated_session_coordinator = sentinel.session_coordinator

        with patch("victor.agent.facades.OrchestrationFacade") as facade_cls:
            AgentRuntimeBootstrapper.create_facades(orch)
            assert orch._orchestration_facade.chat_service is facade_cls.return_value.chat_service

            kwargs = facade_cls.call_args.kwargs

            assert kwargs["get_chat_coordinator"]() is sentinel.chat_coordinator
            assert kwargs["get_tool_coordinator"]() is sentinel.tool_coordinator
            assert kwargs["get_session_coordinator"]() is sentinel.session_coordinator

    def test_create_facades_does_not_bind_deprecated_chat_shim_getters_to_facade(self):
        orch = self._make_mock_orchestrator()

        with patch("victor.agent.facades.OrchestrationFacade") as facade_cls:
            AgentRuntimeBootstrapper.create_facades(orch)
            assert orch._orchestration_facade.chat_service is facade_cls.return_value.chat_service

            kwargs = facade_cls.call_args.kwargs

            assert "get_sync_chat_coordinator" not in kwargs
            assert "get_streaming_chat_coordinator" not in kwargs
            assert "get_unified_chat_coordinator" not in kwargs

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
