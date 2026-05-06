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

        AgentRuntimeBootstrapper.create_facades(orch)

        assert orch._chat_facade.conversation_controller is orch._conversation_controller
        assert orch._tool_facade.tool_pipeline is orch._tool_pipeline
        assert orch._provider_facade.provider_manager is orch._provider_manager
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
        assert kwargs["provider_service"] is getattr(orch, "_provider_service", None)
        assert kwargs["runtime_state_host"] is orch
        assert "provider_coordinator" not in kwargs
        assert "provider_switch_coordinator" not in kwargs

    def test_create_facades_chat_facade_derives_live_runtime_state_from_orchestrator(self):
        orch = self._make_mock_orchestrator()

        with patch("victor.agent.facades.ChatFacade") as facade_cls:
            AgentRuntimeBootstrapper.create_facades(orch)
            assert (
                orch._chat_facade.conversation_controller
                is facade_cls.return_value.conversation_controller
            )

        kwargs = facade_cls.call_args.kwargs
        assert kwargs["runtime_state_host"] is orch

    def test_create_facades_resilience_facade_derives_live_runtime_state_from_orchestrator(self):
        orch = self._make_mock_orchestrator()

        with patch("victor.agent.facades.ResilienceFacade") as facade_cls:
            AgentRuntimeBootstrapper.create_facades(orch)
            assert (
                orch._resilience_facade.recovery_coordinator
                is facade_cls.return_value.recovery_coordinator
            )

        kwargs = facade_cls.call_args.kwargs
        assert kwargs["runtime_state_host"] is orch

    def test_create_facades_workflow_facade_derives_live_runtime_state_from_orchestrator(self):
        orch = self._make_mock_orchestrator()

        with patch("victor.agent.facades.WorkflowFacade") as facade_cls:
            AgentRuntimeBootstrapper.create_facades(orch)
            assert (
                orch._workflow_facade.workflow_runtime
                is facade_cls.return_value.workflow_runtime
            )

        kwargs = facade_cls.call_args.kwargs
        assert kwargs["runtime_state_host"] is orch

    def test_create_facades_orchestration_facade_derives_live_runtime_state_from_orchestrator(
        self,
    ):
        orch = self._make_mock_orchestrator()

        with patch("victor.agent.facades.OrchestrationFacade") as facade_cls:
            AgentRuntimeBootstrapper.create_facades(orch)
            assert orch._orchestration_facade.chat_service is facade_cls.return_value.chat_service

        kwargs = facade_cls.call_args.kwargs
        assert kwargs["runtime_state_host"] is orch

    def test_provider_facade_tracks_current_orchestrator_state_after_materialization(self):
        orch = self._make_mock_orchestrator()
        orch.provider = sentinel.initial_provider
        orch.model = "initial-model"
        orch.provider_name = "initial-provider"
        orch.temperature = 0.3
        orch.max_tokens = 1024
        orch.thinking = True

        AgentRuntimeBootstrapper.create_facades(orch)

        provider_facade = orch._provider_facade
        assert provider_facade.model == "initial-model"

        orch.provider = sentinel.updated_provider
        orch.model = "updated-model"
        orch.provider_name = "updated-provider"
        orch.temperature = 0.8
        orch.max_tokens = 4096
        orch.thinking = False

        assert provider_facade.provider is sentinel.updated_provider
        assert provider_facade.model == "updated-model"
        assert provider_facade.provider_name == "updated-provider"
        assert provider_facade.temperature == 0.8
        assert provider_facade.max_tokens == 4096
        assert provider_facade.thinking is False

    def test_chat_facade_tracks_current_orchestrator_chat_state_after_materialization(self):
        orch = self._make_mock_orchestrator()
        orch._memory_session_id = "initial-memory"
        orch._conversation_embedding_store = sentinel.initial_embedding_store
        orch._system_prompt = "initial prompt"
        orch._context_compactor = sentinel.initial_compactor

        AgentRuntimeBootstrapper.create_facades(orch)

        chat_facade = orch._chat_facade
        assert chat_facade.memory_session_id == "initial-memory"
        assert chat_facade.embedding_store is sentinel.initial_embedding_store
        assert chat_facade.system_prompt == "initial prompt"
        assert chat_facade.context_compactor is sentinel.initial_compactor

        orch._memory_session_id = "updated-memory"
        orch._conversation_embedding_store = sentinel.updated_embedding_store
        orch._system_prompt = "updated prompt"
        orch._context_compactor = sentinel.updated_compactor

        assert chat_facade.memory_session_id == "updated-memory"
        assert chat_facade.embedding_store is sentinel.updated_embedding_store
        assert chat_facade.system_prompt == "updated prompt"
        assert chat_facade.context_compactor is sentinel.updated_compactor

    def test_tool_facade_tracks_current_orchestrator_budget_after_materialization(self):
        orch = self._make_mock_orchestrator()
        orch.tool_budget = 10

        AgentRuntimeBootstrapper.create_facades(orch)

        tool_facade = orch._tool_facade
        assert tool_facade.tool_budget == 10

        orch.tool_budget = 25
        assert tool_facade.tool_budget == 25

    def test_session_facade_tracks_current_orchestrator_session_state_after_materialization(self):
        orch = self._make_mock_orchestrator()
        orch._session_ledger = sentinel.initial_ledger
        orch.active_session_id = "initial-session"
        orch._memory_session_id = "initial-memory"

        AgentRuntimeBootstrapper.create_facades(orch)

        session_facade = orch._session_facade
        assert session_facade.session_ledger is sentinel.initial_ledger
        assert session_facade.active_session_id == "initial-session"
        assert session_facade.memory_session_id == "initial-memory"

        orch._session_ledger = sentinel.updated_ledger
        orch.active_session_id = "updated-session"
        orch._memory_session_id = "updated-memory"

        assert session_facade.session_ledger is sentinel.updated_ledger
        assert session_facade.active_session_id == "updated-session"
        assert session_facade.memory_session_id == "updated-memory"

    def test_resilience_facade_tracks_current_orchestrator_state_after_materialization(self):
        orch = self._make_mock_orchestrator()
        orch._recovery_handler = sentinel.initial_handler
        orch._recovery_integration = sentinel.initial_integration
        orch._cancel_event = sentinel.initial_cancel_event
        orch._is_streaming = False

        AgentRuntimeBootstrapper.create_facades(orch)

        resilience_facade = orch._resilience_facade
        assert resilience_facade.recovery_handler is sentinel.initial_handler
        assert resilience_facade.recovery_integration is sentinel.initial_integration
        assert resilience_facade.cancel_event is sentinel.initial_cancel_event
        assert resilience_facade.is_streaming is False

        orch._recovery_handler = sentinel.updated_handler
        orch._recovery_integration = sentinel.updated_integration
        orch._cancel_event = sentinel.updated_cancel_event
        orch._is_streaming = True

        assert resilience_facade.recovery_handler is sentinel.updated_handler
        assert resilience_facade.recovery_integration is sentinel.updated_integration
        assert resilience_facade.cancel_event is sentinel.updated_cancel_event
        assert resilience_facade.is_streaming is True

    def test_workflow_facade_tracks_current_orchestrator_state_after_materialization(self):
        orch = self._make_mock_orchestrator()
        orch._workflow_registry = sentinel.initial_registry
        orch._coordination_advisor = sentinel.initial_advisor

        AgentRuntimeBootstrapper.create_facades(orch)

        workflow_facade = orch._workflow_facade
        assert workflow_facade.workflow_registry is sentinel.initial_registry
        assert workflow_facade.coordination_advisor is sentinel.initial_advisor

        orch._workflow_registry = sentinel.updated_registry
        orch._coordination_advisor = sentinel.updated_advisor

        assert workflow_facade.workflow_registry is sentinel.updated_registry
        assert workflow_facade.coordination_advisor is sentinel.updated_advisor

    def test_orchestration_facade_tracks_current_orchestrator_state_after_materialization(self):
        orch = self._make_mock_orchestrator()
        orch._chat_stream_adapter = sentinel.initial_chat_stream_adapter
        orch._turn_executor = sentinel.initial_turn_executor
        orch._protocol_adapter = sentinel.initial_protocol_adapter
        orch._iteration_coordinator = sentinel.initial_iteration_coordinator
        orch._observability = sentinel.initial_observability
        orch._runtime_intelligence_integration = sentinel.initial_runtime_intelligence
        orch._subagent_orchestrator = sentinel.initial_subagent

        AgentRuntimeBootstrapper.create_facades(orch)

        orchestration_facade = orch._orchestration_facade
        assert orchestration_facade.chat_stream_adapter is sentinel.initial_chat_stream_adapter
        assert orchestration_facade.turn_executor is sentinel.initial_turn_executor
        assert orchestration_facade.protocol_adapter is sentinel.initial_protocol_adapter
        assert orchestration_facade.iteration_coordinator is sentinel.initial_iteration_coordinator
        assert orchestration_facade.observability is sentinel.initial_observability
        assert (
            orchestration_facade.runtime_intelligence_integration
            is sentinel.initial_runtime_intelligence
        )
        assert orchestration_facade.subagent_orchestrator is sentinel.initial_subagent
        assert hasattr(orchestration_facade, "chat_coordinator") is False
        assert hasattr(orchestration_facade, "tool_coordinator") is False
        assert hasattr(orchestration_facade, "session_coordinator") is False

        orch._chat_stream_adapter = sentinel.updated_chat_stream_adapter
        orch._turn_executor = sentinel.updated_turn_executor
        orch._protocol_adapter = sentinel.updated_protocol_adapter
        orch._iteration_coordinator = sentinel.updated_iteration_coordinator
        orch._observability = sentinel.updated_observability
        orch._runtime_intelligence_integration = sentinel.updated_runtime_intelligence
        orch._subagent_orchestrator = sentinel.updated_subagent

        assert orchestration_facade.chat_stream_adapter is sentinel.updated_chat_stream_adapter
        assert orchestration_facade.turn_executor is sentinel.updated_turn_executor
        assert orchestration_facade.protocol_adapter is sentinel.updated_protocol_adapter
        assert orchestration_facade.iteration_coordinator is sentinel.updated_iteration_coordinator
        assert orchestration_facade.observability is sentinel.updated_observability
        assert (
            orchestration_facade.runtime_intelligence_integration
            is sentinel.updated_runtime_intelligence
        )
        assert orchestration_facade.subagent_orchestrator is sentinel.updated_subagent

    def test_lazy_orchestration_facade_materializes_state_passed_and_runtime_handles(self):
        orch = self._make_mock_orchestrator()
        orch.runtime_intelligence_integration = sentinel.runtime_intelligence_integration
        orch.subagent_orchestrator = sentinel.subagent_orchestrator
        exploration_state_passed = MagicMock(name="exploration_state_passed")
        system_prompt_state_passed = MagicMock(name="system_prompt_state_passed")
        safety_state_passed = MagicMock(name="safety_state_passed")
        coordination_state_passed = MagicMock(name="coordination_state_passed")
        orch._factory.create_exploration_state_passed_coordinator.return_value = (
            exploration_state_passed
        )
        orch._factory.create_system_prompt_state_passed_coordinator.return_value = (
            system_prompt_state_passed
        )
        orch._factory.create_safety_state_passed_coordinator.return_value = safety_state_passed
        orch._factory.create_coordination_state_passed_coordinator.return_value = (
            coordination_state_passed
        )

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
        assert kwargs["get_chat_stream_adapter"] is orch._get_chat_stream_adapter
        assert kwargs["tool_service"] is getattr(orch, "_tool_service", None)
        assert kwargs["session_service"] is getattr(orch, "_session_service", None)
        assert kwargs["context_service"] is getattr(orch, "_context_service", None)
        assert kwargs["provider_service"] is getattr(orch, "_provider_service", None)
        assert kwargs["recovery_service"] is getattr(orch, "_recovery_service", None)
        assert kwargs["turn_executor"] is orch._turn_executor
        assert "deprecated_chat_coordinator" not in kwargs
        assert "get_chat_coordinator" not in kwargs
        assert "get_tool_coordinator" not in kwargs
        assert "get_session_coordinator" not in kwargs
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
        assert kwargs["coordination_state_passed"] is coordination_state_passed
        assert kwargs["presentation"] is orch._presentation
        assert kwargs["vertical_integration_adapter"] is orch._vertical_integration_adapter
        assert kwargs["vertical_context"] is orch._vertical_context
        assert kwargs["observability"] is orch._observability
        assert kwargs["execution_tracer"] is getattr(orch, "_execution_tracer", None)
        assert kwargs["tool_call_tracer"] is getattr(orch, "_tool_call_tracer", None)
        assert kwargs["runtime_state_host"] is orch
        assert callable(kwargs["get_runtime_intelligence_integration"])
        assert callable(kwargs["get_subagent_orchestrator"])
        assert (
            kwargs["get_runtime_intelligence_integration"]()
            is sentinel.runtime_intelligence_integration
        )
        assert kwargs["get_subagent_orchestrator"]() is sentinel.subagent_orchestrator
        orch._factory.create_exploration_state_passed_coordinator.assert_called_once_with()
        orch._factory.create_system_prompt_state_passed_coordinator.assert_called_once_with(
            task_analyzer=orch._task_analyzer
        )
        orch._factory.create_safety_state_passed_coordinator.assert_called_once_with()
        orch._factory.create_coordination_state_passed_coordinator.assert_called_once_with(
            coordination_runtime=orch._coordination_advisor_runtime,
            coordination_advisor=getattr(orch, "_coordination_advisor", None),
            vertical_context=orch._vertical_context,
        )

    def test_create_facades_does_not_bind_removed_coordinator_inputs_to_facade(self):
        orch = self._make_mock_orchestrator()

        with patch("victor.agent.facades.OrchestrationFacade") as facade_cls:
            AgentRuntimeBootstrapper.create_facades(orch)
            assert orch._orchestration_facade.chat_service is facade_cls.return_value.chat_service

            kwargs = facade_cls.call_args.kwargs

            assert "deprecated_chat_coordinator" not in kwargs
            assert "get_tool_coordinator" not in kwargs
            assert "get_session_coordinator" not in kwargs
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
        assert orch._coordination_advisor is None
        assert orch._coordination_advisor_runtime is None
        assert orch._turn_executor is None
        assert orch._protocol_adapter is None
