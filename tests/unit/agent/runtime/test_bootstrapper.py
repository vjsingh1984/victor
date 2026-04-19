"""Tests for AgentRuntimeBootstrapper extraction."""

from unittest.mock import MagicMock, patch

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
        assert orch._sync_chat_coordinator is None
        assert orch._streaming_chat_coordinator is None
        assert orch._unified_chat_coordinator is None
        assert orch._protocol_adapter is None
