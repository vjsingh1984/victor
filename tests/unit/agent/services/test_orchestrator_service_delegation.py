"""Tests verifying orchestrator delegates correctly via both paths.

Tests that the orchestrator methods produce the same results whether
using direct coordinator delegation or service adapter delegation,
based on the USE_SERVICE_LAYER feature flag.
"""

from unittest.mock import MagicMock, patch

import pytest

from victor.core.feature_flags import FeatureFlag, FeatureFlagConfig, FeatureFlagManager


class TestFeatureFlagIntegration:
    """Test that the USE_SERVICE_LAYER flag is properly recognized."""

    def test_flag_exists_in_enum(self):
        assert hasattr(FeatureFlag, "USE_SERVICE_LAYER")
        assert FeatureFlag.USE_SERVICE_LAYER.value == "use_service_layer"

    def test_flag_env_var_name(self):
        assert FeatureFlag.USE_SERVICE_LAYER.get_env_var_name() == "VICTOR_USE_SERVICE_LAYER"

    def test_flag_enabled_by_default(self):
        manager = FeatureFlagManager(FeatureFlagConfig())
        assert manager.is_enabled(FeatureFlag.USE_SERVICE_LAYER) is True

    def test_flag_enabled_via_env(self, monkeypatch):
        monkeypatch.setenv("VICTOR_USE_SERVICE_LAYER", "true")
        manager = FeatureFlagManager(FeatureFlagConfig())
        assert manager.is_enabled(FeatureFlag.USE_SERVICE_LAYER) is True

    def test_flag_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("VICTOR_USE_SERVICE_LAYER", "false")
        manager = FeatureFlagManager(FeatureFlagConfig())
        assert manager.is_enabled(FeatureFlag.USE_SERVICE_LAYER) is False

    def test_flag_runtime_enable(self):
        manager = FeatureFlagManager(FeatureFlagConfig())
        manager.enable(FeatureFlag.USE_SERVICE_LAYER)
        assert manager.is_enabled(FeatureFlag.USE_SERVICE_LAYER) is True


class TestAdapterImports:
    """Test that adapter module imports work correctly."""

    def test_import_tool_adapter(self):
        from victor.agent.services.adapters.tool_adapter import ToolServiceAdapter

        assert ToolServiceAdapter is not None

    def test_import_context_adapter(self):
        from victor.agent.services.adapters.context_adapter import ContextServiceAdapter

        assert ContextServiceAdapter is not None

    def test_import_session_adapter(self):
        from victor.agent.services.adapters.session_adapter import SessionServiceAdapter

        assert SessionServiceAdapter is not None

    def test_import_all_from_package(self):
        from victor.agent.services.adapters import (
            ToolServiceAdapter,
            ContextServiceAdapter,
            SessionServiceAdapter,
        )

        assert all([ToolServiceAdapter, ContextServiceAdapter, SessionServiceAdapter])


class TestServiceProtocolImports:
    """Test that all 6 service protocols are importable."""

    def test_import_all_service_protocols(self):
        from victor.agent.services.protocols import (
            ChatServiceProtocol,
            ContextServiceProtocol,
            ProviderServiceProtocol,
            RecoveryServiceProtocol,
            SessionServiceProtocol,
            ToolServiceProtocol,
        )

        assert all(
            [
                ChatServiceProtocol,
                ToolServiceProtocol,
                SessionServiceProtocol,
                ContextServiceProtocol,
                ProviderServiceProtocol,
                RecoveryServiceProtocol,
            ]
        )

    def test_provider_service_protocol_has_required_methods(self):
        from victor.agent.services.protocols import ProviderServiceProtocol

        required = {
            "switch_provider",
            "switch_model",
            "get_current_provider_info",
            "check_provider_health",
            "get_available_providers",
            "start_health_monitoring",
            "stop_health_monitoring",
            "get_rate_limit_wait_time",
            "get_rate_limit_stats",
            "is_healthy",
        }
        actual = {name for name in dir(ProviderServiceProtocol) if not name.startswith("_")}
        assert required.issubset(actual), f"Missing: {required - actual}"

    def test_tool_service_protocol_has_required_methods(self):
        from victor.agent.services.protocols import ToolServiceProtocol

        required = {
            "select_tools",
            "execute_tool",
            "execute_tools_parallel",
            "get_tool_budget",
            "set_tool_budget",
            "get_tool_usage_stats",
            "reset_tool_budget",
            "process_tool_results",
            "get_available_tools",
            "get_enabled_tools",
            "set_enabled_tools",
            "is_tool_enabled",
            "resolve_tool_alias",
            "parse_and_validate_tool_calls",
            "execute_tool_with_retry",
            "normalize_tool_arguments",
            "build_tool_access_context",
            "validate_tool_call",
            "normalize_arguments_full",
            "on_tool_complete",
            "is_healthy",
        }
        actual = {name for name in dir(ToolServiceProtocol) if not name.startswith("_")}
        assert required.issubset(actual), f"Missing: {required - actual}"

    def test_recovery_service_protocol_has_required_methods(self):
        from victor.agent.services.protocols import RecoveryServiceProtocol

        required = {
            "classify_error",
            "select_recovery_action",
            "execute_recovery",
            "can_retry",
            "should_attempt_recovery",
            "handle_recovery_with_integration",
            "apply_recovery_action",
            "check_natural_completion",
            "handle_empty_response",
            "get_recovery_fallback_message",
            "check_tool_budget",
            "truncate_tool_calls",
            "filter_blocked_tool_calls",
            "check_blocked_threshold",
            "check_force_action",
            "is_healthy",
        }
        actual = {name for name in dir(RecoveryServiceProtocol) if not name.startswith("_")}
        assert required.issubset(actual), f"Missing: {required - actual}"

    def test_context_service_protocol_has_required_methods(self):
        from victor.agent.services.protocols import ContextServiceProtocol

        required = {
            "get_context_metrics",
            "check_context_overflow",
            "compact_context",
            "add_message",
            "get_messages",
            "is_healthy",
        }
        actual = {name for name in dir(ContextServiceProtocol) if not name.startswith("_")}
        assert required.issubset(actual), f"Missing: {required - actual}"


class TestAdapterProtocolConformance:
    """Test that adapters expose the expected interface methods."""

    def test_tool_adapter_has_required_methods(self):
        from victor.agent.services.adapters.tool_adapter import ToolServiceAdapter

        adapter = ToolServiceAdapter(MagicMock())
        assert callable(getattr(adapter, "get_available_tools", None))
        assert callable(getattr(adapter, "get_enabled_tools", None))
        assert callable(getattr(adapter, "set_enabled_tools", None))
        assert callable(getattr(adapter, "is_tool_enabled", None))
        assert callable(getattr(adapter, "resolve_tool_alias", None))
        assert callable(getattr(adapter, "execute_tool_with_retry", None))
        assert callable(getattr(adapter, "parse_and_validate_tool_calls", None))
        assert callable(getattr(adapter, "normalize_tool_arguments", None))
        assert callable(getattr(adapter, "build_tool_access_context", None))
        assert callable(getattr(adapter, "on_tool_complete", None))
        assert callable(getattr(adapter, "is_healthy", None))

    def test_session_adapter_has_required_methods(self):
        from victor.agent.services.adapters.session_adapter import SessionServiceAdapter

        adapter = SessionServiceAdapter(MagicMock())
        assert callable(getattr(adapter, "get_recent_sessions", None))
        assert callable(getattr(adapter, "recover_session", None))
        assert callable(getattr(adapter, "get_session_stats", None))
        assert callable(getattr(adapter, "save_checkpoint", None))
        assert callable(getattr(adapter, "restore_checkpoint", None))
        assert callable(getattr(adapter, "is_healthy", None))

    def test_context_adapter_has_required_methods(self):
        from victor.agent.services.adapters.context_adapter import ContextServiceAdapter

        adapter = ContextServiceAdapter(MagicMock())
        assert callable(getattr(adapter, "get_context_metrics", None))
        assert callable(getattr(adapter, "compact_context", None))
        assert callable(getattr(adapter, "add_message", None))
        assert callable(getattr(adapter, "get_messages", None))
        assert callable(getattr(adapter, "is_healthy", None))


class TestChatServiceBootstrapLaziness:
    """Chat service bootstrap should not force deprecated coordinator init."""

    def test_initialize_services_keeps_chat_and_tool_coordinators_lazy(self):
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.agent.runtime.provider_runtime import LazyRuntimeProxy

        obj = object.__new__(AgentOrchestrator)
        chat_service = MagicMock()
        tool_service = MagicMock()
        session_service = MagicMock()
        context_service = MagicMock()
        provider_service = MagicMock()
        recovery_service = MagicMock()

        obj._container = MagicMock()
        obj._container.get_optional.side_effect = [
            chat_service,
            tool_service,
            session_service,
            context_service,
            provider_service,
            recovery_service,
        ]
        obj._container.is_registered.return_value = True
        obj._conversation_controller = MagicMock()
        obj._streaming_controller = MagicMock()
        obj._provider_coordinator = MagicMock()
        obj._recovery_coordinator = MagicMock()
        obj._recovery_handler = MagicMock()
        obj._recovery_integration = MagicMock()
        obj._provider_manager = MagicMock()
        obj._lifecycle_manager = MagicMock()
        obj._checkpoint_manager = MagicMock()
        obj._session_cost_tracker = MagicMock()
        obj._tool_planner = MagicMock()
        obj._streaming_handler = MagicMock()
        obj._context_compactor = MagicMock()
        obj.unified_tracker = MagicMock()
        obj.settings = MagicMock()
        obj._presentation = MagicMock()
        obj.tools = MagicMock()
        obj.mode_controller = MagicMock()
        obj.argument_normalizer = MagicMock()
        obj._tool_pipeline = MagicMock()
        obj.tool_cache = MagicMock()
        obj.memory_manager = MagicMock()
        obj._memory_session_id = "mem-1"
        obj.tool_budget = 100
        obj._turn_executor = None
        obj._protocol_adapter = None
        obj.observability = MagicMock()

        class TrapChatCoordinator:
            touched = False

            def __getattr__(self, name):
                self.touched = True
                raise AssertionError(f"chat coordinator should remain lazy during bootstrap: {name}")

        trap_chat = TrapChatCoordinator()
        obj._chat_coordinator = LazyRuntimeProxy(
            factory=lambda: trap_chat,
            name="chat_coordinator",
        )

        class TrapToolCoordinator:
            touched = False

            def __getattr__(self, name):
                self.touched = True
                raise AssertionError(f"tool coordinator should remain lazy during bootstrap: {name}")

        trap_tool = TrapToolCoordinator()
        obj._tool_coordinator = LazyRuntimeProxy(
            factory=lambda: trap_tool,
            name="tool_coordinator",
        )

        with (
            patch.object(AgentOrchestrator, "_register_coordinators_for_services"),
            patch.object(AgentOrchestrator, "_bootstrap_service_layer"),
        ):
            obj._initialize_services()

        tool_service.bind_runtime_components.assert_called_once()
        tool_kwargs = tool_service.bind_runtime_components.call_args.kwargs
        assert tool_kwargs["tool_registry"] is obj.tools
        assert tool_kwargs["tool_pipeline"] is obj._tool_pipeline
        assert tool_kwargs["tool_cache"] is obj.tool_cache
        assert tool_kwargs["mode_controller"] is obj.mode_controller
        assert tool_kwargs["tool_planner"] is obj._tool_planner
        assert tool_kwargs["argument_normalizer"] is obj.argument_normalizer
        assert "retry_executor" not in tool_kwargs
        assert "tool_call_parser" not in tool_kwargs
        assert obj._tool_coordinator.initialized is False
        assert trap_tool.touched is False

        chat_service.bind_runtime_components.assert_called_once()
        kwargs = chat_service.bind_runtime_components.call_args.kwargs
        assert kwargs["turn_executor"].initialized is False
        assert callable(kwargs["planning_handler"])
        assert callable(kwargs["stream_chat_handler"])
        assert callable(kwargs["context_limit_handler"])
        assert obj._chat_coordinator.initialized is False
        assert trap_chat.touched is False

        recovery_service.bind_runtime_components.assert_called_once()
        recovery_kwargs = recovery_service.bind_runtime_components.call_args.kwargs
        assert recovery_kwargs["streaming_handler"] is obj._streaming_handler
        assert recovery_kwargs["context_compactor"] is obj._context_compactor
        assert recovery_kwargs["unified_tracker"] is obj.unified_tracker
        assert recovery_kwargs["settings"] is obj.settings
        assert recovery_kwargs["presentation"] is obj._presentation
