"""Tests verifying orchestrator delegates correctly via both paths.

Tests that the orchestrator methods produce the same results whether
using direct coordinator delegation or service adapter delegation,
based on the USE_SERVICE_LAYER feature flag.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.orchestrator import AgentOrchestrator
from victor.core.feature_flags import FeatureFlag, FeatureFlagConfig, FeatureFlagManager
from victor.providers.base import CompletionResponse, Message, StreamChunk


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

        adapter = ToolServiceAdapter(tool_service=MagicMock())
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

        adapter = SessionServiceAdapter(session_service=MagicMock())
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

    @pytest.mark.asyncio
    async def test_orchestrator_protocol_adapter_prefers_execute_tool_calls(self):
        from victor.agent.services.orchestrator_protocol_adapter import OrchestratorProtocolAdapter

        orchestrator = MagicMock()
        orchestrator.execute_tool_calls = AsyncMock(return_value=[{"name": "read", "success": True}])
        orchestrator._handle_tool_calls = AsyncMock(
            side_effect=AssertionError("legacy _handle_tool_calls bridge should not be used")
        )

        adapter = OrchestratorProtocolAdapter(orchestrator)
        result = await adapter.execute_tool_calls([{"name": "read", "arguments": {}}])

        assert result == [{"name": "read", "success": True}]
        orchestrator.execute_tool_calls.assert_awaited_once_with(
            [{"name": "read", "arguments": {}}]
        )

    def test_orchestrator_tool_strategy_event_prefers_metrics_service(self):
        orchestrator = object.__new__(AgentOrchestrator)
        orchestrator._metrics_coordinator = MagicMock()
        orchestrator.model = "test-model"
        orchestrator._is_tool_strategy_v2_enabled = MagicMock(return_value=True)

        provider = MagicMock()
        provider.name = "openai"
        tools = [SimpleNamespace(name="read"), SimpleNamespace(name="git_diff")]

        AgentOrchestrator._emit_tool_strategy_event(
            orchestrator,
            strategy="semantic_selection",
            tool_count=2,
            tool_tokens=128,
            context_window=32768,
            provider=provider,
            reason="small_context_window",
            tools=tools,
        )

        orchestrator._metrics_coordinator.emit_tool_strategy_event.assert_called_once_with(
            strategy="semantic_selection",
            tool_count=2,
            tool_tokens=128,
            context_window=32768,
            provider=provider,
            model="test-model",
            reason="small_context_window",
            tools=tools,
            v2_enabled=True,
        )


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
        obj._deprecated_provider_coordinator = MagicMock()
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
        obj._factory = MagicMock()
        obj._factory.create_service_streaming_runtime.return_value = MagicMock()
        obj._turn_executor = None
        obj._protocol_adapter = None
        obj.observability = MagicMock()

        class TrapChatCoordinator:
            touched = False

            def __getattr__(self, name):
                self.touched = True
                raise AssertionError(
                    f"chat coordinator should remain lazy during bootstrap: {name}"
                )

        trap_chat = TrapChatCoordinator()
        obj._deprecated_chat_coordinator = LazyRuntimeProxy(
            factory=lambda: trap_chat,
            name="chat_coordinator",
        )

        class TrapToolCoordinator:
            touched = False

            def __getattr__(self, name):
                self.touched = True
                raise AssertionError(
                    f"tool coordinator should remain lazy during bootstrap: {name}"
                )

        trap_tool = TrapToolCoordinator()
        obj._deprecated_tool_coordinator = LazyRuntimeProxy(
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
        assert obj._deprecated_tool_coordinator.initialized is False
        assert trap_tool.touched is False

        chat_service.bind_runtime_components.assert_called_once()
        kwargs = chat_service.bind_runtime_components.call_args.kwargs
        assert kwargs["turn_executor"].initialized is False
        assert callable(kwargs["planning_handler"])
        assert (
            kwargs["stream_chat_handler"]
            is obj._factory.create_service_streaming_runtime.return_value.stream_chat
        )
        assert callable(kwargs["context_limit_handler"])
        obj._factory.create_service_streaming_runtime.assert_called_once_with(obj)
        assert obj._deprecated_chat_coordinator.initialized is False
        assert trap_chat.touched is False

        recovery_service.bind_runtime_components.assert_called_once()
        recovery_kwargs = recovery_service.bind_runtime_components.call_args.kwargs
        assert recovery_kwargs["streaming_handler"] is obj._streaming_handler
        assert recovery_kwargs["context_compactor"] is obj._context_compactor
        assert recovery_kwargs["unified_tracker"] is obj.unified_tracker
        assert recovery_kwargs["settings"] is obj.settings
        assert recovery_kwargs["presentation"] is obj._presentation

    @pytest.mark.asyncio
    async def test_chat_service_runtime_handlers_use_service_first_helpers(self):
        from victor.agent.orchestrator import AgentOrchestrator

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
        obj._deprecated_provider_coordinator = MagicMock()
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
        obj._presentation.icon.return_value = "!"
        obj.tools = MagicMock()
        obj.mode_controller = MagicMock()
        obj.argument_normalizer = MagicMock()
        obj._tool_pipeline = MagicMock()
        obj.tool_cache = MagicMock()
        obj.memory_manager = MagicMock()
        obj._memory_session_id = "mem-1"
        obj.tool_budget = 100
        obj._factory = MagicMock()
        obj._factory.create_service_streaming_runtime.return_value = MagicMock()
        obj._turn_executor = None
        obj._protocol_adapter = None
        obj.observability = MagicMock()
        obj._system_added = False
        obj._task_analyzer = MagicMock()
        obj._task_analyzer.analyze.return_value = "task-analysis"
        obj.conversation = MagicMock()
        obj.conversation.messages = [Message(role="user", content="existing")]
        obj.add_message = MagicMock()
        obj._check_context_overflow = MagicMock(return_value=False)
        obj._record_intelligent_outcome = MagicMock()

        class TrapChatCoordinator:
            touched = False

            def __getattr__(self, name):
                self.touched = True
                raise AssertionError(f"chat coordinator should remain lazy during runtime: {name}")

        trap_chat = TrapChatCoordinator()

        from victor.agent.runtime.provider_runtime import LazyRuntimeProxy

        obj._deprecated_chat_coordinator = LazyRuntimeProxy(
            factory=lambda: trap_chat,
            name="chat_coordinator",
        )

        planning_response = CompletionResponse(content="planned", role="assistant")

        with (
            patch.object(AgentOrchestrator, "_register_coordinators_for_services"),
            patch.object(AgentOrchestrator, "_bootstrap_service_layer"),
            patch("victor.agent.services.planning_runtime.PlanningCoordinator") as planning_cls,
        ):
            planning_instance = MagicMock()
            planning_instance.chat_with_planning = AsyncMock(return_value=planning_response)
            planning_cls.return_value = planning_instance

            obj._initialize_services()

            kwargs = chat_service.bind_runtime_components.call_args.kwargs
            planning_handler = kwargs["planning_handler"]
            context_limit_handler = kwargs["context_limit_handler"]

            response = await planning_handler("plan this")
            handled, chunk = await context_limit_handler("plan this", 5, 1000, 1, 0.8)

        assert response is planning_response
        assert handled is False
        assert chunk is None
        planning_instance.chat_with_planning.assert_awaited_once_with(
            "plan this",
            task_analysis="task-analysis",
        )
        obj.conversation.ensure_system_prompt.assert_called_once()
        obj.add_message.assert_any_call("user", "plan this")
        obj.add_message.assert_any_call("assistant", "planned")
        obj._factory.create_service_streaming_runtime.assert_called_once_with(obj)
        assert obj._deprecated_chat_coordinator.initialized is False
        assert trap_chat.touched is False

    @pytest.mark.asyncio
    async def test_planning_handler_skips_duplicate_turn_recording_after_direct_chat_fallback(self):
        from victor.agent.orchestrator import AgentOrchestrator

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
        obj._deprecated_provider_coordinator = MagicMock()
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
        obj._presentation.icon.return_value = "!"
        obj.tools = MagicMock()
        obj.mode_controller = MagicMock()
        obj.argument_normalizer = MagicMock()
        obj._tool_pipeline = MagicMock()
        obj.tool_cache = MagicMock()
        obj.memory_manager = MagicMock()
        obj._memory_session_id = "mem-1"
        obj.tool_budget = 100
        obj._factory = MagicMock()
        obj._factory.create_service_streaming_runtime.return_value = MagicMock()
        obj._turn_executor = None
        obj._protocol_adapter = None
        obj.observability = MagicMock()
        obj._system_added = False
        obj._task_analyzer = MagicMock()
        obj._task_analyzer.analyze.return_value = "task-analysis"
        obj.conversation = MagicMock()
        obj.conversation.messages = [Message(role="system", content="existing")]
        obj.add_message = MagicMock()
        obj._check_context_overflow = MagicMock(return_value=False)
        obj._record_intelligent_outcome = MagicMock()

        direct_response = CompletionResponse(content="already recorded", role="assistant")

        async def _direct_chat(user_message: str) -> CompletionResponse:
            obj.add_message("user", user_message)
            obj.add_message("assistant", direct_response.content)
            obj.conversation.messages.extend(
                [
                    Message(role="user", content=user_message),
                    Message(role="assistant", content=direct_response.content),
                ]
            )
            return direct_response

        async def _direct_chat_with_planning(
            user_message: str,
            task_analysis=None,
        ) -> CompletionResponse:
            assert task_analysis == "task-analysis"
            return await _direct_chat(user_message)

        obj.chat = AsyncMock(side_effect=_direct_chat)

        with (
            patch.object(AgentOrchestrator, "_register_coordinators_for_services"),
            patch.object(AgentOrchestrator, "_bootstrap_service_layer"),
            patch("victor.agent.services.planning_runtime.PlanningCoordinator") as planning_cls,
        ):
            planning_instance = MagicMock()
            planning_instance.chat_with_planning = AsyncMock(side_effect=_direct_chat_with_planning)
            planning_cls.return_value = planning_instance

            obj._initialize_services()

            planning_handler = chat_service.bind_runtime_components.call_args.kwargs["planning_handler"]
            response = await planning_handler("plan this")

        assert response is direct_response
        planning_instance.chat_with_planning.assert_awaited_once_with(
            "plan this",
            task_analysis="task-analysis",
        )
        assert obj.add_message.call_args_list == [
            (("user", "plan this"), {}),
            (("assistant", "already recorded"), {}),
        ]
        obj.conversation.ensure_system_prompt.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_service_stream_handler_uses_service_runtime(self):
        from victor.agent.orchestrator import AgentOrchestrator

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
        obj._deprecated_provider_coordinator = MagicMock()
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
        obj._factory = MagicMock()
        obj._turn_executor = None
        obj._protocol_adapter = None
        obj.observability = MagicMock()

        class TrapChatCoordinator:
            touched = False

            def __getattr__(self, name):
                self.touched = True
                raise AssertionError(
                    f"chat coordinator should remain lazy during streaming: {name}"
                )

        trap_chat = TrapChatCoordinator()

        from victor.agent.runtime.provider_runtime import LazyRuntimeProxy

        obj._deprecated_chat_coordinator = LazyRuntimeProxy(
            factory=lambda: trap_chat,
            name="chat_coordinator",
        )

        stream_chunk = StreamChunk(content="service-stream", is_final=True)

        async def _runtime_stream_chat(user_message: str, **kwargs):
            assert user_message == "hello"
            assert kwargs == {"_preserve_iteration": True}
            yield stream_chunk

        runtime = MagicMock()
        runtime.stream_chat = _runtime_stream_chat
        obj._factory.create_service_streaming_runtime.return_value = runtime

        with (
            patch.object(AgentOrchestrator, "_register_coordinators_for_services"),
            patch.object(AgentOrchestrator, "_bootstrap_service_layer"),
        ):
            obj._initialize_services()

            kwargs = chat_service.bind_runtime_components.call_args.kwargs
            stream_chat_handler = kwargs["stream_chat_handler"]
            chunks = [c async for c in stream_chat_handler("hello", _preserve_iteration=True)]

        assert chunks == [stream_chunk]
        obj._factory.create_service_streaming_runtime.assert_called_once_with(obj)
        assert obj._deprecated_chat_coordinator.initialized is False
        assert trap_chat.touched is False

    @pytest.mark.asyncio
    async def test_orchestrator_stream_chat_uses_service_runtime_pipeline(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        obj._apply_skill_for_turn = MagicMock()
        obj._turn_executor = None
        obj.conversation = MagicMock()
        obj.conversation.messages = []

        runtime = MagicMock()
        pipeline = object()
        runtime.get_pipeline.return_value = pipeline
        obj._get_service_streaming_runtime = MagicMock(return_value=runtime)

        chunk = StreamChunk(content="loop", is_final=True)

        loop_instance = MagicMock()

        async def _loop_stream_chat(user_message: str, streaming_pipeline=None):
            assert user_message == "hello"
            assert streaming_pipeline is pipeline
            yield chunk

        loop_instance.stream_chat = _loop_stream_chat

        flag_manager = MagicMock()
        flag_manager.is_enabled.return_value = True

        with (
            patch("victor.core.feature_flags.get_feature_flag_manager", return_value=flag_manager),
            patch("victor.framework.agentic_loop.AgenticLoop", return_value=loop_instance),
        ):
            chunks = [c async for c in obj.stream_chat("hello")]

        assert chunks == [chunk]
        obj._apply_skill_for_turn.assert_called_once_with("hello")
        obj._get_service_streaming_runtime.assert_called_once_with()
        runtime.get_pipeline.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_orchestrator_stream_chat_executes_tools_without_chatservice_fallback(self):
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.agent.streaming.context import StreamingChatContext
        from victor.agent.streaming.tool_execution import create_tool_execution_handler

        obj = object.__new__(AgentOrchestrator)
        obj._apply_skill_for_turn = MagicMock()
        obj._turn_executor = None
        obj.conversation = MagicMock()
        obj.conversation.messages = []
        obj.create_recovery_context = lambda stream_ctx: {"stream_ctx": stream_ctx}
        obj._recovery_service = None
        obj._recovery_coordinator = SimpleNamespace(
            check_tool_budget=lambda *_args, **_kwargs: None,
            truncate_tool_calls=lambda _ctx, tool_calls, remaining: (tool_calls, remaining),
            filter_blocked_tool_calls=lambda _ctx, tool_calls: (tool_calls, [], 0),
            check_blocked_threshold=lambda _ctx, _all_blocked: None,
        )
        obj._chunk_generator = SimpleNamespace(
            generate_tool_start_chunk=lambda *_args, **_kwargs: StreamChunk(content="start"),
            generate_tool_result_chunks=lambda _tool_result: [StreamChunk(content="done")],
        )
        obj.reminder_manager = SimpleNamespace(
            update_state=MagicMock(),
            get_consolidated_reminder=lambda: None,
        )
        obj.unified_tracker = SimpleNamespace(unique_resources=set())
        obj.settings = SimpleNamespace(tool_call_budget_warning_threshold=250)
        obj._check_progress_with_handler = lambda _stream_ctx: None
        obj._handle_force_completion_with_handler = lambda _stream_ctx: None

        async def _noop_async_generator(_stream_ctx):
            if False:
                yield None

        obj._handle_budget_exhausted = _noop_async_generator
        obj._handle_force_final_response = _noop_async_generator
        obj.execute_tool_calls = AsyncMock(
            return_value=[
                {
                    "name": "read",
                    "success": True,
                    "args": {"path": "victor/agent/orchestrator.py"},
                    "elapsed": 1.0,
                }
            ]
        )
        obj._session_accessor = SimpleNamespace(observed_files=set())

        fallback_calls = []

        async def _fallback_stream_chat(user_message: str, **kwargs):
            fallback_calls.append((user_message, kwargs))
            yield StreamChunk(content="fallback", is_final=True)

        obj._chat_service = SimpleNamespace(stream_chat=_fallback_stream_chat)

        runtime = MagicMock()
        runtime.get_pipeline.return_value = SimpleNamespace(
            _tool_execution_handler=create_tool_execution_handler(obj)
        )
        obj._get_service_streaming_runtime = MagicMock(return_value=runtime)

        loop_instance = MagicMock()

        async def _loop_stream_chat(user_message: str, streaming_pipeline=None):
            result = await streaming_pipeline._tool_execution_handler.execute_tools(
                stream_ctx=StreamingChatContext(user_message=user_message),
                tool_calls=[
                    {
                        "name": "read",
                        "arguments": {"path": "victor/agent/orchestrator.py"},
                    }
                ],
                user_message=user_message,
                full_content="read the file",
                tool_calls_used=0,
                tool_budget=5,
            )
            for chunk in result.chunks:
                yield chunk

        loop_instance.stream_chat = _loop_stream_chat

        flag_manager = MagicMock()
        flag_manager.is_enabled.return_value = True

        with (
            patch("victor.core.feature_flags.get_feature_flag_manager", return_value=flag_manager),
            patch("victor.framework.agentic_loop.AgenticLoop", return_value=loop_instance),
        ):
            chunks = [c async for c in obj.stream_chat("hello")]

        assert [chunk.content for chunk in chunks] == ["start", "done"]
        obj.execute_tool_calls.assert_awaited_once_with(
            [{"name": "read", "arguments": {"path": "victor/agent/orchestrator.py"}}]
        )
        assert fallback_calls == []

    @pytest.mark.asyncio
    async def test_orchestrator_stream_chat_fallback_preserves_current_iteration(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        obj._apply_skill_for_turn = MagicMock()
        obj._turn_executor = None
        obj.conversation = MagicMock()
        obj.conversation.messages = [Message(role="system", content="existing")]
        obj._current_stream_context = SimpleNamespace(total_iterations=3)

        fallback_calls = []

        async def _fallback_stream_chat(user_message: str, **kwargs):
            fallback_calls.append((user_message, kwargs))
            yield StreamChunk(content="fallback", is_final=True)

        obj._chat_service = SimpleNamespace(stream_chat=_fallback_stream_chat)

        runtime = MagicMock()
        runtime.get_pipeline.return_value = object()
        obj._get_service_streaming_runtime = MagicMock(return_value=runtime)

        loop_instance = MagicMock()

        async def _loop_stream_chat(user_message: str, streaming_pipeline=None):
            assert user_message == "hello"
            assert streaming_pipeline is runtime.get_pipeline.return_value
            raise RuntimeError("stream failed mid-iteration")
            yield  # pragma: no cover

        loop_instance.stream_chat = _loop_stream_chat

        flag_manager = MagicMock()
        flag_manager.is_enabled.return_value = True

        with (
            patch("victor.core.feature_flags.get_feature_flag_manager", return_value=flag_manager),
            patch("victor.framework.agentic_loop.AgenticLoop", return_value=loop_instance),
        ):
            chunks = [c async for c in obj.stream_chat("hello")]

        assert [chunk.content for chunk in chunks] == ["fallback"]
        assert fallback_calls == [
            ("hello", {"_preserve_iteration": True, "_current_iteration": 3})
        ]

    @pytest.mark.asyncio
    async def test_orchestrator_chat_agentic_loop_records_single_user_and_assistant_turn(self):
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.agent.services.orchestrator_protocol_adapter import OrchestratorProtocolAdapter

        obj = object.__new__(AgentOrchestrator)
        obj._chat_service = SimpleNamespace(
            chat=AsyncMock(side_effect=AssertionError("chat service fallback should not be used"))
        )
        obj._system_added = False

        class FakeConversation:
            def __init__(self):
                self.messages = []

            def ensure_system_prompt(self):
                if not self.messages or self.messages[0].role != "system":
                    self.messages.insert(0, Message(role="system", content="system prompt"))

        conversation = FakeConversation()
        obj.conversation = conversation

        def _record_message(role: str, content: str) -> None:
            conversation.messages.append(Message(role=role, content=content))

        obj.add_message = MagicMock(side_effect=_record_message)

        response = CompletionResponse(content="done", role="assistant")
        chat_context = OrchestratorProtocolAdapter(obj)
        tool_context = SimpleNamespace(tool_calls_used=99)
        provider_context = SimpleNamespace(
            task_classifier=SimpleNamespace(classify=MagicMock(return_value="task-analysis"))
        )

        async def _execute_turn(**kwargs):
            assert kwargs == {
                "user_message": "hello",
                "task_classification": "task-analysis",
                "is_qa_task": False,
                "intent": None,
                "temperature_override": None,
            }
            chat_context.add_message("assistant", response.content)
            return SimpleNamespace(response=response)

        turn_executor = SimpleNamespace(
            _chat_context=chat_context,
            _tool_context=tool_context,
            _provider_context=provider_context,
            _is_question_only=MagicMock(return_value=False),
            _run_parallel_exploration=AsyncMock(),
            execute_turn=AsyncMock(side_effect=_execute_turn),
            _ensure_complete_response=AsyncMock(
                side_effect=AssertionError("completion fallback should not be used")
            ),
        )
        obj._turn_executor = turn_executor

        loop_instance = MagicMock()

        async def _loop_run(query: str, context=None):
            assert query == "hello"
            assert context == {
                "_task_classification": "task-analysis",
                "_is_qa_task": False,
            }
            action_result = await turn_executor.execute_turn(
                user_message=query,
                task_classification=context["_task_classification"],
                is_qa_task=context["_is_qa_task"],
                intent=None,
                temperature_override=None,
            )
            return SimpleNamespace(iterations=[SimpleNamespace(action_result=action_result)])

        loop_instance.run = AsyncMock(side_effect=_loop_run)

        flag_manager = MagicMock()
        flag_manager.is_enabled.return_value = True

        with (
            patch("victor.core.feature_flags.get_feature_flag_manager", return_value=flag_manager),
            patch("victor.framework.agentic_loop.AgenticLoop", return_value=loop_instance),
        ):
            result = await obj.chat("hello")

        assert result is response
        obj._chat_service.chat.assert_not_awaited()
        turn_executor._run_parallel_exploration.assert_awaited_once_with(
            "hello",
            "task-analysis",
        )
        assert [message.role for message in conversation.messages] == [
            "system",
            "user",
            "assistant",
        ]
        assert [message.content for message in conversation.messages] == [
            "system prompt",
            "hello",
            "done",
        ]
        assert obj.add_message.call_args_list == [
            (("user", "hello"), {}),
            (("assistant", "done"), {}),
        ]

    def test_get_service_streaming_runtime_prefers_factory_and_caches(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        runtime = MagicMock(name="service_streaming_runtime")
        obj._factory = MagicMock()
        obj._factory.create_service_streaming_runtime.return_value = runtime

        first = obj._get_service_streaming_runtime()
        second = obj._get_service_streaming_runtime()

        assert first is runtime
        assert second is runtime
        obj._factory.create_service_streaming_runtime.assert_called_once_with(obj)
