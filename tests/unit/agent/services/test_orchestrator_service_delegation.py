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

    def test_import_context_adapter(self):
        from victor.agent.services.adapters.context_adapter import ContextServiceAdapter

        assert ContextServiceAdapter is not None

    def test_import_all_from_package(self):
        from victor.agent.services.adapters import ContextServiceAdapter

        assert ContextServiceAdapter is not None


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

    def test_context_adapter_has_required_methods(self):
        from victor.agent.services.adapters.context_adapter import ContextServiceAdapter

        adapter = ContextServiceAdapter(MagicMock())
        assert callable(getattr(adapter, "get_context_metrics", None))
        assert callable(getattr(adapter, "check_context_overflow", None))
        assert callable(getattr(adapter, "compact_context", None))
        assert callable(getattr(adapter, "add_message", None))
        assert callable(getattr(adapter, "add_messages", None))
        assert callable(getattr(adapter, "get_messages", None))
        assert callable(getattr(adapter, "clear_messages", None))
        assert callable(getattr(adapter, "get_max_tokens", None))
        assert callable(getattr(adapter, "set_max_tokens", None))
        assert callable(getattr(adapter, "estimate_tokens", None))
        assert callable(getattr(adapter, "is_healthy", None))

    @pytest.mark.asyncio
    async def test_context_adapter_supports_controller_backed_chat_runtime_operations(self):
        from victor.agent.conversation.controller import ConversationController
        from victor.agent.services.adapters.context_adapter import ContextServiceAdapter

        controller = ConversationController()
        controller.set_system_prompt("system prompt")
        adapter = ContextServiceAdapter(controller)

        adapter.add_message("user", "hello")

        assert [message.role for message in controller.messages] == ["system", "user"]
        assert await adapter.check_context_overflow() is False

        adapter.clear_messages(retain_system=True)

        assert [message.role for message in controller.messages] == ["system"]
        assert controller.messages[0].content == "system prompt"

    def test_bootstrap_new_services_registers_context_adapter_backed_by_controller(self):
        from victor.agent.conversation.controller import ConversationController
        from victor.agent.services.adapters.context_adapter import ContextServiceAdapter
        from victor.agent.services.protocols import ContextServiceProtocol
        from victor.core.bootstrap_services import bootstrap_new_services
        from victor.core.container import ServiceContainer

        container = ServiceContainer()
        controller = ConversationController()
        controller.set_system_prompt("system prompt")
        streaming = MagicMock()
        flag_manager = MagicMock()
        flag_manager.is_enabled.return_value = False

        with patch("victor.core.feature_flags.get_feature_flag_manager", return_value=flag_manager):
            bootstrap_new_services(
                container,
                conversation_controller=controller,
                streaming_coordinator=streaming,
            )

        context_service = container.get_optional(ContextServiceProtocol)

        assert isinstance(context_service, ContextServiceAdapter)
        context_service.add_message("user", "hello")
        assert [message.role for message in controller.messages] == ["system", "user"]

    def test_protocol_adapter_runtime_state_setters_proxy_to_host_orchestrator(self):
        from victor.agent.services.orchestrator_protocol_adapter import OrchestratorProtocolAdapter

        orchestrator = SimpleNamespace(temperature=0.2, tool_budget=5)
        adapter = OrchestratorProtocolAdapter(orchestrator)

        adapter.temperature = 0.6
        adapter.tool_budget = 12

        assert orchestrator.temperature == 0.6
        assert orchestrator.tool_budget == 12

    @pytest.mark.asyncio
    async def test_orchestrator_protocol_adapter_prefers_execute_tool_calls(self):
        from victor.agent.services.orchestrator_protocol_adapter import OrchestratorProtocolAdapter

        orchestrator = MagicMock()
        orchestrator.execute_tool_calls = AsyncMock(
            return_value=[{"name": "read", "success": True}]
        )
        orchestrator._handle_tool_calls = AsyncMock(
            side_effect=AssertionError("legacy _handle_tool_calls bridge should not be used")
        )

        adapter = OrchestratorProtocolAdapter(orchestrator)
        result = await adapter.execute_tool_calls([{"name": "read", "arguments": {}}])

        assert result == [{"name": "read", "success": True}]
        orchestrator.execute_tool_calls.assert_awaited_once_with(
            [{"name": "read", "arguments": {}}]
        )

    @pytest.mark.asyncio
    async def test_orchestrator_protocol_adapter_exposes_sync_chat_runtime_surface(self):
        from victor.agent.services.orchestrator_protocol_adapter import OrchestratorProtocolAdapter
        from victor.agent.services.protocols.chat_runtime import SyncChatRuntimeProtocol

        response = CompletionResponse(content="adapter-chat", role="assistant")
        orchestrator = MagicMock()
        orchestrator.chat = AsyncMock(return_value=response)
        orchestrator.get_last_skill_match_info.return_value = {"skill_match": "python"}

        adapter = OrchestratorProtocolAdapter(orchestrator)
        result = await adapter.chat("hello", use_planning=True)

        assert isinstance(adapter, SyncChatRuntimeProtocol)
        assert result is response
        assert adapter.get_last_skill_match_info() == {"skill_match": "python"}
        orchestrator.chat.assert_awaited_once_with("hello", use_planning=True)
        orchestrator.get_last_skill_match_info.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_orchestrator_protocol_adapter_exposes_chat_compat_runtime_surface(self):
        from victor.agent.services.orchestrator_protocol_adapter import OrchestratorProtocolAdapter
        from victor.agent.services.protocols.chat_runtime import ChatCompatRuntimeProtocol

        stream_chunk = StreamChunk(content="stream", is_final=True)
        orchestrator = MagicMock()

        async def _stream_chat(user_message: str, **kwargs):
            assert user_message == "hello"
            assert kwargs == {"mode": "compat"}
            yield stream_chunk

        orchestrator.stream_chat = _stream_chat
        orchestrator._handle_context_and_iteration_limits_runtime = AsyncMock(
            return_value=(True, stream_chunk)
        )

        adapter = OrchestratorProtocolAdapter(orchestrator)
        chunks = [chunk async for chunk in adapter.stream_chat("hello", mode="compat")]
        handled, result_chunk = await adapter._handle_context_and_iteration_limits_runtime(
            "hello",
            5,
            1000,
            1,
            0.8,
        )

        assert isinstance(adapter, ChatCompatRuntimeProtocol)
        assert chunks == [stream_chunk]
        assert handled is True
        assert result_chunk is stream_chunk
        orchestrator._handle_context_and_iteration_limits_runtime.assert_awaited_once_with(
            "hello",
            5,
            1000,
            1,
            0.8,
        )

    @pytest.mark.asyncio
    async def test_orchestrator_protocol_adapter_exposes_service_chat_runtime_handlers(self):
        from victor.agent.services.orchestrator_protocol_adapter import OrchestratorProtocolAdapter

        response = CompletionResponse(content="planned", role="assistant")
        orchestrator = MagicMock()
        orchestrator._run_planning_chat_runtime = AsyncMock(return_value=response)
        orchestrator._handle_context_and_iteration_limits_runtime = AsyncMock(
            return_value=(False, None)
        )

        adapter = OrchestratorProtocolAdapter(orchestrator)
        planning_response = await adapter._run_planning_chat_runtime("plan this")
        handled, chunk = await adapter._handle_context_and_iteration_limits_runtime(
            "plan this",
            5,
            1000,
            1,
            0.8,
        )

        assert planning_response is response
        assert handled is False
        assert chunk is None
        orchestrator._run_planning_chat_runtime.assert_awaited_once_with("plan this")
        orchestrator._handle_context_and_iteration_limits_runtime.assert_awaited_once_with(
            "plan this",
            5,
            1000,
            1,
            0.8,
        )

    def test_orchestrator_protocol_adapter_exposes_planning_runtime_surface(self):
        from victor.agent.services.orchestrator_protocol_adapter import OrchestratorProtocolAdapter

        provider = MagicMock(name="provider")
        profile = MagicMock(name="profile")
        orchestrator = MagicMock()
        orchestrator.provider = provider
        orchestrator.model = "gpt-test"
        orchestrator.max_tokens = 2048
        orchestrator.profile = profile
        orchestrator._planning_model_override = "openai:o4-mini"
        orchestrator._system_prompt_override = "original system prompt"

        adapter = OrchestratorProtocolAdapter(orchestrator)
        adapter.set_system_prompt("new system prompt")

        assert adapter.provider is provider
        assert adapter.model == "gpt-test"
        assert adapter.max_tokens == 2048
        assert adapter.profile is profile
        assert adapter.planning_model_override == "openai:o4-mini"
        assert adapter._system_prompt_override == "original system prompt"
        orchestrator.set_system_prompt.assert_called_once_with("new system prompt")

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
        assert kwargs["planning_handler"].__self__ is obj._protocol_adapter
        assert (
            kwargs["stream_chat_handler"]
            is obj._factory.create_service_streaming_runtime.return_value.stream_chat
        )
        assert callable(kwargs["context_limit_handler"])
        assert kwargs["context_limit_handler"].__self__ is obj._protocol_adapter
        obj._factory.create_service_streaming_runtime.assert_called_once_with(obj._protocol_adapter)
        assert obj._deprecated_chat_coordinator.initialized is False
        assert trap_chat.touched is False

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
        obj._factory.create_service_streaming_runtime.assert_called_once_with(obj._protocol_adapter)
        assert obj._deprecated_chat_coordinator.initialized is False
        assert trap_chat.touched is False

    @pytest.mark.asyncio
    async def test_run_planning_chat_runtime_constructs_planning_coordinator_with_protocol_adapter(
        self,
    ):
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.agent.services.orchestrator_protocol_adapter import OrchestratorProtocolAdapter

        obj = object.__new__(AgentOrchestrator)
        planning_response = CompletionResponse(content="planned", role="assistant")
        planning_instance = MagicMock(name="planning_coordinator")
        planning_instance.chat_with_planning = AsyncMock(return_value=planning_response)

        protocol_adapter = OrchestratorProtocolAdapter(obj)
        obj._protocol_adapter = protocol_adapter
        obj._service_planning_coordinator = None
        obj._task_analyzer = MagicMock()
        obj._task_analyzer.analyze.return_value = "task-analysis"
        obj._get_conversation_message_count = MagicMock(side_effect=[0, 1])

        with patch("victor.agent.services.planning_runtime.PlanningCoordinator") as planning_cls:
            planning_cls.return_value = planning_instance

            response = await AgentOrchestrator._run_planning_chat_runtime(obj, "plan this")

        assert response is planning_response
        assert obj._service_planning_coordinator is planning_instance
        planning_cls.assert_called_once_with(protocol_adapter)
        planning_instance.chat_with_planning.assert_awaited_once_with(
            "plan this",
            task_analysis="task-analysis",
        )

    def test_get_planning_chat_runtime_prefers_cached_helper_and_protocol_adapter(self):
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.agent.services.planning_chat_runtime import PlanningChatRuntime

        obj = object.__new__(AgentOrchestrator)
        adapter = MagicMock(name="protocol_adapter")
        obj._protocol_adapter = adapter

        first = obj._get_planning_chat_runtime()
        second = obj._get_planning_chat_runtime()

        assert isinstance(first, PlanningChatRuntime)
        assert first is second
        assert first._runtime is adapter

    @pytest.mark.asyncio
    async def test_run_planning_chat_runtime_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        response = CompletionResponse(content="planned", role="assistant")
        helper.run = AsyncMock(return_value=response)
        obj._planning_chat_runtime = helper

        result = await AgentOrchestrator._run_planning_chat_runtime(obj, "plan this")

        assert result is response
        helper.run.assert_awaited_once_with("plan this")

    def test_get_task_guidance_runtime_prefers_cached_helper_and_protocol_adapter(self):
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.agent.services.task_guidance_runtime import TaskGuidanceRuntime

        obj = object.__new__(AgentOrchestrator)
        adapter = MagicMock(name="protocol_adapter")
        obj._protocol_adapter = adapter

        first = obj._get_task_guidance_runtime()
        second = obj._get_task_guidance_runtime()

        assert isinstance(first, TaskGuidanceRuntime)
        assert first is second
        assert first._runtime is adapter

    def test_prepare_task_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        helper.prepare_task.return_value = ("classification", 7)
        obj._task_guidance_runtime = helper

        result = AgentOrchestrator._prepare_task(obj, "plan this", "analysis")

        assert result == ("classification", 7)
        helper.prepare_task.assert_called_once_with("plan this", "analysis")

    def test_apply_intent_guard_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        obj._task_guidance_runtime = helper

        AgentOrchestrator._apply_intent_guard(obj, "read this file")

        helper.apply_intent_guard.assert_called_once_with("read this file")

    def test_apply_task_guidance_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        obj._task_guidance_runtime = helper

        AgentOrchestrator._apply_task_guidance(
            obj,
            user_message="analyze architecture",
            unified_task_type="analysis",
            is_analysis_task=True,
            is_action_task=False,
            needs_execution=False,
            max_exploration_iterations=8,
        )

        helper.apply_task_guidance.assert_called_once_with(
            user_message="analyze architecture",
            unified_task_type="analysis",
            is_analysis_task=True,
            is_action_task=False,
            needs_execution=False,
            max_exploration_iterations=8,
        )

    def test_get_context_limit_runtime_prefers_cached_helper_and_protocol_adapter(self):
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.agent.services.context_limit_runtime import ContextLimitRuntime

        obj = object.__new__(AgentOrchestrator)
        adapter = MagicMock(name="protocol_adapter")
        obj._protocol_adapter = adapter

        first = obj._get_context_limit_runtime()
        second = obj._get_context_limit_runtime()

        assert isinstance(first, ContextLimitRuntime)
        assert first is second
        assert first._runtime is adapter

    @pytest.mark.asyncio
    async def test_handle_context_and_iteration_limits_runtime_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        helper.handle_limits = AsyncMock(return_value=(False, None))
        obj._context_limit_runtime = helper
        obj._check_context_overflow = MagicMock(return_value=False)

        result = await AgentOrchestrator._handle_context_and_iteration_limits_runtime(
            obj,
            "plan this",
            5,
            1000,
            1,
            0.8,
        )

        assert result == (False, None)
        helper.handle_limits.assert_awaited_once_with("plan this", 5, 1000, 1, 0.8)

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

            planning_handler = chat_service.bind_runtime_components.call_args.kwargs[
                "planning_handler"
            ]
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
        obj._factory.create_service_streaming_runtime.assert_called_once_with(obj._protocol_adapter)
        assert obj._deprecated_chat_coordinator.initialized is False
        assert trap_chat.touched is False

    @pytest.mark.asyncio
    async def test_orchestrator_stream_chat_delegates_to_chat_service(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        obj._apply_skill_for_turn = MagicMock()
        chunk = StreamChunk(content="service", is_final=True)

        async def _stream_chat(user_message: str, **kwargs):
            assert user_message == "hello"
            assert kwargs == {}
            yield chunk

        obj._chat_service = SimpleNamespace(stream_chat=_stream_chat)

        chunks = [c async for c in obj.stream_chat("hello")]

        assert chunks == [chunk]
        obj._apply_skill_for_turn.assert_called_once_with("hello")

    @pytest.mark.asyncio
    async def test_orchestrator_stream_chat_does_not_inject_legacy_loop_kwargs(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        obj._apply_skill_for_turn = MagicMock()
        calls = []

        async def _stream_chat(user_message: str, **kwargs):
            calls.append((user_message, kwargs))
            yield StreamChunk(content="service", is_final=True)

        obj._chat_service = SimpleNamespace(stream_chat=_stream_chat)

        chunks = [c async for c in obj.stream_chat("hello")]

        assert [chunk.content for chunk in chunks] == ["service"]
        assert calls == [("hello", {})]

    @pytest.mark.asyncio
    async def test_orchestrator_stream_chat_propagates_chat_service_errors(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        obj._apply_skill_for_turn = MagicMock()

        async def _stream_chat(_user_message: str, **_kwargs):
            raise RuntimeError("stream failed")
            yield  # pragma: no cover

        obj._chat_service = SimpleNamespace(stream_chat=_stream_chat)

        with pytest.raises(RuntimeError, match="stream failed"):
            _ = [c async for c in obj.stream_chat("hello")]

    @pytest.mark.asyncio
    async def test_orchestrator_chat_delegates_to_chat_service(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        response = CompletionResponse(content="done", role="assistant")
        obj._chat_service = SimpleNamespace(chat=AsyncMock(return_value=response))

        result = await obj.chat("hello")

        assert result is response
        obj._chat_service.chat.assert_awaited_once_with("hello", use_planning=False)

    @pytest.mark.asyncio
    async def test_orchestrator_chat_passes_use_planning_through_service(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        response = CompletionResponse(content="planned", role="assistant")
        obj._chat_service = SimpleNamespace(chat=AsyncMock(return_value=response))

        result = await obj.chat("hello", use_planning=True)

        assert result is response
        obj._chat_service.chat.assert_awaited_once_with("hello", use_planning=True)

    @pytest.mark.asyncio
    async def test_orchestrator_chat_propagates_chat_service_errors(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        obj._chat_service = SimpleNamespace(chat=AsyncMock(side_effect=RuntimeError("chat failed")))

        with pytest.raises(RuntimeError, match="chat failed"):
            await obj.chat("hello")

    def test_get_service_streaming_runtime_prefers_factory_and_caches(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        adapter = MagicMock(name="protocol_adapter")
        runtime = MagicMock(name="service_streaming_runtime")
        obj._protocol_adapter = adapter
        obj._factory = MagicMock()
        obj._factory.create_service_streaming_runtime.return_value = runtime

        first = obj._get_service_streaming_runtime()
        second = obj._get_service_streaming_runtime()

        assert first is runtime
        assert second is runtime
        obj._factory.create_service_streaming_runtime.assert_called_once_with(adapter)

    def test_orchestrator_protocol_adapter_proxies_streaming_runtime_host_state(self):
        from types import SimpleNamespace

        from victor.agent.services.orchestrator_protocol_adapter import OrchestratorProtocolAdapter

        orchestrator = SimpleNamespace(
            _current_stream_context="ctx",
            _container=MagicMock(name="container"),
        )
        adapter = OrchestratorProtocolAdapter(orchestrator)

        assert adapter._current_stream_context == "ctx"
        assert adapter._container is orchestrator._container

        adapter._runtime_intelligence = "runtime-intelligence"
        assert orchestrator._runtime_intelligence == "runtime-intelligence"

        del adapter._current_stream_context
        assert not hasattr(orchestrator, "_current_stream_context")
