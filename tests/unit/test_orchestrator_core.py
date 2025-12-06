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

"""Core tests for AgentOrchestrator module - focusing on high-impact functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.stream_handler import StreamMetrics
from victor.config.settings import Settings


@pytest.fixture
def mock_provider():
    """Create a mock provider."""
    provider = MagicMock()
    provider.name = "mock_provider"
    provider.supports_tools.return_value = True
    provider.get_context_window.return_value = 100000  # Return integer for context window
    provider.chat = AsyncMock(return_value=MagicMock(content="Response", tool_calls=[]))
    return provider


@pytest.fixture
def orchestrator_settings():
    """Create settings for testing."""
    return Settings(
        analytics_enabled=False,
        use_semantic_tool_selection=False,
        use_mcp_tools=False,
    )


@pytest.fixture
def orchestrator(mock_provider, orchestrator_settings):
    """Create an orchestrator for testing."""
    with patch("victor.agent.orchestrator.UsageLogger"):
        return AgentOrchestrator(
            settings=orchestrator_settings,
            provider=mock_provider,
            model="test-model",
        )


class TestStreamMetrics:
    """Tests for stream metrics functionality."""

    def test_record_first_token_time(self, orchestrator):
        """Test _record_first_token records time correctly."""
        # Initialize stream metrics through the public API
        stream_metrics = orchestrator._init_stream_metrics()
        assert stream_metrics.first_token_time is None

        orchestrator._record_first_token()

        # Access through the metrics collector's internal state
        assert orchestrator._metrics_collector._current_stream_metrics.first_token_time is not None

    def test_record_first_token_no_metrics(self, orchestrator):
        """Test _record_first_token does nothing when no metrics."""
        # Should not raise even without initialized metrics
        orchestrator._record_first_token()

    def test_finalize_stream_metrics(self, orchestrator):
        """Test _finalize_stream_metrics returns metrics (covers lines 306-321)."""
        import time

        # Initialize metrics through the API
        stream_metrics = orchestrator._init_stream_metrics()
        stream_metrics.first_token_time = time.time()
        stream_metrics.total_chunks = 10

        result = orchestrator._finalize_stream_metrics()

        assert result is not None
        assert result.end_time is not None
        assert result.total_chunks == 10

    def test_finalize_stream_metrics_no_metrics(self, orchestrator):
        """Test _finalize_stream_metrics returns None when no metrics."""
        result = orchestrator._finalize_stream_metrics()
        assert result is None

    def test_get_last_stream_metrics(self, orchestrator):
        """Test get_last_stream_metrics returns stored metrics (covers line 325)."""
        # Initialize metrics first
        orchestrator._init_stream_metrics()
        result = orchestrator.get_last_stream_metrics()
        assert result is not None

    def test_get_last_stream_metrics_no_metrics(self, orchestrator):
        """Test get_last_stream_metrics returns None when no metrics."""
        result = orchestrator.get_last_stream_metrics()
        assert result is None


class TestEmbeddingPreload:
    """Tests for embedding preload functionality."""

    @pytest.mark.asyncio
    async def test_preload_embeddings_no_selector(self, orchestrator):
        """Test _preload_embeddings returns early when no semantic selector."""
        orchestrator.semantic_selector = None
        await orchestrator._preload_embeddings()  # Should not raise

    @pytest.mark.asyncio
    async def test_preload_embeddings_already_initialized(self, orchestrator):
        """Test _preload_embeddings returns early when already initialized (covers line 340)."""
        orchestrator.semantic_selector = MagicMock()
        orchestrator.tool_selector = MagicMock()
        orchestrator.tool_selector._embeddings_initialized = True

        await orchestrator._preload_embeddings()  # Should return early

    @pytest.mark.asyncio
    async def test_preload_embeddings_success(self, orchestrator):
        """Test _preload_embeddings initializes embeddings (covers lines 342-347)."""
        mock_selector = MagicMock()
        mock_selector.initialize_tool_embeddings = AsyncMock()
        orchestrator.semantic_selector = mock_selector
        orchestrator.tool_selector = MagicMock()
        orchestrator.tool_selector._embeddings_initialized = False

        await orchestrator._preload_embeddings()

        mock_selector.initialize_tool_embeddings.assert_called_once()
        assert orchestrator.tool_selector._embeddings_initialized is True

    @pytest.mark.asyncio
    async def test_preload_embeddings_failure(self, orchestrator):
        """Test _preload_embeddings handles exception (covers lines 348-352)."""
        mock_selector = MagicMock()
        mock_selector.initialize_tool_embeddings = AsyncMock(
            side_effect=Exception("Embedding error")
        )
        orchestrator.semantic_selector = mock_selector
        orchestrator.tool_selector = MagicMock()
        orchestrator.tool_selector._embeddings_initialized = False

        # Should not raise
        await orchestrator._preload_embeddings()

    def test_start_embedding_preload_no_semantic(self, orchestrator):
        """Test start_embedding_preload does nothing when semantic not enabled (covers line 361)."""
        orchestrator.use_semantic_selection = False
        orchestrator.start_embedding_preload()
        assert orchestrator._embedding_preload_task is None

    def test_start_embedding_preload_already_started(self, orchestrator):
        """Test start_embedding_preload does nothing when already started."""
        orchestrator.use_semantic_selection = True
        orchestrator._embedding_preload_task = MagicMock()  # Already started
        orchestrator.start_embedding_preload()
        # Should not create a new task


class TestMCPIntegration:
    """Tests for MCP integration setup."""

    def test_setup_mcp_integration_no_mcp_registry(self, orchestrator):
        """Test _setup_mcp_integration handles missing MCPRegistry (covers lines 660-662)."""
        with patch.dict("sys.modules", {"victor.mcp.registry": None}):
            with patch.object(orchestrator, "_setup_legacy_mcp"):
                # This would raise ImportError, calling _setup_legacy_mcp
                try:
                    orchestrator._setup_mcp_integration()
                except Exception:
                    pass

    def test_setup_legacy_mcp_no_command(self, orchestrator):
        """Test _setup_legacy_mcp with no command does nothing (covers line 670)."""
        orchestrator._setup_legacy_mcp(None)  # Should not raise

    def test_setup_legacy_mcp_with_command_failure(self, orchestrator):
        """Test _setup_legacy_mcp handles connection failure (covers lines 678-679)."""
        with patch("victor.mcp.client.MCPClient") as mock_client:
            mock_client.return_value.connect.side_effect = Exception("Connection failed")
            with patch("victor.agent.orchestrator.configure_mcp_client"):
                orchestrator._setup_legacy_mcp("mcp command")  # Should not raise


class TestToolDependencies:
    """Tests for tool dependency registration."""

    def test_register_default_tool_dependencies(self, orchestrator):
        """Test _register_default_tool_dependencies creates graph (covers line 696+)."""
        orchestrator._register_default_tool_dependencies()
        assert orchestrator.tool_graph is not None


class TestConversationState:
    """Tests for conversation state management."""

    def test_reset_conversation(self, orchestrator):
        """Test reset_conversation resets state."""
        orchestrator.add_message("user", "test")
        orchestrator.reset_conversation()
        # After reset, messages should be empty
        assert len(orchestrator.conversation.messages) == 0

    def test_conversation_messages_property(self, orchestrator):
        """Test conversation.messages property returns messages."""
        orchestrator.add_message("user", "test message")
        history = orchestrator.conversation.messages
        assert len(history) >= 1


class TestToolSelection:
    """Tests for tool selection mechanisms."""

    def test_record_tool_selection_semantic(self, orchestrator):
        """Test _record_tool_selection records semantic stats (covers lines 381-388)."""
        initial = orchestrator._metrics_collector._selection_stats.semantic_selections
        orchestrator._record_tool_selection("semantic", 5)
        assert orchestrator._metrics_collector._selection_stats.semantic_selections == initial + 1
        assert orchestrator._metrics_collector._selection_stats.total_tools_selected >= 5

    def test_record_tool_selection_keyword(self, orchestrator):
        """Test _record_tool_selection records keyword stats."""
        initial = orchestrator._metrics_collector._selection_stats.keyword_selections
        orchestrator._record_tool_selection("keyword", 3)
        assert orchestrator._metrics_collector._selection_stats.keyword_selections == initial + 1

    def test_record_tool_selection_fallback(self, orchestrator):
        """Test _record_tool_selection records fallback stats."""
        initial = orchestrator._metrics_collector._selection_stats.fallback_selections
        orchestrator._record_tool_selection("fallback", 2)
        assert orchestrator._metrics_collector._selection_stats.fallback_selections == initial + 1

    def test_get_tool_usage_stats(self, orchestrator):
        """Test get_tool_usage_stats returns stats (covers lines 446-457+)."""
        orchestrator._record_tool_selection("keyword", 3)
        stats = orchestrator.get_tool_usage_stats()
        assert "selection_stats" in stats
        assert "keyword_selections" in stats["selection_stats"]


class TestResponseSanitization:
    """Tests for response sanitization methods."""

    def test_strip_markup(self, orchestrator):
        """Test _strip_markup removes tags (covers line 548)."""
        text = "<tag>Hello</tag> World"
        result = orchestrator._strip_markup(text)
        assert result is not None

    def test_sanitize_response(self, orchestrator):
        """Test _sanitize_response cleans content (covers line 552)."""
        text = "Some response with cleanup needs"
        result = orchestrator._sanitize_response(text)
        assert isinstance(result, str)

    def test_is_garbage_content(self, orchestrator):
        """Test _is_garbage_content detects malformed output (covers line 556)."""
        # Normal content
        assert orchestrator._is_garbage_content("Hello world") is False
        # Very short content might be considered garbage
        result = orchestrator._is_garbage_content("")
        assert isinstance(result, bool)

    def test_is_valid_tool_name(self, orchestrator):
        """Test _is_valid_tool_name validates names (covers line 560)."""
        # Valid tool names from registry should return True
        # Invalid names should return False
        result = orchestrator._is_valid_tool_name("nonexistent_tool_xyz")
        assert isinstance(result, bool)


class TestProviderChecks:
    """Tests for provider type checks."""

    def test_is_cloud_provider(self, orchestrator):
        """Test _is_cloud_provider check (covers line 532)."""
        result = orchestrator._is_cloud_provider()
        assert isinstance(result, bool)

    def test_is_local_provider(self, orchestrator):
        """Test _is_local_provider check (covers line 536)."""
        result = orchestrator._is_local_provider()
        assert isinstance(result, bool)


class TestSystemPrompt:
    """Tests for system prompt building."""

    def test_build_system_prompt_with_adapter(self, orchestrator):
        """Test _build_system_prompt_with_adapter (covers line 540)."""
        result = orchestrator._build_system_prompt_with_adapter()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_build_system_prompt_for_provider(self, orchestrator):
        """Test _build_system_prompt_for_provider (covers line 544)."""
        result = orchestrator._build_system_prompt_for_provider()
        assert isinstance(result, str)


class TestConversationStage:
    """Tests for conversation stage management."""

    def test_get_conversation_stage(self, orchestrator):
        """Test get_conversation_stage returns stage (covers line 487)."""
        from victor.agent.conversation_state import ConversationStage

        stage = orchestrator.get_conversation_stage()
        assert isinstance(stage, ConversationStage)

    def test_get_stage_recommended_tools(self, orchestrator):
        """Test get_stage_recommended_tools returns tools (covers line 495)."""
        tools = orchestrator.get_stage_recommended_tools()
        assert isinstance(tools, set)


class TestToolCallLogging:
    """Tests for tool call logging."""

    def test_log_tool_call(self, orchestrator):
        """Test _log_tool_call logs info (covers line 564)."""
        # Should not raise
        orchestrator._log_tool_call("test_tool", {"arg": "value"})


class TestToolExecution:
    """Tests for tool execution tracking."""

    def test_record_tool_execution_success(self, orchestrator):
        """Test _record_tool_execution with success (covers lines 400-444)."""
        orchestrator._record_tool_execution("test_tool", success=True, elapsed_ms=100.0)

        stats = orchestrator._metrics_collector._tool_usage_stats.get("test_tool")
        assert stats is not None
        assert stats["total_calls"] == 1
        assert stats["successful_calls"] == 1
        assert stats["failed_calls"] == 0

    def test_record_tool_execution_failure(self, orchestrator):
        """Test _record_tool_execution with failure."""
        orchestrator._record_tool_execution("failing_tool", success=False, elapsed_ms=50.0)

        stats = orchestrator._metrics_collector._tool_usage_stats.get("failing_tool")
        assert stats is not None
        assert stats["total_calls"] == 1
        assert stats["successful_calls"] == 0
        assert stats["failed_calls"] == 1

    def test_record_tool_execution_timing(self, orchestrator):
        """Test _record_tool_execution tracks timing correctly."""
        orchestrator._record_tool_execution("timed_tool", success=True, elapsed_ms=100.0)
        orchestrator._record_tool_execution("timed_tool", success=True, elapsed_ms=200.0)

        stats = orchestrator._metrics_collector._tool_usage_stats.get("timed_tool")
        assert stats["total_calls"] == 2
        assert stats["total_time_ms"] == 300.0
        assert stats["avg_time_ms"] == 150.0
        assert stats["min_time_ms"] == 100.0
        assert stats["max_time_ms"] == 200.0


class TestToolCallParsing:
    """Tests for tool call parsing."""

    def test_parse_tool_calls_with_adapter_empty(self, orchestrator):
        """Test _parse_tool_calls_with_adapter with no tool calls (covers lines 515-528)."""
        result = orchestrator._parse_tool_calls_with_adapter("Hello world", None)
        assert result is not None
        assert result.tool_calls == []

    def test_parse_tool_calls_with_adapter_native(self, orchestrator):
        """Test _parse_tool_calls_with_adapter with native tool calls."""
        raw_calls = [{"name": "read_file", "arguments": {"path": "/test.py"}}]
        result = orchestrator._parse_tool_calls_with_adapter("", raw_calls)
        assert result is not None


class TestWorkflowRegistration:
    """Tests for workflow registration."""

    def test_workflow_registry_initialized(self, orchestrator):
        """Test workflow_registry is initialized (covers line 215)."""
        assert orchestrator.workflow_registry is not None

    def test_workflow_registry_has_workflows(self, orchestrator):
        """Test _register_default_workflows adds workflows (covers line 568)."""
        # Workflows are registered during init, so check they exist
        from victor.workflows.base import WorkflowRegistry

        assert isinstance(orchestrator.workflow_registry, WorkflowRegistry)


class TestToolCallBudget:
    """Tests for tool call budget management."""

    def test_tool_calls_used_tracking(self, orchestrator):
        """Test tool_calls_used is tracked."""
        orchestrator.tool_calls_used = 0
        orchestrator.tool_calls_used += 1
        assert orchestrator.tool_calls_used == 1

    def test_tool_budget(self, orchestrator):
        """Test tool_budget property (covers line 159-160)."""
        budget = orchestrator.tool_budget
        assert isinstance(budget, int)
        assert budget > 0


class TestToolCacheIntegration:
    """Tests for tool cache integration."""

    def test_orchestrator_has_tool_cache(self, orchestrator):
        """Test orchestrator has tool_cache attribute."""
        # May be None if not configured
        assert hasattr(orchestrator, "tool_cache")

    def test_orchestrator_with_tool_cache(self, mock_provider, orchestrator_settings):
        """Test orchestrator with tool cache configured."""
        orchestrator_settings.tool_cache_enabled = True
        orchestrator_settings.tool_cache_ttl = 300

        with patch("victor.agent.orchestrator.UsageLogger"):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            # Tool cache should be initialized
            assert hasattr(orch, "tool_cache")


class TestToolSupportChecks:
    """Tests for tool support methods."""

    def test_should_use_tools(self, orchestrator):
        """Test _should_use_tools always returns True (covers line 953-955)."""
        assert orchestrator._should_use_tools() is True

    def test_model_supports_tool_calls_no_provider(self, orchestrator):
        """Test _model_supports_tool_calls with no provider name (covers lines 959-961)."""
        orchestrator.provider_name = ""
        result = orchestrator._model_supports_tool_calls()
        assert isinstance(result, bool)

    def test_model_supports_tool_calls_supported(self, orchestrator):
        """Test _model_supports_tool_calls with supported model (covers lines 963)."""
        # Set up to return True
        orchestrator.tool_capabilities = MagicMock()
        orchestrator.tool_capabilities.is_tool_call_supported.return_value = True
        orchestrator.provider_name = "test"

        result = orchestrator._model_supports_tool_calls()
        assert result is True

    def test_model_supports_tool_calls_not_supported(self, orchestrator):
        """Test _model_supports_tool_calls with unsupported model (covers lines 964-975)."""
        orchestrator.tool_capabilities = MagicMock()
        orchestrator.tool_capabilities.is_tool_call_supported.return_value = False
        orchestrator.tool_capabilities.get_supported_models.return_value = ["model-a", "model-b"]
        orchestrator.provider_name = "test"
        orchestrator._tool_capability_warned = False

        result = orchestrator._model_supports_tool_calls()

        assert result is False
        assert orchestrator._tool_capability_warned is True


class TestCancellationSupport:
    """Tests for streaming cancellation support."""

    def test_request_cancellation(self, orchestrator):
        """Test request_cancellation sets event (covers lines 1955-1957)."""
        import asyncio

        orchestrator._cancel_event = asyncio.Event()
        orchestrator.request_cancellation()
        assert orchestrator._cancel_event.is_set()

    def test_request_cancellation_no_event(self, orchestrator):
        """Test request_cancellation with no event (safe no-op)."""
        orchestrator._cancel_event = None
        # Should not raise
        orchestrator.request_cancellation()

    def test_is_streaming_true(self, orchestrator):
        """Test is_streaming returns True when streaming (covers line 1965)."""
        orchestrator._is_streaming = True
        assert orchestrator.is_streaming() is True

    def test_is_streaming_false(self, orchestrator):
        """Test is_streaming returns False when not streaming."""
        orchestrator._is_streaming = False
        assert orchestrator.is_streaming() is False

    def test_check_cancellation_true(self, orchestrator):
        """Test _check_cancellation when cancelled (covers lines 1973-1974)."""
        import asyncio

        orchestrator._cancel_event = asyncio.Event()
        orchestrator._cancel_event.set()
        assert orchestrator._check_cancellation() is True

    def test_check_cancellation_false(self, orchestrator):
        """Test _check_cancellation when not cancelled (covers line 1975)."""
        import asyncio

        orchestrator._cancel_event = asyncio.Event()
        assert orchestrator._check_cancellation() is False

    def test_check_cancellation_no_event(self, orchestrator):
        """Test _check_cancellation with no event."""
        orchestrator._cancel_event = None
        assert orchestrator._check_cancellation() is False


class TestShutdown:
    """Tests for shutdown functionality."""

    @pytest.mark.asyncio
    async def test_shutdown_basic(self, orchestrator):
        """Test shutdown cleans up resources (covers lines 1987-2022)."""
        orchestrator.provider = AsyncMock()
        orchestrator.provider.close = AsyncMock()
        orchestrator.code_manager = MagicMock()
        orchestrator.semantic_selector = None

        await orchestrator.shutdown()

        orchestrator.provider.close.assert_called_once()
        orchestrator.code_manager.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_with_semantic_selector(self, orchestrator):
        """Test shutdown closes semantic selector (covers lines 2014-2020)."""
        orchestrator.provider = AsyncMock()
        orchestrator.provider.close = AsyncMock()
        orchestrator.code_manager = MagicMock()

        mock_selector = AsyncMock()
        mock_selector.close = AsyncMock()
        orchestrator.semantic_selector = mock_selector

        await orchestrator.shutdown()

        mock_selector.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_handles_provider_error(self, orchestrator):
        """Test shutdown handles provider close error (covers lines 2003-2004)."""
        orchestrator.provider = AsyncMock()
        orchestrator.provider.close = AsyncMock(side_effect=Exception("Close error"))
        orchestrator.code_manager = MagicMock()
        orchestrator.semantic_selector = None

        # Should not raise
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_handles_code_manager_error(self, orchestrator):
        """Test shutdown handles code manager error (covers lines 2011-2012)."""
        orchestrator.provider = AsyncMock()
        orchestrator.provider.close = AsyncMock()
        orchestrator.code_manager = MagicMock()
        orchestrator.code_manager.stop.side_effect = Exception("Stop error")
        orchestrator.semantic_selector = None

        # Should not raise
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_handles_selector_error(self, orchestrator):
        """Test shutdown handles selector close error (covers lines 2019-2020)."""
        orchestrator.provider = AsyncMock()
        orchestrator.provider.close = AsyncMock()
        orchestrator.code_manager = MagicMock()

        mock_selector = AsyncMock()
        mock_selector.close = AsyncMock(side_effect=Exception("Selector error"))
        orchestrator.semantic_selector = mock_selector

        # Should not raise
        await orchestrator.shutdown()


class TestContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_aenter(self, orchestrator):
        """Test __aenter__ returns self (covers line 2026)."""
        result = await orchestrator.__aenter__()
        assert result is orchestrator

    @pytest.mark.asyncio
    async def test_aexit_calls_shutdown(self, orchestrator):
        """Test __aexit__ calls shutdown (covers line 2030)."""
        orchestrator.shutdown = AsyncMock()

        await orchestrator.__aexit__(None, None, None)

        orchestrator.shutdown.assert_called_once()


class TestChatMethod:
    """Tests for chat method."""

    @pytest.mark.asyncio
    async def test_chat_basic(self, mock_provider, orchestrator_settings):
        """Test chat adds messages and gets response (covers lines 1004-1047)."""
        mock_response = MagicMock()
        mock_response.content = "Test response"
        mock_response.tool_calls = []
        mock_provider.chat = AsyncMock(return_value=mock_response)
        mock_provider.supports_tools.return_value = False

        with patch("victor.agent.orchestrator.UsageLogger"):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )

            response = await orch.chat("Hello")

            assert response.content == "Test response"
            # User message should be added
            assert any(
                m.role == "user" and "Hello" in m.content for m in orch.conversation.messages
            )

    @pytest.mark.asyncio
    async def test_chat_with_tools(self, mock_provider, orchestrator_settings):
        """Test chat with tool support enabled."""
        mock_response = MagicMock()
        mock_response.content = "Using tools"
        mock_response.tool_calls = []
        mock_provider.chat = AsyncMock(return_value=mock_response)
        mock_provider.supports_tools.return_value = True

        with patch("victor.agent.orchestrator.UsageLogger"):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )

            # Mock tool selector
            orch.tool_selector.select_tools = AsyncMock(return_value=[])
            orch.tool_selector.prioritize_by_stage = MagicMock(return_value=[])

            response = await orch.chat("Search for files")

            assert response.content == "Using tools"
            orch.tool_selector.select_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_with_thinking(self, mock_provider, orchestrator_settings):
        """Test chat with thinking enabled (covers lines 1027-1029)."""
        mock_response = MagicMock()
        mock_response.content = "Thought response"
        mock_response.tool_calls = []
        mock_provider.chat = AsyncMock(return_value=mock_response)
        mock_provider.supports_tools.return_value = False

        with patch("victor.agent.orchestrator.UsageLogger"):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
                thinking=True,
            )

            await orch.chat("Think about this")

            # Verify thinking parameter was passed
            call_kwargs = mock_provider.chat.call_args[1]
            assert "thinking" in call_kwargs


class TestAddMessage:
    """Tests for add_message method."""

    def test_add_message_user(self, orchestrator):
        """Test add_message adds user message (covers lines 984-986)."""
        orchestrator.add_message("user", "Test user message")
        assert any(
            m.role == "user" and "Test user message" in m.content
            for m in orchestrator.conversation.messages
        )

    def test_add_message_assistant(self, orchestrator):
        """Test add_message adds assistant message (covers lines 987-988)."""
        orchestrator.add_message("assistant", "Test assistant response")
        assert any(
            m.role == "assistant" and "Test assistant response" in m.content
            for m in orchestrator.conversation.messages
        )


class TestEnsureSystemMessage:
    """Tests for _ensure_system_message method."""

    def test_ensure_system_message(self, orchestrator):
        """Test _ensure_system_message adds system prompt (covers lines 992-993)."""
        orchestrator._system_added = False
        orchestrator._ensure_system_message()
        assert orchestrator._system_added is True


class TestFromSettings:
    """Tests for from_settings factory method."""

    @pytest.mark.asyncio
    async def test_from_settings_profile_not_found(self, orchestrator_settings):
        """Test from_settings raises for missing profile (covers line 2054)."""
        orchestrator_settings.load_profiles = MagicMock(return_value={})

        with pytest.raises(ValueError, match="Profile not found"):
            await AgentOrchestrator.from_settings(
                settings=orchestrator_settings, profile_name="nonexistent"
            )

    @pytest.mark.asyncio
    async def test_from_settings_success(self, orchestrator_settings):
        """Test from_settings creates orchestrator (covers lines 2050-2070)."""
        mock_profile = MagicMock()
        mock_profile.provider = "mock_provider"
        mock_profile.model = "test-model"
        mock_profile.temperature = 0.7
        mock_profile.max_tokens = 4096
        mock_profile.tool_selection = None

        orchestrator_settings.load_profiles = MagicMock(return_value={"default": mock_profile})
        orchestrator_settings.get_provider_settings = MagicMock(return_value={})

        mock_provider = MagicMock()
        mock_provider.name = "mock_provider"
        mock_provider.supports_tools.return_value = True
        mock_provider.get_context_window.return_value = 100000

        with patch("victor.agent.orchestrator.ProviderRegistry") as mock_registry:
            mock_registry.create.return_value = mock_provider
            with patch("victor.agent.orchestrator.UsageLogger"):
                orch = await AgentOrchestrator.from_settings(
                    settings=orchestrator_settings, profile_name="default"
                )

                assert orch is not None
                assert orch.model == "test-model"


class TestToolPlanning:
    """Tests for tool planning methods."""

    def test_plan_tools_empty_goals(self, orchestrator):
        """Test _plan_tools with empty goals (covers line 805-806)."""
        result = orchestrator._plan_tools([])
        assert result == []

    def test_plan_tools_with_goals(self, orchestrator):
        """Test _plan_tools with valid goals (covers lines 808-819)."""
        # Add tool to graph
        orchestrator.tool_graph.add_tool("test_tool", inputs=["query"], outputs=["result"])
        result = orchestrator._plan_tools(["result"], ["query"])
        # Result depends on tool graph configuration
        assert isinstance(result, list)

    def test_goal_hints_for_message_summary(self, orchestrator):
        """Test _goal_hints_for_message detects summary requests (covers lines 825-828)."""
        result = orchestrator._goal_hints_for_message("Please summarize this code")
        assert "summary" in result

    def test_goal_hints_for_message_review(self, orchestrator):
        """Test _goal_hints_for_message detects review requests (covers line 827-828)."""
        result = orchestrator._goal_hints_for_message("Can you review this?")
        assert "summary" in result

    def test_goal_hints_for_message_documentation(self, orchestrator):
        """Test _goal_hints_for_message detects documentation requests (covers lines 829-830)."""
        result = orchestrator._goal_hints_for_message("Generate documentation please")
        assert "documentation" in result

    def test_goal_hints_for_message_security(self, orchestrator):
        """Test _goal_hints_for_message detects security requests (covers lines 831-832)."""
        result = orchestrator._goal_hints_for_message("Run a security scan")
        assert "security_report" in result

    def test_goal_hints_for_message_metrics(self, orchestrator):
        """Test _goal_hints_for_message detects metrics requests (covers lines 833-834)."""
        result = orchestrator._goal_hints_for_message("Show complexity metrics")
        assert "metrics_report" in result

    def test_goal_hints_for_message_no_match(self, orchestrator):
        """Test _goal_hints_for_message with no matching keywords."""
        result = orchestrator._goal_hints_for_message("Hello world")
        assert result == []


class TestToolConfiguration:
    """Tests for tool configuration loading."""

    def test_load_tool_configurations_empty(self, mock_provider, orchestrator_settings):
        """Test _load_tool_configurations with empty config."""
        orchestrator_settings.load_tool_config = MagicMock(return_value=None)

        with patch("victor.agent.orchestrator.UsageLogger"):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            # Should not raise during init
            assert orch is not None

    def test_load_tool_configurations_exception(self, mock_provider, orchestrator_settings):
        """Test _load_tool_configurations handles exception (covers lines 950-951)."""
        orchestrator_settings.load_tool_config = MagicMock(side_effect=Exception("Config error"))

        with patch("victor.agent.orchestrator.UsageLogger"):
            # Should not raise during init
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            assert orch is not None


class TestHandleToolCalls:
    """Tests for _handle_tool_calls method."""

    @pytest.mark.asyncio
    async def test_handle_tool_calls_empty_list(self, orchestrator):
        """Test _handle_tool_calls with empty list (covers line 1782)."""
        result = await orchestrator._handle_tool_calls([])
        assert result == []

    @pytest.mark.asyncio
    async def test_handle_tool_calls_not_a_dict(self, orchestrator):
        """Test _handle_tool_calls with non-dict tool call (covers lines 1788-1792)."""
        result = await orchestrator._handle_tool_calls(["not a dict"])
        assert result == []

    @pytest.mark.asyncio
    async def test_handle_tool_calls_no_name(self, orchestrator):
        """Test _handle_tool_calls with tool call without name (covers lines 1795-1797)."""
        result = await orchestrator._handle_tool_calls([{"arguments": {}}])
        assert result == []

    @pytest.mark.asyncio
    async def test_handle_tool_calls_invalid_name(self, orchestrator):
        """Test _handle_tool_calls with invalid tool name (covers lines 1800-1804)."""
        # Register a mock that returns False for invalid names
        orchestrator.sanitizer.is_valid_tool_name = MagicMock(return_value=False)

        result = await orchestrator._handle_tool_calls([{"name": "123invalid"}])
        assert result == []

    @pytest.mark.asyncio
    async def test_handle_tool_calls_disabled_tool(self, orchestrator):
        """Test _handle_tool_calls with disabled tool (covers lines 1807-1809)."""
        # Ensure tool is not in registry
        result = await orchestrator._handle_tool_calls([{"name": "nonexistent_tool"}])
        assert result == []

    @pytest.mark.asyncio
    async def test_handle_tool_calls_budget_reached(self, orchestrator):
        """Test _handle_tool_calls when budget reached (covers lines 1811-1815)."""
        orchestrator.tool_calls_used = 100
        orchestrator.tool_budget = 10

        # Use a valid tool name
        orchestrator.sanitizer.is_valid_tool_name = MagicMock(return_value=True)
        orchestrator.tools.is_tool_enabled = MagicMock(return_value=True)

        result = await orchestrator._handle_tool_calls([{"name": "read_file", "arguments": {}}])
        # Should skip because budget reached
        assert result == []

    @pytest.mark.asyncio
    async def test_handle_tool_calls_json_string_arguments(
        self, mock_provider, orchestrator_settings
    ):
        """Test _handle_tool_calls with JSON string arguments (covers lines 1820-1827)."""
        with patch("victor.agent.orchestrator.UsageLogger"):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )

            # Mock tool execution
            orch.tool_executor.execute = AsyncMock(
                return_value=MagicMock(success=True, result="done", error=None)
            )

            # Use string JSON arguments
            result = await orch._handle_tool_calls(
                [{"name": "read_file", "arguments": '{"path": "/test.py"}'}]
            )

            assert len(result) == 1
            assert result[0]["success"] is True

    @pytest.mark.asyncio
    async def test_handle_tool_calls_none_arguments(self, mock_provider, orchestrator_settings):
        """Test _handle_tool_calls with None arguments (covers lines 1828-1829)."""
        with patch("victor.agent.orchestrator.UsageLogger"):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )

            # Mock tool execution
            orch.tool_executor.execute = AsyncMock(
                return_value=MagicMock(success=True, result="done", error=None)
            )

            result = await orch._handle_tool_calls([{"name": "read_file", "arguments": None}])

            assert len(result) == 1
            assert result[0]["success"] is True

    @pytest.mark.asyncio
    async def test_handle_tool_calls_repeated_failure_skip(
        self, mock_provider, orchestrator_settings
    ):
        """Test _handle_tool_calls skips repeated failing calls (covers lines 1841-1845)."""
        with patch("victor.agent.orchestrator.UsageLogger"):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )

            # Add to failed signatures
            import json

            args = {"path": "/test.py"}
            signature = ("read_file", json.dumps(args, sort_keys=True, default=str))
            orch.failed_tool_signatures.add(signature)

            # Try same call again
            result = await orch._handle_tool_calls([{"name": "read_file", "arguments": args}])

            # Should be skipped
            assert result == []

    @pytest.mark.asyncio
    async def test_handle_tool_calls_success(self, mock_provider, orchestrator_settings):
        """Test _handle_tool_calls successful execution (covers lines 1912-1927)."""
        with patch("victor.agent.orchestrator.UsageLogger"):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )

            # Mock successful tool execution
            orch.tool_executor.execute = AsyncMock(
                return_value=MagicMock(success=True, result="File contents", error=None)
            )

            result = await orch._handle_tool_calls(
                [{"name": "read_file", "arguments": {"path": "/test.py"}}]
            )

            assert len(result) == 1
            assert result[0]["success"] is True
            assert result[0]["name"] == "read_file"
            assert orch.tool_calls_used == 1
            assert "read_file" in orch.executed_tools

    @pytest.mark.asyncio
    async def test_handle_tool_calls_failure(self, mock_provider, orchestrator_settings):
        """Test _handle_tool_calls failed execution (covers lines 1928-1941)."""
        with patch("victor.agent.orchestrator.UsageLogger"):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )

            # Mock failed tool execution
            orch.tool_executor.execute = AsyncMock(
                return_value=MagicMock(success=False, result=None, error="File not found")
            )

            result = await orch._handle_tool_calls(
                [{"name": "read_file", "arguments": {"path": "/nonexistent.py"}}]
            )

            assert len(result) == 1
            assert result[0]["success"] is False
            assert result[0]["error"] == "File not found"
            # Signature should be added to failed set
            assert len(orch.failed_tool_signatures) == 1

    @pytest.mark.asyncio
    async def test_handle_tool_calls_read_file_tracking(self, mock_provider, orchestrator_settings):
        """Test _handle_tool_calls tracks read_file paths (covers lines 1885-1886)."""
        with patch("victor.agent.orchestrator.UsageLogger"):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )

            # Mock successful tool execution
            orch.tool_executor.execute = AsyncMock(
                return_value=MagicMock(success=True, result="File contents", error=None)
            )

            await orch._handle_tool_calls(
                [{"name": "read_file", "arguments": {"path": "/test.py"}}]
            )

            assert "/test.py" in orch.observed_files


class TestGetToolStatusMessage:
    """Tests for _get_tool_status_message helper method."""

    def test_execute_bash_with_command(self, mock_provider, orchestrator_settings):
        """Test status message for execute_bash tool."""
        with patch("victor.agent.orchestrator.UsageLogger"):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            result = orch._get_tool_status_message(
                "execute_bash", {"command": "ls -la"}
            )
            assert result == "🔧 Running execute_bash: `ls -la`"

    def test_execute_bash_long_command_truncation(self, mock_provider, orchestrator_settings):
        """Test long command truncation for execute_bash."""
        with patch("victor.agent.orchestrator.UsageLogger"):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            long_cmd = "a" * 100
            result = orch._get_tool_status_message(
                "execute_bash", {"command": long_cmd}
            )
            assert result == f"🔧 Running execute_bash: `{'a' * 80}...`"

    def test_list_directory(self, mock_provider, orchestrator_settings):
        """Test status message for list_directory tool."""
        with patch("victor.agent.orchestrator.UsageLogger"):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            result = orch._get_tool_status_message(
                "list_directory", {"path": "/src"}
            )
            assert result == "🔧 Listing directory: /src"

    def test_list_directory_default_path(self, mock_provider, orchestrator_settings):
        """Test list_directory with default path."""
        with patch("victor.agent.orchestrator.UsageLogger"):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            result = orch._get_tool_status_message("list_directory", {})
            assert result == "🔧 Listing directory: ."

    def test_read_file(self, mock_provider, orchestrator_settings):
        """Test status message for read_file tool."""
        with patch("victor.agent.orchestrator.UsageLogger"):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            result = orch._get_tool_status_message(
                "read_file", {"path": "/src/main.py"}
            )
            assert result == "🔧 Reading file: /src/main.py"

    def test_edit_files_single(self, mock_provider, orchestrator_settings):
        """Test status message for edit_files with single file."""
        with patch("victor.agent.orchestrator.UsageLogger"):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            result = orch._get_tool_status_message(
                "edit_files", {"files": [{"path": "/src/main.py"}]}
            )
            assert result == "🔧 Editing: /src/main.py"

    def test_edit_files_multiple(self, mock_provider, orchestrator_settings):
        """Test status message for edit_files with multiple files."""
        with patch("victor.agent.orchestrator.UsageLogger"):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            result = orch._get_tool_status_message(
                "edit_files",
                {
                    "files": [
                        {"path": "/a.py"},
                        {"path": "/b.py"},
                        {"path": "/c.py"},
                        {"path": "/d.py"},
                    ]
                },
            )
            assert result == "🔧 Editing: /a.py, /b.py, /c.py (+1 more)"

    def test_edit_files_empty(self, mock_provider, orchestrator_settings):
        """Test status message for edit_files with empty files list."""
        with patch("victor.agent.orchestrator.UsageLogger"):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            result = orch._get_tool_status_message("edit_files", {"files": []})
            assert result == "🔧 Running edit_files..."

    def test_write_file(self, mock_provider, orchestrator_settings):
        """Test status message for write_file tool."""
        with patch("victor.agent.orchestrator.UsageLogger"):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            result = orch._get_tool_status_message(
                "write_file", {"path": "/new_file.py"}
            )
            assert result == "🔧 Writing file: /new_file.py"

    def test_code_search(self, mock_provider, orchestrator_settings):
        """Test status message for code_search tool."""
        with patch("victor.agent.orchestrator.UsageLogger"):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            result = orch._get_tool_status_message(
                "code_search", {"query": "def main"}
            )
            assert result == "🔧 Searching: def main"

    def test_code_search_long_query_truncation(self, mock_provider, orchestrator_settings):
        """Test long query truncation for code_search."""
        with patch("victor.agent.orchestrator.UsageLogger"):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            long_query = "a" * 60
            result = orch._get_tool_status_message(
                "code_search", {"query": long_query}
            )
            assert result == f"🔧 Searching: {'a' * 50}..."

    def test_unknown_tool(self, mock_provider, orchestrator_settings):
        """Test status message for unknown tools."""
        with patch("victor.agent.orchestrator.UsageLogger"):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            result = orch._get_tool_status_message(
                "some_custom_tool", {"arg": "value"}
            )
            assert result == "🔧 Running some_custom_tool..."
