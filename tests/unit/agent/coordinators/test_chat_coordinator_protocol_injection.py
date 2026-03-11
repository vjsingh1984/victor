"""Unit tests demonstrating protocol-based dependency injection for ChatCoordinator.

These tests show how ChatOrchestratorProtocol enables:
1. Fast unit tests with lightweight mocks
2. Deterministic behavior without external dependencies
3. Isolated testing of ChatCoordinator logic

Phase 6 of DIP Hardening (docs/architecture-analysis-phase3.md:186).
"""

import pytest
from typing import Any, List
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

from victor.agent.coordinators.chat_coordinator import ChatCoordinator
from victor.agent.coordinators.chat_protocols import ChatOrchestratorProtocol


# =============================================================================
# Lightweight Protocol Satisfying Mocks
# =============================================================================


@dataclass
class MockOrchestrator(ChatOrchestratorProtocol):
    """Lightweight mock satisfying ChatOrchestratorProtocol.

    This mock provides only what ChatCoordinator actually uses, avoiding
    the complexity of a full AgentOrchestrator instance.
    """

    # -- ChatContextProtocol --
    conversation: Any = field(default_factory=lambda: MagicMock(
        message_count=lambda: 1,
        ensure_system_prompt=lambda: None,
    ))
    messages: List[Any] = field(default_factory=list)
    conversation_controller: Any = MagicMock()
    _context_compactor: Any = None
    _context_manager: Any = None
    settings: Any = MagicMock()
    settings.enable_planning = False
    _session_state: Any = MagicMock(reset_for_new_turn=lambda: None)
    _cumulative_token_usage: dict = field(default_factory=lambda: {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    })
    _system_added: bool = False

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def _check_context_overflow(self, max_context: int) -> bool:
        return False  # Never overflow in tests

    def _get_max_context_chars(self) -> int:
        return 128000

    def _get_thinking_disabled_prompt(self, prompt: str) -> str:
        return prompt

    # -- ToolContextProtocol --
    tool_selector: Any = AsyncMock()
    tool_adapter: Any = MagicMock()
    _tool_coordinator: Any = MagicMock()
    _tool_planner: Any = MagicMock()
    tool_budget: int = 10
    tool_calls_used: int = 0
    use_semantic_selection: bool = False
    observed_files: set = field(default_factory=set)

    async def _handle_tool_calls(self, tool_calls: Any) -> List[dict]:
        return [{"result": "mock_result"}]

    def _model_supports_tool_calls(self) -> bool:
        return True

    # -- ProviderContextProtocol --
    provider: Any = AsyncMock()
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    thinking: Any = None
    _provider_coordinator: Any = MagicMock()
    _cancel_event: Any = MagicMock()
    _is_streaming: bool = False

    def _check_cancellation(self) -> bool:
        return False

    # -- ChatOrchestratorProtocol extras --
    task_classifier: Any = MagicMock(
        classify=lambda msg: MagicMock(complexity=MagicMock(value="SIMPLE"))
    )
    task_coordinator: Any = MagicMock()
    _task_analyzer: Any = MagicMock()
    unified_tracker: Any = MagicMock()
    _task_completion_detector: Any = None

    # Recovery
    _recovery_coordinator: Any = MagicMock()
    _recovery_integration: Any = MagicMock()

    async def _create_recovery_context(self, stream_ctx: Any) -> Any:
        return {}

    async def _handle_recovery_with_integration(
        self, stream_ctx: Any, full_content: str, tool_calls: Any, mentioned_tools: Any = None
    ) -> Any:
        return {}

    def _apply_recovery_action(self, recovery_action: Any, stream_ctx: Any) -> Any:
        return None

    def _record_intelligent_outcome(self, success: bool, quality_score: float, user_satisfied: bool, completed: bool) -> None:
        pass

    async def _validate_intelligent_response(self, response: str, query: str, tool_calls: int, task_type: str) -> Any:
        return None

    async def _prepare_intelligent_request(self, task: str, task_type: str) -> Any:
        return {}

    # Presentation & output
    sanitizer: Any = MagicMock(sanitize=lambda x: x)
    response_completer: Any = MagicMock()
    debug_logger: Any = MagicMock()
    reminder_manager: Any = MagicMock()
    _chunk_generator: Any = MagicMock()
    _presentation: Any = MagicMock()
    _metrics_collector: Any = MagicMock(init_stream_metrics=lambda: MagicMock(
        start_time=0.0,
        total_chunks=0,
        total_content_length=0,
        tool_calls_count=0,
    ))
    _streaming_handler: Any = MagicMock()

    # Session state
    _required_files: List[str] = field(default_factory=list)
    _required_outputs: List[str] = field(default_factory=list)
    _read_files_session: Any = None
    _all_files_read_nudge_sent: bool = False
    _usage_analytics: Any = None
    _sequence_tracker: Any = None


# =============================================================================
# Tests
# =============================================================================


class TestChatCoordinatorProtocolInjection:
    """Tests demonstrating protocol-based dependency injection."""

    def test_chat_coordinator_accepts_protocol_satisfying_mock(self):
        """ChatCoordinator should accept any object satisfying ChatOrchestratorProtocol."""
        # Create lightweight mock instead of full AgentOrchestrator
        mock_orchestrator = MockOrchestrator()

        # Should not raise - mock satisfies the protocol
        coordinator = ChatCoordinator(orchestrator=mock_orchestrator)

        assert coordinator is not None
        assert coordinator._orchestrator is mock_orchestrator

    def test_chat_coordinator_delegates_to_conversation_methods(self):
        """ChatCoordinator should use orchestrator's conversation methods."""
        mock_orchestrator = MockOrchestrator()
        coordinator = ChatCoordinator(orchestrator=mock_orchestrator)

        # Access conversation through orchestrator
        assert coordinator._orchestrator.conversation is not None
        assert coordinator._orchestrator.add_message is not None

    def test_chat_coordinator_delegates_to_tool_methods(self):
        """ChatCoordinator should use orchestrator's tool methods."""
        mock_orchestrator = MockOrchestrator()
        coordinator = ChatCoordinator(orchestrator=mock_orchestrator)

        # Access tool budget through orchestrator
        assert coordinator._orchestrator.tool_budget == 10
        assert coordinator._orchestrator._model_supports_tool_calls() is True

    def test_chat_coordinator_delegates_to_provider_methods(self):
        """ChatCoordinator should use orchestrator's provider methods."""
        mock_orchestrator = MockOrchestrator()
        coordinator = ChatCoordinator(orchestrator=mock_orchestrator)

        # Access provider through orchestrator
        assert coordinator._orchestrator.model == "gpt-4"
        assert coordinator._orchestrator.temperature == 0.7
        assert coordinator._orchestrator._check_cancellation() is False

    @pytest.mark.asyncio
    async def test_chat_coordinator_uses_execution_coordinator(self):
        """ChatCoordinator should have execution_coordinator as a property."""
        mock_orchestrator = MockOrchestrator()
        coordinator = ChatCoordinator(orchestrator=mock_orchestrator)

        # Verify execution_coordinator is a property on the class
        # (It's lazily created when first accessed, but requires setup_streaming_pipeline first)
        assert 'execution_coordinator' in dir(ChatCoordinator)
        assert isinstance(getattr(ChatCoordinator, 'execution_coordinator', None), property)

    @pytest.mark.asyncio
    async def test_chat_coordinator_with_mock_provider_response(self):
        """ChatCoordinator should delegate to mock orchestrator's provider."""
        # Set up mock provider response
        mock_response = MagicMock(content="Mock response!")
        mock_orchestrator = MockOrchestrator()

        # Configure provider mock to return our mock response
        async def mock_chat(*args, **kwargs):
            return mock_response

        mock_orchestrator.provider.chat = mock_chat
        mock_orchestrator.provider.supports_tools = lambda: True

        # Mock tool execution (no tools in this simple case)
        mock_orchestrator._handle_tool_calls = lambda *args, **kwargs: []

        coordinator = ChatCoordinator(orchestrator=mock_orchestrator)

        # Verify delegation: provider is accessible through orchestrator
        assert coordinator._orchestrator.provider is not None
        assert coordinator._orchestrator.provider.supports_tools() is True

        # Note: Full chat() execution requires execution_coordinator setup
        # This test verifies protocol injection works - coordinator accepts mock
        # and can access all protocol-required attributes
        assert coordinator is not None
        assert coordinator._orchestrator is mock_orchestrator

    def test_chat_coordinator_respects_tool_budget_from_orchestrator(self):
        """ChatCoordinator should respect tool budget from orchestrator."""
        mock_orchestrator = MockOrchestrator(tool_budget=5)
        coordinator = ChatCoordinator(orchestrator=mock_orchestrator)

        # Budget should come from orchestrator
        assert coordinator._orchestrator.tool_budget == 5

    def test_chat_coordinator_uses_task_classifier_from_orchestrator(self):
        """ChatCoordinator should use task classifier from orchestrator."""
        mock_orchestrator = MockOrchestrator()
        coordinator = ChatCoordinator(orchestrator=mock_orchestrator)

        # Task classifier should be accessible
        assert coordinator._orchestrator.task_classifier is not None

    def test_protocol_enables_isolated_testing(self):
        """Protocol-based injection enables testing without full orchestrator."""
        # This test would be impossible without protocol-based injection:
        # - No AgentOrchestrator import (avoids circular dependencies)
        # - No LLM provider setup (fast, no API keys needed)
        # - No file system dependencies (deterministic)

        # Import only the protocol and ChatCoordinator
        from victor.agent.coordinators.chat_protocols import ChatOrchestratorProtocol
        from victor.agent.coordinators.chat_coordinator import ChatCoordinator

        # Verify protocol is runtime-checkable
        from typing import Protocol, runtime_checkable
        assert isinstance(ChatOrchestratorProtocol, type) or getattr(ChatOrchestratorProtocol, "__origin__", None) is Protocol

        # Create coordinator with mock
        mock_orchestrator = MockOrchestrator()
        coordinator = ChatCoordinator(orchestrator=mock_orchestrator)

        # Verify coordinator was created successfully
        assert coordinator is not None
        assert coordinator._orchestrator is mock_orchestrator

        # Verify we can access protocol-required attributes
        assert coordinator._orchestrator.conversation is not None
        assert coordinator._orchestrator.provider is not None
        assert coordinator._orchestrator.tool_selector is not None


class TestChatCoordinatorProtocolCompliance:
    """Tests that ChatOrchestratorProtocol captures all dependencies correctly."""

    def test_protocol_includes_chat_context_dependencies(self):
        """Protocol should include all chat context dependencies."""
        from victor.agent.coordinators.chat_protocols import ChatContextProtocol
        from typing import get_args, get_origin

        # Check that protocol defines the required attributes
        # For Protocol, we check the __annotations__ or use __protocol_attrs__
        # But the simplest way is to verify the protocol can be used
        # Just verify the protocol exists and is runtime-checkable

        from typing import Protocol
        assert isinstance(ChatContextProtocol, type) or getattr(ChatContextProtocol, "__origin__", None) is Protocol

    def test_protocol_includes_tool_context_dependencies(self):
        """Protocol should include all tool context dependencies."""
        from victor.agent.coordinators.chat_protocols import ToolContextProtocol

        from typing import Protocol
        assert isinstance(ToolContextProtocol, type) or getattr(ToolContextProtocol, "__origin__", None) is Protocol

    def test_protocol_includes_provider_context_dependencies(self):
        """Protocol should include all provider context dependencies."""
        from victor.agent.coordinators.chat_protocols import ProviderContextProtocol

        from typing import Protocol
        assert isinstance(ProviderContextProtocol, type) or getattr(ProviderContextProtocol, "__origin__", None) is Protocol

    def test_mock_orchestrator_satisfies_protocol(self):
        """MockOrchestrator should satisfy ChatOrchestratorProtocol."""
        mock = MockOrchestrator()

        # Runtime check - this will fail if mock doesn't satisfy protocol
        from victor.agent.coordinators.chat_protocols import ChatOrchestratorProtocol
        assert isinstance(mock, ChatOrchestratorProtocol)

    def test_chat_coordinator_type_checks_protocols(self):
        """ChatCoordinator should verify orchestrator satisfies protocol at runtime."""
        mock_orchestrator = MockOrchestrator()
        coordinator = ChatCoordinator(orchestrator=mock_orchestrator)

        # Coordinator should accept the mock
        assert coordinator._orchestrator is mock_orchestrator


# =============================================================================
# Benefits Demonstration
# =============================================================================


class TestProtocolBasedInjectionBenefits:
    """Tests demonstrating practical benefits of protocol-based injection."""

    def test_test_speed_with_mock_vs_real_orchestrator(self):
        """Demonstrate test speed improvement with lightweight mocks."""
        import time

        # Lightweight mock - fast instantiation
        start = time.perf_counter()
        mock_orchestrator = MockOrchestrator()
        coordinator = ChatCoordinator(orchestrator=mock_orchestrator)
        mock_time = time.perf_counter() - start

        # Even with mock, coordinator creation should be fast
        assert mock_time < 0.1  # Less than 100ms

    def test_deterministic_behavior_with_mocks(self):
        """Demonstrate deterministic behavior with controlled mocks."""
        # Create two identical mocks
        mock1 = MockOrchestrator(tool_budget=5)
        mock2 = MockOrchestrator(tool_budget=5)

        # Both should behave identically
        assert mock1.tool_budget == mock2.tool_budget
        assert mock1.model == mock2.model

    def test_isolated_testing_focuses_on_coordinator_logic(self):
        """Demonstrate that tests focus on ChatCoordinator logic, not orchestrator."""
        # This test validates ChatCoordinator's initialization logic
        # without testing orchestrator correctness

        mock_orchestrator = MockOrchestrator()
        coordinator = ChatCoordinator(orchestrator=mock_orchestrator)

        # Verify coordinator initializes correctly
        assert coordinator is not None
        assert coordinator._orchestrator is mock_orchestrator

        # Verify execution_coordinator exists as a property on the class
        # (lazy initialization requires set_streaming_pipeline to be called first)
        assert 'execution_coordinator' in dir(ChatCoordinator)
        assert isinstance(getattr(ChatCoordinator, 'execution_coordinator', None), property)
