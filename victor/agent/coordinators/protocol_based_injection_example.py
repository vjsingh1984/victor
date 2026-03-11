"""Example demonstrating protocol-based dependency injection for ChatCoordinator.

This example shows how protocol-based injection enables clean testing
with lightweight mocks instead of full orchestrator instances.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

# =============================================================================
# Lightweight Protocol Implementations for Testing
# =============================================================================


@dataclass
class MockConversation:
    """Mock conversation for testing."""

    messages: List[Any] = field(default_factory=list)

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    @property
    def message_count(self) -> int:
        return len(self.messages)


@dataclass
class MockToolSelector:
    """Mock tool selector for testing."""

    select_calls: List[Any] = field(default_factory=list)

    async def select_tools(self, context: str, **kwargs) -> List[Any]:
        self.select_calls.append((context, kwargs))
        return []  # Return empty tools list


@dataclass
class MockProvider:
    """Mock LLM provider for testing."""

    chat_responses: List[Any] = field(default_factory=list)

    async def chat(self, messages: Any, **kwargs) -> Any:
        response = self.chat_responses.pop(0) if self.chat_responses else None
        return response

    def supports_tools(self) -> bool:
        return True


@dataclass
class MockTaskCoordinator:
    """Mock task coordinator for testing."""

    prepare_task_calls: List[Any] = field(default_factory=list)

    def prepare_task(self, user_message: str, task_type: Any, controller: Any) -> tuple:
        self.prepare_task_calls.append((user_message, task_type, controller))
        # Return (task_classification, complexity_tool_budget)
        return (None, 10)


@dataclass
class MockContainer:
    """Mock container for testing."""

    _services: Dict[type, Any] = field(default_factory=dict)

    def get(self, service_type: type) -> Any:
        return self._services.get(service_type)


# =============================================================================
# Mock Orchestrator satisfying ChatOrchestratorProtocol
# =============================================================================


@dataclass
class MockChatOrchestrator:
    """Lightweight mock satisfying ChatOrchestratorProtocol.

    This mock provides only the attributes that ChatCoordinator actually uses,
    without the full complexity of AgentOrchestrator. This enables:
    - Fast unit tests (no heavy initialization)
    - Deterministic behavior (controlled responses)
    - Isolated testing (no external dependencies)
    """

    # -- ChatContextProtocol --
    conversation: Any = field(default_factory=MockConversation)
    messages: List[Any] = field(default_factory=list)
    conversation_controller: Any = None
    _context_compactor: Any = None
    _context_manager: Any = None
    settings: Any = None
    _session_state: Any = None
    _cumulative_token_usage: Dict[str, int] = field(default_factory=dict)
    _system_added: bool = False

    def add_message(self, role: str, content: str) -> None:
        self.conversation.add_message(role, content)

    def _check_context_overflow(self, max_context: int) -> bool:
        return len(self.messages) > 100  # Simple mock

    def _get_max_context_chars(self) -> int:
        return 128000  # Default GPT-4 context

    def _get_thinking_disabled_prompt(self, prompt: str) -> str:
        return f"{prompt} (thinking disabled)"

    # -- ToolContextProtocol --
    tool_selector: Any = field(default_factory=MockToolSelector)
    tool_adapter: Any = None
    _tool_coordinator: Any = None
    _tool_planner: Any = None
    tool_budget: int = 10
    tool_calls_used: int = 0
    use_semantic_selection: bool = False
    observed_files: set = field(default_factory=set)

    async def _handle_tool_calls(self, tool_calls: Any) -> List[Dict[str, Any]]:
        return [{"result": "mock_tool_result"}]

    def _model_supports_tool_calls(self) -> bool:
        return True

    # -- ProviderContextProtocol --
    provider: Any = field(default_factory=MockProvider)
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    thinking: Any = None
    _provider_coordinator: Any = None
    _cancel_event: Any = None
    _is_streaming: bool = False

    def _check_cancellation(self) -> bool:
        return False

    # -- ChatOrchestratorProtocol extras --
    task_classifier: Any = None
    task_coordinator: Any = field(default_factory=MockTaskCoordinator)
    _task_analyzer: Any = None
    unified_tracker: Any = None
    _task_completion_detector: Any = None

    # Recovery methods (delegated)
    _recovery_coordinator: Any = None
    _recovery_integration: Any = None

    async def _create_recovery_context(self, stream_ctx: Any) -> Any:
        return {"mock": "recovery_context"}

    async def _handle_recovery_with_integration(
        self, stream_ctx: Any, full_content: str, tool_calls: Any, mentioned_tools: Any = None
    ) -> Any:
        return {"mock": "recovery_action"}

    def _apply_recovery_action(self, recovery_action: Any, stream_ctx: Any) -> Any:
        return None

    def _record_intelligent_outcome(
        self, success: bool, quality_score: float, user_satisfied: bool, completed: bool
    ) -> None:
        pass

    async def _validate_intelligent_response(
        self, response: str, query: str, tool_calls: int, task_type: str
    ) -> Any:
        return {"valid": True}

    async def _prepare_intelligent_request(self, task: str, task_type: str) -> Any:
        return {"mock": "intelligent_context"}

    # Presentation & output
    sanitizer: Any = None
    response_completer: Any = None
    debug_logger: Any = None
    reminder_manager: Any = None
    _chunk_generator: Any = None
    _presentation: Any = None
    _metrics_collector: Any = None
    _streaming_handler: Any = None

    # Session state
    _required_files: List[str] = field(default_factory=list)
    _required_outputs: List[str] = field(default_factory=list)
    _read_files_session: Any = None
    _all_files_read_nudge_sent: bool = False
    _usage_analytics: Any = None
    _sequence_tracker: Any = None
    _container: Any = field(default_factory=MockContainer)


# =============================================================================
# Usage Example
# =============================================================================


async def example_chat_with_mock_orchestrator():
    """Example: Using ChatCoordinator with a lightweight mock orchestrator."""
    from victor.agent.coordinators.chat_coordinator import ChatCoordinator

    # Create lightweight mock instead of full AgentOrchestrator
    mock_orchestrator = MockChatOrchestrator(
        tool_budget=5,
        model="gpt-4",
    )

    # Set up mock responses
    mock_orchestrator.provider.chat_responses.append(
        # Simple completion response
        type("MockResponse", (), {"content": "Hello from mock!"})()
    )

    # Create ChatCoordinator with mock
    coordinator = ChatCoordinator(orchestrator=mock_orchestrator)

    # Use chat method - no LLM calls, fully deterministic
    result = await coordinator.chat("Hello, coordinator!")

    print(f"Response: {result.content}")
    print(
        f"Mock task coordinator calls: {len(mock_orchestrator.task_coordinator.prepare_task_calls)}"
    )


# =============================================================================
# Benefits of Protocol-Based Injection
# =============================================================================


def print_benefits():
    """Print benefits of protocol-based dependency injection."""

    benefits = """
    Benefits of Protocol-Based Injection for ChatCoordinator
    ================================================================

    1. FAST TESTS
       - No heavy AgentOrchestrator initialization
       - No LLM provider setup
       - No file system dependencies
       - Tests run in milliseconds vs seconds

    2. DETERMINISTIC BEHAVIOR
       - Controlled mock responses
       - No external API calls
       - Reproducible test scenarios

    3. ISOLATED TESTING
       - Test only ChatCoordinator logic
       - No dependency on orchestrator correctness
       - Failures point to ChatCoordinator bugs

    4. CLEAR CONTRACTS
       - ChatOrchestratorProtocol documents exact dependencies
       - Compile-time checking via @runtime_checkable
       - IDE autocomplete for all dependencies

    5. FLEXIBLE MOCKS
       - Override only what you need for the test
       - Real implementation for unused methods
       - Easy to create scenario-specific mocks

    Example Test Speed Comparison:
    ------------------------------------
    With full AgentOrchestrator:  ~2-5 seconds per test
    With MockChatOrchestrator:    ~10-50 milliseconds per test
    Speedup:                     40-500x faster

    Before Protocol-Based Injection:
    ------------------------------------
    def test_chat_coordinator():
        orchestrator = AgentOrchestrator(...)  # Slow initialization
        coordinator = ChatCoordinator(orchestrator)
        result = await coordinator.chat("test")
        assert result.content  # Hard to mock, depends on LLM

    After Protocol-Based Injection:
    ------------------------------------
    def test_chat_coordinator():
        mock = MockChatOrchestrator()  # Fast, no I/O
        mock.provider.chat_responses.append(MockResponse("test"))
        coordinator = ChatCoordinator(mock)
        result = await coordinator.chat("test")
        assert result.content == "test"  # Deterministic

    """

    print(benefits)


if __name__ == "__main__":
    import asyncio

    print_benefits()
    print("\n" + "=" * 60 + "\n")
    asyncio.run(example_chat_with_mock_orchestrator())
