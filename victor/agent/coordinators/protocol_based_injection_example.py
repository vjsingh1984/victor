"""Examples for protocol-based injection during the chat migration.

New code should target `ChatServiceProtocol` / `ChatService`. The legacy
`ChatCoordinator` has been removed along with `chat_compat` module.

Note: The `chat_compat` module was removed as part of the service-first
architecture migration. Use `ChatService` from `victor.agent.services` instead.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from victor.agent.services.protocols import ChatServiceProtocol
from victor.agent.services.protocols.chat_runtime import ChatOrchestratorProtocol
from victor.core.async_utils import run_sync

# =============================================================================
# Lightweight Protocol Implementations for Testing
# =============================================================================


@dataclass
class MockConversation:
    """Mock conversation for testing."""

    messages: List[Any] = field(default_factory=list)

    def add_message(self, role: str, content: str, **metadata: Any) -> None:
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


@dataclass
class MockChatService(ChatServiceProtocol):
    """Lightweight mock satisfying the canonical ChatServiceProtocol."""

    responses: List[Any] = field(default_factory=list)
    streamed_chunks: List[Any] = field(default_factory=list)
    reset_calls: int = 0

    async def chat(self, user_message: str, *, stream: bool = False, **kwargs) -> Any:
        if self.responses:
            return self.responses.pop(0)
        return type("MockResponse", (), {"content": f"Echo: {user_message}"})()

    async def stream_chat(self, user_message: str, **kwargs):
        for chunk in self.streamed_chunks:
            yield chunk

    async def chat_with_planning(
        self, user_message: str, use_planning: Optional[bool] = None
    ) -> Any:
        return await self.chat(user_message, use_planning=use_planning)

    async def handle_context_and_iteration_limits(
        self,
        user_message: str,
        max_total_iterations: int,
        max_context: int,
        total_iterations: int,
        last_quality_score: float,
    ) -> tuple[bool, Optional[Any]]:
        return False, None

    def reset_conversation(self) -> None:
        self.reset_calls += 1

    @staticmethod
    def persist_message(
        role: str,
        content: str,
        memory_manager: Optional[Any] = None,
        memory_session_id: Optional[str] = None,
        usage_logger: Optional[Any] = None,
    ) -> None:
        return None

    def is_healthy(self) -> bool:
        return True


# =============================================================================
# Legacy Mock Orchestrator satisfying ChatOrchestratorProtocol
# =============================================================================


@dataclass
class MockChatOrchestrator(ChatOrchestratorProtocol):
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

    def add_message(self, role: str, content: str, **metadata: Any) -> None:
        self.conversation.add_message(role, content, **metadata)

    def _check_context_overflow(self, max_context: int) -> bool:
        return len(self.messages) > 100  # Simple mock

    def _get_max_context_chars(self) -> int:
        return 128000  # Default GPT-4 context

    def _get_thinking_disabled_prompt(self, prompt: str) -> str:
        return f"{prompt} (thinking disabled)"

    # -- ToolContextProtocol --
    tool_selector: Any = field(default_factory=MockToolSelector)
    tool_adapter: Any = None
    _tool_planner: Any = None
    tool_budget: int = 10
    tool_calls_used: int = 0
    use_semantic_selection: bool = False
    observed_files: set = field(default_factory=set)

    async def execute_tool_calls(self, tool_calls: Any) -> List[Dict[str, Any]]:
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

    def create_recovery_context(self, stream_ctx: Any) -> Any:
        return {"mock": "recovery_context"}

    async def _handle_recovery_with_integration(
        self,
        stream_ctx: Any,
        full_content: str,
        tool_calls: Any,
        mentioned_tools: Any = None,
    ) -> Any:
        return {"mock": "recovery_action"}

    def _apply_recovery_action(self, recovery_action: Any, stream_ctx: Any) -> Any:
        return None

    def _record_runtime_intelligence_outcome(
        self, success: bool, quality_score: float, user_satisfied: bool, completed: bool
    ) -> None:
        pass

    async def _validate_runtime_intelligence_response(
        self, response: str, query: str, tool_calls: int, task_type: str
    ) -> Any:
        return {"valid": True}

    async def _prepare_runtime_intelligence_request(self, task: str, task_type: str) -> Any:
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


async def example_chat_service_protocol_with_mock():
    """Canonical example: use ChatServiceProtocol with a lightweight mock."""
    mock_service = MockChatService(
        responses=[type("MockResponse", (), {"content": "Hello from chat service!"})()]
    )

    result = await mock_service.chat("Hello, service!")

    print(f"Response: {result.content}")
    print(f"Healthy: {mock_service.is_healthy()}")


# =============================================================================
# Benefits of Protocol-Based Injection
# =============================================================================


def print_benefits():
    """Print benefits of protocol-based dependency injection."""

    benefits = """
    Benefits of Protocol-Based Injection During Chat Migration
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
       - Test ChatServiceProtocol / ChatService directly
       - Use protocol-based mocks for fast testing
       - Failures point to the active service runtime

    4. CLEAR CONTRACTS
       - ChatServiceProtocol is the canonical contract for new code
       - ChatOrchestratorProtocol remains useful for legacy integrations
       - @runtime_checkable keeps both boundaries explicit

    5. CLEAR RUNTIME OWNERSHIP
       - ChatService / ServiceStreamingRuntime own the live chat path
       - Service-first architecture with protocol boundaries

    6. FLEXIBLE MOCKS
       - Override only what you need for the test
       - Real implementation for unused methods
       - Easy to create scenario-specific mocks

    Example Test Speed Comparison:
    ------------------------------------
    With full AgentOrchestrator:  ~2-5 seconds per test
    With MockChatService:        ~10-50 milliseconds per test
    Speedup:                     40-500x faster

    Canonical Service-First Example:
    ------------------------------------
    async def test_chat_service():
        service = MockChatService()
        result = await service.chat("test")
        assert result.content == "Echo: test"

    """

    print(benefits)


def main() -> None:
    """Run the protocol-based injection example."""
    print_benefits()
    print("\n" + "=" * 60 + "\n")
    run_sync(example_chat_service_protocol_with_mock())


if __name__ == "__main__":
    main()
