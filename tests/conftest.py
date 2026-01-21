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

"""Shared pytest fixtures and configuration."""

import multiprocessing
import socket
import sys

import pytest


def is_ollama_available() -> bool:
    """Check if Ollama server is running at localhost:11434.

    Use this in tests that require Ollama with:
        @pytest.mark.skipif(not is_ollama_available(), reason="Ollama not available")

    Returns:
        True if Ollama is reachable, False otherwise.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", 11434))
        return result == 0
    finally:
        sock.close()


def requires_ollama():
    """Pytest marker to skip tests when Ollama is not available.

    Usage:
        @requires_ollama()
        def test_something():
            ...
    """
    return pytest.mark.skipif(
        not is_ollama_available(), reason="Ollama server not available at localhost:11434"
    )


from unittest.mock import MagicMock, patch

# On macOS, use 'spawn' start method to avoid semaphore leak warnings
# This is a known issue with Python multiprocessing on macOS
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("spawn", force=False)
    except RuntimeError:
        pass  # Already set


@pytest.fixture
def mock_code_execution_manager():
    """Mock CodeSandbox to avoid Docker startup during tests."""
    with patch("victor.tools.code_executor_tool.CodeSandbox") as mock_cem:
        mock_instance = MagicMock()
        mock_instance.start.return_value = None
        mock_instance.stop.return_value = None
        mock_instance.docker_available = False
        mock_instance.container = None
        mock_cem.return_value = mock_instance
        yield mock_cem


@pytest.fixture
def mock_docker_client():
    """Mock Docker client for tests that need Docker functionality."""
    with patch("docker.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_from_env.return_value = mock_client
        yield mock_client


@pytest.fixture(autouse=True)
def reset_singletons(request):
    """Reset singleton classifiers before each test for isolation.

    This prevents test pollution from cached singleton state, especially
    embedding services and classifiers that cache model instances.
    """
    # Skip reset for cache, workflow compiler, and server tests to avoid hanging during cleanup
    test_path = str(request.node.fspath)
    is_problematic_test = (
        "test_cache" in test_path
        or "test_unified_workflow_compiler" in test_path
        or "test_server_feature_parity" in test_path
    )

    def _reset_all():
        # Reset TaskTypeClassifier singleton
        try:
            from victor.storage.embeddings.task_classifier import TaskTypeClassifier

            TaskTypeClassifier.reset_instance()
        except ImportError:
            pass

        # Reset EmbeddingService singleton
        try:
            from victor.storage.embeddings.service import EmbeddingService

            EmbeddingService.reset_instance()
        except ImportError:
            pass

        # Reset IntentClassifier singleton
        try:
            from victor.storage.embeddings.intent_classifier import IntentClassifier

            IntentClassifier.reset_instance()
        except ImportError:
            pass

        # Reset SharedToolRegistry singleton
        try:
            from victor.agent.shared_tool_registry import SharedToolRegistry

            SharedToolRegistry.reset_instance()
        except ImportError:
            pass

        # Reset EventBus singleton (cancels pending async tasks to prevent leaks)
        try:
            from victor.observability.event_bus import EventBus

            EventBus.reset_instance()
        except ImportError:
            pass

        # Reset ProgressiveToolsRegistry singleton
        try:
            from victor.tools.progressive_registry import ProgressiveToolsRegistry

            ProgressiveToolsRegistry.reset_instance()
        except ImportError:
            pass

        # Reset VerticalRegistry (remove test verticals that may pollute other tests)
        # We only remove test-specific verticals (like test_vertical, mock_vertical, etc.)
        # and keep all production verticals intact
        try:
            from victor.core.verticals import VerticalRegistry

            # Remove test verticals (heuristic: names containing "test", "mock", "temp")
            test_vertical_names = [
                name
                for name in VerticalRegistry._registry.keys()
                if any(
                    keyword in name.lower() for keyword in ["test", "mock", "temp", "dummy", "fake"]
                )
            ]

            for name in test_vertical_names:
                VerticalRegistry.unregister(name)
        except ImportError:
            pass

        # Reset ToolSelectionCache singleton
        try:
            from victor.tools.caches.selection_cache import reset_tool_selection_cache

            reset_tool_selection_cache()
        except ImportError:
            pass

        # Reset VerticalBase config cache
        # Clear config cache to prevent stale config across tests
        try:
            from victor.core.verticals.base import VerticalBase

            VerticalBase.clear_config_cache(clear_all=True)
        except ImportError:
            pass

        # Reset UniversalRegistry singletons (clear all cached registries)
        # Prevents cross-test pollution from cached registry instances
        try:
            from victor.core.registries import UniversalRegistry

            UniversalRegistry._instances.clear()
        except ImportError:
            pass

        # Reset all registry type caches
        try:
            from victor.core.registries.universal_registry import CacheStrategy

            # Clear any thread-local or instance caches
            # Note: We don't clear individual registries here as they may be
            # legitimately used across tests
            pass
        except ImportError:
            pass

    # Reset before test
    if not is_problematic_test:
        _reset_all()

    yield

    # Reset after test
    if not is_problematic_test:
        _reset_all()


@pytest.fixture(autouse=True)
def isolate_environment_variables(monkeypatch):
    """Isolate tests from environment variables and .env files.

    This fixture prevents tests from loading actual API keys from:
    - Environment variables
    - .env files
    - System keyring
    - profiles.yaml

    This ensures tests are deterministic and don't leak credentials.
    """
    # Mock env file loading to prevent .env file from being loaded
    monkeypatch.setenv("VICTOR_SKIP_ENV_FILE", "1")

    # Clear API key environment variables
    api_key_vars = [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "XAI_API_KEY",
        "MOONSHOT_API_KEY",
        "DEEPSEEK_API_KEY",
        "GROQ_API_KEY",
        "VICTOR_ANTHROPIC_KEY",
        "VICTOR_GOOGLE_KEY",
        "VICTOR_OPENAI_KEY",
    ]

    for var in api_key_vars:
        monkeypatch.delenv(var, raising=False)

    # Mock the API key manager to return None for all providers
    monkeypatch.setattr("victor.config.api_keys.get_api_key", lambda provider: None)


@pytest.fixture(autouse=True)
def auto_mock_docker_for_orchestrator(request):
    """Automatically mock Docker for tests that create AgentOrchestrator.

    This fixture is auto-used but only applies mocking when the test
    fixture requests 'orchestrator' or matches certain patterns.
    """
    # Check if test needs orchestrator
    test_name = request.node.name
    test_path = str(request.node.fspath)

    # Tests that create orchestrator need Docker mocking
    needs_mock = any(
        [
            "orchestrator" in test_name.lower(),
            "tool_selection" in test_path,
            "tool_cache" in test_path,
            "goal_inference" in test_path,
            "tool_dependency" in test_path,
            "tool_call_matrix" in test_path,
            "thinking_mode" in test_path,
            "model_capability" in test_path,
            "test_orchestrator" in test_path,
            "integration" in test_path,
            "test_file_editor_tool.py" in test_path,
        ]
    )

    if needs_mock:
        with patch("victor.tools.code_executor_tool.CodeSandbox") as mock_cem:
            mock_instance = MagicMock()
            mock_instance.start.return_value = None
            mock_instance.stop.return_value = None
            mock_instance.docker_available = False
            mock_instance.container = None
            mock_cem.return_value = mock_instance
            yield
    else:
        yield


# ============ Workflow Fixtures ============


@pytest.fixture
def empty_workflow_graph():
    """Empty StateGraph workflow for testing.

    Creates an empty WorkflowGraph instance that can be populated with
    nodes and edges for testing workflow building logic.
    """
    from dataclasses import dataclass
    from victor.workflows.graph_dsl import WorkflowGraph, State

    @dataclass
    class TestState(State):
        """Minimal state for testing."""

        value: str = ""

    return WorkflowGraph(TestState, name="test_workflow")


@pytest.fixture
def linear_workflow_graph():
    """Linear A -> B -> C workflow graph.

    Creates a complete linear workflow with three nodes that can
    be used to test sequential execution patterns.
    """
    from dataclasses import dataclass
    from victor.workflows.graph_dsl import WorkflowGraph, State

    @dataclass
    class LinearState(State):
        """State for linear workflow."""

        value: str = ""
        step: int = 0

    def handler_a(state: LinearState) -> LinearState:
        state.step = 1
        state.value = "a"
        return state

    def handler_b(state: LinearState) -> LinearState:
        state.step = 2
        state.value += "->b"
        return state

    def handler_c(state: LinearState) -> LinearState:
        state.step = 3
        state.value += "->c"
        return state

    graph = WorkflowGraph(LinearState, name="linear")
    graph.add_node("a", handler_a)
    graph.add_node("b", handler_b)
    graph.add_node("c", handler_c)
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.set_entry_point("a")
    graph.set_finish_point("c")
    return graph


@pytest.fixture
def branching_workflow_graph():
    """Workflow graph with conditional branching.

    Creates a workflow with a decision node that routes to different
    branches based on state, useful for testing conditional routing.
    """
    from dataclasses import dataclass
    from victor.workflows.graph_dsl import WorkflowGraph, State

    @dataclass
    class BranchState(State):
        """State for branching workflow."""

        value: str = ""
        branch: str = "default"

    def router(state: BranchState) -> str:
        return state.branch

    graph = WorkflowGraph(BranchState, name="branching")
    graph.add_node("start", lambda s: s)
    graph.add_node("branch_a", lambda s: s)
    graph.add_node("branch_b", lambda s: s)
    graph.add_node("merge", lambda s: s)
    graph.set_entry_point("start")
    graph.add_conditional_edges(
        "start", router, {"a": "branch_a", "b": "branch_b", "default": "merge"}
    )
    graph.add_edge("branch_a", "merge")
    graph.add_edge("branch_b", "merge")
    graph.set_finish_point("merge")
    return graph


# ============ Multi-Agent Fixtures ============


@pytest.fixture
def mock_team_member():
    """Mock team member for testing.

    Creates a mock TeamMember with executor role for testing
    team coordination without actual agent execution.
    """
    from unittest.mock import AsyncMock, MagicMock
    from victor.agent.subagents.base import SubAgentRole

    member = MagicMock()
    member.id = "test_member"
    member.role = SubAgentRole.EXECUTOR
    member.name = "Test Executor"
    member.goal = "Test goal"
    member.tool_budget = 15
    member.is_manager = False
    member.priority = 0
    return member


@pytest.fixture
def team_member_specs():
    """List of TeamMemberSpec instances for testing.

    Provides a standard set of team member specifications
    representing a typical research/execute/review pipeline.
    """
    from victor.framework.teams import TeamMemberSpec

    return [
        TeamMemberSpec(role="researcher", goal="Research the codebase"),
        TeamMemberSpec(role="executor", goal="Implement changes"),
        TeamMemberSpec(role="reviewer", goal="Review the implementation"),
    ]


@pytest.fixture
def mock_team_coordinator():
    """Mock TeamCoordinator for testing without orchestrator.

    Creates a mock coordinator with stubbed methods for testing
    team-related functionality in isolation.
    """
    from unittest.mock import AsyncMock, MagicMock

    coordinator = MagicMock()
    coordinator.execute_team = AsyncMock(
        return_value=MagicMock(
            success=True,
            final_output="Test output",
            member_results={},
            total_tool_calls=5,
            total_duration=10.0,
        )
    )
    coordinator.set_progress_callback = MagicMock()
    return coordinator


# ============ HITL Fixtures ============


@pytest.fixture
def hitl_executor():
    """HITLExecutor for testing human-in-the-loop workflows.

    Creates an executor with a default handler that can be
    used to test HITL node execution flow.
    """
    from victor.workflows.hitl import HITLExecutor, DefaultHITLHandler

    return HITLExecutor(handler=DefaultHITLHandler())


@pytest.fixture
def auto_approve_handler():
    """Handler that auto-approves all HITL requests for testing.

    Returns a handler function that automatically approves all
    requests, useful for testing workflow flow without interruption.
    """
    from victor.workflows.hitl import HITLResponse, HITLStatus

    async def handler(request):
        return HITLResponse(
            request_id=request.request_id,
            status=HITLStatus.APPROVED,
            approved=True,
        )

    return handler


@pytest.fixture
def auto_reject_handler():
    """Handler that auto-rejects all HITL requests for testing.

    Returns a handler function that automatically rejects all
    requests, useful for testing rejection flow handling.
    """
    from victor.workflows.hitl import HITLResponse, HITLStatus

    async def handler(request):
        return HITLResponse(
            request_id=request.request_id,
            status=HITLStatus.REJECTED,
            approved=False,
            reason="Auto-rejected for testing",
        )

    return handler


@pytest.fixture
def hitl_approval_node():
    """Pre-configured HITLNode for approval testing."""
    from victor.workflows.hitl import HITLNode, HITLNodeType, HITLFallback

    return HITLNode(
        id="test_approval",
        name="Test Approval",
        hitl_type=HITLNodeType.APPROVAL,
        prompt="Approve this action?",
        timeout=5.0,
        fallback=HITLFallback.ABORT,
    )


# ============ Mode Config Fixtures ============


@pytest.fixture
def default_mode_config():
    """Default mode configuration for testing.

    Returns a standard mode definition with typical settings.
    """
    from victor.core.mode_config import ModeDefinition

    return ModeDefinition(
        name="test_mode",
        tool_budget=20,
        max_iterations=40,
        temperature=0.7,
        description="Test mode for unit tests",
    )


@pytest.fixture
def mode_config_registry():
    """Fresh ModeConfigRegistry for testing.

    Creates a new registry instance (not the singleton) to allow
    isolated testing of mode configuration logic.
    """
    from victor.core.mode_config import ModeConfigRegistry

    # Create a fresh instance, bypassing singleton
    registry = ModeConfigRegistry()
    return registry


@pytest.fixture
def registered_mode_registry():
    """ModeConfigRegistry with test verticals registered.

    Creates a registry with pre-configured test verticals for
    testing vertical-specific mode lookups.
    """
    from victor.core.mode_config import ModeConfigRegistry, ModeDefinition

    registry = ModeConfigRegistry()
    registry.register_vertical(
        "test_vertical",
        modes={
            "custom": ModeDefinition(
                name="custom",
                tool_budget=25,
                max_iterations=50,
                description="Custom test mode",
            ),
        },
        task_budgets={"test_task": 15, "complex_task": 30},
        default_mode="standard",
        default_budget=12,
    )
    return registry


# ============ Utility Fixtures ============


@pytest.fixture
def temp_workflow_context():
    """Temporary workflow context dictionary for testing.

    Provides a basic context structure that can be passed to
    workflow nodes during execution.
    """
    return {
        "input": "test input",
        "files": [],
        "analysis": None,
        "output": None,
    }


# ============ Provider Fixtures ============


@pytest.fixture
def mock_completion_response():
    """Mock CompletionResponse for testing.

    Returns a factory function that creates mock responses.
    """
    from tests.utils.test_helpers import create_test_completion_response

    return create_test_completion_response


@pytest.fixture
def mock_provider():
    """Mock LLM provider for testing.

    Creates a provider with pre-configured async methods.
    """
    from tests.utils.test_helpers import create_mock_provider

    return create_mock_provider()


@pytest.fixture
def mock_streaming_provider():
    """Mock streaming provider for testing.

    Creates a provider that supports streaming responses.
    """
    from tests.utils.test_helpers import create_mock_provider

    return create_mock_provider(supports_streaming=True)


@pytest.fixture
def mock_tool_provider():
    """Mock tool-calling provider for testing.

    Creates a provider that supports tool calling.
    """
    from tests.utils.test_helpers import create_mock_provider

    return create_mock_provider(supports_tools=True)


@pytest.fixture
def mock_messages():
    """Mock messages for testing.

    Creates a standard set of test messages.
    """
    from tests.utils.test_helpers import create_test_messages

    return create_test_messages()


@pytest.fixture
def mock_conversation():
    """Mock conversation history for testing.

    Creates a multi-turn conversation.
    """
    from tests.utils.test_helpers import create_test_conversation

    return create_test_conversation()


# ============ Orchestrator Fixtures ============


@pytest.fixture
def mock_orchestrator():
    """Mock orchestrator for testing.

    Creates a mock orchestrator with common methods stubbed.
    """
    from tests.utils.test_helpers import create_mock_orchestrator

    return create_mock_orchestrator()


@pytest.fixture
def mock_orchestrator_with_provider(mock_provider):
    """Mock orchestrator with a specific provider.

    Creates an orchestrator using the given mock provider.
    """
    from tests.utils.test_helpers import create_mock_orchestrator

    return create_mock_orchestrator(provider=mock_provider)


# ============ Tool Registry Fixtures ============


@pytest.fixture
def mock_tool_registry():
    """Mock tool registry for testing.

    Creates a registry with common methods stubbed.
    """
    from tests.utils.test_helpers import create_mock_tool_registry

    return create_mock_tool_registry()


@pytest.fixture
def mock_tool_registry_with_tools():
    """Mock tool registry with pre-configured tools.

    Creates a registry with a set of test tools.
    """
    from tests.utils.test_helpers import create_mock_tool_registry

    tools = ["read", "write", "grep", "bash"]
    return create_mock_tool_registry(tools=tools)


# ============ Event Bus Fixtures ============


@pytest.fixture
def mock_event_bus():
    """Mock event bus for testing.

    Creates an event bus with async methods stubbed.
    """
    from tests.utils.test_helpers import create_mock_event_bus

    return create_mock_event_bus()


# ============ Settings Fixtures ============


@pytest.fixture
def mock_settings():
    """Mock settings object for testing.

    Creates a settings object with common configuration.
    """
    from tests.utils.test_helpers import create_test_settings

    return create_test_settings()


# ============ Tool Definition Fixtures ============


@pytest.fixture
def mock_tool_definition():
    """Mock tool definition for testing.

    Creates a standard tool definition structure.
    """
    from tests.utils.test_helpers import create_test_tool_definition

    return create_test_tool_definition()


@pytest.fixture
def mock_tool_definitions():
    """List of mock tool definitions for testing.

    Creates multiple tool definitions.
    """
    from tests.utils.test_helpers import create_test_tool_definition

    return [
        create_test_tool_definition(name="read", description="Read a file"),
        create_test_tool_definition(name="write", description="Write a file"),
        create_test_tool_definition(name="grep", description="Search for text"),
    ]


# ============ Test Data Fixtures ============


@pytest.fixture
def sample_codebase_path(tmp_path):
    """Create a sample codebase structure for testing.

    Creates a temporary directory with common project files.
    """
    import os

    # Create directory structure
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "docs").mkdir()

    # Create sample files
    (tmp_path / "README.md").write_text("# Test Project\n\nThis is a test project.")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")
    (tmp_path / ".gitignore").write_text("*.pyc\n__pycache__/")

    # Create Python source file
    (tmp_path / "src" / "main.py").write_text(
        """
def hello():
    print("Hello, World!")

if __name__ == "__main__":
    hello()
"""
    )

    # Create test file
    (tmp_path / "tests" / "test_main.py").write_text(
        """
def test_hello():
    assert True
"""
    )

    return str(tmp_path)


@pytest.fixture
def sample_python_file(tmp_path):
    """Create a sample Python file for testing.

    Creates a temporary Python file with sample code.
    """
    python_file = tmp_path / "sample.py"
    python_file.write_text(
        '''
"""Sample Python module for testing."""

from typing import List


class SampleClass:
    """A sample class for testing."""

    def __init__(self, name: str):
        """Initialize the sample class."""
        self.name = name

    def greet(self) -> str:
        """Return a greeting message."""
        return f"Hello, {self.name}!"

    def process_items(self, items: List[str]) -> List[str]:
        """Process a list of items."""
        return [item.upper() for item in items]


def sample_function(x: int, y: int) -> int:
    """Add two numbers together."""
    return x + y


if __name__ == "__main__":
    cls = SampleClass("World")
    print(cls.greet())
    print(sample_function(1, 2))
'''
    )
    return str(python_file)


# ============ Network Mocking Fixtures ============


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for testing.

    Mocks httpx.Client to prevent actual network calls.
    """
    from unittest.mock import patch, MagicMock

    with patch("httpx.AsyncClient") as mock_client:
        client = MagicMock()
        mock_client.return_value = client
        yield client


@pytest.fixture
def mock_network(request):
    """Mock network access for testing.

    Automatically applies to tests marked with 'requires_network'.
    """
    import httpx
    from unittest.mock import patch, AsyncMock

    # Only apply if test has the marker
    if "requires_network" in request.keywords:
        with patch("httpx.AsyncClient") as mock_client:
            client = AsyncMock()
            mock_client.return_value = client
            yield mock_client
    else:
        yield None


# ============ Performance Testing Fixtures ============


@pytest.fixture
def performance_threshold():
    """Performance threshold for benchmark tests.

    Returns default threshold in seconds.
    """
    return 1.0  # 1 second default threshold


@pytest.fixture
def measure_time():
    """Context manager to measure execution time.

    Usage:
        with measure_time() as timer:
            # do work
        elapsed = timer.elapsed
    """
    import time

    class TimeTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.elapsed = None

        def __enter__(self):
            self.start_time = time.time()
            return self

        def __exit__(self, *args):
            self.end_time = time.time()
            self.elapsed = self.end_time - self.start_time

    return TimeTimer


# ============ Database Fixtures ============


@pytest.fixture
def mock_database():
    """Mock database connection for testing.

    Creates a mock database without requiring actual DB.
    """
    from unittest.mock import AsyncMock, MagicMock

    db = MagicMock()
    db.connect = AsyncMock()
    db.disconnect = AsyncMock()
    db.execute = AsyncMock(return_value=[])
    db.fetch_one = AsyncMock(return_value=None)
    db.fetch_all = AsyncMock(return_value=[])
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    return db


@pytest.fixture
def mock_cache():
    """Mock cache for testing.

    Creates an in-memory cache for testing.
    """
    from unittest.mock import AsyncMock, MagicMock

    cache = MagicMock()
    cache._data = {}

    def get_mock(key):
        return cache._data.get(key)

    def set_mock(key, value):
        cache._data[key] = value

    cache.get = MagicMock(side_effect=get_mock)
    cache.set = MagicMock(side_effect=set_mock)
    cache.delete = AsyncMock()
    cache.clear = MagicMock(side_effect=lambda: cache._data.clear())
    cache.exists = MagicMock(side_effect=lambda k: k in cache._data)

    return cache


# ============ Assertion Helper Fixtures ============


@pytest.fixture
def assert_valid_completion():
    """Fixture that provides completion validation function."""
    from tests.utils.test_helpers import assert_completion_valid

    return assert_completion_valid


@pytest.fixture
def assert_provider_called_with():
    """Fixture that provides provider call assertion function."""
    from tests.utils.test_helpers import assert_provider_called

    return assert_provider_called_with


# ============ Integration Test Fixtures ============


@pytest.fixture
def integration_test_config():
    """Configuration for integration tests.

    Provides common settings for integration tests.
    """
    return {
        "timeout": 30,
        "retry_attempts": 3,
        "retry_delay": 1.0,
        "cleanup": True,
        "verbose": True,
    }


@pytest.fixture
def test_environment_setup(tmp_path):
    """Set up a complete test environment.

    Creates a temporary directory with configuration, source code,
    and test files for integration testing.
    """
    import os

    # Create project structure
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    # Create .victor directory
    (project_dir / ".victor").mkdir()

    # Create config file
    config_content = """
provider: anthropic
model: claude-sonnet-4-5
temperature: 0.7
max_tokens: 4096
"""
    (project_dir / ".victor" / "config.yaml").write_text(config_content)

    # Create source directory
    src_dir = project_dir / "src"
    src_dir.mkdir()

    # Create sample source file
    (src_dir / "main.py").write_text(
        """
def main():
    print("Hello from test project!")

if __name__ == "__main__":
    main()
"""
    )

    # Create tests directory
    tests_dir = project_dir / "tests"
    tests_dir.mkdir()

    (tests_dir / "test_main.py").write_text(
        """
def test_main():
    assert True
"""
    )

    # Change to project directory
    original_cwd = os.getcwd()
    os.chdir(str(project_dir))

    yield str(project_dir)

    # Cleanup: restore original directory
    os.chdir(original_cwd)
