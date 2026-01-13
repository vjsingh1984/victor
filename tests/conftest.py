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
def reset_singletons():
    """Reset singleton classifiers before each test for isolation.

    This prevents test pollution from cached singleton state, especially
    embedding services and classifiers that cache model instances.
    """

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
                name for name in VerticalRegistry._registry.keys()
                if any(keyword in name.lower() for keyword in ["test", "mock", "temp", "dummy", "fake"])
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

    # Reset before test
    _reset_all()

    yield

    # Reset after test
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
