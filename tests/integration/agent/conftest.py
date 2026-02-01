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

"""Fixtures for orchestrator integration tests.

Provides mock implementations of LLM providers, tools, settings,
and both legacy and refactored orchestrator instances.
"""

import asyncio
import os
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers.

    Adds the 'requires_coordinator' marker for tests that require
    the coordinator-based orchestrator (USE_COORDINATOR_ORCHESTRATOR=true).
    """
    use_coordinator = os.getenv("USE_COORDINATOR_ORCHESTRATOR", "false").lower() == "true"

    # Add marker documentation
    config.addinivalue_line(
        "markers",
        "requires_coordinator: marks tests requiring coordinator-based orchestrator "
        "(run with USE_COORDINATOR_ORCHESTRATOR=true)",
    )


# ============================================================================
# Basic Mocks (Settings, Provider, Container)
# ============================================================================


@pytest.fixture
def test_settings():
    """Create test settings for orchestrator creation.

    Returns a mock Settings object with typical configuration.
    """
    settings = MagicMock()
    settings.temperature = 0.7
    settings.max_tokens = 4096
    settings.thinking = False
    settings.tool_selection = {"strategy": "hybrid"}
    settings.provider = "anthropic"
    settings.model = "claude-sonnet-4-5"
    settings.timeout = 120.0
    settings.stream = False
    return settings


@pytest.fixture
def test_provider():
    """Create mock LLM provider for testing.

    Returns a mock BaseProvider with chat and streaming capabilities.
    """
    provider = MagicMock()
    provider.name = "anthropic"
    provider.model = "claude-sonnet-4-5"

    # Mock supports_tools
    provider.supports_tools = MagicMock(return_value=True)

    # Mock chat response
    async def mock_chat(messages, **kwargs):
        return MagicMock(
            content="Test response from LLM",
            usage=MagicMock(input_tokens=100, output_tokens=50),
            tool_calls=None,  # No tool calls by default
        )

    provider.chat = AsyncMock(side_effect=mock_chat)

    # Mock streaming response
    async def mock_stream_chat(messages, **kwargs):
        chunks = [
            MagicMock(content="Test", delta="Test", usage=None),
            MagicMock(content=" response", delta=" response", usage=None),
            MagicMock(
                content="",
                delta="",
                usage=MagicMock(input_tokens=100, output_tokens=50),
            ),
        ]
        for chunk in chunks:
            yield chunk

    provider.stream_chat = mock_stream_chat

    return provider


@pytest.fixture
def test_container():
    """Create mock ServiceContainer for testing.

    Returns a mock container with service resolution for common dependencies.
    """
    from victor.tools.base import CostTier

    container = MagicMock()

    # Mock get_service method
    def get_service(service_type):
        service_str = str(service_type)

        # ToolPipeline
        if "ToolPipeline" in service_str:
            pipeline = MagicMock()
            pipeline.execute_tool_calls = AsyncMock(return_value=[])
            return pipeline

        # ConversationController
        elif "ConversationController" in service_str:
            controller = MagicMock()
            controller.add_message = MagicMock()
            controller.get_messages = MagicMock(return_value=[])
            controller.clear = MagicMock()
            return controller

        # StreamingController
        elif "StreamingController" in service_str:
            return MagicMock()

        # ToolRegistrar
        elif "ToolRegistrar" in service_str:
            registrar = MagicMock()
            registrar.get_all_tools = MagicMock(
                return_value={
                    "test_tool": MagicMock(
                        name="test_tool",
                        description="A test tool",
                        parameters={
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                            },
                        },
                        cost_tier=CostTier.FREE,
                    )
                }
            )
            return registrar

        # MemoryManager
        elif "MemoryManager" in service_str:
            return MagicMock()

        # ContextCompactor
        elif "ContextCompactor" in service_str:
            compactor = MagicMock()
            compactor.compact = AsyncMock(side_effect=lambda conv, max_tokens: conv)
            return compactor

        # UsageAnalytics
        elif "UsageAnalytics" in service_str:
            return MagicMock()

        # ToolSequenceTracker
        elif "ToolSequenceTracker" in service_str:
            tracker = MagicMock()
            tracker.add_tool_call = MagicMock()
            tracker.get_sequence = MagicMock(return_value=[])
            return tracker

        # MetricsCollector
        elif "MetricsCollector" in service_str:
            return MagicMock()

        # TaskAnalyzer
        elif "TaskAnalyzer" in service_str:
            return MagicMock()

        # ToolSelector
        elif "ToolSelector" in service_str:
            return MagicMock()

        # SearchRouter
        elif "SearchRouter" in service_str:
            return MagicMock()

        return None

    container.get_service = MagicMock(side_effect=get_service)
    return container


# ============================================================================
# Orchestrator Factories and Instances
# ============================================================================


@pytest.fixture
def legacy_orchestrator_factory(test_settings, test_provider, test_container):
    """Create legacy OrchestratorFactory for testing.

    Returns a factory that creates legacy (non-coordinator) orchestrators.
    """
    from victor.agent.orchestrator_factory import OrchestratorFactory

    factory = OrchestratorFactory(
        settings=test_settings,
        provider=test_provider,
        model="claude-sonnet-4-5",
    )
    # Attach container for later use
    factory._test_container = test_container
    return factory


@pytest.fixture
def legacy_orchestrator(legacy_orchestrator_factory, test_container):
    """Create legacy orchestrator instance for testing.

    Returns an AgentOrchestrator instance without coordinators.
    """

    # Access factory attributes using public properties
    settings = legacy_orchestrator_factory.settings
    provider = legacy_orchestrator_factory.provider
    model = legacy_orchestrator_factory.model

    # Create a simple mock orchestrator for testing
    # (Full orchestrator creation requires many dependencies)
    orchestrator = MagicMock()
    orchestrator.settings = settings
    orchestrator.provider = provider
    orchestrator.model = model
    orchestrator.session_id = "test_session_legacy"
    orchestrator.active_session_id = "test_session_legacy"

    return orchestrator


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def mock_message():
    """Create a simple test message."""
    return "Hello, this is a test message!"


@pytest.fixture
def test_conversation_history():
    """Create test conversation history for testing.

    Returns a list of messages representing a typical conversation.
    """
    return [
        {"role": "user", "content": "Hello, can you help me?"},
        {"role": "assistant", "content": "Of course! How can I assist you today?"},
        {"role": "user", "content": "I need help with debugging my code."},
        {"role": "assistant", "content": "I'd be happy to help with debugging."},
    ]


@pytest.fixture
def test_conversation_long():
    """Create long conversation history for context compaction testing.

    Returns a list of messages that exceeds typical context budget.
    """
    messages = []
    for i in range(50):
        messages.append(
            {
                "role": "user",
                "content": f"This is message number {i} with some content to fill space.",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": f"Response number {i} with helpful information and debugging tips.",
            }
        )
    return messages


@pytest.fixture
def test_analytics_events():
    """Create test analytics events for testing.

    Returns a list of sample analytics events.
    """
    from victor.protocols import AnalyticsEvent

    return [
        AnalyticsEvent(
            event_type="tool_call",
            data={"tool": "read_file", "file_path": "/src/main.py", "duration": 0.5},
            timestamp=datetime.utcnow().isoformat(),
            session_id="test_session",
        ),
        AnalyticsEvent(
            event_type="tool_call",
            data={"tool": "write_file", "file_path": "/src/test.py", "duration": 1.2},
            timestamp=datetime.utcnow().isoformat(),
            session_id="test_session",
        ),
        AnalyticsEvent(
            event_type="llm_call",
            data={
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "tokens": 150,
            },
            timestamp=datetime.utcnow().isoformat(),
            session_id="test_session",
        ),
        AnalyticsEvent(
            event_type="context_compaction",
            data={
                "original_tokens": 5000,
                "compacted_tokens": 2500,
                "strategy": "truncation",
            },
            timestamp=datetime.utcnow().isoformat(),
            session_id="test_session",
        ),
    ]


@pytest.fixture
def test_session_id():
    """Generate unique test session ID."""
    import uuid

    return f"test_session_{uuid.uuid4().hex[:8]}"


# ============================================================================
# Mock Tools
# ============================================================================


@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    return [
        {"role": "user", "content": "Hello, can you help me?"},
        {"role": "assistant", "content": "Of course! How can I assist you today?"},
        {"role": "user", "content": "I need help with debugging my code."},
    ]


@pytest.fixture
def mock_tools():
    """Create multiple mock tools for testing."""

    read_tool = MagicMock()
    read_tool.name = "read_file"
    read_tool.execute = AsyncMock(return_value={"content": "file content"})

    search_tool = MagicMock()
    search_tool.name = "search"
    search_tool.execute = AsyncMock(return_value={"results": []})

    return {"read_file": read_tool, "search": search_tool}


@pytest.fixture
def mock_tool():
    """Create a mock tool for testing.

    Returns a mock BaseTool implementation.
    """
    from victor.tools.base import CostTier

    tool = MagicMock()
    tool.name = "test_tool"
    tool.description = "A test tool for integration testing"
    tool.parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
        },
    }
    tool.cost_tier = CostTier.FREE

    # Mock execute
    async def mock_execute(**kwargs):
        return {"result": "Tool executed successfully", "kwargs": kwargs}

    tool.execute = AsyncMock(side_effect=mock_execute)

    return tool


@pytest.fixture
def mock_tool_registry():
    """Create mock tool registry for testing.

    Returns a mock tool registry with sample tools.
    """
    from victor.tools.base import CostTier

    registry = MagicMock()

    # Mock get_all_tools
    def get_all_tools():
        return {
            "read_file": MagicMock(
                name="read_file",
                description="Read file contents",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
                cost_tier=CostTier.FREE,
            ),
            "write_file": MagicMock(
                name="write_file",
                description="Write to file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
                cost_tier=CostTier.FREE,
            ),
            "execute_bash": MagicMock(
                name="execute_bash",
                description="Execute bash command",
                parameters={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                },
                cost_tier=CostTier.LOW,
            ),
        }

    registry.get_all_tools = MagicMock(side_effect=get_all_tools)
    return registry


# ============================================================================
# Mock Coordinators
# ============================================================================


@pytest.fixture
def mock_config_coordinator():
    """Create mock ConfigCoordinator for testing."""
    config_coord = MagicMock()

    async def mock_load_config(session_id):
        return {
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
            "temperature": 0.7,
            "max_tokens": 4096,
        }

    async def mock_validate_config(config):
        from victor.agent.coordinators.config_coordinator import ValidationResult

        return ValidationResult(valid=True, errors=[])

    config_coord.load_config = AsyncMock(side_effect=mock_load_config)
    config_coord.validate_config = AsyncMock(side_effect=mock_validate_config)

    return config_coord


@pytest.fixture
def mock_prompt_coordinator():
    """Create mock PromptCoordinator for testing."""
    from victor.protocols import PromptContext

    prompt_coord = MagicMock()

    async def mock_build_system_prompt(context: PromptContext):
        return "You are a helpful AI assistant for testing."

    async def mock_build_task_hint(task, mode="build"):
        return f"Task: {task}, Mode: {mode}"

    prompt_coord.build_system_prompt = AsyncMock(side_effect=mock_build_system_prompt)
    prompt_coord.build_task_hint = AsyncMock(side_effect=mock_build_task_hint)

    return prompt_coord


@pytest.fixture
def mock_context_coordinator():
    """Create mock ContextCoordinator for testing."""
    from victor.protocols import CompactionContext, ContextBudget, CompactionResult

    context_coord = MagicMock()

    async def mock_is_within_budget(context: CompactionContext, budget: ContextBudget):
        return context.get("token_count", 0) < budget.get("max_tokens", 4096)

    async def mock_compact_context(context: CompactionContext, budget: ContextBudget):
        messages = context.get("messages", [])
        compacted = messages[-10:]  # Keep last 10
        return CompactionResult(
            compacted_context={**context, "messages": compacted},
            tokens_saved=context.get("token_count", 0) - 1000,
            messages_removed=len(messages) - len(compacted),
            strategy_used="truncation",
            metadata={"reserve_messages": 10},
        )

    context_coord.is_within_budget = AsyncMock(side_effect=mock_is_within_budget)
    context_coord.compact_context = AsyncMock(side_effect=mock_compact_context)

    return context_coord


@pytest.fixture
def mock_analytics_coordinator():
    """Create mock AnalyticsCoordinator for testing."""
    from victor.protocols import AnalyticsEvent, ExportResult

    analytics_coord = MagicMock()
    analytics_coord._events = []

    async def mock_track_event(event: AnalyticsEvent):
        analytics_coord._events.append(event)

    async def mock_export_analytics():
        return ExportResult(
            success=True,
            exporter_type="mock",
            records_exported=len(analytics_coord._events),
        )

    async def mock_query_analytics(query):
        # Filter events by query
        events = analytics_coord._events
        if query.session_id:
            events = [e for e in events if e.session_id == query.session_id]
        if query.event_types:
            events = [e for e in events if e.event_type in query.event_types]
        return events

    analytics_coord.track_event = AsyncMock(side_effect=mock_track_event)
    analytics_coord.export_analytics = AsyncMock(side_effect=mock_export_analytics)
    analytics_coord.query_analytics = AsyncMock(side_effect=mock_query_analytics)

    return analytics_coord


# ============================================================================
# Environment and Temporary Fixtures
# ============================================================================


@pytest.fixture
def temp_workspace(tmp_path):
    """Create temporary workspace for file-based testing.

    Creates a temporary directory structure for testing file operations.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Create test files
    (workspace / "main.py").write_text("print('Hello, World!')\n")
    (workspace / "test.py").write_text("def test_hello():\n    assert True\n")

    # Create subdirectory
    src_dir = workspace / "src"
    src_dir.mkdir()
    (src_dir / "utils.py").write_text("def helper():\n    pass\n")

    return workspace


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set mock environment variables for testing."""
    monkeypatch.setenv("VICTOR_PROVIDER", "anthropic")
    monkeypatch.setenv("VICTOR_MODEL", "claude-sonnet-4-5")
    monkeypatch.setenv("VICTOR_TEMPERATURE", "0.7")
    monkeypatch.setenv("VICTOR_MAX_TOKENS", "4096")


# ============================================================================
# Performance Monitoring
# ============================================================================


@pytest.fixture
def performance_monitor():
    """Monitor performance of coordinator operations.

    Returns a performance monitor for tracking coordinator overhead.
    """
    import time

    class PerformanceMonitor:
        def __init__(self):
            self.metrics: dict[str, list[float]] = {}

        def track(self, operation: str):
            """Decorator to track operation timing."""

            def decorator(func):
                async def wrapper(*args, **kwargs):
                    start = time.time()
                    result = await func(*args, **kwargs)
                    duration = time.time() - start
                    if operation not in self.metrics:
                        self.metrics[operation] = []
                    self.metrics[operation].append(duration)
                    return result

                return wrapper

            return decorator

        def get_average_time(self, operation: str) -> float:
            """Get average execution time for an operation."""
            if operation not in self.metrics or not self.metrics[operation]:
                return 0.0
            return sum(self.metrics[operation]) / len(self.metrics[operation])

        def get_overhead_percentage(self, base_time: float, operation: str) -> float:
            """Calculate coordinator overhead as percentage."""
            avg_time = self.get_average_time(operation)
            if avg_time == 0:
                return 0.0
            return ((avg_time - base_time) / base_time) * 100

    return PerformanceMonitor()


# ============================================================================
# Test Helpers
# ============================================================================


@pytest.fixture
def test_helpers():
    """Helper functions for integration testing.

    Returns a collection of utility functions for testing.
    """

    class Helpers:
        @staticmethod
        async def wait_for_condition(condition, timeout: float = 5.0, interval: float = 0.1):
            """Wait for a condition to become true."""
            start = asyncio.get_event_loop().time()
            while await condition() if asyncio.iscoroutinefunction(condition) else condition():
                if asyncio.get_event_loop().time() - start >= timeout:
                    raise TimeoutError(f"Condition not met within {timeout}s")
                await asyncio.sleep(interval)

        @staticmethod
        def create_mock_message(role: str, content: str) -> dict[str, Any]:
            """Create a mock message for testing."""
            return {"role": role, "content": content}

        @staticmethod
        def create_mock_tool_call(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            """Create a mock tool call for testing."""
            return {
                "id": f"call_{tool_name}_123",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": arguments,
                },
            }

        @staticmethod
        def estimate_tokens(text: str) -> int:
            """Estimate token count (rough approximation)."""
            return int(len(text.split()) * 1.3)  # Rough estimate

    return Helpers()
