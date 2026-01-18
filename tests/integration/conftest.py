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

"""Integration test fixtures for refactored orchestrator tests.

These fixtures provide mock implementations and test utilities for testing
the coordinator-based orchestrator architecture.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest


@pytest.fixture
def mock_settings():
    """Mock Settings object for testing.

    Creates a mock Settings with typical configuration values.
    """
    settings = MagicMock()
    settings.temperature = 0.7
    settings.max_tokens = 4096
    settings.thinking = False
    settings.tool_selection = {"strategy": "hybrid"}
    settings.provider = "anthropic"
    settings.model = "claude-sonnet-4-5"
    return settings


@pytest.fixture
def mock_provider():
    """Mock BaseProvider for testing.

    Creates a mock provider with chat and streaming capabilities.
    """
    provider = MagicMock()
    provider.name = "anthropic"
    provider.supports_tools = MagicMock(return_value=True)

    # Mock chat response
    async def mock_chat(messages, **kwargs):
        return MagicMock(
            content="Test response",
            usage=MagicMock(input_tokens=100, output_tokens=50),
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
def mock_container():
    """Mock ServiceContainer for testing.

    Creates a mock container with service resolution.
    """
    container = MagicMock()

    # Mock get_service method
    def get_service(service_type):
        if "ToolPipeline" in str(service_type):
            return MagicMock(execute_tool_calls=AsyncMock(return_value=[]))
        elif "ConversationController" in str(service_type):
            return MagicMock()
        return None

    container.get_service = MagicMock(side_effect=get_service)
    return container


@pytest.fixture
def test_session_id():
    """Test session ID for isolation."""
    return "test_session_12345"


@pytest.fixture
def test_conversation_history():
    """Create test conversation history for testing.

    Returns a list of messages representing a typical conversation.
    """
    return [
        {"role": "user", "content": "Hello, can you help me?"},
        {"role": "assistant", "content": "Of course! How can I assist you today?"},
        {"role": "user", "content": "I need help with debugging my code."},
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
def mock_prompt_contributors():
    """Create mock prompt contributors for testing.

    Returns a list of mock IPromptContributor implementations.
    """
    contributors = []

    # System prompt contributor
    system_contributor = MagicMock()
    system_contributor.priority = MagicMock(return_value=100)
    system_contributor.contribute = AsyncMock(return_value="You are a helpful AI assistant.")
    contributors.append(system_contributor)

    # Task hint contributor
    task_contributor = MagicMock()
    task_contributor.priority = MagicMock(return_value=75)
    task_contributor.contribute = AsyncMock(return_value="Focus on clarity and accuracy.")
    contributors.append(task_contributor)

    return contributors


@pytest.fixture
def mock_compaction_strategies():
    """Create mock compaction strategies for testing.

    Returns a list of mock ICompactionStrategy implementations.
    """
    from victor.protocols import CompactionResult, CompactionContext, ContextBudget

    strategies = []

    # Strategy 1: Truncation
    truncation_strategy = MagicMock()
    truncation_strategy.__class__.__name__ = "TruncationCompactionStrategy"

    async def mock_truncation_can_apply(context: CompactionContext, budget: ContextBudget):
        return context.get("token_count", 0) > budget.get("max_tokens", 4096)

    async def mock_truncation_compact(context: CompactionContext, budget: ContextBudget):
        messages = context.get("messages", [])
        compacted = messages[-10:]  # Keep last 10 messages
        return CompactionResult(
            compacted_context={**context, "messages": compacted, "token_count": 2000},
            tokens_saved=context.get("token_count", 0) - 2000,
            messages_removed=len(messages) - len(compacted),
            strategy_used="truncation",
            metadata={"reserve_messages": 10},
        )

    truncation_strategy.can_apply = AsyncMock(side_effect=mock_truncation_can_apply)
    truncation_strategy.compact = AsyncMock(side_effect=mock_truncation_compact)
    strategies.append(truncation_strategy)

    return strategies


@pytest.fixture
def mock_analytics_exporters():
    """Create mock analytics exporters for testing.

    Returns a list of mock IAnalyticsExporter implementations.
    """
    from victor.protocols import ExportResult

    exporters = []

    # Console exporter
    console_exporter = MagicMock()
    console_exporter.exporter_type = MagicMock(return_value="console")

    async def mock_console_export(data):
        return ExportResult(
            success=True,
            exporter_type="console",
            records_exported=len(data.get("events", [])),
        )

    console_exporter.export = AsyncMock(side_effect=mock_console_export)
    exporters.append(console_exporter)

    # File exporter
    file_exporter = MagicMock()
    file_exporter.exporter_type = MagicMock(return_value="file")

    async def mock_file_export(data):
        return ExportResult(
            success=True,
            exporter_type="file",
            records_exported=len(data.get("events", [])),
        )

    file_exporter.export = AsyncMock(side_effect=mock_file_export)
    exporters.append(file_exporter)

    return exporters


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
            data={"provider": "anthropic", "model": "claude-sonnet-4-5", "tokens": 150},
            timestamp=datetime.utcnow().isoformat(),
            session_id="test_session",
        ),
    ]


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
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
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


@pytest.fixture
def mock_context_compactor():
    """Create mock context compactor for testing.

    Returns a mock context compactor with typical behavior.
    """
    compactor = MagicMock()

    # Mock compact method
    async def mock_compact(conversation, max_tokens):
        # Simple truncation strategy
        messages = conversation.get("messages", [])
        if len(messages) > 10:
            compacted = {
                "messages": messages[-10:],
                "compacted": True,
                "original_count": len(messages),
            }
            return compacted
        return conversation

    compactor.compact = AsyncMock(side_effect=mock_compact)
    compactor.get_statistics = MagicMock(
        return_value={
            "total_compactions": 5,
            "total_tokens_saved": 15000,
            "last_compaction": datetime.utcnow().isoformat(),
        }
    )

    return compactor


@pytest.fixture
def performance_monitor():
    """Monitor performance of coordinator operations.

    Returns a performance monitor for tracking coordinator overhead.
    """
    import time

    class PerformanceMonitor:
        def __init__(self):
            self.metrics: Dict[str, List[float]] = {}

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

        def get_max_time(self, operation: str) -> float:
            """Get max execution time for an operation."""
            if operation not in self.metrics or not self.metrics[operation]:
                return 0.0
            return max(self.metrics[operation])

        def get_overhead_percentage(self, base_time: float, operation: str) -> float:
            """Calculate coordinator overhead as percentage."""
            avg_time = self.get_average_time(operation)
            if avg_time == 0:
                return 0.0
            return ((avg_time - base_time) / base_time) * 100

    return PerformanceMonitor()


@pytest.fixture
def integration_test_helpers():
    """Helper functions for integration testing.

    Returns a collection of utility functions for testing.
    """

    class Helpers:
        @staticmethod
        def assert_coordinator_interactions(
            orchestrator,
            expected_coordinators: List[str],
        ):
            """Assert that orchestrator has expected coordinators."""
            for coord_name in expected_coordinators:
                assert hasattr(
                    orchestrator, f"{coord_name}_coordinator"
                ), f"Missing {coord_name}_coordinator"

        @staticmethod
        async def wait_for_condition(
            condition,
            timeout: float = 5.0,
            interval: float = 0.1,
        ):
            """Wait for a condition to become true."""
            start = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start < timeout:
                if await condition() if asyncio.iscoroutinefunction(condition) else condition():
                    return True
                await asyncio.sleep(interval)
            raise TimeoutError(f"Condition not met within {timeout}s")

        @staticmethod
        def create_mock_message(role: str, content: str) -> Dict[str, Any]:
            """Create a mock message for testing."""
            return {"role": role, "content": content}

        @staticmethod
        def create_mock_tool_call(
            tool_name: str,
            arguments: Dict[str, Any],
        ) -> Dict[str, Any]:
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
            return len(text.split()) * 1.3  # Rough estimate

    return Helpers()
