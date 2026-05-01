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

"""Tests for tool helper services and the canonical tool surface."""

import importlib
import pytest

from victor.agent.services.tools import (
    ToolTrackerService,
    ToolTrackerServiceConfig,
    ToolSelectorService,
    ToolSelectorServiceConfig,
    ToolResultProcessor,
    ToolResultProcessorConfig,
)
from victor.agent.services.tool_service import ToolService, ToolServiceConfig
from victor.tools.base import BaseTool, ToolResult

# =============================================================================
# Mock Tools
# =============================================================================


class MockSearchTool(BaseTool):
    """Mock search tool."""

    name = "search"
    description = "Search for files"

    def parameters(self):
        return {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        }

    def execute(self, query: str):
        return ToolResult(success=True, output=f"Search results for: {query}")


class MockReadTool(BaseTool):
    """Mock read tool."""

    name = "read_file"
    description = "Read a file"

    def parameters(self):
        return {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path"}},
            "required": ["path"],
        }

    def execute(self, path: str):
        return ToolResult(success=True, output=f"File content from: {path}")


class MockToolSelector:
    """Mock selector compatible with canonical ToolService."""

    def __init__(self):
        self.enabled_tools = set()

    async def select(self, context, max_tools):
        available = list(getattr(context, "available_tools", []))
        return available[:max_tools]

    def set_enabled_tools(self, tools):
        self.enabled_tools = set(tools)


class MockToolExecutor:
    """Mock executor compatible with canonical ToolService."""

    def __init__(self, tool_registry):
        self.tool_registry = tool_registry

    async def execute(self, tool_name, arguments):
        return self.tool_registry[tool_name].execute(**arguments)


class MockToolRegistrar:
    """Mock registrar compatible with canonical ToolService."""

    def __init__(self, tool_registry):
        self.tool_registry = tool_registry

    def get_tool_names(self):
        return list(self.tool_registry.keys())

    def get_registered_tools(self):
        return list(self.tool_registry.keys())

    def list_tools(self):
        return list(self.tool_registry.values())

    def get_tool(self, tool_name):
        return self.tool_registry.get(tool_name)

    def get_all_tools(self):
        return list(self.tool_registry.values())


# =============================================================================
# ToolTrackerService Tests
# =============================================================================


class TestToolTrackerService:
    """Tests for ToolTrackerService."""

    def test_initialization(self):
        """Test service initialization."""
        config = ToolTrackerServiceConfig(initial_budget=50)
        tracker = ToolTrackerService(config)

        assert tracker.budget_limit == 50
        assert tracker.budget_used == 0
        assert tracker.get_remaining_budget() == 50

    def test_budget_consumption(self):
        """Test budget consumption."""
        config = ToolTrackerServiceConfig(initial_budget=10)
        tracker = ToolTrackerService(config)

        tracker.consume_budget(3)
        assert tracker.get_remaining_budget() == 7
        assert tracker.budget_used == 3

        tracker.consume_budget(5)
        assert tracker.get_remaining_budget() == 2

    def test_budget_exhausted(self):
        """Test budget exhaustion."""
        config = ToolTrackerServiceConfig(initial_budget=5)
        tracker = ToolTrackerService(config)

        assert tracker.is_budget_exhausted() is False

        tracker.consume_budget(5)
        assert tracker.is_budget_exhausted() is True

    def test_budget_reset(self):
        """Test budget reset."""
        config = ToolTrackerServiceConfig(initial_budget=10)
        tracker = ToolTrackerService(config)

        tracker.consume_budget(5)
        assert tracker.get_remaining_budget() == 5

        tracker.reset_tool_budget()
        assert tracker.get_remaining_budget() == 10
        assert tracker.budget_used == 0

    def test_usage_tracking(self):
        """Test usage tracking."""
        config = ToolTrackerServiceConfig()
        tracker = ToolTrackerService(config)

        # Record executions
        tracker.record_execution("search", success=True, duration_ms=100)
        tracker.record_execution("search", success=True, duration_ms=150)
        tracker.record_execution("read", success=False, duration_ms=50)

        # Check counts
        assert tracker.get_tool_call_count("search") == 2
        assert tracker.get_tool_call_count("read") == 1
        assert tracker.get_tool_error_count("search") == 0
        assert tracker.get_tool_error_count("read") == 1

    def test_usage_stats(self):
        """Test usage statistics."""
        config = ToolTrackerServiceConfig()
        tracker = ToolTrackerService(config)

        tracker.record_execution("search", success=True, duration_ms=100)
        tracker.record_execution("read", success=True, duration_ms=200)

        stats = tracker.get_tool_usage_stats()

        assert stats["total_calls"] == 2
        assert stats["total_errors"] == 0
        assert stats["success_rate"] == 100.0
        assert "average_durations_ms" in stats


# =============================================================================
# ToolSelectorService Tests
# =============================================================================


class TestToolSelectorService:
    """Tests for ToolSelectorService."""

    def test_initialization(self):
        """Test service initialization."""
        config = ToolSelectorServiceConfig()
        selector = ToolSelectorService(
            config=config,
            available_tools={"search", "read_file", "write_file"},
        )

        assert selector.available_tools == {"search", "read_file", "write_file"}

    def test_enable_disable_tools(self):
        """Test enabling and disabling tools."""
        config = ToolSelectorServiceConfig()
        selector = ToolSelectorService(
            config=config,
            available_tools={"search", "read_file"},
        )

        # Enable tools
        selector.enable_tool("search")
        assert selector.is_tool_enabled("search") is True
        assert selector.is_tool_enabled("read_file") is False

        # Disable tool
        selector.disable_tool("search")
        assert selector.is_tool_enabled("search") is False

    def test_set_enabled_tools(self):
        """Test setting enabled tools."""
        config = ToolSelectorServiceConfig()
        selector = ToolSelectorService(
            config=config,
            available_tools={"search", "read_file", "write_file"},
        )

        selector.set_enabled_tools({"search", "read_file"})

        assert selector.is_tool_enabled("search") is True
        assert selector.is_tool_enabled("read_file") is True
        assert selector.is_tool_enabled("write_file") is False

    def test_get_enabled_tools(self):
        """Test getting enabled tools."""
        config = ToolSelectorServiceConfig(default_enabled_tools={"search", "read_file"})
        selector = ToolSelectorService(
            config=config,
            available_tools={"search", "read_file", "write_file"},
        )

        enabled = selector.get_enabled_tools()
        assert enabled == {"search", "read_file"}

    @pytest.mark.asyncio
    async def test_select_tools(self):
        """Test tool selection."""
        config = ToolSelectorServiceConfig()
        selector = ToolSelectorService(
            config=config,
            available_tools={"search", "read_file", "write_file"},
        )

        # Enable all tools
        selector.set_enabled_tools({"search", "read_file", "write_file"})

        # Select tools for search query
        selected = await selector.select_tools(
            "Search for files",
            available_tools={"search", "read_file", "write_file"},
        )

        # Should select search tool
        assert "search" in selected

    def test_filter_hallucinated_tools(self):
        """Test filtering hallucinated tools."""
        config = ToolSelectorServiceConfig(enable_hallucination_filter=True)
        selector = ToolSelectorService(
            config=config,
            available_tools={"search", "read_file"},
        )

        tool_calls = [
            {"name": "search", "arguments": {"query": "test"}},
            {"name": "hallucinated_tool", "arguments": {}},
            {"name": "read_file", "arguments": {"path": "/test"}},
        ]

        filtered = selector.filter_hallucinated_tools(
            tool_calls, known_tools={"search", "read_file"}
        )

        assert len(filtered) == 2
        assert all(call["name"] in {"search", "read_file"} for call in filtered)


# =============================================================================
# ToolResultProcessor Tests
# =============================================================================


class TestToolResultProcessor:
    """Tests for ToolResultProcessor."""

    def test_process_result(self):
        """Test result processing."""
        config = ToolResultProcessorConfig()
        processor = ToolResultProcessor(config)

        result = ToolResult(success=True, output="Test output", error=None)

        processed = processor.process_result(result)

        assert processed["success"] is True
        assert processed["output"] == "Test output"
        assert processed["has_output"] is True
        assert processed["has_error"] is False

    def test_format_result_for_llm_success(self):
        """Test formatting successful result for LLM."""
        config = ToolResultProcessorConfig()
        processor = ToolResultProcessor(config)

        result = ToolResult(success=True, output="Test output", error=None)

        formatted = processor.format_result_for_llm(result)

        assert "✓ Success" in formatted
        assert "Test output" in formatted

    def test_format_result_for_llm_error(self):
        """Test formatting error result for LLM."""
        config = ToolResultProcessorConfig()
        processor = ToolResultProcessor(config)

        result = ToolResult(success=False, output=None, error="Test error")

        formatted = processor.format_result_for_llm(result)

        assert "✗ Error" in formatted
        assert "Test error" in formatted

    def test_aggregate_results(self):
        """Test result aggregation."""
        config = ToolResultProcessorConfig()
        processor = ToolResultProcessor(config)

        results = [
            ToolResult(success=True, output="Output 1", error=None),
            ToolResult(success=True, output="Output 2", error=None),
            ToolResult(success=False, output=None, error="Error 3"),
        ]

        aggregated = processor.aggregate_results(results)

        assert aggregated["total"] == 3
        assert aggregated["successful"] == 2
        assert aggregated["failed"] == 1
        assert aggregated["success_rate"] == pytest.approx(66.67, rel=0.01)

    def test_extract_insights(self):
        """Test insight extraction."""
        config = ToolResultProcessorConfig(enable_insights=True)
        processor = ToolResultProcessor(config)

        results = [
            ToolResult(success=True, output="Output 1", error=None),
            ToolResult(success=True, output="Output 2", error=None),
        ]

        insights = processor.extract_insights(results)

        assert len(insights) > 0
        # Check that we have a success rate insight
        assert any("succeeded" in insight.lower() for insight in insights)


# =============================================================================
# Canonical ToolService Surface Tests
# =============================================================================


class TestCanonicalToolServiceSurface:
    """Tests for the canonical ToolService surface."""

    def test_helper_package_does_not_export_tool_service(self):
        """Helper package should not export a duplicate ToolService."""
        tools_pkg = importlib.import_module("victor.agent.services.tools")

        assert not hasattr(tools_pkg, "ToolService")
        assert not hasattr(tools_pkg, "ToolServiceConfig")

    def test_legacy_facade_module_removed(self):
        """Legacy facade module should point callers to the canonical service."""
        with pytest.raises(ImportError, match="tool_service\\.ToolService"):
            importlib.import_module("victor.agent.services.tools.tool_service_facade")

    def _build_service(self, tool_registry, **config_kwargs):
        return ToolService(
            config=ToolServiceConfig(**config_kwargs),
            tool_selector=MockToolSelector(),
            tool_executor=MockToolExecutor(tool_registry),
            tool_registrar=MockToolRegistrar(tool_registry),
        )

    def test_initialization(self):
        """Test canonical service initialization."""
        tool_registry = {
            "search": MockSearchTool(),
            "read_file": MockReadTool(),
        }

        service = self._build_service(tool_registry)

        assert service.get_available_tools() == {"search", "read_file"}

    def test_budget_management(self):
        """Test budget management through canonical service."""
        tool_registry = {"search": MockSearchTool()}

        service = self._build_service(tool_registry, default_tool_budget=20)

        assert service.get_tool_budget() == 20
        assert service.get_remaining_budget() == 20

        service.consume_budget(5)
        assert service.get_remaining_budget() == 15

    def test_tool_enablement(self):
        """Test tool enablement through canonical service."""
        tool_registry = {
            "search": MockSearchTool(),
            "read_file": MockReadTool(),
        }

        service = self._build_service(tool_registry)

        service.set_enabled_tools({"search"})

        assert service.is_tool_enabled("search") is True
        assert service.is_tool_enabled("read_file") is False

    def test_is_healthy(self):
        """Test health check."""
        tool_registry = {"search": MockSearchTool()}

        service = self._build_service(tool_registry)

        assert service.is_healthy() is True

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        """Test tool execution through canonical service."""
        tool_registry = {"search": MockSearchTool()}

        service = self._build_service(tool_registry, default_tool_budget=10)
        service.set_enabled_tools({"search"})

        result = await service.execute_tool("search", {"query": "test"})

        assert result.success is True
        assert "test" in result.output

    def test_usage_stats(self):
        """Test usage stats through canonical service."""
        tool_registry = {"search": MockSearchTool()}

        service = self._build_service(tool_registry)

        service.consume_budget()
        service._track_tool_usage("search", success=True)

        stats = service.get_tool_usage_stats()

        assert stats["total_calls"] == 1
        assert stats["success_rate"] == 1.0
