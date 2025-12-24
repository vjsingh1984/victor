"""Tests for tool dependency planning functionality.

These tests verify that tool selection correctly orders tools and includes
keyword-matching tools based on user messages using ToolMetadataRegistry.
"""

from unittest.mock import patch, MagicMock

import pytest

from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings
from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    StreamChunk,
    ToolDefinition,
)


class _DummyProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "dummy"

    def supports_tools(self) -> bool:  # type: ignore[override]
        return True

    def supports_streaming(self) -> bool:  # type: ignore[override]
        return False

    async def chat(self, *args, **kwargs) -> CompletionResponse:
        return CompletionResponse(content="", role="assistant", model="dummy")

    async def stream(self, *args, **kwargs):
        if False:
            yield StreamChunk()  # pragma: no cover

    async def close(self) -> None:
        return None


def _orch():
    return AgentOrchestrator(
        Settings(
            analytics_enabled=False,
            tool_selection_strategy="keyword",
            tool_cache_enabled=False,
        ),
        _DummyProvider(),
        "dummy",
    )


def test_core_tools_always_selected():
    """Test that core tools are always included in keyword selection."""
    orch = _orch()
    try:
        tools = orch.tool_selector.select_keywords("do something")
        names = [t.name for t in tools]
        # Core/critical tools should be included (read, ls, shell, edit, search)
        # Note: 'write' is NOT a critical tool - 'edit' is for file modifications
        assert "read" in names
        assert "ls" in names
    finally:
        import asyncio

        asyncio.run(orch.shutdown())


def test_planned_tools_prepended_to_selection():
    """Test that pre-planned tools are included first in selection."""
    orch = _orch()
    try:
        # Create planned tools
        planned = [
            ToolDefinition(name="search", description="Search", parameters={}),
            ToolDefinition(name="read", description="Read", parameters={}),
            ToolDefinition(name="docs_coverage", description="Docs", parameters={}),
        ]
        tools = orch.tool_selector.select_keywords("summarize", planned_tools=planned)
        names = [t.name for t in tools]
        # Planned tools should be at the start
        assert names[:3] == ["search", "read", "docs_coverage"]
    finally:
        orch.shutdown()


def test_keyword_matching_uses_registry():
    """Test that keyword matching uses ToolMetadataRegistry.

    get_tools_from_message() scans the user message for keywords defined
    in @tool decorators via the metadata registry.
    """
    from victor.agent.tool_selection import get_tools_from_message

    # Without registered tools, returns empty set
    tools = get_tools_from_message("document this code with docstrings")
    assert isinstance(tools, set)


def test_docs_keyword_matching_with_mock_registry():
    """Test that keyword matching works when registry has matching tools."""
    with patch(
        "victor.agent.tool_selection.get_tools_from_message",
        return_value={"docs_coverage"},
    ):
        orch = _orch()
        try:
            tools = orch.tool_selector.select_keywords("document the codebase")
            names = [t.name for t in tools]
            assert "docs_coverage" in names
        finally:
            orch.shutdown()


def test_registry_keyword_lookup():
    """Test that ToolMetadataRegistry correctly looks up tools by keywords."""
    from victor.tools.metadata_registry import ToolMetadataRegistry, ToolMetadataEntry
    from victor.tools.base import Priority, AccessMode, DangerLevel, CostTier, ExecutionCategory

    # Create a mock tool with keywords
    mock_tool = MagicMock()
    mock_tool.name = "docs_coverage"
    mock_tool.description = "Analyze documentation coverage"
    mock_tool.priority = Priority.MEDIUM
    mock_tool.access_mode = AccessMode.READONLY
    mock_tool.danger_level = DangerLevel.SAFE
    mock_tool.cost_tier = CostTier.FREE
    mock_tool.keywords = ["document", "docstring", "docs"]
    mock_tool.category = "docs"
    mock_tool.aliases = set()
    # New decorator-driven metadata fields
    mock_tool.stages = []
    mock_tool.mandatory_keywords = []
    mock_tool.task_types = []
    mock_tool.progress_params = []
    mock_tool.execution_category = ExecutionCategory.READ_ONLY

    registry = ToolMetadataRegistry()
    registry.register(mock_tool)

    # Test keyword lookup
    tools = registry.get_tools_matching_text("document this code with docstrings")
    assert "docs_coverage" in tools


def test_registry_keyword_lookup_case_insensitive():
    """Test that keyword lookup is case-insensitive."""
    from victor.tools.metadata_registry import ToolMetadataRegistry
    from victor.tools.base import Priority, AccessMode, DangerLevel, CostTier, ExecutionCategory

    mock_tool = MagicMock()
    mock_tool.name = "review_tool"
    mock_tool.description = "Code review"
    mock_tool.priority = Priority.MEDIUM
    mock_tool.access_mode = AccessMode.READONLY
    mock_tool.danger_level = DangerLevel.SAFE
    mock_tool.cost_tier = CostTier.FREE
    mock_tool.keywords = ["review", "quality"]
    mock_tool.category = "review"
    mock_tool.aliases = set()
    # New decorator-driven metadata fields
    mock_tool.stages = []
    mock_tool.mandatory_keywords = []
    mock_tool.task_types = []
    mock_tool.progress_params = []
    mock_tool.execution_category = ExecutionCategory.READ_ONLY

    registry = ToolMetadataRegistry()
    registry.register(mock_tool)

    # Test case-insensitive matching
    tools_lower = registry.get_tools_matching_text("review this code")
    tools_upper = registry.get_tools_matching_text("REVIEW this code")
    tools_mixed = registry.get_tools_matching_text("Review This Code")

    assert "review_tool" in tools_lower
    assert "review_tool" in tools_upper
    assert "review_tool" in tools_mixed
