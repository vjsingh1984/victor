"""Tests for tool dependency planning functionality.

These tests verify that tool selection correctly orders tools and includes
keyword-matching tools based on user messages using ToolMetadataRegistry.

Updated to use the new IToolSelector protocol with ToolSelectionContext.
"""

from unittest.mock import patch, MagicMock

import pytest

from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings
from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    StreamChunk,
)
from victor.protocols import ToolSelectionContext


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


@pytest.mark.asyncio
async def test_core_tools_always_selected():
    """Test that core tools are always included in keyword selection."""
    orch = _orch()
    try:
        # Use new IToolSelector API with ToolSelectionContext
        context = ToolSelectionContext(
            task_description="do something", conversation_stage="initial"
        )
        tools = await orch.tool_selector.select_tools("do something", context)
        names = [t.name for t in tools]
        # Core/critical tools should be included (read, ls, shell, edit, search)
        # Note: 'write' is NOT a critical tool - 'edit' is for file modifications
        assert "read" in names
        assert "ls" in names
    finally:
        await orch.shutdown()


@pytest.mark.asyncio
async def test_planned_tools_prepended_to_selection():
    """Test that pre-planned tools are included first in selection."""
    orch = _orch()
    try:
        # Use real tools that exist in the registry
        # "read", "ls", "graph" are core tools that are registered
        # Note: "code_search" requires code_search_tool import which isn't loaded by default
        context = ToolSelectionContext(
            task_description="summarize",
            conversation_stage="initial",
            planned_tools=["graph", "read", "ls"],  # Use real tool names that are registered
        )

        tools = await orch.tool_selector.select_tools("summarize", context)
        names = [t.name for t in tools]

        # Planned tools should be included and appear early in selection
        assert "graph" in names
        assert "read" in names
        assert "ls" in names

        # Verify planned tools come before non-planned tools
        graph_idx = names.index("graph")
        read_idx = names.index("read")
        ls_idx = names.index("ls")

        # All planned tools should be in the first 5 positions
        assert max(graph_idx, read_idx, ls_idx) < 5
    finally:
        await orch.shutdown()


def test_keyword_matching_uses_registry():
    """Test that keyword matching uses ToolMetadataRegistry.

    get_tools_from_message() scans the user message for keywords defined
    in @tool decorators via the metadata registry.
    """
    from victor.agent.tool_selection import get_tools_from_message

    # Without registered tools, returns empty set
    tools = get_tools_from_message("document this code with docstrings")
    assert isinstance(tools, set)


@pytest.mark.asyncio
async def test_docs_keyword_matching_with_mock_registry():
    """Test that keyword matching works when registry has matching tools."""
    with patch(
        "victor.agent.tool_selection.get_tools_from_message",
        return_value={"docs_coverage"},
    ):
        orch = _orch()
        try:
            # Use new IToolSelector API
            context = ToolSelectionContext(
                task_description="document the codebase", conversation_stage="initial"
            )
            tools = await orch.tool_selector.select_tools("document the codebase", context)
            names = [t.name for t in tools]
            assert "docs_coverage" in names
        finally:
            await orch.shutdown()


def test_registry_keyword_lookup():
    """Test that ToolMetadataRegistry correctly looks up tools by keywords."""
    from victor.tools.metadata_registry import ToolMetadataRegistry
    from victor.tools.enums import AccessMode, CostTier, DangerLevel, ExecutionCategory, Priority

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
    from victor.tools.enums import AccessMode, CostTier, DangerLevel, ExecutionCategory, Priority

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
