"""Tests for tool selection fallback behavior.

Updated to use the new IToolSelector protocol with ToolSelectionContext.
Tests verify that tool selection provides appropriate fallback behavior when
semantic selection or keyword matching returns no results.
"""

from typing import List
import asyncio
from unittest.mock import MagicMock

import pytest

from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings
from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    StreamChunk,
    ToolDefinition,
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

    async def chat(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: List[ToolDefinition] | None = None,
        **kwargs,
    ) -> CompletionResponse:
        return CompletionResponse(content="", role="assistant", model=model)

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: List[ToolDefinition] | None = None,
        **kwargs,
    ):
        if False:
            yield StreamChunk()  # pragma: no cover

    async def close(self) -> None:
        return None


@pytest.fixture()
def orchestrator() -> AgentOrchestrator:
    settings = Settings(analytics_enabled=False, tool_selection_strategy="keyword")
    orch = AgentOrchestrator(settings=settings, provider=_DummyProvider(), model="dummy")
    try:
        yield orch
    finally:
        import asyncio

        asyncio.run(orch.shutdown())


@pytest.mark.asyncio
async def test_keyword_selection_returns_core_tools(orchestrator: AgentOrchestrator) -> None:
    """Test that keyword selection includes core tools even for generic queries."""
    # Generic query that doesn't match specific keywords
    context = ToolSelectionContext(
        task_description="zzz",
        conversation_stage="initial"
    )

    tools = await orchestrator.tool_selector.select_tools("zzz", context)

    assert tools, "keyword selection should return some tools"
    # Should include core tools (read, ls, etc.) even for generic queries
    tool_names = {t.name for t in tools}
    assert "read" in tool_names or "ls" in tool_names


@pytest.mark.asyncio
async def test_keyword_selection_limits_tool_count(orchestrator: AgentOrchestrator) -> None:
    """Test that keyword selection doesn't return excessive tools."""
    # Query that could potentially match many tools
    context = ToolSelectionContext(
        task_description="analyze everything",
        conversation_stage="initial"
    )

    tools = await orchestrator.tool_selector.select_tools("analyze everything", context)

    # Should have a reasonable limit
    assert len(tools) <= 50, "should not return excessive number of tools"


@pytest.mark.asyncio
async def test_keyword_selection_includes_stage_relevant_tools(orchestrator: AgentOrchestrator) -> None:
    """Test that keyword selection considers conversation stage."""
    # Execution stage query
    context = ToolSelectionContext(
        task_description="modify the file",
        conversation_stage="execution"
    )

    tools = await orchestrator.tool_selector.select_tools("modify the file", context)

    tool_names = {t.name for t in tools}
    # Should include execution tools like edit, write
    assert any(name in tool_names for name in ["edit", "write", "shell"])


# NOTE: The following tests were removed because they tested internal
# implementation details (prioritize_by_stage, select_semantic) that are
# no longer exposed through the IToolSelector protocol.
#
# The new architecture uses a unified select_tools() method that handles
# all selection strategies (keyword, semantic, hybrid) internally.
# Fallback behavior is now built into the strategy factory.
#
# Tests should verify behavior through the public API (select_tools),
# not test internal implementation details.
