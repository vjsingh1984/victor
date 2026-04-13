"""TDD tests for tool selection caching in streaming pipeline."""

import pytest
from unittest.mock import AsyncMock, MagicMock


class TestToolSelectionCaching:

    @pytest.mark.asyncio
    async def test_same_context_reuses_tools(self):
        from victor.agent.streaming.pipeline import StreamingChatPipeline

        orch = MagicMock()
        orch._select_tools_for_turn = AsyncMock(
            return_value=[MagicMock(name="read"), MagicMock(name="write")]
        )
        pipeline = StreamingChatPipeline.__new__(StreamingChatPipeline)
        pipeline._last_tool_context = None
        pipeline._last_tools = None
        tools1 = await pipeline._get_tools_cached(orch, "fix the bug", None)
        assert orch._select_tools_for_turn.call_count == 1
        tools2 = await pipeline._get_tools_cached(orch, "fix the bug", None)
        assert orch._select_tools_for_turn.call_count == 1
        assert tools1 is tools2

    @pytest.mark.asyncio
    async def test_different_context_invalidates(self):
        from victor.agent.streaming.pipeline import StreamingChatPipeline

        orch = MagicMock()
        orch._select_tools_for_turn = AsyncMock(return_value=[MagicMock(name="read")])
        pipeline = StreamingChatPipeline.__new__(StreamingChatPipeline)
        pipeline._last_tool_context = None
        pipeline._last_tools = None
        await pipeline._get_tools_cached(orch, "fix the bug", None)
        assert orch._select_tools_for_turn.call_count == 1
        await pipeline._get_tools_cached(orch, "add a feature", None)
        assert orch._select_tools_for_turn.call_count == 2

    @pytest.mark.asyncio
    async def test_none_tools_not_cached(self):
        from victor.agent.streaming.pipeline import StreamingChatPipeline

        orch = MagicMock()
        orch._select_tools_for_turn = AsyncMock(return_value=None)
        pipeline = StreamingChatPipeline.__new__(StreamingChatPipeline)
        pipeline._last_tool_context = None
        pipeline._last_tools = None
        tools = await pipeline._get_tools_cached(orch, "hello", None)
        assert tools is None
        assert orch._select_tools_for_turn.call_count == 1
        await pipeline._get_tools_cached(orch, "hello", None)
        assert orch._select_tools_for_turn.call_count == 2
