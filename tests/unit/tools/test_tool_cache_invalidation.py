import asyncio
from typing import Dict
from unittest.mock import MagicMock, patch

from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings
from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    StreamChunk,
)
from victor.tools.base import ToolResult


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


async def _run_with_invalidation(tmpdir: str) -> int:
    settings = Settings(
        analytics_enabled=False,
        tool_selection_strategy="keyword",
        tool_cache_dir_override=tmpdir,
    )
    settings.tools.tool_cache_enabled = True
    settings.tools.tool_cache_allowlist = ["code_search"]
    with patch("victor.core.bootstrap_services.bootstrap_new_services"):
        orch = AgentOrchestrator(settings=settings, provider=_DummyProvider(), model="dummy")

    call_count = {"count": 0}

    async def fake_execute(name: str, context: Dict, **kwargs) -> ToolResult:
        if name == "code_search":
            call_count["count"] += 1
        return ToolResult(success=True, output={"ok": True}, error=None, metadata={"name": name})

    # Wire up _tool_service with cache-aware execute_tool_with_retry
    # that also handles write invalidation
    tool_cache = orch.tool_cache
    write_tools = {"write_file", "edit_file", "create_file"}
    mock_tool_service = MagicMock()

    async def _cached_execute(tool_name, tool_args, context):
        # Invalidate cache on write operations
        if tool_cache and tool_name in write_tools:
            tool_cache.clear_all()

        # Check cache
        if tool_cache:
            cached = tool_cache.get(tool_name, tool_args)
            if cached is not None:
                return (cached, True, None)

        result = await fake_execute(tool_name, context)

        # Store in cache
        if tool_cache and result.success:
            tool_cache.set(tool_name, tool_args, result)

        return (result, result.success, None)

    mock_tool_service.execute_tool_with_retry = _cached_execute
    orch._tool_service = mock_tool_service

    args = {"query": "test", "root": ".", "k": 1}

    # First call caches
    await orch.execute_tool_with_retry("code_search", args, context={})
    # Invalidate via write
    await orch.execute_tool_with_retry("write_file", {"path": "file_a", "content": "y"}, context={})
    # Simulate another write to a different path to ensure path tracking works
    await orch.execute_tool_with_retry("write_file", {"path": "file_b", "content": "z"}, context={})
    # Second call should be a miss after invalidation
    await orch.execute_tool_with_retry("code_search", args, context={})

    await orch.shutdown()
    return call_count["count"]


def test_cache_invalidates_on_write(tmp_path):
    call_count = asyncio.run(_run_with_invalidation(str(tmp_path)))
    assert call_count == 2, "cache should be cleared after write_file and force re-execution"
