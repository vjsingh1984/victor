import asyncio
from typing import Dict

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
        use_semantic_tool_selection=False,
        tool_cache_enabled=True,
        tool_cache_allowlist=["code_search"],
        tool_cache_dir=tmpdir,
    )
    orch = AgentOrchestrator(settings=settings, provider=_DummyProvider(), model="dummy")

    call_count = {"count": 0}

    async def fake_execute(name: str, context: Dict, **kwargs) -> ToolResult:
        if name == "code_search":
            call_count["count"] += 1
        return ToolResult(success=True, output={"ok": True}, error=None, metadata={"name": name})

    orch.tools.execute = fake_execute  # type: ignore[assignment]

    args = {"query": "test", "root": ".", "k": 1}

    # First call caches
    await orch._execute_tool_with_retry("code_search", args, context={})
    # Invalidate via write
    await orch._execute_tool_with_retry(
        "write_file", {"path": "file_a", "content": "y"}, context={}
    )
    # Simulate another write to a different path to ensure path tracking works
    await orch._execute_tool_with_retry(
        "write_file", {"path": "file_b", "content": "z"}, context={}
    )
    # Second call should be a miss after invalidation
    await orch._execute_tool_with_retry("code_search", args, context={})

    orch.shutdown()
    return call_count["count"]


def test_cache_invalidates_on_write(tmp_path):
    call_count = asyncio.run(_run_with_invalidation(str(tmp_path)))
    assert call_count == 2, "cache should be cleared after write_file and force re-execution"
