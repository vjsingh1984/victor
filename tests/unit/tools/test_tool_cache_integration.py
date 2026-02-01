import asyncio

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


async def _run_twice_with_cache(tmpdir: str) -> int:
    # Only allowlist code_search to hit cache for this test
    settings = Settings(
        analytics_enabled=False,
        tool_selection_strategy="keyword",
        tool_cache_enabled=True,
        tool_cache_allowlist=["code_search"],
        tool_cache_dir=tmpdir,  # Use correct setting name
    )
    orch = AgentOrchestrator(settings=settings, provider=_DummyProvider(), model="dummy")

    call_count = {"count": 0}

    async def fake_execute(name: str, context: dict, **kwargs) -> ToolResult:
        call_count["count"] += 1
        return ToolResult(success=True, output={"ok": True})

    # Monkeypatch tool execution
    orch.tools.execute = fake_execute  # type: ignore[assignment]

    args = {"query": "test", "root": ".", "k": 1}

    # First call populates cache
    await orch._execute_tool_with_retry("code_search", args, context={})
    # Second call should hit cache, not invoke fake_execute again
    await orch._execute_tool_with_retry("code_search", args, context={})

    await orch.shutdown()
    return call_count["count"]


def test_tool_cache_hits_on_repeat_calls(tmp_path):
    call_count = asyncio.run(_run_twice_with_cache(str(tmp_path)))
    assert call_count == 1, "tool execution should run once and then be served from cache"
