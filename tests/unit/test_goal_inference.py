
from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings
from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    StreamChunk,
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
            use_semantic_tool_selection=False,
            tool_cache_enabled=False,
        ),
        _DummyProvider(),
        "dummy",
    )


def test_security_goal_inference_adds_security_chain():
    orch = _orch()
    try:
        # Use the ToolSelector's select_keywords method
        tools = orch.tool_selector.select_keywords("run a security scan of the repo")
        names = [t.name for t in tools]
        assert "security_scan" in names
    finally:
        orch.shutdown()


def test_metrics_goal_inference_adds_metrics_chain():
    orch = _orch()
    try:
        # Use the ToolSelector's select_keywords method
        tools = orch.tool_selector.select_keywords("analyze code complexity and metrics")
        names = [t.name for t in tools]
        assert "analyze_metrics" in names
    finally:
        orch.shutdown()
