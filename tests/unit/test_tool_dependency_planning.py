
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


def test_plan_tools_orders_search_read_analyze():
    orch = AgentOrchestrator(
        Settings(
            analytics_enabled=False,
            use_semantic_tool_selection=False,
            tool_cache_enabled=False,
        ),
        _DummyProvider(),
        "dummy",
    )
    try:
        planned = orch._plan_tools(["summary"], available_inputs=["query"])
        names = [t.name for t in planned]
        assert names == ["code_search", "read_file", "analyze_docs"]
    finally:
        orch.shutdown()


def test_keyword_selection_includes_planned_chain():
    orch = AgentOrchestrator(
        Settings(
            analytics_enabled=False,
            use_semantic_tool_selection=False,
            tool_cache_enabled=False,
        ),
        _DummyProvider(),
        "dummy",
    )
    try:
        # Use the ToolSelector's select_keywords method with planned tools
        goals = orch._goal_hints_for_message("summarize the codebase")
        planned_tools = orch._plan_tools(goals, available_inputs=["query"]) if goals else None
        tools = orch.tool_selector.select_keywords(
            "summarize the codebase", planned_tools=planned_tools
        )
        names = [t.name for t in tools]
        # Planned tools should lead the list in order
        assert names[:3] == ["code_search", "read_file", "analyze_docs"]
    finally:
        orch.shutdown()
