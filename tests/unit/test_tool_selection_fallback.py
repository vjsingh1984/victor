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
    settings = Settings(analytics_enabled=False, use_semantic_tool_selection=False)
    orch = AgentOrchestrator(settings=settings, provider=_DummyProvider(), model="dummy")
    try:
        yield orch
    finally:
        orch.shutdown()


def test_prioritize_tools_stage_minimizes_broadcast(orchestrator: AgentOrchestrator) -> None:
    """If stage pruning removes everything, ensure we return a minimal slice instead of all tools."""
    tools = [
        ToolDefinition(name=f"custom{i}", description="desc", parameters={}) for i in range(12)
    ]

    # Use the ToolSelector's prioritize_by_stage method
    pruned = orchestrator.tool_selector.prioritize_by_stage("unrelated task", tools)

    assert pruned  # not empty
    assert len(pruned) <= orchestrator.tool_selector.fallback_max_tools
    # Should not return the entire original list
    assert len(pruned) < len(tools)


def test_prioritize_tools_stage_prefers_core_fallback(orchestrator: AgentOrchestrator) -> None:
    """When no stage tools match, core tools should be preferred if present."""
    tools = [
        ToolDefinition(name="read_file", description="desc", parameters={}),
        ToolDefinition(name="execute_bash", description="desc", parameters={}),
        ToolDefinition(name="custom", description="desc", parameters={}),
    ]

    # Use the ToolSelector's prioritize_by_stage method
    pruned = orchestrator.tool_selector.prioritize_by_stage("random", tools)

    names = {t.name for t in pruned}
    # Core tools should be included
    assert "read_file" in names or "execute_bash" in names


def test_semantic_fallback_uses_core_tools(
    orchestrator: AgentOrchestrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Semantic selector returning no tools should trigger core+keyword fallback, not broadcast all."""

    async def _fake_select(*args, **kwargs):  # pragma: no cover - executed in test
        return []

    async def _fake_init(*args, **kwargs):
        return None

    # Force semantic path
    orchestrator.use_semantic_selection = True
    orchestrator.semantic_selector = MagicMock()
    orchestrator.semantic_selector.initialize_tool_embeddings = _fake_init  # type: ignore[assignment]
    orchestrator.semantic_selector.select_relevant_tools_with_context = _fake_select  # type: ignore[assignment]

    # Also update the tool_selector to use the mocked semantic_selector
    orchestrator.tool_selector.semantic_selector = orchestrator.semantic_selector
    orchestrator.tool_selector._embeddings_initialized = False

    # Simulate a message that won't match keywords either (minimal fallback)
    selected = asyncio.run(orchestrator.tool_selector.select_semantic("zzz"))

    assert selected, "fallback should return some tools"
    assert len(selected) <= orchestrator.tool_selector.fallback_max_tools
