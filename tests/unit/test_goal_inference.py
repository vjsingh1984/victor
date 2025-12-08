from unittest.mock import patch

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
    """Test that security-related keywords trigger category detection."""
    from victor.agent.tool_selection import detect_categories_from_message

    # Verify category detection works for security keywords
    categories = detect_categories_from_message("run a security scan of the repo")
    assert "security" in categories


def test_metrics_goal_inference_adds_metrics_chain():
    """Test that metrics-related keywords trigger category detection."""
    from victor.agent.tool_selection import detect_categories_from_message

    # Verify category detection works for metrics keywords
    categories = detect_categories_from_message("analyze code complexity and metrics")
    assert "metrics" in categories


def test_security_keyword_tool_selection_with_mocked_registry():
    """Test that security tools are selected when category lookup returns tools."""
    # Patch get_tools_for_categories to return security tools
    with patch(
        "victor.agent.tool_selection.get_tools_for_categories",
        return_value={"scan"},
    ):
        orch = _orch()
        try:
            tools = orch.tool_selector.select_keywords("run a security scan of the repo")
            names = [t.name for t in tools]
            assert "scan" in names
        finally:
            orch.shutdown()


def test_metrics_keyword_tool_selection_with_mocked_registry():
    """Test that metrics tools are selected when category lookup returns tools."""
    # Patch get_tools_for_categories to return metrics tools
    with patch(
        "victor.agent.tool_selection.get_tools_for_categories",
        return_value={"metrics"},
    ):
        orch = _orch()
        try:
            tools = orch.tool_selector.select_keywords("analyze code complexity and metrics")
            names = [t.name for t in tools]
            assert "metrics" in names
        finally:
            orch.shutdown()
