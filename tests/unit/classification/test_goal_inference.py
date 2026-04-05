from unittest.mock import MagicMock, patch

from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings
from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
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
    """Test that security tools are selected when keyword matching returns tools.

    select_keywords uses get_tools_from_message for keyword-based tool lookup,
    then filters against the set of registered tools. We mock both the keyword
    matcher and the tool registry to ensure the "scan" tool appears in the
    final selection.
    """
    mock_scan_tool = MagicMock()
    mock_scan_tool.name = "scan"
    mock_scan_tool.description = "Security scan tool"
    mock_scan_tool.parameters = {}

    scan_tool_def = ToolDefinition(name="scan", description="Security scan tool", parameters={})

    with (
        patch(
            "victor.agent.tool_selection.get_tools_from_message",
            return_value={"scan"},
        ),
    ):
        orch = _orch()
        try:
            # Clear any vertical-set enabled_tools so we hit the fallback path
            orch.tool_selector._enabled_tools = set()
            # Inject our mock tool into the tool registry
            orch.tool_selector.tools.list_tools = MagicMock(return_value=[mock_scan_tool])
            tools = orch.tool_selector.select_keywords("run a security scan of the repo")
            names = [t.name for t in tools]
            assert "scan" in names
        finally:
            import asyncio

            asyncio.run(orch.shutdown())


def test_metrics_keyword_tool_selection_with_mocked_registry():
    """Test that metrics tools are selected when keyword matching returns tools.

    select_keywords uses get_tools_from_message for keyword-based tool lookup,
    then filters against the set of registered tools. We mock both the keyword
    matcher and the tool registry to ensure the "metrics" tool appears in the
    final selection.
    """
    mock_metrics_tool = MagicMock()
    mock_metrics_tool.name = "metrics"
    mock_metrics_tool.description = "Metrics tool"
    mock_metrics_tool.parameters = {}

    with (
        patch(
            "victor.agent.tool_selection.get_tools_from_message",
            return_value={"metrics"},
        ),
    ):
        orch = _orch()
        try:
            orch.tool_selector._enabled_tools = set()
            orch.tool_selector.tools.list_tools = MagicMock(return_value=[mock_metrics_tool])
            tools = orch.tool_selector.select_keywords("analyze code complexity and metrics")
            names = [t.name for t in tools]
            assert "metrics" in names
        finally:
            import asyncio

            asyncio.run(orch.shutdown())
