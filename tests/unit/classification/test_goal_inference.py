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
        return CompletionResponse(content="", role="assistant", model="dummy")  # type: ignore[call-arg]

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


# NOTE: The following tests were removed because they tested deprecated functionality.
# The old ToolSelector.select_keywords() method is no longer available in the new
# IToolSelector interface. Tool selection now uses the unified strategy factory
# (keyword, semantic, hybrid) with the select_tools() method and ToolSelectionContext.
# Tests for keyword-based tool selection are covered by other integration tests.
