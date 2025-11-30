import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings
from victor.providers.base import BaseProvider, Message
from victor.tools.base import ToolRegistry, BaseTool
from pathlib import Path


class DummyProvider(BaseProvider):
    """Minimal provider stub for orchestrator unit tests."""

    def __init__(self):
        super().__init__(api_key=None)

    @property
    def name(self) -> str:  # pragma: no cover - trivial
        return "dummy"

    def supports_tools(self) -> bool:  # pragma: no cover - trivial
        return True

    def supports_streaming(self) -> bool:  # pragma: no cover - trivial
        return False

    async def chat(
        self, messages, *, model, temperature=0.7, max_tokens=4096, tools=None, **kwargs
    ):
        raise NotImplementedError

    async def stream(
        self, messages, *, model, temperature=0.7, max_tokens=4096, tools=None, **kwargs
    ):
        raise NotImplementedError

    async def close(self):  # pragma: no cover - trivial
        return None


class AlwaysFailTool(BaseTool):
    """Tool that always fails, capturing attempts."""

    name = "always_fail"
    description = "Always fails"
    parameters = {}

    def __init__(self):
        super().__init__()
        self.attempts = 0

    async def execute(self, context, **kwargs):
        self.attempts += 1
        return SimpleNamespace(success=False, output=None, error="boom")


@pytest.mark.asyncio
async def test_repeated_failing_call_is_skipped_after_first_failure(monkeypatch):
    """Orchestrator should skip identical failing tool calls to break loops."""
    settings = Settings(
        analytics_enabled=False,
        analytics_log_file=str(Path("tmp_usage.log")),
        tool_cache_dir=str(Path("tmp_cache")),
    )
    settings.tool_cache_enabled = False
    provider = DummyProvider()

    # Minimal profile/model fields to construct orchestrator
    orch = AgentOrchestrator(
        settings=settings,
        provider=provider,
        model="dummy-model",
        temperature=0.0,
        max_tokens=10,
    )

    # Inject a failing tool
    failing_tool = AlwaysFailTool()
    orch.tools = ToolRegistry()
    orch.tools.register(failing_tool)

    # Simulate a single tool call repeated twice
    tool_calls = [
        {"name": "always_fail", "arguments": {"foo": "bar"}},
        {"name": "always_fail", "arguments": {"foo": "bar"}},
    ]

    # Execute tool calls
    results = await orch._handle_tool_calls(tool_calls)

    # First call executed (with retries), second skipped due to repeat signature
    assert failing_tool.attempts == 3  # retries inside ToolExecutionManager
    assert orch.executed_tools.count("always_fail") == 1
    assert len(results) == 1  # second call skipped pre-execution
    assert results[0]["success"] is False
