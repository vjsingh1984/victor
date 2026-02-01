# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration tests for ToolExecutor with orchestrator."""

import pytest
from unittest.mock import MagicMock, patch

from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.tool_executor import ToolExecutor, ToolExecutionResult
from victor.config.settings import Settings


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(analytics_enabled=False, tool_cache_enabled=False)


@pytest.fixture
def mock_provider():
    """Create a mock provider."""
    provider = MagicMock()
    provider.name = "mock"
    provider.supports_tools.return_value = True
    provider.supports_streaming.return_value = True
    return provider


@pytest.fixture
def orchestrator(settings, mock_provider):
    """Create orchestrator with mocked provider."""
    with patch("victor.agent.orchestrator_imports.UsageLogger"):
        orc = AgentOrchestrator(
            settings=settings,
            provider=mock_provider,
            model="test-model",
        )
    # Enable the canonical name for read/write tools used in tests
    orc.tools.enable_tool("read")
    orc.tools.enable_tool("write")
    return orc


class TestToolExecutorIntegration:
    """Tests for ToolExecutor integration with orchestrator."""

    def test_orchestrator_has_tool_executor(self, orchestrator):
        """Verify orchestrator initializes with ToolExecutor."""
        assert hasattr(orchestrator, "tool_executor")
        assert isinstance(orchestrator.tool_executor, ToolExecutor)

    def test_tool_executor_shares_registry(self, orchestrator):
        """Verify ToolExecutor uses same tool registry as orchestrator."""
        assert orchestrator.tool_executor.tools is orchestrator.tools

    def test_tool_executor_shares_normalizer(self, orchestrator):
        """Verify ToolExecutor uses same argument normalizer."""
        assert orchestrator.tool_executor.normalizer is orchestrator.argument_normalizer

    @pytest.mark.asyncio
    async def test_handle_tool_calls_uses_executor(self, orchestrator):
        """Test that _handle_tool_calls routes through ToolExecutor."""
        # Mock the tool_executor.execute method
        mock_result = ToolExecutionResult(
            tool_name="read",  # Now using canonical name
            success=True,
            result="file contents here",
            error=None,
            execution_time=0.05,
        )

        with patch.object(
            orchestrator.tool_executor, "execute", return_value=mock_result
        ) as mock_exec:
            tool_calls = [{"name": "read_file", "arguments": {"path": "/tmp/test.txt"}}]
            results = await orchestrator._handle_tool_calls(tool_calls)

            # Verify executor was called
            mock_exec.assert_called_once()
            call_kwargs = mock_exec.call_args
            # Now we expect the canonical name "read" (resolved from "read_file")
            assert call_kwargs.kwargs["tool_name"] == "read"

            # Verify result was processed
            assert len(results) == 1
            assert results[0]["success"] is True

    @pytest.mark.asyncio
    async def test_tool_executor_handles_failure(self, orchestrator):
        """Test that ToolExecutor failure is properly handled."""
        mock_result = ToolExecutionResult(
            tool_name="read",  # Canonical name
            success=False,
            result=None,
            error="File not found",
            execution_time=0.01,
        )

        with patch.object(orchestrator.tool_executor, "execute", return_value=mock_result):
            tool_calls = [{"name": "read_file", "arguments": {"path": "/nonexistent"}}]
            results = await orchestrator._handle_tool_calls(tool_calls)

            assert len(results) == 1
            assert results[0]["success"] is False
            assert "error" in results[0]

    @pytest.mark.asyncio
    async def test_context_passed_to_executor(self, orchestrator):
        """Test that proper context is passed to tool executor."""
        mock_result = ToolExecutionResult(
            tool_name="list_directory",
            success=True,
            result=["file1.txt", "file2.py"],
            error=None,
        )

        with patch.object(
            orchestrator.tool_executor, "execute", return_value=mock_result
        ) as mock_exec:
            tool_calls = [{"name": "list_directory", "arguments": {"root": "."}}]
            await orchestrator._handle_tool_calls(tool_calls)

            # Verify context was passed
            call_kwargs = mock_exec.call_args.kwargs
            assert "context" in call_kwargs
            context = call_kwargs["context"]
            assert "code_manager" in context
            assert "provider" in context
            assert "settings" in context


class TestToolExecutorCaching:
    """Tests for ToolExecutor caching behavior."""

    @pytest.fixture
    def orchestrator_with_cache(self, mock_provider):
        """Create orchestrator with cache enabled."""
        settings = Settings(
            analytics_enabled=False,
            tool_cache_enabled=True,
            tool_cache_allowlist=["read_file", "list_directory"],
        )
        with patch("victor.agent.orchestrator_imports.UsageLogger"):
            orc = AgentOrchestrator(
                settings=settings,
                provider=mock_provider,
                model="test-model",
            )
        return orc

    def test_tool_executor_has_cache(self, orchestrator_with_cache):
        """Verify ToolExecutor receives cache from orchestrator."""
        assert orchestrator_with_cache.tool_executor.cache is not None
        assert orchestrator_with_cache.tool_executor.cache is orchestrator_with_cache.tool_cache
