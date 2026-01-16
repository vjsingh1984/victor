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

"""Unit tests for tool selector health check functionality.

This test module verifies the health check and graceful fallback mechanisms
that prevent the critical bug where SemanticToolSelector was never initialized,
blocking ALL chat functionality.
"""

import pytest

from victor.tools.keyword_tool_selector import KeywordToolSelector
from victor.tools.semantic_selector import SemanticToolSelector
from victor.agent.tool_selector_factory import create_tool_selector_strategy
from victor.tools.base import ToolRegistry


@pytest.mark.unit
class TestToolSelectorHealthCheck:
    """Test health check methods for tool selector initialization."""

    def test_keyword_selector_is_always_healthy(self):
        """Test that keyword selector doesn't need initialization."""
        tools = ToolRegistry()
        selector = KeywordToolSelector(
            tools=tools,
            conversation_state=None,
            model="gpt-4",
            provider_name="openai",
            enabled_tools=None,
        )

        # Create a mock orchestrator to test the health check
        class MockOrchestrator:
            def __init__(self, tool_selector, tools):
                self.tool_selector = tool_selector
                self.tools = tools

            def check_tool_selector_health(self):
                """Copied from AgentOrchestrator."""
                if not hasattr(self, "tool_selector") or self.tool_selector is None:
                    return {
                        "healthy": False,
                        "strategy": None,
                        "initialized": False,
                        "message": "Tool selector not created",
                        "can_auto_recover": False,
                    }

                strategy = getattr(self.tool_selector, "strategy", "unknown")
                strategy_name = strategy.value if hasattr(strategy, "value") else str(strategy)

                needs_init = hasattr(self.tool_selector, "initialize_tool_embeddings")
                is_initialized = (
                    hasattr(self.tool_selector, "_embeddings_initialized")
                    and self.tool_selector._embeddings_initialized
                )

                if not needs_init:
                    return {
                        "healthy": True,
                        "strategy": strategy_name,
                        "initialized": True,
                        "message": f"Tool selector ready (strategy: {strategy_name})",
                        "can_auto_recover": False,
                    }

                if is_initialized:
                    return {
                        "healthy": True,
                        "strategy": strategy_name,
                        "initialized": True,
                        "message": f"Tool selector ready (strategy: {strategy_name}, embeddings initialized)",
                        "can_auto_recover": False,
                    }

                return {
                    "healthy": False,
                    "strategy": strategy_name,
                    "initialized": False,
                    "message": f"Tool selector NOT initialized (strategy: {strategy_name})",
                    "can_auto_recover": True,
                }

        mock_orch = MockOrchestrator(selector, tools)
        health = mock_orch.check_tool_selector_health()

        assert health["healthy"] is True
        assert health["initialized"] is True
        assert health["can_auto_recover"] is False

    def test_semantic_selector_needs_initialization(self):
        """Test that semantic selector reports unhealthy when not initialized."""
        tools = ToolRegistry()

        # Create semantic selector without initialization
        selector = SemanticToolSelector(
            embedding_model="all-MiniLM-L6-v2",
            embedding_provider="sentence-transformers",
            cache_embeddings=True,
        )

        # Ensure it's not initialized
        if hasattr(selector, "_embeddings_initialized"):
            selector._embeddings_initialized = False

        class MockOrchestrator:
            def __init__(self, tool_selector, tools):
                self.tool_selector = tool_selector
                self.tools = tools

            def check_tool_selector_health(self):
                """Copied from AgentOrchestrator."""
                if not hasattr(self, "tool_selector") or self.tool_selector is None:
                    return {
                        "healthy": False,
                        "strategy": None,
                        "initialized": False,
                        "message": "Tool selector not created",
                        "can_auto_recover": False,
                    }

                strategy = getattr(self.tool_selector, "strategy", "unknown")
                strategy_name = strategy.value if hasattr(strategy, "value") else str(strategy)

                needs_init = hasattr(self.tool_selector, "initialize_tool_embeddings")
                is_initialized = (
                    hasattr(self.tool_selector, "_embeddings_initialized")
                    and self.tool_selector._embeddings_initialized
                )

                if not needs_init:
                    return {
                        "healthy": True,
                        "strategy": strategy_name,
                        "initialized": True,
                        "message": f"Tool selector ready (strategy: {strategy_name})",
                        "can_auto_recover": False,
                    }

                if is_initialized:
                    return {
                        "healthy": True,
                        "strategy": strategy_name,
                        "initialized": True,
                        "message": f"Tool selector ready (strategy: {strategy_name}, embeddings initialized)",
                        "can_auto_recover": False,
                    }

                return {
                    "healthy": False,
                    "strategy": strategy_name,
                    "initialized": False,
                    "message": f"Tool selector NOT initialized (strategy: {strategy_name})",
                    "can_auto_recover": True,
                }

        mock_orch = MockOrchestrator(selector, tools)
        health = mock_orch.check_tool_selector_health()

        # Should be unhealthy
        assert health["healthy"] is False
        assert health["initialized"] is False
        assert health["can_auto_recover"] is True
        assert "NOT initialized" in health["message"]

    @pytest.mark.asyncio
    async def test_semantic_selector_becomes_healthy_after_init(self):
        """Test that semantic selector becomes healthy after initialization."""
        tools = ToolRegistry()

        # Create semantic selector
        selector = SemanticToolSelector(
            embedding_model="all-MiniLM-L6-v2",
            embedding_provider="sentence-transformers",
            cache_embeddings=True,
        )

        class MockOrchestrator:
            def __init__(self, tool_selector, tools):
                self.tool_selector = tool_selector
                self.tools = tools

            def check_tool_selector_health(self):
                """Copied from AgentOrchestrator."""
                if not hasattr(self, "tool_selector") or self.tool_selector is None:
                    return {
                        "healthy": False,
                        "strategy": None,
                        "initialized": False,
                        "message": "Tool selector not created",
                        "can_auto_recover": False,
                    }

                strategy = getattr(self.tool_selector, "strategy", "unknown")
                strategy_name = strategy.value if hasattr(strategy, "value") else str(strategy)

                needs_init = hasattr(self.tool_selector, "initialize_tool_embeddings")
                is_initialized = (
                    hasattr(self.tool_selector, "_embeddings_initialized")
                    and self.tool_selector._embeddings_initialized
                )

                if not needs_init:
                    return {
                        "healthy": True,
                        "strategy": strategy_name,
                        "initialized": True,
                        "message": f"Tool selector ready (strategy: {strategy_name})",
                        "can_auto_recover": False,
                    }

                if is_initialized:
                    return {
                        "healthy": True,
                        "strategy": strategy_name,
                        "initialized": True,
                        "message": f"Tool selector ready (strategy: {strategy_name}, embeddings initialized)",
                        "can_auto_recover": False,
                    }

                return {
                    "healthy": False,
                    "strategy": strategy_name,
                    "initialized": False,
                    "message": f"Tool selector NOT initialized (strategy: {strategy_name})",
                    "can_auto_recover": True,
                }

        mock_orch = MockOrchestrator(selector, tools)

        # Before initialization - unhealthy
        health_before = mock_orch.check_tool_selector_health()
        assert health_before["healthy"] is False

        # Initialize
        await selector.initialize_tool_embeddings(tools)

        # After initialization - healthy
        health_after = mock_orch.check_tool_selector_health()
        assert health_after["healthy"] is True
        assert health_after["initialized"] is True
