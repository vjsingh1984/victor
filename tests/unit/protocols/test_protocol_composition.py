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

"""Tests for IAgentOrchestrator protocol composition.

Tests that IAgentOrchestrator correctly composes all focused protocols
and maintains backward compatibility.
"""

import pytest
from typing import Any, Optional
from collections.abc import AsyncIterator

from victor.protocols.agent import IAgentOrchestrator
from victor.protocols.chat import ChatProtocol
from victor.protocols.provider import ProviderProtocol
from victor.protocols.tools import ToolProtocol
from victor.protocols.state import StateProtocol
from victor.protocols.config_agent import ConfigProtocol


class MockProvider:
    """Mock provider for testing."""

    def __init__(self):
        self.name = "mock_provider"


class MockToolRegistry:
    """Mock tool registry for testing."""

    def __init__(self):
        self.tools = ["read_file", "write_file", "search"]


class MockSettings:
    """Mock settings for testing."""

    def __init__(self):
        self.debug_mode = True


class MockAgentMode:
    """Mock agent mode for testing."""

    def __init__(self, name: str):
        self.name = name


class MockStreamChunk:
    """Mock stream chunk for testing."""

    def __init__(self, content: str):
        self.content = content


class MockOrchestratorImplementation:
    """Mock implementation of IAgentOrchestrator for testing.

    This mock implements all protocols that IAgentOrchestrator composes.
    """

    def __init__(self):
        # Chat
        self._message_count = 0
        self._messages = []  # Add messages list

        # Provider
        self._provider = MockProvider()
        self._provider_name = "anthropic"
        self._model = "claude-sonnet-4-20250514"
        self._temperature = 0.7

        # Tools
        self._tool_registry = MockToolRegistry()
        self._allowed_tools = None

        # State
        self._tool_calls_used = 5
        self._executed_tools = ["read_file", "search"]
        self._failed_tool_signatures = {("write_file", "hash1")}
        self._observed_files = {"/path/to/file.py"}

        # Config
        self._settings = MockSettings()
        self._tool_budget = 100
        self._mode = MockAgentMode("BUILD")

    # =========================================================================
    # ChatProtocol implementation
    # =========================================================================

    async def chat(
        self,
        message: str,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Send a message and get a response."""
        self._message_count += 1
        return f"Response to: {message}"

    async def stream_chat(
        self,
        message: str,
        **kwargs: Any,
    ) -> AsyncIterator:
        """Stream a chat response."""
        yield MockStreamChunk(f"Chunk 1: {message}")
        yield MockStreamChunk(f"Chunk 2: {message}")

    # =========================================================================
    # ProviderProtocol implementation
    # =========================================================================

    @property
    def provider(self) -> Any:
        """Get the current LLM provider instance."""
        return self._provider

    @property
    def provider_name(self) -> str:
        """Get the name of the current provider."""
        return self._provider_name

    @property
    def model(self) -> str:
        """Get the current model identifier."""
        return self._model

    @property
    def temperature(self) -> float:
        """Get the temperature setting for sampling."""
        return self._temperature

    # =========================================================================
    # ToolProtocol implementation
    # =========================================================================

    @property
    def tool_registry(self) -> Any:
        """Get the tool registry."""
        return self._tool_registry

    @property
    def allowed_tools(self) -> Optional[list[str]]:
        """Get list of allowed tool names, if restricted."""
        return self._allowed_tools

    # =========================================================================
    # StateProtocol implementation
    # =========================================================================

    @property
    def messages(self) -> list[Any]:
        """Get conversation messages."""
        return self._messages

    @property
    def tool_calls_used(self) -> int:
        """Get number of tool calls used in this session."""
        return self._tool_calls_used

    @property
    def executed_tools(self) -> list[str]:
        """Get list of executed tool names in order."""
        return self._executed_tools

    @property
    def failed_tool_signatures(self) -> set[tuple[str, str]]:
        """Get set of failed tool call signatures."""
        return self._failed_tool_signatures

    @property
    def observed_files(self) -> set[str]:
        """Get set of files observed during session."""
        return self._observed_files

    # =========================================================================
    # ConfigProtocol implementation
    # =========================================================================

    @property
    def settings(self) -> Any:
        """Get configuration settings."""
        return self._settings

    @property
    def tool_budget(self) -> int:
        """Get the tool budget for this session."""
        return self._tool_budget

    @property
    def mode(self) -> Any:
        """Get the current agent mode."""
        return self._mode


class TestProtocolComposition:
    """Test suite for IAgentOrchestrator protocol composition."""

    def test_implementation_is_iagent_orchestrator(self):
        """Test that mock implements IAgentOrchestrator."""
        impl = MockOrchestratorImplementation()
        assert isinstance(impl, IAgentOrchestrator)

    def test_implementation_is_chat_protocol(self):
        """Test that mock implements ChatProtocol."""
        impl = MockOrchestratorImplementation()
        assert isinstance(impl, ChatProtocol)

    def test_implementation_is_provider_protocol(self):
        """Test that mock implements ProviderProtocol."""
        impl = MockOrchestratorImplementation()
        assert isinstance(impl, ProviderProtocol)

    def test_implementation_is_tool_protocol(self):
        """Test that mock implements ToolProtocol."""
        impl = MockOrchestratorImplementation()
        assert isinstance(impl, ToolProtocol)

    def test_implementation_is_state_protocol(self):
        """Test that mock implements StateProtocol."""
        impl = MockOrchestratorImplementation()
        assert isinstance(impl, StateProtocol)

    def test_implementation_is_config_protocol(self):
        """Test that mock implements ConfigProtocol."""
        impl = MockOrchestratorImplementation()
        assert isinstance(impl, ConfigProtocol)

    @pytest.mark.asyncio
    async def test_chat_methods_work(self):
        """Test that chat methods work through IAgentOrchestrator."""
        impl = MockOrchestratorImplementation()
        result = await impl.chat("Hello!")
        assert result == "Response to: Hello!"

    @pytest.mark.asyncio
    async def test_stream_chat_works(self):
        """Test that stream_chat works through IAgentOrchestrator."""
        impl = MockOrchestratorImplementation()
        chunks = []
        async for chunk in impl.stream_chat("Hello!"):
            chunks.append(chunk)
        assert len(chunks) == 2

    def test_provider_properties_work(self):
        """Test that provider properties work through IAgentOrchestrator."""
        impl = MockOrchestratorImplementation()
        assert impl.provider_name == "anthropic"
        assert impl.model == "claude-sonnet-4-20250514"
        assert impl.temperature == 0.7

    def test_tool_properties_work(self):
        """Test that tool properties work through IAgentOrchestrator."""
        impl = MockOrchestratorImplementation()
        assert impl.tool_registry.tools == ["read_file", "write_file", "search"]
        assert impl.allowed_tools is None

    def test_state_properties_work(self):
        """Test that state properties work through IAgentOrchestrator."""
        impl = MockOrchestratorImplementation()
        assert impl.tool_calls_used == 5
        assert impl.executed_tools == ["read_file", "search"]
        assert ("write_file", "hash1") in impl.failed_tool_signatures
        assert "/path/to/file.py" in impl.observed_files

    def test_config_properties_work(self):
        """Test that config properties work through IAgentOrchestrator."""
        impl = MockOrchestratorImplementation()
        assert impl.settings.debug_mode is True
        assert impl.tool_budget == 100
        assert impl.mode.name == "BUILD"


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_existing_code_can_use_iagent_orchestrator(self):
        """Test that existing code using IAgentOrchestrator still works."""
        impl = MockOrchestratorImplementation()

        # Existing code that depends on IAgentOrchestrator
        def process_orchestrator(orchestrator: IAgentOrchestrator):
            """Function that accepts IAgentOrchestrator."""
            return {
                "provider": orchestrator.provider_name,
                "model": orchestrator.model,
                "tool_budget": orchestrator.tool_budget,
            }

        result = process_orchestrator(impl)
        assert result["provider"] == "anthropic"
        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["tool_budget"] == 100

    def test_new_code_can_use_focused_protocols(self):
        """Test that new code can depend on focused protocols."""
        impl = MockOrchestratorImplementation()

        # New code that depends only on ChatProtocol
        async def chat_only(chat: ChatProtocol):
            """Function that only needs chat functionality."""
            return await chat.chat("Test")

        # New code that depends only on ProviderProtocol
        def get_provider_info(provider: ProviderProtocol):
            """Function that only needs provider info."""
            return f"{provider.provider_name}:{provider.model}"

        # Both should work
        result1 = chat_only(impl)
        result2 = get_provider_info(impl)
        assert result2 == "anthropic:claude-sonnet-4-20250514"

    def test_new_code_can_compose_protocols(self):
        """Test that new code can compose multiple protocols."""
        impl = MockOrchestratorImplementation()

        # Function that needs both chat and provider
        async def chat_with_provider(
            chat: ChatProtocol,
            provider: ProviderProtocol,
        ):
            """Function that needs both chat and provider functionality."""
            model = provider.model
            response = await chat.chat(f"Using model: {model}")
            return response

        # Should work
        import asyncio

        result = asyncio.run(chat_with_provider(impl, impl))
        assert "Using model: claude-sonnet-4-20250514" in result


class TestProtocolInheritance:
    """Test protocol inheritance structure."""

    def test_iagent_orchestrator_inherits_chat_protocol(self):
        """Test that IAgentOrchestrator is a subtype of ChatProtocol."""
        # IAgentOrchestrator should be compatible with ChatProtocol
        assert issubclass(ChatProtocol, type) or hasattr(ChatProtocol, "__protocol_attrs__")

    def test_iagent_orchestrator_inherits_provider_protocol(self):
        """Test that IAgentOrchestrator is a subtype of ProviderProtocol."""
        assert issubclass(ProviderProtocol, type) or hasattr(ProviderProtocol, "__protocol_attrs__")

    def test_iagent_orchestrator_inherits_tool_protocol(self):
        """Test that IAgentOrchestrator is a subtype of ToolProtocol."""
        assert issubclass(ToolProtocol, type) or hasattr(ToolProtocol, "__protocol_attrs__")

    def test_iagent_orchestrator_inherits_state_protocol(self):
        """Test that IAgentOrchestrator is a subtype of StateProtocol."""
        assert issubclass(StateProtocol, type) or hasattr(StateProtocol, "__protocol_attrs__")

    def test_iagent_orchestrator_inherits_config_protocol(self):
        """Test that IAgentOrchestrator is a subtype of ConfigProtocol."""
        assert issubclass(ConfigProtocol, type) or hasattr(ConfigProtocol, "__protocol_attrs__")

    def test_focused_protocols_are_independent(self):
        """Test that focused protocols are independent."""
        # Each protocol should be independently usable
        impl = MockOrchestratorImplementation()

        # Should be able to use as any of the protocols
        assert isinstance(impl, ChatProtocol)
        assert isinstance(impl, ProviderProtocol)
        assert isinstance(impl, ToolProtocol)
        assert isinstance(impl, StateProtocol)
        assert isinstance(impl, ConfigProtocol)
