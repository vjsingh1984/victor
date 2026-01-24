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

"""Test helper functions for creating test data and mocks."""

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock

from victor.agent.protocols import ToolExecutorProtocol
from victor.providers.base import BaseProvider, Message, StreamChunk


def create_test_completion_response(
    content: str = "Test response",
    role: str = "assistant",
    model: str = "test-model",
    stop_reason: str = "stop",
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    usage: Optional[Dict[str, int]] = None,
) -> Mock:
    """Create a mock CompletionResponse for testing.

    Args:
        content: Response content text
        role: Message role (default: "assistant")
        model: Model name
        stop_reason: Reason for stopping generation
        tool_calls: Optional list of tool calls
        usage: Optional token usage info

    Returns:
        Mock object with CompletionResponse structure
    """
    response = Mock()
    response.content = content
    response.role = role
    response.model = model
    response.stop_reason = stop_reason
    response.tool_calls = tool_calls
    response.usage = usage or {"prompt_tokens": 10, "completion_tokens": 20}
    response.raw_response = None
    response.metadata = None
    return response


def create_test_stream_chunk(
    content: str = "chunk",
    is_final: bool = False,
    stop_reason: Optional[str] = None,
    usage: Optional[Dict[str, int]] = None,
) -> StreamChunk:
    """Create a StreamChunk for testing streaming responses.

    Args:
        content: Content for this chunk
        is_final: Whether this is the final chunk
        stop_reason: Reason for stopping (optional)
        usage: Token usage (optional)

    Returns:
        StreamChunk object
    """
    return StreamChunk(
        content=content,
        is_final=is_final,
        stop_reason=stop_reason,
        usage=usage,
    )


def create_test_messages(
    user_message: str = "Hello, assistant!",
    system_prompt: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> List[Message]:
    """Create test messages for provider chat calls.

    Args:
        user_message: The user's message
        system_prompt: Optional system prompt
        conversation_history: Optional conversation history

    Returns:
        List of Message objects
    """
    messages = []

    if system_prompt:
        messages.append(Message(role="system", content=system_prompt))

    if conversation_history:
        for msg in conversation_history:
            messages.append(Message(role=msg["role"], content=msg["content"]))

    messages.append(Message(role="user", content=user_message))
    return messages


def create_test_tool_definition(
    name: str = "test_tool",
    description: str = "A test tool",
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a test tool definition.

    Args:
        name: Tool name
        description: Tool description
        parameters: Optional tool parameters (JSON Schema)

    Returns:
        Tool definition dictionary
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters
            or {
                "type": "object",
                "properties": {
                    "param1": {
                        "type": "string",
                        "description": "First parameter",
                    }
                },
                "required": ["param1"],
            },
        },
    }


def create_mock_provider(
    model: str = "test-model",
    provider_name: str = "mock",
    response_content: str = "Mock response",
    supports_tools: bool = True,
    supports_streaming: bool = True,
) -> Mock:
    """Create a mock provider for testing.

    Args:
        model: Model name
        provider_name: Provider name
        response_content: Default response content
        supports_tools: Whether provider supports tools
        supports_streaming: Whether provider supports streaming

    Returns:
        Mock provider with async methods
    """
    provider = Mock(spec=BaseProvider)
    provider.name = provider_name
    provider.model = model

    # Mock chat method
    provider.chat = AsyncMock(return_value=create_test_completion_response(response_content))

    # Mock stream method
    async def mock_stream(*args, **kwargs):
        yield create_test_stream_chunk(content=response_content, is_final=False)
        yield create_test_stream_chunk(content="", is_final=True, stop_reason="stop")

    provider.stream = mock_stream

    # Mock capability checks
    provider.supports_tools = Mock(return_value=supports_tools)
    provider.supports_streaming = Mock(return_value=supports_streaming)

    return provider


def create_mock_orchestrator(
    provider: Optional[Mock] = None,
    tool_registry: Optional[Mock] = None,
) -> Mock:
    """Create a mock orchestrator for testing.

    Args:
        provider: Optional mock provider
        tool_registry: Optional mock tool registry

    Returns:
        Mock orchestrator with common methods
    """
    orchestrator = Mock()

    # Provider setup
    if provider:
        orchestrator.provider = provider
    else:
        orchestrator.provider = create_mock_provider()

    # Tool registry setup
    if tool_registry:
        orchestrator.tool_registry = tool_registry
    else:
        orchestrator.tool_registry = Mock(spec=ToolExecutorProtocol)

    # Mock common methods
    orchestrator.chat = AsyncMock(return_value=create_test_completion_response())
    orchestrator.stream_chat = AsyncMock()
    orchestrator.run_tool = AsyncMock(return_value={"result": "success"})
    orchestrator.get_available_tools = Mock(return_value=[])

    return orchestrator


def create_mock_tool_result(
    tool_name: str = "test_tool", result: Any = "success"
) -> Dict[str, Any]:
    """Create a mock tool execution result.

    Args:
        tool_name: Name of the tool
        result: Tool execution result

    Returns:
        Tool result dictionary
    """
    return {
        "tool": tool_name,
        "result": result,
        "success": True,
        "error": None,
    }


def assert_completion_valid(response: Mock, content_check: Optional[str] = None) -> None:
    """Assert that a completion response is valid.

    Args:
        response: Completion response to validate
        content_check: Optional string to check in content

    Raises:
        AssertionError: If response is invalid
    """
    assert response is not None, "Response should not be None"
    assert hasattr(response, "content"), "Response should have content attribute"
    assert hasattr(response, "role"), "Response should have role attribute"
    assert response.role == "assistant", "Response role should be assistant"
    assert response.content, "Response content should not be empty"

    if content_check:
        assert content_check in response.content, f"Response should contain '{content_check}'"


def assert_provider_called(
    mock_provider: Mock,
    call_count: Optional[int] = None,
    min_calls: int = 1,
) -> None:
    """Assert that a mock provider was called.

    Args:
        mock_provider: Mock provider to check
        call_count: Exact expected call count (optional)
        min_calls: Minimum expected call count

    Raises:
        AssertionError: If provider was not called as expected
    """
    assert mock_provider.chat.called, "Provider chat should have been called"

    if call_count is not None:
        assert (
            mock_provider.chat.call_count == call_count
        ), f"Expected {call_count} calls, got {mock_provider.chat.call_count}"
    else:
        assert (
            mock_provider.chat.call_count >= min_calls
        ), f"Expected at least {min_calls} calls, got {mock_provider.chat.call_count}"


def assert_tool_called(mock_tool: Mock, call_count: int = 1) -> None:
    """Assert that a mock tool was called.

    Args:
        mock_tool: Mock tool to check
        call_count: Expected call count

    Raises:
        AssertionError: If tool was not called as expected
    """
    assert mock_tool.called, f"Tool {mock_tool} should have been called"
    assert (
        mock_tool.call_count == call_count
    ), f"Expected {call_count} calls, got {mock_tool.call_count}"


def create_test_conversation() -> List[Dict[str, str]]:
    """Create a test conversation history.

    Returns:
        List of conversation messages
    """
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]


def create_test_settings(
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-5",
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> Mock:
    """Create a mock settings object.

    Args:
        provider: Provider name
        model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens

    Returns:
        Mock settings object
    """
    settings = Mock()
    settings.provider = provider
    settings.model = model
    settings.temperature = temperature
    settings.max_tokens = max_tokens
    settings.api_key = None
    settings.timeout = 30
    return settings


def create_mock_event_bus() -> Mock:
    """Create a mock event bus for testing.

    Returns:
        Mock event bus with async methods
    """
    event_bus = Mock()
    event_bus.publish = AsyncMock()
    event_bus.subscribe = Mock()
    event_bus.unsubscribe = Mock()
    event_bus.connect = AsyncMock()
    event_bus.disconnect = AsyncMock()
    return event_bus


def create_mock_tool_registry(tools: Optional[List[str]] = None) -> Mock:
    """Create a mock tool registry.

    Args:
        tools: Optional list of tool names

    Returns:
        Mock tool registry
    """
    registry = Mock(spec=ToolExecutorProtocol)
    registry.get_all_tools = Mock(return_value=tools or [])
    registry.get_tool = Mock(return_value=Mock())
    registry.execute_tool = AsyncMock(return_value={"result": "success"})
    return registry
