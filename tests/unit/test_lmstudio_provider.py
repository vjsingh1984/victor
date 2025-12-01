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

"""Comprehensive tests for LMStudio provider (OpenAI-compatible).

LMStudio uses the OpenAI provider with a custom base_url.
These tests ensure LMStudio-specific configurations work correctly.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.providers.openai_provider import OpenAIProvider
from victor.providers.base import (
    Message,
    ToolDefinition,
    ProviderError,
)


@pytest.fixture
def lmstudio_provider():
    """Create OpenAIProvider configured for LMStudio."""
    return OpenAIProvider(
        api_key="lm-studio",  # Placeholder key for LMStudio
        base_url="http://localhost:1234/v1",  # LMStudio default port
        timeout=300,  # Longer timeout for local models
        max_retries=3,
    )


@pytest.fixture
def vllm_provider():
    """Create OpenAIProvider configured for vLLM."""
    return OpenAIProvider(
        api_key="EMPTY",  # vLLM doesn't require API key
        base_url="http://localhost:8000/v1",  # vLLM default port
        timeout=300,
        max_retries=3,
    )


class TestLMStudioProviderInitialization:
    """Test LMStudio provider initialization."""

    @pytest.mark.asyncio
    async def test_lmstudio_initialization(self):
        """Test LMStudio provider initialization."""
        provider = OpenAIProvider(
            api_key="lm-studio",
            base_url="http://localhost:1234/v1",
            timeout=300,
        )

        assert provider.api_key == "lm-studio"
        assert provider.base_url == "http://localhost:1234/v1"
        assert provider.timeout == 300
        assert provider.client is not None

    @pytest.mark.asyncio
    async def test_lmstudio_custom_port(self):
        """Test LMStudio with custom port."""
        provider = OpenAIProvider(
            api_key="lm-studio",
            base_url="http://localhost:5678/v1",
        )

        assert provider.base_url == "http://localhost:5678/v1"

    @pytest.mark.asyncio
    async def test_vllm_initialization(self):
        """Test vLLM provider initialization."""
        provider = OpenAIProvider(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1",
        )

        assert provider.api_key == "EMPTY"
        assert provider.base_url == "http://localhost:8000/v1"


class TestLMStudioBasicChat:
    """Test basic chat functionality with LMStudio."""

    @pytest.mark.asyncio
    async def test_lmstudio_chat_simple(self, lmstudio_provider):
        """Test simple chat with LMStudio."""
        mock_message = MagicMock()
        mock_message.content = "Here's a Python function:\n\ndef add(a, b):\n    return a + b"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 15
        mock_usage.completion_tokens = 30
        mock_usage.total_tokens = 45

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model_dump = lambda: {}

        with patch.object(
            lmstudio_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_response

            messages = [Message(role="user", content="Write a Python function to add two numbers")]
            response = await lmstudio_provider.chat(
                messages=messages,
                model="local-model",  # LMStudio uses this as placeholder
                temperature=0.3,
                max_tokens=2048,
            )

            assert (
                response.content == "Here's a Python function:\n\ndef add(a, b):\n    return a + b"
            )
            assert response.role == "assistant"
            assert response.stop_reason == "stop"
            assert response.usage["prompt_tokens"] == 15
            assert response.usage["completion_tokens"] == 30

    @pytest.mark.asyncio
    async def test_lmstudio_chat_coding_task(self, lmstudio_provider):
        """Test LMStudio with coding task."""
        mock_message = MagicMock()
        mock_message.content = (
            "```python\nclass Calculator:\n    def add(self, a, b):\n        return a + b\n```"
        )
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_response.model_dump = lambda: {}

        with patch.object(
            lmstudio_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_response

            messages = [
                Message(role="system", content="You are an expert Python programmer"),
                Message(role="user", content="Create a Calculator class"),
            ]
            response = await lmstudio_provider.chat(
                messages=messages,
                model="Qwen2.5-Coder-7B-Instruct",
                temperature=0.2,
            )

            assert "Calculator" in response.content
            assert "def add" in response.content

    @pytest.mark.asyncio
    async def test_lmstudio_low_temperature(self, lmstudio_provider):
        """Test LMStudio with low temperature for deterministic output."""
        mock_message = MagicMock()
        mock_message.content = (
            "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
        )
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_response.model_dump = lambda: {}

        with patch.object(
            lmstudio_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_response

            messages = [Message(role="user", content="Write a recursive factorial function")]
            response = await lmstudio_provider.chat(
                messages=messages,
                model="local-model",
                temperature=0.1,  # Very low for deterministic output
            )

            # Verify temperature was passed
            call_args = mock_create.call_args
            assert call_args.kwargs["temperature"] == 0.1
            assert "factorial" in response.content


class TestLMStudioToolCalling:
    """Test tool calling with LMStudio."""

    @pytest.mark.asyncio
    async def test_lmstudio_with_tools(self, lmstudio_provider):
        """Test LMStudio with tool definitions."""
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_local_123"
        mock_tool_call.function = MagicMock()
        mock_tool_call.function.name = "write_file"
        mock_tool_call.function.arguments = json.dumps(
            {"file_path": "/tmp/test.py", "content": "print('Hello')"}
        )

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_response.model_dump = lambda: {}

        with patch.object(
            lmstudio_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_response

            tools = [
                ToolDefinition(
                    name="write_file",
                    description="Write content to a file",
                    parameters={
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["file_path", "content"],
                    },
                )
            ]

            messages = [Message(role="user", content="Create a test.py file that prints Hello")]
            response = await lmstudio_provider.chat(
                messages=messages,
                model="local-model",
                tools=tools,
            )

            assert response.tool_calls is not None
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0]["name"] == "write_file"
            assert "test.py" in response.tool_calls[0]["arguments"]

    @pytest.mark.asyncio
    async def test_lmstudio_multiple_tools(self, lmstudio_provider):
        """Test LMStudio with multiple tool definitions."""
        mock_message = MagicMock()
        mock_message.content = "I'll use these tools"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_response.model_dump = lambda: {}

        with patch.object(
            lmstudio_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_response

            tools = [
                ToolDefinition(
                    name="read_file",
                    description="Read a file",
                    parameters={"type": "object", "properties": {"path": {"type": "string"}}},
                ),
                ToolDefinition(
                    name="write_file",
                    description="Write a file",
                    parameters={"type": "object", "properties": {"path": {"type": "string"}}},
                ),
                ToolDefinition(
                    name="execute_bash",
                    description="Execute bash command",
                    parameters={"type": "object", "properties": {"command": {"type": "string"}}},
                ),
            ]

            messages = [Message(role="user", content="Help me with file operations")]
            await lmstudio_provider.chat(
                messages=messages,
                model="local-model",
                tools=tools,
            )

            # Verify all tools were passed
            call_args = mock_create.call_args
            assert len(call_args.kwargs["tools"]) == 3


class TestLMStudioStreaming:
    """Test streaming functionality with LMStudio."""

    @pytest.mark.asyncio
    async def test_lmstudio_streaming(self, lmstudio_provider):
        """Test streaming with LMStudio."""
        # Create mock stream chunks
        chunks_data = [
            ("def ", False),
            ("add(a, ", False),
            ("b):\n    ", False),
            ("return ", False),
            ("a + b", False),
            ("", True),  # Final chunk
        ]

        mock_chunks = []
        for content, is_final in chunks_data:
            mock_delta = MagicMock()
            mock_delta.content = content
            mock_choice = MagicMock()
            mock_choice.delta = mock_delta
            mock_choice.finish_reason = "stop" if is_final else None
            mock_chunk = MagicMock()
            mock_chunk.choices = [mock_choice]
            mock_chunks.append(mock_chunk)

        async def async_iter():
            for chunk in mock_chunks:
                yield chunk

        with patch.object(
            lmstudio_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = async_iter()

            messages = [Message(role="user", content="Write an add function")]
            chunks = []

            async for chunk in lmstudio_provider.stream(
                messages=messages,
                model="local-model",
            ):
                chunks.append(chunk)

            # Verify chunks
            assert len(chunks) == 6
            assert chunks[0].content == "def "
            assert chunks[-1].is_final is True

            # Verify stream=True was passed
            call_args = mock_create.call_args
            assert call_args.kwargs["stream"] is True

    @pytest.mark.asyncio
    async def test_lmstudio_streaming_with_tools(self, lmstudio_provider):
        """Test streaming with tools."""
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock(delta=MagicMock(content="test"), finish_reason=None)]

        async def async_iter():
            yield mock_chunk

        with patch.object(
            lmstudio_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = async_iter()

            tools = [
                ToolDefinition(
                    name="calculate",
                    description="Calculate",
                    parameters={"type": "object"},
                )
            ]

            messages = [Message(role="user", content="Calculate")]
            chunks = []

            async for chunk in lmstudio_provider.stream(
                messages=messages,
                model="local-model",
                tools=tools,
            ):
                chunks.append(chunk)

            # Verify tools were passed
            call_args = mock_create.call_args
            assert "tools" in call_args.kwargs


class TestVLLMProvider:
    """Test vLLM-specific configurations."""

    @pytest.mark.asyncio
    async def test_vllm_chat(self, vllm_provider):
        """Test chat with vLLM."""
        mock_message = MagicMock()
        mock_message.content = "vLLM response"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_response.model_dump = lambda: {}

        with patch.object(
            vllm_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_response

            messages = [Message(role="user", content="Test")]
            response = await vllm_provider.chat(
                messages=messages,
                model="Qwen/Qwen2.5-Coder-7B-Instruct",
            )

            assert response.content == "vLLM response"

    @pytest.mark.asyncio
    async def test_vllm_with_full_model_name(self, vllm_provider):
        """Test vLLM with HuggingFace model name."""
        mock_message = MagicMock()
        mock_message.content = "Response"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_response.model_dump = lambda: {}

        with patch.object(
            vllm_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_response

            messages = [Message(role="user", content="Test")]
            await vllm_provider.chat(
                messages=messages,
                model="deepseek-ai/deepseek-coder-6.7b-instruct",
            )

            # Verify model name was passed correctly
            call_args = mock_create.call_args
            assert call_args.kwargs["model"] == "deepseek-ai/deepseek-coder-6.7b-instruct"

    @pytest.mark.asyncio
    async def test_vllm_custom_port(self):
        """Test vLLM with custom port."""
        provider = OpenAIProvider(
            api_key="EMPTY",
            base_url="http://localhost:9000/v1",
        )

        assert provider.base_url == "http://localhost:9000/v1"

    @pytest.mark.asyncio
    async def test_vllm_coding_task(self, vllm_provider):
        """Test vLLM with coding task."""
        mock_message = MagicMock()
        mock_message.content = "```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 20
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 70

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model_dump = lambda: {}

        with patch.object(
            vllm_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_response

            messages = [
                Message(role="system", content="You are an expert Python programmer"),
                Message(role="user", content="Write a fibonacci function"),
            ]
            response = await vllm_provider.chat(
                messages=messages,
                model="codellama/CodeLlama-7b-Instruct-hf",
                temperature=0.2,
            )

            assert "fibonacci" in response.content
            assert response.usage["prompt_tokens"] == 20
            assert response.usage["completion_tokens"] == 50


class TestVLLMToolCalling:
    """Test tool calling with vLLM."""

    @pytest.mark.asyncio
    async def test_vllm_with_tools(self, vllm_provider):
        """Test vLLM with tool definitions."""
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_vllm_123"
        mock_tool_call.function = MagicMock()
        mock_tool_call.function.name = "execute_code"
        mock_tool_call.function.arguments = json.dumps(
            {"language": "python", "code": "print('Hello from vLLM')"}
        )

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_response.model_dump = lambda: {}

        with patch.object(
            vllm_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_response

            tools = [
                ToolDefinition(
                    name="execute_code",
                    description="Execute code in specified language",
                    parameters={
                        "type": "object",
                        "properties": {
                            "language": {"type": "string"},
                            "code": {"type": "string"},
                        },
                        "required": ["language", "code"],
                    },
                )
            ]

            messages = [Message(role="user", content="Execute a hello world in Python")]
            response = await vllm_provider.chat(
                messages=messages,
                model="Qwen/Qwen2.5-Coder-7B-Instruct",
                tools=tools,
            )

            assert response.tool_calls is not None
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0]["name"] == "execute_code"
            assert "python" in response.tool_calls[0]["arguments"].lower()

    @pytest.mark.asyncio
    async def test_vllm_multiple_tools(self, vllm_provider):
        """Test vLLM with multiple tool definitions."""
        mock_message = MagicMock()
        mock_message.content = "I can use these tools to help"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_response.model_dump = lambda: {}

        with patch.object(
            vllm_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_response

            tools = [
                ToolDefinition(
                    name="search_code",
                    description="Search codebase",
                    parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                ),
                ToolDefinition(
                    name="analyze_code",
                    description="Analyze code quality",
                    parameters={"type": "object", "properties": {"file": {"type": "string"}}},
                ),
                ToolDefinition(
                    name="refactor_code",
                    description="Refactor code",
                    parameters={"type": "object", "properties": {"code": {"type": "string"}}},
                ),
            ]

            messages = [Message(role="user", content="Help me with code analysis")]
            await vllm_provider.chat(
                messages=messages,
                model="Qwen/Qwen2.5-Coder-7B-Instruct",
                tools=tools,
            )

            # Verify all tools were passed
            call_args = mock_create.call_args
            assert len(call_args.kwargs["tools"]) == 3


class TestVLLMStreaming:
    """Test streaming functionality with vLLM."""

    @pytest.mark.asyncio
    async def test_vllm_streaming(self, vllm_provider):
        """Test streaming with vLLM."""
        # Create mock stream chunks
        chunks_data = [
            ("class ", False),
            ("Person:\n    ", False),
            ("def __init__(", False),
            ("self, name):\n        ", False),
            ("self.name = name", False),
            ("", True),  # Final chunk
        ]

        mock_chunks = []
        for content, is_final in chunks_data:
            mock_delta = MagicMock()
            mock_delta.content = content
            mock_choice = MagicMock()
            mock_choice.delta = mock_delta
            mock_choice.finish_reason = "stop" if is_final else None
            mock_chunk = MagicMock()
            mock_chunk.choices = [mock_choice]
            mock_chunks.append(mock_chunk)

        async def async_iter():
            for chunk in mock_chunks:
                yield chunk

        with patch.object(
            vllm_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = async_iter()

            messages = [Message(role="user", content="Create a Person class")]
            chunks = []

            async for chunk in vllm_provider.stream(
                messages=messages,
                model="Qwen/Qwen2.5-Coder-7B-Instruct",
            ):
                chunks.append(chunk)

            # Verify chunks
            assert len(chunks) == 6
            assert chunks[0].content == "class "
            assert chunks[-1].is_final is True

            # Verify stream=True was passed
            call_args = mock_create.call_args
            assert call_args.kwargs["stream"] is True

    @pytest.mark.asyncio
    async def test_vllm_streaming_with_tools(self, vllm_provider):
        """Test vLLM streaming with tools."""
        mock_chunk = MagicMock()
        mock_chunk.choices = [
            MagicMock(delta=MagicMock(content="analyzing..."), finish_reason=None)
        ]

        async def async_iter():
            yield mock_chunk

        with patch.object(
            vllm_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = async_iter()

            tools = [
                ToolDefinition(
                    name="analyze",
                    description="Analyze data",
                    parameters={"type": "object"},
                )
            ]

            messages = [Message(role="user", content="Analyze this")]
            chunks = []

            async for chunk in vllm_provider.stream(
                messages=messages,
                model="Qwen/Qwen2.5-Coder-7B-Instruct",
                tools=tools,
            ):
                chunks.append(chunk)

            # Verify tools were passed
            call_args = mock_create.call_args
            assert "tools" in call_args.kwargs

    @pytest.mark.asyncio
    async def test_vllm_streaming_long_response(self, vllm_provider):
        """Test vLLM streaming with long response."""
        # Simulate a long code generation
        chunks_data = [(f"line {i}\n", False) for i in range(20)]
        chunks_data.append(("", True))

        mock_chunks = []
        for content, is_final in chunks_data:
            mock_delta = MagicMock()
            mock_delta.content = content
            mock_choice = MagicMock()
            mock_choice.delta = mock_delta
            mock_choice.finish_reason = "stop" if is_final else None
            mock_chunk = MagicMock()
            mock_chunk.choices = [mock_choice]
            mock_chunks.append(mock_chunk)

        async def async_iter():
            for chunk in mock_chunks:
                yield chunk

        with patch.object(
            vllm_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = async_iter()

            messages = [Message(role="user", content="Generate long code")]
            chunks = []

            async for chunk in vllm_provider.stream(
                messages=messages,
                model="Qwen/Qwen2.5-Coder-7B-Instruct",
                max_tokens=8192,
            ):
                chunks.append(chunk)

            # Verify we got all chunks
            assert len(chunks) == 21
            assert chunks[-1].is_final is True


class TestVLLMAdvancedFeatures:
    """Test advanced vLLM features."""

    @pytest.mark.asyncio
    async def test_vllm_large_context(self, vllm_provider):
        """Test vLLM with large context window."""
        mock_message = MagicMock()
        mock_message.content = "Response with large context"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_response.model_dump = lambda: {}

        with patch.object(
            vllm_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_response

            # Create large context with many messages
            messages = [
                Message(role="system", content="You are a code analysis expert"),
            ]
            for i in range(15):
                messages.append(Message(role="user", content=f"Analyze function {i}"))
                messages.append(Message(role="assistant", content=f"Analysis for function {i}"))

            messages.append(Message(role="user", content="Summarize all analyses"))

            await vllm_provider.chat(
                messages=messages,
                model="Qwen/Qwen2.5-Coder-7B-Instruct",
                max_tokens=16384,  # Large context
            )

            # Verify all messages were passed
            call_args = mock_create.call_args
            assert len(call_args.kwargs["messages"]) == 32

    @pytest.mark.asyncio
    async def test_vllm_custom_parameters(self, vllm_provider):
        """Test vLLM with custom sampling parameters."""
        mock_message = MagicMock()
        mock_message.content = "Response"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_response.model_dump = lambda: {}

        with patch.object(
            vllm_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_response

            messages = [Message(role="user", content="Generate code")]
            await vllm_provider.chat(
                messages=messages,
                model="Qwen/Qwen2.5-Coder-7B-Instruct",
                temperature=0.1,
                top_p=0.9,
                presence_penalty=0.5,
                frequency_penalty=0.5,
            )

            # Verify custom parameters were passed
            call_args = mock_create.call_args
            assert call_args.kwargs["temperature"] == 0.1
            assert "top_p" in call_args.kwargs
            assert "presence_penalty" in call_args.kwargs
            assert "frequency_penalty" in call_args.kwargs

    @pytest.mark.asyncio
    async def test_vllm_low_temperature(self, vllm_provider):
        """Test vLLM with very low temperature for deterministic output."""
        mock_message = MagicMock()
        mock_message.content = "def sort_list(lst):\n    return sorted(lst)"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_response.model_dump = lambda: {}

        with patch.object(
            vllm_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_response

            messages = [Message(role="user", content="Write a function to sort a list")]
            response = await vllm_provider.chat(
                messages=messages,
                model="Qwen/Qwen2.5-Coder-7B-Instruct",
                temperature=0.0,  # Deterministic
            )

            # Verify temperature was passed
            call_args = mock_create.call_args
            assert call_args.kwargs["temperature"] == 0.0
            assert "sort" in response.content.lower()

    @pytest.mark.asyncio
    async def test_vllm_close(self, vllm_provider):
        """Test closing the vLLM provider."""
        with patch.object(
            vllm_provider.client,
            "close",
            new_callable=AsyncMock,
        ) as mock_close:
            await vllm_provider.close()
            mock_close.assert_called_once()


class TestErrorHandling:
    """Test error handling for local providers."""

    @pytest.mark.asyncio
    async def test_lmstudio_connection_error(self, lmstudio_provider):
        """Test connection error when LMStudio is not running."""
        with patch.object(
            lmstudio_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.side_effect = Exception("Connection refused: http://localhost:1234")

            messages = [Message(role="user", content="Test")]

            with pytest.raises(ProviderError) as exc_info:
                await lmstudio_provider.chat(
                    messages=messages,
                    model="local-model",
                )

            assert "Connection refused" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_vllm_model_not_loaded(self, vllm_provider):
        """Test error when vLLM model is not loaded."""
        with patch.object(
            vllm_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.side_effect = Exception("Model not found")

            messages = [Message(role="user", content="Test")]

            with pytest.raises(ProviderError):
                await vllm_provider.chat(
                    messages=messages,
                    model="non-existent-model",
                )

    @pytest.mark.asyncio
    async def test_lmstudio_timeout(self, lmstudio_provider):
        """Test timeout handling."""
        with patch.object(
            lmstudio_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.side_effect = Exception("Request timeout")

            messages = [Message(role="user", content="Test")]

            with pytest.raises(ProviderError) as exc_info:
                await lmstudio_provider.chat(
                    messages=messages,
                    model="local-model",
                )

            assert "timeout" in str(exc_info.value).lower()


class TestProviderProperties:
    """Test provider properties for local models."""

    @pytest.mark.asyncio
    async def test_lmstudio_name(self, lmstudio_provider):
        """Test provider name."""
        assert lmstudio_provider.name == "openai"

    @pytest.mark.asyncio
    async def test_lmstudio_supports_tools(self, lmstudio_provider):
        """Test tools support."""
        assert lmstudio_provider.supports_tools() is True

    @pytest.mark.asyncio
    async def test_lmstudio_supports_streaming(self, lmstudio_provider):
        """Test streaming support."""
        assert lmstudio_provider.supports_streaming() is True

    @pytest.mark.asyncio
    async def test_vllm_supports_tools(self, vllm_provider):
        """Test vLLM tools support."""
        assert vllm_provider.supports_tools() is True

    @pytest.mark.asyncio
    async def test_lmstudio_base_url(self, lmstudio_provider):
        """Test base URL is correctly set."""
        assert lmstudio_provider.base_url == "http://localhost:1234/v1"

    @pytest.mark.asyncio
    async def test_vllm_base_url(self, vllm_provider):
        """Test vLLM base URL."""
        assert vllm_provider.base_url == "http://localhost:8000/v1"


class TestAdvancedFeatures:
    """Test advanced features with local providers."""

    @pytest.mark.asyncio
    async def test_lmstudio_context_window(self, lmstudio_provider):
        """Test with large context."""
        mock_message = MagicMock()
        mock_message.content = "Response with context"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_response.model_dump = lambda: {}

        with patch.object(
            lmstudio_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_response

            # Create long context
            messages = [
                Message(role="system", content="You are a helpful assistant"),
            ]
            for i in range(10):
                messages.append(Message(role="user", content=f"Message {i}"))
                messages.append(Message(role="assistant", content=f"Response {i}"))

            messages.append(Message(role="user", content="Final question"))

            await lmstudio_provider.chat(
                messages=messages,
                model="local-model",
                max_tokens=8192,
            )

            # Verify all messages were passed
            call_args = mock_create.call_args
            assert len(call_args.kwargs["messages"]) == 22

    @pytest.mark.asyncio
    async def test_lmstudio_custom_parameters(self, lmstudio_provider):
        """Test with custom parameters."""
        mock_message = MagicMock()
        mock_message.content = "Response"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_response.model_dump = lambda: {}

        with patch.object(
            lmstudio_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_response

            messages = [Message(role="user", content="Test")]
            await lmstudio_provider.chat(
                messages=messages,
                model="local-model",
                temperature=0.1,
                top_p=0.95,
                presence_penalty=0.2,
                frequency_penalty=0.3,
            )

            # Verify custom parameters were passed
            call_args = mock_create.call_args
            assert call_args.kwargs["temperature"] == 0.1
            assert "top_p" in call_args.kwargs
            assert "presence_penalty" in call_args.kwargs
            assert "frequency_penalty" in call_args.kwargs

    @pytest.mark.asyncio
    async def test_lmstudio_close(self, lmstudio_provider):
        """Test closing the provider."""
        with patch.object(
            lmstudio_provider.client,
            "close",
            new_callable=AsyncMock,
        ) as mock_close:
            await lmstudio_provider.close()
            mock_close.assert_called_once()


class TestModelSharing:
    """Test model sharing scenarios."""

    @pytest.mark.asyncio
    async def test_lmstudio_with_ollama_model(self, lmstudio_provider):
        """Test LMStudio using a model shared from Ollama via Gollama."""
        mock_message = MagicMock()
        mock_message.content = "Response from shared model"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_response.model_dump = lambda: {}

        with patch.object(
            lmstudio_provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_response

            messages = [Message(role="user", content="Test with shared model")]
            response = await lmstudio_provider.chat(
                messages=messages,
                model="qwen2.5-coder-7b",  # Model linked from Ollama
                temperature=0.3,
            )

            assert response.content == "Response from shared model"

            # Verify model name was passed
            call_args = mock_create.call_args
            assert call_args.kwargs["model"] == "qwen2.5-coder-7b"
