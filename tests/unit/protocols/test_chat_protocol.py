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

"""Tests for ChatProtocol.

Tests the ChatProtocol interface and conformance.
"""

import pytest
from typing import Any
from collections.abc import AsyncIterator

from victor.protocols.chat import ChatProtocol


class MockChatImplementation:
    """Mock implementation of ChatProtocol for testing."""

    async def chat(
        self,
        message: str,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Mock chat implementation."""
        return f"Response to: {message}"

    async def stream_chat(
        self,
        message: str,
        **kwargs: Any,
    ) -> AsyncIterator:
        """Mock stream chat implementation."""
        yield f"Chunk 1: {message}"
        yield f"Chunk 2: {message}"


class TestChatProtocol:
    """Test suite for ChatProtocol."""

    @pytest.mark.asyncio
    async def test_chat_method(self):
        """Test that chat method works correctly."""
        impl = MockChatImplementation()
        result = await impl.chat("Hello!")
        assert result == "Response to: Hello!"

    @pytest.mark.asyncio
    async def test_stream_chat_method(self):
        """Test that stream_chat method works correctly."""
        impl = MockChatImplementation()
        chunks = []
        async for chunk in impl.stream_chat("Hello!"):
            chunks.append(chunk)
        assert len(chunks) == 2
        assert chunks[0] == "Chunk 1: Hello!"
        assert chunks[1] == "Chunk 2: Hello!"

    def test_protocol_conformance(self):
        """Test that mock implements ChatProtocol."""
        impl = MockChatImplementation()
        # This should not raise an error
        assert isinstance(impl, ChatProtocol)

    @pytest.mark.asyncio
    async def test_chat_with_stream_false(self):
        """Test chat with stream=False (default)."""
        impl = MockChatImplementation()
        result = await impl.chat("Test", stream=False)
        assert result == "Response to: Test"

    @pytest.mark.asyncio
    async def test_chat_with_kwargs(self):
        """Test chat with additional kwargs."""
        impl = MockChatImplementation()
        result = await impl.chat("Test", temperature=0.7, max_tokens=100)
        assert result == "Response to: Test"

    @pytest.mark.asyncio
    async def test_stream_chat_with_kwargs(self):
        """Test stream_chat with additional kwargs."""
        impl = MockChatImplementation()
        chunks = []
        async for chunk in impl.stream_chat("Test", temperature=0.7):
            chunks.append(chunk)
        assert len(chunks) == 2


class TestChatProtocolTypeChecking:
    """Test type checking and protocol compliance."""

    def test_chat_protocol_is_protocol(self):
        """Test that ChatProtocol is a Protocol."""
        from typing import Protocol

        assert issubclass(ChatProtocol, Protocol)

    def test_chat_protocol_has_chat_method(self):
        """Test that ChatProtocol defines chat method."""
        assert hasattr(ChatProtocol, "__annotations__")
        # Check that chat is in the protocol
        assert "chat" in dir(ChatProtocol)

    def test_chat_protocol_has_stream_chat_method(self):
        """Test that ChatProtocol defines stream_chat method."""
        assert hasattr(ChatProtocol, "__annotations__")
        # Check that stream_chat is in the protocol
        assert "stream_chat" in dir(ChatProtocol)
