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

"""Chat protocol for LLM interaction.

This module defines the ChatProtocol for LLM chat and streaming interfaces.
This protocol isolates chat-related functionality following the Interface
Segregation Principle (ISP).

Design Principles:
    - ISP: Protocol contains only chat-related methods
    - DIP: Depend on this abstraction, not concrete implementations
    - OCP: Extend via protocol composition, not modification
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Protocol, runtime_checkable

if TYPE_CHECKING:
    from victor.providers.base import StreamChunk


@runtime_checkable
class ChatProtocol(Protocol):
    """LLM chat and streaming interface.

    This protocol defines methods for interacting with LLM providers
    through both synchronous and streaming chat interfaces.

    Implementations:
        - AgentOrchestrator (via IAgentOrchestrator)
        - Mock implementations for testing
    """

    async def chat(
        self,
        message: str,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Send a message and get a response.

        Args:
            message: User message to send to the LLM
            stream: Whether to stream the response
            **kwargs: Additional provider-specific options

        Returns:
            Response from the LLM (string or structured response)

        Examples:
            >>> response = await orchestrator.chat("Hello!")
            >>> print(response)
        """
        ...

    async def stream_chat(
        self,
        message: str,
        **kwargs: Any,
    ) -> AsyncIterator["StreamChunk"]:
        """Stream a chat response.

        Args:
            message: User message to send to the LLM
            **kwargs: Additional provider-specific options

        Yields:
            StreamChunk objects with response content

        Examples:
            >>> async for chunk in orchestrator.stream_chat("Explain Python"):
            ...     print(chunk.content, end="")
        """
        ...


__all__ = [
    "ChatProtocol",
]
