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

"""OpenAI-compatible utilities for Victor providers.

This module provides shared conversion functions for providers that use
OpenAI-compatible APIs (OpenAI, xAI, LMStudio, vLLM, Ollama).

These utilities help reduce code duplication across provider implementations.
"""

import json
import logging
from typing import Any, Optional

from victor.providers.base import Message, ToolDefinition

logger = logging.getLogger(__name__)


def convert_tools_to_openai_format(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Convert standard tools to OpenAI function calling format.

    This format is used by OpenAI, xAI, LMStudio, vLLM, and Ollama.

    Args:
        tools: Standard tool definitions

    Returns:
        OpenAI-formatted tools list
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


def convert_tools_to_anthropic_format(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Convert standard tools to Anthropic format.

    Args:
        tools: Standard tool definitions

    Returns:
        Anthropic-formatted tools list
    """
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }
        for tool in tools
    ]


def convert_messages_to_openai_format(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert standard messages to OpenAI format.

    Args:
        messages: Standard message objects

    Returns:
        OpenAI-formatted messages list
    """
    result = []
    for msg in messages:
        formatted: dict[str, Any] = {
            "role": msg.role,
            "content": msg.content or "",
        }

        # Add tool calls if present
        if msg.tool_calls:
            formatted["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": (
                            json.dumps(tc.get("arguments", {}))
                            if isinstance(tc.get("arguments"), dict)
                            else tc.get("arguments", "{}")
                        ),
                    },
                }
                for tc in msg.tool_calls
            ]

        # Add tool_call_id for tool responses
        if msg.role == "tool" and hasattr(msg, "tool_call_id") and msg.tool_call_id:
            formatted["tool_call_id"] = msg.tool_call_id

        # Add name if present (for function responses)
        if hasattr(msg, "name") and msg.name:
            formatted["name"] = msg.name

        result.append(formatted)

    return result


def parse_openai_tool_calls(
    tool_calls_data: Optional[list[dict[str, Any]]],
) -> Optional[list[dict[str, Any]]]:
    """Parse OpenAI-format tool calls into standard dictionaries.

    Args:
        tool_calls_data: Raw tool calls from OpenAI-compatible API

    Returns:
        List of tool call dictionaries, or None if no tool calls
    """
    if not tool_calls_data:
        return None

    tool_calls = []
    for tc in tool_calls_data:
        func = tc.get("function", {})
        args_str = func.get("arguments", "{}")

        # Parse arguments
        try:
            arguments = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            arguments = {"raw": args_str}

        tool_calls.append(
            {
                "id": tc.get("id", ""),
                "name": func.get("name", ""),
                "arguments": arguments,
            }
        )

    return tool_calls if tool_calls else None


def parse_openai_stream_chunk(chunk_data: dict[str, Any]) -> dict[str, Any]:
    """Parse OpenAI-compatible streaming chunk.

    Args:
        chunk_data: Raw chunk data

    Returns:
        Parsed chunk with content, tool_calls, finish_reason
    """
    result: dict[str, Any] = {
        "content": None,
        "tool_calls": None,
        "finish_reason": None,
    }

    choices = chunk_data.get("choices", [])
    if not choices:
        return result

    choice = choices[0]
    delta = choice.get("delta", {})

    # Extract content
    if "content" in delta:
        result["content"] = delta["content"]

    # Extract tool calls delta
    if "tool_calls" in delta:
        result["tool_calls"] = delta["tool_calls"]

    # Extract finish reason
    if "finish_reason" in choice and choice["finish_reason"]:
        result["finish_reason"] = choice["finish_reason"]

    return result
