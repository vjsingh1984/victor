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
from typing import Any, Dict, List, Optional

import httpx

from victor.providers.base import Message, ToolDefinition

logger = logging.getLogger(__name__)


def convert_tools_to_openai_format(tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
    """Convert standard tools to OpenAI function calling format.

    This format is used by OpenAI, xAI, LMStudio, vLLM, and Ollama.

    Args:
        tools: Standard tool definitions

    Returns:
        OpenAI-formatted tools list
    """
    result = [
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
    # Log tool schemas sent to provider
    tool_sigs = []
    for t in result:
        fn = t["function"]
        props = fn.get("parameters", {}).get("properties", {})
        tool_sigs.append(f"{fn['name']}({list(props.keys())})")
    logger.debug(
        "[ToolSchemas→LLM] OpenAI format: %d tools: %s",
        len(result),
        ", ".join(tool_sigs),
    )
    return result


def convert_tools_to_anthropic_format(
    tools: List[ToolDefinition],
) -> List[Dict[str, Any]]:
    """Convert standard tools to Anthropic format.

    Args:
        tools: Standard tool definitions

    Returns:
        Anthropic-formatted tools list
    """
    result = [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }
        for tool in tools
    ]
    # Log tool schemas sent to provider
    tool_sigs = []
    for t in result:
        props = t.get("input_schema", {}).get("properties", {})
        tool_sigs.append(f"{t['name']}({list(props.keys())})")
    logger.debug(
        "[ToolSchemas→LLM] Anthropic format: %d tools: %s",
        len(result),
        ", ".join(tool_sigs),
    )
    return result


def convert_messages_to_openai_format(messages: List[Message]) -> List[Dict[str, Any]]:
    """Convert standard messages to OpenAI format.

    Args:
        messages: Standard message objects

    Returns:
        OpenAI-formatted messages list
    """
    result = []
    for msg in messages:
        formatted: Dict[str, Any] = {
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
    tool_calls_data: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
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


def parse_openai_stream_chunk(chunk_data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse OpenAI-compatible streaming chunk.

    Args:
        chunk_data: Raw chunk data

    Returns:
        Parsed chunk with content, tool_calls, finish_reason
    """
    result: Dict[str, Any] = {
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


def fix_orphaned_tool_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fix orphaned tool_calls/responses after compaction.

    OpenAI-compatible APIs require strict pairing:
    - Every ``tool_calls[].id`` in an assistant message must have a matching
      ``role=tool`` response with the same ``tool_call_id``.
    - Every ``role=tool`` message must have a corresponding assistant
      ``tool_calls`` entry.

    After context compaction, either side may be missing. This function:
    1. Strips ``tool_calls`` from assistant messages whose responses are gone.
    2. Removes ``role=tool`` messages whose assistant ``tool_calls`` are gone.

    Shared utility for all OpenAI-compatible providers (Z.AI, DeepSeek,
    OpenAI, xAI, Groq, LMStudio, etc.).

    Args:
        messages: List of formatted message dicts (already converted to
                  OpenAI format with ``role``, ``content``, ``tool_calls``,
                  ``tool_call_id`` keys).

    Returns:
        Cleaned message list with consistent tool_calls/response pairing.
    """
    # Collect all tool_call IDs declared by assistant messages
    declared_tool_call_ids: set = set()
    for msg in messages:
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                if "id" in tc:
                    declared_tool_call_ids.add(tc["id"])

    # Collect all tool_call IDs present in tool responses
    present_response_ids = {
        m.get("tool_call_id") for m in messages if m.get("role") == "tool" and m.get("tool_call_id")
    }

    # Strip assistant tool_calls whose responses were compacted away
    for msg in messages:
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            tc_ids = {tc["id"] for tc in msg["tool_calls"] if "id" in tc}
            if not tc_ids.issubset(present_response_ids):
                del msg["tool_calls"]
                if msg.get("content") is None:
                    msg["content"] = ""

    # Remove orphaned tool responses whose assistant tool_calls were
    # compacted away
    messages = [
        m
        for m in messages
        if not (
            m.get("role") == "tool"
            and m.get("tool_call_id")
            and m["tool_call_id"] not in declared_tool_call_ids
        )
    ]

    return messages


def build_openai_messages(messages: List[Message]) -> List[Dict[str, Any]]:
    """Build a fully-validated OpenAI-format message list from standard Message objects.

    Extracted from ZAIProvider._build_request_payload — the most complete message
    serialization in the codebase. Shared across all OpenAI-compatible providers.

    Handles:
    - Assistant messages: serializes tool_calls to OpenAI ``{id, type, function}`` format;
      sets content=None when tool_calls are present (required by GLM/OpenAI spec)
    - Tool response messages: copies tool_call_id; generates a fallback MD5 ID when
      the original is missing or orphaned (prevents 400 errors after context compaction)
    - Second-pass orphaned cleanup via fix_orphaned_tool_messages()

    Args:
        messages: Standard Message objects from the conversation history

    Returns:
        Cleaned OpenAI-format message list ready to send to any /v1/chat/completions API
    """
    formatted: List[Dict[str, Any]] = []
    valid_tool_call_ids: set = set()

    for msg in messages:
        entry: Dict[str, Any] = {
            "role": msg.role,
            "content": msg.content if msg.content is not None else "",
        }

        if msg.role == "assistant":
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                openai_tcs = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        tc_id = tc.get("id", "")
                        tc_name = tc.get("name", "")
                        tc_args = tc.get("arguments", {})
                        if isinstance(tc_args, dict):
                            tc_args = json.dumps(tc_args)
                        openai_tcs.append(
                            {
                                "id": tc_id,
                                "type": "function",
                                "function": {"name": tc_name, "arguments": tc_args},
                            }
                        )
                        valid_tool_call_ids.add(tc_id)
                if openai_tcs:
                    entry["tool_calls"] = openai_tcs
                    if not entry["content"]:
                        entry["content"] = None

        elif msg.role == "tool":
            tool_call_id = getattr(msg, "tool_call_id", None)
            # Skip tool messages without tool_call_id - they would cause 400 errors
            # and fix_orphaned_tool_messages() can't properly clean them up.
            if not tool_call_id:
                continue
            entry["tool_call_id"] = tool_call_id
            # name field for function responses (required by some providers)
            if hasattr(msg, "name") and msg.name:
                entry["name"] = msg.name

        formatted.append(entry)

    return fix_orphaned_tool_messages(formatted)


def handle_httpx_status_error(
    exc: "httpx.HTTPStatusError",
    provider_name: str,
) -> Exception:
    """Map an httpx.HTTPStatusError to a typed ProviderError subclass.

    Extracted from ZAIProvider — provides richer 400 error messages by parsing
    the JSON error body for the ``error.message`` field. Shared across all
    httpx-based OpenAI-compatible providers.

    Args:
        exc: The HTTPStatusError raised by httpx
        provider_name: Provider name for error messages (e.g., "zai", "xai")

    Returns:
        The mapped exception (caller should raise it).
    """
    from victor.providers.base import ProviderAuthError, ProviderError, ProviderRateLimitError

    status = exc.response.status_code
    raw = ""
    try:
        raw = exc.response.text[:500]
    except Exception:
        pass

    if status == 401:
        return ProviderAuthError(
            message=f"Authentication failed: {raw or 'HTTP 401'}",
            provider=provider_name,
            status_code=401,
        )
    if status == 429:
        return ProviderRateLimitError(
            message=f"Rate limit exceeded: {raw or 'HTTP 429'}",
            provider=provider_name,
            status_code=429,
        )
    if status == 400:
        msg = raw
        try:
            body = json.loads(raw)
            msg = body.get("error", {}).get("message", raw)
        except Exception:
            pass
        return ProviderError(
            message=f"{provider_name} request format error: {msg}",
            provider=provider_name,
            status_code=400,
            raw_error=exc,
        )
    return ProviderError(
        message=f"{provider_name} HTTP error {status}: {raw}",
        provider=provider_name,
        status_code=status,
        raw_error=exc,
    )


def accumulate_tool_call_delta(
    delta: Dict[str, Any],
    accumulated: List[Dict[str, Any]],
) -> None:
    """Accumulate a streaming tool-call delta into the running accumulator list.

    Extracted from ZAIProvider._parse_stream_chunk — handles OpenAI SSE streaming
    format where tool call data arrives in fragments across multiple chunks.

    Each entry in ``accumulated`` has shape: ``{id, name, arguments}`` where
    ``arguments`` is a string being built up by concatenation.

    Args:
        delta: The ``delta`` dict from an SSE chunk (``choices[0].delta``)
        accumulated: Mutable list; extended and mutated in place
    """
    tool_call_deltas = delta.get("tool_calls", [])
    for tc_delta in tool_call_deltas:
        idx = tc_delta.get("index", 0)
        while len(accumulated) <= idx:
            accumulated.append({"id": "", "name": "", "arguments": ""})
        if "id" in tc_delta:
            accumulated[idx]["id"] = tc_delta["id"]
        func = tc_delta.get("function", {})
        if "name" in func:
            accumulated[idx]["name"] = func["name"]
        if "arguments" in func:
            accumulated[idx]["arguments"] += func["arguments"]
