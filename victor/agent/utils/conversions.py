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

"""Type conversion utilities for agent orchestrator.

This module provides utility functions for converting between different
data types used throughout the orchestrator. These converters support
backward compatibility and normalize data access patterns.

Functions:
- token_usage_to_dict: Convert TokenUsage objects to dict format
- validation_result_to_dict: Convert ValidationResult to dict format
- message_to_dict: Convert Message objects to dict format
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def token_usage_to_dict(usage: Any) -> dict[str, int]:
    """Convert TokenUsage to dictionary format.

    This function handles different TokenUsage implementations and provides
    a consistent dictionary representation. It safely handles None, dict,
    and object inputs.

    Args:
        usage: TokenUsage instance, dict, or None

    Returns:
        Dictionary with token usage data containing:
        - prompt_tokens: Number of tokens in prompts (default: 0)
        - completion_tokens: Number of tokens in completions (default: 0)
        - total_tokens: Total number of tokens (default: 0)

    Example:
        >>> token_usage_to_dict(None)
        {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        >>> token_usage_to_dict({'prompt_tokens': 10, 'completion_tokens': 20})
        {'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30}
    """
    if usage is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    # Try to extract attributes from object
    prompt_tokens = getattr(usage, "prompt_tokens", 0)
    completion_tokens = getattr(usage, "completion_tokens", 0)
    total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def validation_result_to_dict(result: Any) -> dict[str, Any]:
    """Convert ValidationResult to dictionary format.

    This function normalizes different ValidationResult implementations
    into a consistent dictionary format for backward compatibility.

    Args:
        result: ValidationResult instance or None

    Returns:
        Dictionary with validation data containing:
        - is_valid: Boolean indicating validation status
        - errors: List of error messages (default: [])
        - warnings: List of warning messages (default: [])

    Example:
        >>> validation_result_to_dict(None)
        {'is_valid': False, 'errors': [], 'warnings': []}
    """
    if result is None:
        return {
            "is_valid": False,
            "errors": [],
            "warnings": [],
        }

    return {
        "is_valid": getattr(result, "is_valid", False),
        "errors": list(getattr(result, "errors", [])),
        "warnings": list(getattr(result, "warnings", [])),
    }


def message_to_dict(message: Any) -> dict[str, Any]:
    """Convert Message object to dictionary format.

    This function handles different Message implementations and provides
    a consistent dictionary representation for message objects.

    Args:
        message: Message object with role and content attributes

    Returns:
        Dictionary with message data containing:
        - role: Message role (user, assistant, system)
        - content: Message content string
        - Additional fields if present on the message object

    Example:
        >>> msg = type('Message', (), {'role': 'user', 'content': 'Hello'})()
        >>> message_to_dict(msg)
        {'role': 'user', 'content': 'Hello'}
    """
    if message is None:
        return {
            "role": "system",
            "content": "",
        }

    if isinstance(message, dict):
        return message.copy()

    # Extract standard fields
    result = {
        "role": getattr(message, "role", "system"),
        "content": getattr(message, "content", ""),
    }

    # Extract any additional fields
    for attr in dir(message):
        if not attr.startswith("_") and attr not in result:
            try:
                value = getattr(message, attr)
                if not callable(value):
                    result[attr] = value
            except Exception:
                # Skip attributes that can't be accessed
                pass

    return result


def stream_metrics_to_dict(metrics: Any) -> dict[str, Any]:
    """Convert StreamMetrics to dictionary format.

    This function normalizes StreamMetrics objects into dictionaries
    for consistent access to streaming metrics data.

    Args:
        metrics: StreamMetrics object or None

    Returns:
        Dictionary with streaming metrics data

    Example:
        >>> stream_metrics_to_dict(None)
        {'duration_ms': 0, 'tokens_per_second': 0.0, 'total_chunks': 0}
    """
    if metrics is None:
        return {
            "duration_ms": 0,
            "tokens_per_second": 0.0,
            "total_chunks": 0,
        }

    if isinstance(metrics, dict):
        return metrics.copy()

    return {
        "duration_ms": getattr(metrics, "duration_ms", 0),
        "tokens_per_second": getattr(metrics, "tokens_per_second", 0.0),
        "total_chunks": getattr(metrics, "total_chunks", 0),
        "first_chunk_time_ms": getattr(metrics, "first_chunk_time_ms", 0),
    }


def tool_result_to_dict(result: Any) -> dict[str, Any]:
    """Convert tool execution result to dictionary format.

    This function normalizes different tool result implementations
    into a consistent dictionary format.

    Args:
        result: Tool execution result (any type)

    Returns:
        Dictionary with standardized execution data containing:
        - success: Boolean indicating success (default: False)
        - output: Result output if available
        - error: Error message if available

    Example:
        >>> tool_result_to_dict(None)
        {'success': False, 'output': None, 'error': None}
    """
    if result is None:
        return {
            "success": False,
            "output": None,
            "error": None,
        }

    if isinstance(result, dict):
        return {
            "success": result.get("success", False),
            "output": result.get("output"),
            "error": result.get("error"),
        }

    return {
        "success": getattr(result, "success", False),
        "output": getattr(result, "output", None),
        "error": getattr(result, "error", None),
    }


__all__ = [
    "token_usage_to_dict",
    "validation_result_to_dict",
    "message_to_dict",
    "stream_metrics_to_dict",
    "tool_result_to_dict",
]
