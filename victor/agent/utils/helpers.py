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

"""General helper functions for agent orchestrator.

This module provides utility functions for common operations like:
- String manipulation and formatting
- File path extraction from text
- Output requirement extraction
- Tool output formatting for logging

These helpers were extracted from AgentOrchestrator to improve
testability and reusability.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Regular expressions for extracting file paths
FILE_PATH_PATTERNS = [
    # Common file path patterns
    r"[`']?([a-zA-Z0-9_./-]+\.[a-zA-Z0-9]+)[`']?",  # code.py, ./file.txt
    r'["\']([a-zA-Z0-9_./-]+)["\']',  # "path/to/file"
    r"\b([a-zA-Z]:[\\/][a-zA-Z0-9_./-]+)\b",  # Windows paths C:\path\to\file
    r"\b([a-zA-Z0-9_./-]+/[a-zA-Z0-9_./-]+)\b",  # Unix paths path/to/file
]

# Regular expressions for extracting output requirements
OUTPUT_REQUIREMENT_PATTERNS = [
    r"(?:output|save|write|export|generate)\s+(?:to|as|in)\s+[`']?([^`'\n]+)[`']?",
    r"create\s+(?:a\s+)?(?:file|document|output)\s+(?:called|named)?\s*[`']?([^`'\n]+)[`']?",
    r"file\s*:?\s*[`']?([^`'\n]+)[`']?",
]


def extract_file_paths_from_text(text: str) -> List[str]:
    """Extract file paths mentioned in text.

    This function scans text for common file path patterns and returns
    a list of unique file paths that were mentioned.

    Args:
        text: The text to scan for file paths

    Returns:
        List of unique file paths found in the text

    Example:
        >>> extract_file_paths_from_text("Edit file.py and update ./config.json")
        ['file.py', './config.json']
    """
    if not text:
        return []

    paths = set()
    for pattern in FILE_PATH_PATTERNS:
        matches = re.findall(pattern, text)
        paths.update(matches)

    # Filter out common false positives
    false_positives = {
        "http",
        "https",
        "www",
        "com",
        "org",
        "net",
        "and",
        "or",
        "the",
        "to",
        "from",
    }

    valid_paths = [
        path
        for path in paths
        if path.lower() not in false_positives
        and len(path) > 2
        and ("." in path or "/" in path or "\\" in path)
    ]

    return sorted(valid_paths)


def extract_output_requirements_from_text(text: str) -> List[str]:
    """Extract output file requirements from text.

    This function scans text for patterns indicating where output should
    be saved or what files should be created.

    Args:
        text: The text to scan for output requirements

    Returns:
        List of unique output file paths or descriptions found in the text

    Example:
        >>> extract_output_requirements_from_text("Save the output to results.json")
        ['results.json']
    """
    if not text:
        return []

    requirements = set()
    for pattern in OUTPUT_REQUIREMENT_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        requirements.update(matches)

    # Clean up requirements
    cleaned = [
        req.strip().strip("'\"`") for req in requirements if req.strip() and len(req.strip()) > 2
    ]

    return sorted(set(cleaned))


def format_tool_output_for_log(
    tool_name: str,
    args: Dict[str, Any],
    output: Any,
    max_length: int = 500,
) -> str:
    """Format tool execution output for logging.

    This function creates a concise, log-friendly representation of
    tool execution results. It truncates large outputs and structures
    the information for readability.

    Args:
        tool_name: Name of the tool that was executed
        args: Arguments passed to the tool
        output: Raw output from the tool
        max_length: Maximum length for output string (default: 500)

    Returns:
        Formatted string suitable for logging

    Example:
        >>> format_tool_output_for_log("read_file", {"path": "test.py"}, "content...")
        "Tool: read_file | Args: {'path': 'test.py'} | Output: content..."
    """
    # Format tool name
    tool_str = f"Tool: {tool_name}"

    # Format arguments (truncate if too long)
    args_str = str(args)
    if len(args_str) > 200:
        args_str = args_str[:200] + "..."
    args_str = f"Args: {args_str}"

    # Format output (truncate if too long)
    output_str = str(output)
    if len(output_str) > max_length:
        output_str = output_str[:max_length] + "..."

    # Handle special output types
    if isinstance(output, dict):
        if "error" in output:
            output_str = f"Error: {output['error']}"
        elif "output" in output:
            output_str = str(output["output"])
            if len(output_str) > max_length:
                output_str = output_str[:max_length] + "..."
    elif isinstance(output, list):
        output_str = f"[{len(output)} items]"
        if len(output) > 0 and len(str(output)) < max_length:
            output_str = str(output)

    output_str = f"Output: {output_str}"

    # Combine all parts
    return " | ".join([tool_str, args_str, output_str])


def truncate_for_display(
    text: str,
    max_lines: int = 10,
    max_chars: int = 1000,
    indicator: str = "...",
) -> str:
    """Truncate text for display purposes.

    This function limits text output by both line count and character count,
    adding an indicator when truncation occurs.

    Args:
        text: The text to truncate
        max_lines: Maximum number of lines to keep (default: 10)
        max_chars: Maximum number of characters to keep (default: 1000)
        indicator: String to append when truncated (default: "...")

    Returns:
        Truncated text with indicator if truncation occurred

    Example:
        >>> truncate_for_display("line1\\nline2\\nline3", max_lines=2)
        'line1\\nline2...'
    """
    if not text:
        return ""

    # Truncate by lines
    lines = text.split("\n")
    if len(lines) > max_lines:
        text = "\n".join(lines[:max_lines])
        if len(text) + len(indicator) <= max_chars:
            return text + indicator
        return text[: max_chars - len(indicator)] + indicator

    # Truncate by characters
    if len(text) > max_chars:
        return text[: max_chars - len(indicator)] + indicator

    return text


def sanitize_string_for_log(text: str, max_length: int = 200) -> str:
    """Sanitize a string for safe logging.

    This function removes or replaces characters that might cause
    issues in logs and truncates to a reasonable length.

    Args:
        text: The text to sanitize
        max_length: Maximum length for the sanitized string

    Returns:
        Sanitized string safe for logging

    Example:
        >>> sanitize_string_for_log("Text with\\nnewlines\\r\\r\\n")
        'Text with newlines '
    """
    if not text:
        return ""

    # Replace problematic characters
    sanitized = text.replace("\n", " ")
    sanitized = sanitized.replace("\r", " ")
    sanitized = sanitized.replace("\t", " ")

    # Remove control characters
    sanitized = "".join(char for char in sanitized if ord(char) >= 32 or char == " ")

    # Normalize whitespace
    sanitized = " ".join(sanitized.split())

    # Truncate if necessary
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."

    return sanitized


def build_tool_call_summary(
    tool_name: str,
    success: bool,
    duration_ms: float | None = None,
    error: str | None = None,
) -> str:
    """Build a concise summary of a tool call.

    This function creates a one-line summary of tool execution for
    logging or display purposes.

    Args:
        tool_name: Name of the tool that was called
        success: Whether the tool call was successful
        duration_ms: Optional execution duration in milliseconds
        error: Optional error message if not successful

    Returns:
        Concise summary string

    Example:
        >>> build_tool_call_summary("read_file", True, 45.2)
        'read_file: SUCCESS (45.2ms)'
        >>> build_tool_call_summary("write_file", False, error="Permission denied")
        'write_file: FAILED (Permission denied)'
    """
    status = "SUCCESS" if success else "FAILED"

    parts = [f"{tool_name}: {status}"]

    if duration_ms is not None:
        parts.append(f"({duration_ms:.1f}ms)")
    elif error:
        parts.append(f"({error})")

    return " ".join(parts)


__all__ = [
    "extract_file_paths_from_text",
    "extract_output_requirements_from_text",
    "format_tool_output_for_log",
    "truncate_for_display",
    "sanitize_string_for_log",
    "build_tool_call_summary",
]
