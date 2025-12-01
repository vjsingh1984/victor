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

"""Response sanitization for Victor.

Handles cleaning and validation of model responses, including:
- Removing malformed XML/HTML tags
- Filtering training data leakage patterns
- Validating tool names
- Detecting garbage content from local models
"""

import logging
import re
from typing import List

logger = logging.getLogger(__name__)


class ResponseSanitizer:
    """Sanitizes and validates model responses.

    Handles common issues from local models:
    - Repeated </function> or </parameter> tags
    - XML-like formatting artifacts
    - JSON-like tool call attempts in plain text
    - Instruction/example leakage from model training
    """

    # Patterns indicating training data leakage
    LEAKAGE_PATTERNS: List[str] = [
        r"Do not invent any new or additional parameters.*",
        r"The parameter value should be passed as a string.*",
        r"If you want to call multiple functions.*",
        r"Do NOT surround the function call.*",
        r"All parameters are required unless.*",
        r"The agent is not allowed to directly access.*",
        r"Begin by calling list_directory.*",
        r"execute_bash\(command=.*\)",
        r"No files read yet\. Avoid file-specific claims.*",
        r'list_directory\(path="[^"]*"\)',
        r'read_file\(path="[^"]*"\)',
    ]

    # Patterns indicating garbage/malformed output
    GARBAGE_PATTERNS: List[str] = [
        r"FUNCTION_CALL\s*\{",  # Raw function call syntax
        r"</function>\s*</function>",  # Repeated closing tags
        r"<parameter[^>]*>",  # Raw parameter tags
        r'^\s*\{\s*"name":\s*"[^"]+",\s*"arguments":',  # Raw JSON tool calls
        r"^\s*<IMPORTANT>",  # Instruction leakage
        r"^\s*<important>",  # Instruction leakage (lowercase)
        r"</important>",  # Instruction leakage closing tag
        r"^\s*Do NOT",  # Instruction leakage
        r"^\s*NEVER\s+",  # Instruction leakage
        r"^\s*ALWAYS include",  # Instruction leakage
        r"^\s*- Do NOT",  # List-style instruction leakage
        r"^\s*- Use lowercase",  # Formatting instruction leakage
        r"^\s*- The parameters",  # Parameter instruction leakage
        r"the XML tag",  # Format instruction leakage
        r"backticks.*markdown",  # Format instruction leakage
        r"JSON parsing",  # Technical instruction leakage
        r"file system structure",  # Security instruction leakage
        r"\[TOOL_REQUEST\]",  # LMStudio default format leakage
        r"\[END_TOOL_REQUEST\]",  # LMStudio default format leakage
    ]

    # Patterns for invalid/hallucinated tool names
    INVALID_TOOL_PATTERNS: List[str] = [
        r"^example_",
        r"^func_",
        r"^function_",
        r"^tool_name",
        r"^my_",
        r"^test_tool",
        r"^sample_",
        r"/$",  # Ends with slash
        r"^<",  # Starts with XML tag
        r">$",  # Ends with XML tag
        r"\s",  # Contains whitespace
        r"^\d",  # Starts with number
    ]

    def strip_markup(self, text: str) -> str:
        """Remove simple XML/HTML-like tags to salvage plain text.

        Args:
            text: Text potentially containing markup

        Returns:
            Plain text with markup removed
        """
        if not text:
            return text
        cleaned = re.sub(r"<[^>]+>", " ", text)
        return " ".join(cleaned.split())

    def sanitize(self, text: str) -> str:
        """Sanitize model response by removing malformed patterns.

        Handles common issues from local models:
        - Repeated </function> or </parameter> tags
        - XML-like formatting artifacts
        - JSON-like tool call attempts in plain text
        - Instruction/example leakage from model training

        Args:
            text: Raw response text from the model

        Returns:
            Cleaned text suitable for display
        """
        if not text:
            return text

        original_len = len(text)

        # Remove repeated closing tags (</function>, </parameter>, etc.)
        # These indicate the model is confused about tool calling format
        text = re.sub(r"(</\w+>\s*){3,}", "", text)

        # Remove orphaned XML-like tags
        text = re.sub(r"</?function[^>]*>", "", text)
        text = re.sub(r"</?parameter[^>]*>", "", text)
        text = re.sub(r"</?tool[^>]*>", "", text)
        text = re.sub(r"</?IMPORTANT[^>]*>", "", text)

        # Remove training data leakage patterns
        for pattern in self.LEAKAGE_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

        # Remove JSON-like tool call attempts embedded in text
        # Pattern: {"name": "tool_name", ...}
        text = re.sub(r'\{"name":\s*"[^"]+",\s*"arguments":\s*\{[^}]*\}\}', "", text)

        # Remove lines that are just tool call syntax
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip lines that look like raw tool call attempts
            if stripped.startswith('{"name":') or stripped.startswith("</"):
                continue
            # Skip lines that are just parameter= syntax
            if re.match(r"^(parameter=|<parameter)", stripped):
                continue
            cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)

        # Clean up excessive whitespace
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        text = text.strip()

        # Log if significant content was removed (indicates model confusion)
        if len(text) < original_len * 0.5 and original_len > 100:
            logger.warning(
                f"Sanitization removed {original_len - len(text)} chars "
                f"({100 - len(text) * 100 // original_len}% of response) - "
                "model may be confused about tool calling format"
            )

        return text

    def is_garbage_content(self, content: str) -> bool:
        """Detect if content is garbage/malformed output from local models.

        Args:
            content: Content chunk to check

        Returns:
            True if content appears to be garbage
        """
        if not content:
            return False

        for pattern in self.GARBAGE_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                return True

        return False

    def is_valid_tool_name(self, name: str) -> bool:
        """Check if a tool name is valid and not a hallucination.

        Args:
            name: Tool name to validate

        Returns:
            True if the tool name appears valid
        """
        if not name or not isinstance(name, str):
            return False

        # Reject common hallucinated/example tool names
        for pattern in self.INVALID_TOOL_PATTERNS:
            if re.search(pattern, name, re.IGNORECASE):
                logger.debug(f"Rejecting invalid tool name: {name}")
                return False

        # Must be alphanumeric with underscores only
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", name):
            logger.debug(f"Rejecting malformed tool name: {name}")
            return False

        return True


# Module-level singleton for convenience
_sanitizer = ResponseSanitizer()


def sanitize_response(text: str) -> str:
    """Sanitize model response (convenience function).

    Args:
        text: Raw response text

    Returns:
        Cleaned text
    """
    return _sanitizer.sanitize(text)


def is_garbage_content(content: str) -> bool:
    """Detect garbage content (convenience function).

    Args:
        content: Content to check

    Returns:
        True if garbage detected
    """
    return _sanitizer.is_garbage_content(content)


def is_valid_tool_name(name: str) -> bool:
    """Validate tool name (convenience function).

    Args:
        name: Tool name to validate

    Returns:
        True if valid
    """
    return _sanitizer.is_valid_tool_name(name)


def strip_markup(text: str) -> str:
    """Strip markup from text (convenience function).

    Args:
        text: Text with markup

    Returns:
        Plain text
    """
    return _sanitizer.strip_markup(text)
