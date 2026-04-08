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

"""Tool result deduplication for repeated file reads.

Replaces older duplicate file read results with compact stubs to reduce
context consumption without losing awareness that the file was read.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from victor.config.orchestrator_constants import (
    DeduplicationConfig,
    DEDUPLICATION_CONFIG,
)
from victor.providers.base import Message

logger = logging.getLogger(__name__)

# Pattern to extract path from TOOL_OUTPUT markers
_TOOL_OUTPUT_PATH_RE = re.compile(
    r'<TOOL_OUTPUT\s+tool="(\w+)"(?:\s+path="([^"]*)")?', re.IGNORECASE
)


class ToolResultDeduplicator:
    """Deduplicates repeated file reads in conversation history.

    Only replaces .content of identified messages — never removes messages
    or changes message order. Minimum content threshold prevents stubbing
    small results.
    """

    def __init__(self, config: Optional[DeduplicationConfig] = None):
        self._config = config or DEDUPLICATION_CONFIG

    @property
    def config(self) -> DeduplicationConfig:
        return self._config

    def should_deduplicate(self, tool_name: str, args: Dict[str, Any]) -> bool:
        if not self._config.enabled:
            return False
        return tool_name.lower() in self._config.dedup_tool_names

    def deduplicate_in_place(
        self, messages: List[Message], new_tool_name: str, new_args: Dict[str, Any]
    ) -> int:
        """Replace older duplicate file read contents with stubs.

        Args:
            messages: Conversation message list (modified in place)
            new_tool_name: Name of the tool that just executed
            new_args: Arguments of the tool that just executed

        Returns:
            Number of messages stubbed
        """
        if not self._config.enabled:
            return 0

        target_path = new_args.get("path", new_args.get("file_path", ""))
        if not target_path:
            return 0

        stubbed = 0
        # Scan messages except the last one (which is the new result)
        for i in range(len(messages) - 1):
            msg = messages[i]
            if msg.role != "user":
                continue
            if len(msg.content) < self._config.min_content_chars_to_dedup:
                continue

            # Check for TOOL_OUTPUT marker with matching path
            match = _TOOL_OUTPUT_PATH_RE.search(msg.content)
            if not match:
                continue

            matched_tool = match.group(1)
            matched_path = match.group(2) or ""

            if matched_tool.lower() not in self._config.dedup_tool_names:
                continue
            if matched_path != target_path:
                continue

            # Build stub
            lines = msg.content.count("\n") + 1
            stub = self._config.stub_template.format(path=target_path, lines=lines)

            # Replace content
            messages[i] = Message(role=msg.role, content=stub)
            stubbed += 1
            logger.debug(f"Deduplicated file read for {target_path} at message {i}")

        return stubbed
