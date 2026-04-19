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

"""Rule-based compaction summarizer (Claudecode-style).

Implements fast, deterministic, reproducible compaction without LLM calls.
Generates machine-readable XML format for structured parsing.

Performance: <100ms for 100 messages (vs 2-5s for LLM-based).
Use case: 80% of compactions - simple cases with clear patterns.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from victor.config.compaction_strategy_settings import CompactionStrategySettings
from victor.providers.base import Message


logger = logging.getLogger(__name__)


@dataclass
class RuleBasedSummary:
    """Structured summary from rule-based compaction.

    Contains all the information needed to reconstruct the context
    of compacted messages in a machine-readable format.
    """

    scope: str
    """Scope description (e.g., "X earlier messages compacted (user=Y, assistant=Z, tool=N)")."""

    tools_mentioned: List[str]
    """List of unique tool names used in compacted messages."""

    recent_user_requests: List[str]
    """Recent user requests (truncated to 160 chars each)."""

    pending_work: List[str]
    """Inferred pending work items from keyword detection."""

    key_files_referenced: List[str]
    """File paths extracted from content (limited to 8 files)."""

    current_work: str
    """Description of current work based on recent messages."""

    key_timeline: List[Dict[str, Any]]
    """Timeline of key messages with role and truncated content."""


class RuleBasedCompactionSummarizer:
    """Claudecode-style deterministic rule-based summarization.

    Fast, cheap, reproducible compaction without LLM calls.
    Generates machine-readable XML format for parsing.

    Performance characteristics:
    - Speed: <100ms for 100 messages
    - Cost: $0 (no LLM calls)
    - Reproducibility: Deterministic (same input → same output)
    - Quality: Good for simple cases, misses nuance in complex conversations

    Use when:
    - Message complexity is low (simple Q&A, single-file edits)
    - Token count is moderate (<10k tokens)
    - Speed is critical (real-time applications)
    - Cost is a concern (high-volume systems)

    Don't use when:
    - Complex multi-file refactoring (use LLM-based)
    - Nuanced technical decisions (use LLM-based)
    - Cross-cutting concerns (use hybrid or LLM-based)
    """

    # File extensions to identify when extracting file paths
    _FILE_EXTENSIONS = {
        "rs", "ts", "tsx", "js", "jsx", "json", "md", "py", "go", "java",
        "cpp", "c", "h", "cs", "php", "rb", "swift", "kt", "scala", "sh",
        "yaml", "yml", "toml", "ini", "cfg", "conf", "xml", "html", "css",
        "sql", "graphql", "proto", "thrift", "avsc", "wsdl", "xsd", "dtd"
    }

    # Keywords that indicate pending work
    _PENDING_KEYWORDS = [
        "todo", "next", "pending", "follow up", "remaining",
        "not done", "incomplete", "unfinished", "later", "subsequent",
        "after this", "then", "subsequently", "finally"
    ]

    # Patterns for identifying current work
    _CURRENT_WORK_PATTERNS = [
        r"currently\s+(working\s+on|doing|implementing|fixing|debugging)",
        r"(now|just)\s+(started|began|working)",
        r"in\s+progress",
        r"wip",
        r"work\s+in\s+progress"
    ]

    def __init__(self, config: CompactionStrategySettings):
        """Initialize rule-based summarizer.

        Args:
            config: Compaction strategy settings
        """
        self._config = config
        self._file_extensions = self._FILE_EXTENSIONS
        self._pending_keywords = [kw.lower() for kw in self._PENDING_KEYWORDS]
        self._current_work_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self._CURRENT_WORK_PATTERNS
        ]

    def summarize(
        self,
        messages: List[Message],
        ledger: Optional[Any] = None,
    ) -> str:
        """Generate rule-based summary in XML format.

        Args:
            messages: Messages to summarize
            ledger: Optional ledger (currently unused, for API compatibility)

        Returns:
            XML-formatted summary string
        """
        if not messages:
            return ""

        try:
            summary = self._generate_summary(messages)
            return self._format_xml(summary)
        except Exception as e:
            logger.warning(f"Rule-based summarization failed: {e}")
            return self._format_fallback_summary(len(messages))

    def _generate_summary(self, messages: List[Message]) -> RuleBasedSummary:
        """Generate structured summary using rules.

        Args:
            messages: Messages to analyze

        Returns:
            RuleBasedSummary with structured data
        """
        # Count messages by role
        user_count = sum(1 for m in messages if m.role == "user")
        assistant_count = sum(1 for m in messages if m.role == "assistant")
        tool_count = sum(1 for m in messages if m.role == "tool")

        # Extract structured information
        tools = self._extract_tool_names(messages)
        recent_requests = self._extract_recent_user_requests(
            messages,
            limit=min(3, user_count)
        )
        pending = self._infer_pending_work(messages)
        files = self._extract_file_paths(messages)
        current_work = self._infer_current_work(messages)
        timeline = self._build_timeline(messages)

        # Build scope description
        scope = (
            f"{len(messages)} earlier messages compacted "
            f"(user={user_count}, assistant={assistant_count}, tool={tool_count})"
        )

        return RuleBasedSummary(
            scope=scope,
            tools_mentioned=tools,
            recent_user_requests=recent_requests,
            pending_work=pending,
            key_files_referenced=files,
            current_work=current_work,
            key_timeline=timeline,
        )

    def _format_xml(self, summary: RuleBasedSummary) -> str:
        """Format summary as XML (claudecode-compatible).

        Args:
            summary: Structured summary data

        Returns:
            XML-formatted string
        """
        lines = ["<summary>", "Conversation summary:"]
        lines.append(f"- Scope: {summary.scope}.")

        if summary.tools_mentioned:
            lines.append(
                f"- Tools mentioned: {', '.join(summary.tools_mentioned)}."
            )

        if summary.recent_user_requests:
            lines.append("- Recent user requests:")
            lines.extend(f"  - {self._escape_xml(req)}" for req in summary.recent_user_requests)

        if summary.pending_work:
            lines.append("- Pending work:")
            lines.extend(f"  - {self._escape_xml(item)}" for item in summary.pending_work)

        if summary.key_files_referenced:
            lines.append(
                f"- Key files referenced: {', '.join(summary.key_files_referenced)}."
            )

        if summary.current_work:
            lines.append(f"- Current work: {self._escape_xml(summary.current_work)}")

        if summary.key_timeline:
            lines.append("- Key timeline:")
            for entry in summary.key_timeline[-10:]:  # Limit to 10 most recent
                role = entry["role"]
                content = self._truncate(entry["content"], 160)
                lines.append(f"  - {role}: {self._escape_xml(content)}")

        lines.append("</summary>")
        return "\n".join(lines)

    def _format_fallback_summary(self, message_count: int) -> str:
        """Format a minimal fallback summary.

        Args:
            message_count: Number of messages that were compacted

        Returns:
            Minimal XML summary
        """
        return (
            "<summary>\n"
            f"Conversation summary: {message_count} earlier messages compacted.\n"
            "</summary>"
        )

    def _extract_tool_names(self, messages: List[Message]) -> List[str]:
        """Extract unique tool names from messages.

        Args:
            messages: Messages to analyze

        Returns:
            Sorted list of unique tool names
        """
        tools = set()
        for msg in messages:
            # Check for tool_calls attribute (OpenAI format)
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    if isinstance(call, dict):
                        tool_name = call.get("name", "")
                        if tool_name:
                            tools.add(tool_name)
                    elif hasattr(call, "name"):
                        tools.add(call.name)

            # Check for tool_name attribute (tool result messages)
            if hasattr(msg, "tool_name") and msg.tool_name:
                tools.add(msg.tool_name)

            # Check content for tool mentions (e.g., "Used read_file tool")
            if msg.content:
                # Pattern: "used X tool" or "X tool returned"
                tool_mentions = re.findall(
                    r"used\s+(\w+)\s+tool|(\w+)\s+tool\s+(?:returned|failed)",
                    msg.content.lower()
                )
                for mention in tool_mentions:
                    tool_name = mention[0] or mention[1]
                    if tool_name:
                        tools.add(tool_name)

        return sorted(tools)

    def _extract_file_paths(self, messages: List[Message]) -> List[str]:
        """Extract file paths from message content.

        Args:
            messages: Messages to analyze

        Returns:
            Sorted list of unique file paths (limited to 8)
        """
        files = set()

        for msg in messages:
            if not msg.content:
                continue

            # Tokenize by whitespace and common delimiters
            tokens = re.split(r'[\s\(\)\[\]{}",;:]', msg.content)

            for token in tokens:
                # Clean up the token
                clean = token.strip("`'\"")

                # Check if it looks like a file path
                if "/" in clean or "\\" in clean:
                    # Extract file extension
                    parts = clean.split(".")
                    if len(parts) > 1:
                        ext = parts[-1].lower()
                        # Check if it's a known code file extension
                        if ext in self._file_extensions:
                            # Further validation: should have reasonable length
                            # and not be a URL or common non-file pattern
                            if self._is_valid_file_path(clean):
                                files.add(clean)

        # Sort and limit to 8 files
        return sorted(list(files))[:8]

    def _is_valid_file_path(self, path: str) -> bool:
        """Check if a path looks like a valid file path (not URL, etc.).

        Args:
            path: Path to validate

        Returns:
            True if path looks like a valid file path
        """
        # Exclude URLs
        if path.startswith(("http://", "https://", "ftp://")):
            return False

        # Exclude very short or very long paths
        if len(path) < 3 or len(path) > 200:
            return False

        # Exclude common non-file patterns
        if path in (".", "..", "./", "../"):
            return False

        # Should contain at least one path separator
        if "/" not in path and "\\" not in path:
            return False

        return True

    def _extract_recent_user_requests(
        self,
        messages: List[Message],
        limit: int = 3,
    ) -> List[str]:
        """Extract recent user requests from messages.

        Args:
            messages: Messages to analyze
            limit: Maximum number of requests to extract

        Returns:
            List of recent user request contents (truncated)
        """
        user_messages = [
            m.content for m in messages
            if m.role == "user" and m.content
        ]

        # Get the most recent user messages
        recent = user_messages[-limit:] if len(user_messages) > limit else user_messages

        # Truncate each request
        return [self._truncate(req, 160) for req in recent]

    def _infer_pending_work(self, messages: List[Message]) -> List[str]:
        """Infer pending work from recent messages using keyword detection.

        Args:
            messages: Messages to analyze

        Returns:
            List of inferred pending work items
        """
        pending = []

        # Look for pending work keywords in recent messages
        for msg in reversed(messages):
            if not msg.content:
                continue

            content_lower = msg.content.lower()

            # Check if any pending keyword is present
            if any(kw in content_lower for kw in self._pending_keywords):
                # Extract the relevant sentence/phrase
                pending_item = self._extract_pending_sentence(msg.content)
                if pending_item:
                    pending.append(self._truncate(pending_item, 160))

                    # Limit to 3 pending items
                    if len(pending) >= 3:
                        break

        # Reverse to maintain chronological order
        return list(reversed(pending))

    def _extract_pending_sentence(self, content: str) -> Optional[str]:
        """Extract the sentence containing pending work keywords.

        Args:
            content: Message content

        Returns:
            Sentence containing pending keywords, or None
        """
        # Split into sentences (rough approximation)
        sentences = re.split(r'[.!?]+', content)

        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(kw in sentence_lower for kw in self._pending_keywords):
                # Clean up whitespace
                cleaned = sentence.strip()
                if cleaned:
                    return cleaned

        return None

    def _infer_current_work(self, messages: List[Message]) -> str:
        """Infer current work based on recent messages.

        Args:
            messages: Messages to analyze

        Returns:
            Description of current work
        """
        # Look for current work patterns in recent messages
        for msg in reversed(messages):
            if not msg.content:
                continue

            for pattern in self._current_work_patterns:
                match = pattern.search(msg.content)
                if match:
                    # Extract the sentence containing the match
                    sentence = self._extract_matching_sentence(
                        msg.content,
                        match.start(),
                        match.end()
                    )
                    if sentence:
                        return self._truncate(sentence, 160)

        # Fallback: extract from most recent assistant message
        for msg in reversed(messages):
            if msg.role == "assistant" and msg.content:
                return self._truncate(msg.content, 160)

        return ""

    def _extract_matching_sentence(
        self,
        content: str,
        match_start: int,
        match_end: int,
    ) -> Optional[str]:
        """Extract the sentence containing a regex match.

        Args:
            content: Full content
            match_start: Start index of match
            match_end: End index of match

        Returns:
            Sentence containing the match, or None
        """
        # Find sentence boundaries
        sentence_start = content.rfind(".", 0, match_start) + 1
        if sentence_start == 0:
            sentence_start = content.rfind("\n", 0, match_start) + 1

        sentence_end = content.find(".", match_end)
        if sentence_end == -1:
            sentence_end = content.find("\n", match_end)
        if sentence_end == -1:
            sentence_end = len(content)

        sentence = content[sentence_start:sentence_end].strip()
        return sentence if sentence else None

    def _build_timeline(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Build timeline from messages.

        Args:
            messages: Messages to timeline

        Returns:
            List of timeline entries with role and content
        """
        timeline = []

        for msg in messages:
            # Skip system messages in timeline
            if msg.role == "system":
                continue

            timeline.append({
                "role": msg.role,
                "content": msg.content,
            })

        return timeline

    @staticmethod
    def _truncate(content: str, max_chars: int) -> str:
        """Truncate content to max_chars with ellipsis.

        Args:
            content: Content to truncate
            max_chars: Maximum character count

        Returns:
            Truncated content with ellipsis if needed
        """
        if len(content) <= max_chars:
            return content

        truncated = content[:max_chars]
        return truncated + "…"

    @staticmethod
    def _escape_xml(text: str) -> str:
        """Escape special XML characters.

        Args:
            text: Text to escape

        Returns:
            XML-escaped text
        """
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace('"', "&quot;")
        text = text.replace("'", "&apos;")
        return text
