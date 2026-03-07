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

"""Structured session state ledger.

Maintains an append-only log of high-signal events (file reads, modifications,
decisions, recommendations, pending actions) that survives context compaction
and is rendered into the LLM context each turn.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from victor.config.orchestrator_constants import SessionLedgerConfig, SESSION_LEDGER_CONFIG

logger = logging.getLogger(__name__)

# Patterns for extracting decisions from assistant responses
DECISION_PATTERNS = [
    re.compile(r"(?:I (?:will|'ll|am going to|decided to))\s+(.+?)(?:\.|$)", re.IGNORECASE),
    re.compile(r"(?:conclusion|decision)\s*:\s*(.+?)(?:\.|$)", re.IGNORECASE),
]

# Patterns for extracting recommendations from assistant responses
RECOMMENDATION_PATTERNS = [
    re.compile(r"(?:I recommend|I suggest|should)\s+(.+?)(?:\.|$)", re.IGNORECASE),
    re.compile(r"(?:recommendation|suggestion)\s*:\s*(.+?)(?:\.|$)", re.IGNORECASE),
]

# Patterns for detecting TOOL_OUTPUT markers
TOOL_OUTPUT_PATTERN = re.compile(
    r'<TOOL_OUTPUT\s+tool="(\w+)"(?:\s+path="([^"]*)")?[^>]*>', re.IGNORECASE
)


@dataclass(frozen=True)
class LedgerEntry:
    """A single entry in the session ledger."""

    timestamp: float
    category: str  # "file_read", "file_modified", "decision", "recommendation", "pending_action"
    key: str  # file path or entry ID
    summary: str
    turn_index: int
    resolved: bool = False


class SessionLedger:
    """Append-only log of high-signal session events.

    Survives compaction by being rendered as a context block each turn.
    Entries are evicted oldest-first when max_entries is exceeded.
    """

    def __init__(self, config: Optional[SessionLedgerConfig] = None):
        self._config = config or SESSION_LEDGER_CONFIG
        self._entries: List[LedgerEntry] = []
        self._files_read: Dict[str, str] = {}  # path -> summary

    @property
    def entries(self) -> List[LedgerEntry]:
        return list(self._entries)

    @property
    def config(self) -> SessionLedgerConfig:
        return self._config

    def _add_entry(self, entry: LedgerEntry) -> None:
        self._entries.append(entry)
        while len(self._entries) > self._config.max_entries:
            self._entries.pop(0)

    def record_file_read(self, path: str, summary: str, turn_index: int) -> None:
        truncated = summary[: self._config.file_summary_max_len]
        self._files_read[path] = truncated
        self._add_entry(
            LedgerEntry(
                timestamp=time.time(),
                category="file_read",
                key=path,
                summary=truncated,
                turn_index=turn_index,
            )
        )

    def record_file_modified(self, path: str, change_summary: str, turn_index: int) -> None:
        truncated = change_summary[: self._config.file_summary_max_len]
        self._add_entry(
            LedgerEntry(
                timestamp=time.time(),
                category="file_modified",
                key=path,
                summary=truncated,
                turn_index=turn_index,
            )
        )

    def record_decision(self, decision: str, turn_index: int) -> None:
        self._add_entry(
            LedgerEntry(
                timestamp=time.time(),
                category="decision",
                key=f"decision_{turn_index}_{len(self._entries)}",
                summary=decision[: self._config.file_summary_max_len],
                turn_index=turn_index,
            )
        )

    def record_recommendation(self, recommendation: str, turn_index: int) -> None:
        self._add_entry(
            LedgerEntry(
                timestamp=time.time(),
                category="recommendation",
                key=f"rec_{turn_index}_{len(self._entries)}",
                summary=recommendation[: self._config.file_summary_max_len],
                turn_index=turn_index,
            )
        )

    def record_pending_action(self, action: str, turn_index: int) -> None:
        self._add_entry(
            LedgerEntry(
                timestamp=time.time(),
                category="pending_action",
                key=f"action_{turn_index}_{len(self._entries)}",
                summary=action[: self._config.file_summary_max_len],
                turn_index=turn_index,
            )
        )

    def resolve_pending_action(self, action_key: str) -> None:
        for i, entry in enumerate(self._entries):
            if entry.category == "pending_action" and entry.key == action_key:
                self._entries[i] = LedgerEntry(
                    timestamp=entry.timestamp,
                    category=entry.category,
                    key=entry.key,
                    summary=entry.summary,
                    turn_index=entry.turn_index,
                    resolved=True,
                )
                return

    def update_from_tool_result(
        self, tool_name: str, args: Dict[str, Any], result: str, turn_index: int
    ) -> None:
        """Parse tool results to extract file reads and modifications."""
        tool_lower = tool_name.lower()

        # Read-type tools
        if tool_lower in ("read", "cat", "read_file"):
            path = args.get("path", args.get("file_path", ""))
            if path:
                # Extract first meaningful line as summary
                lines = result.split("\n") if result else []
                first_line = ""
                for line in lines[:10]:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#") and len(stripped) > 5:
                        first_line = stripped
                        break
                summary = f"{len(lines)} lines"
                if first_line:
                    summary += f": {first_line[:60]}"
                self.record_file_read(path, summary, turn_index)

        # Write/edit type tools
        elif tool_lower in ("write", "edit", "write_file", "create_file"):
            path = args.get("path", args.get("file_path", ""))
            if path:
                summary = f"Modified via {tool_lower}"
                if "content" in args:
                    summary += f" ({len(str(args['content']))} chars)"
                self.record_file_modified(path, summary, turn_index)

        # Also check for TOOL_OUTPUT markers in result
        if result:
            for match in TOOL_OUTPUT_PATTERN.finditer(result):
                matched_tool = match.group(1)
                matched_path = match.group(2) or ""
                if matched_tool in ("read", "cat") and matched_path:
                    if matched_path not in self._files_read:
                        self.record_file_read(matched_path, "read via tool output", turn_index)

    def update_from_assistant_response(self, content: str, turn_index: int) -> None:
        """Extract decisions and recommendations from assistant responses."""
        if not content:
            return

        for pattern in DECISION_PATTERNS:
            for match in pattern.finditer(content):
                decision = match.group(1).strip()
                if len(decision) > 10:
                    self.record_decision(decision, turn_index)
                    break  # One decision per pattern group

        for pattern in RECOMMENDATION_PATTERNS:
            for match in pattern.finditer(content):
                rec = match.group(1).strip()
                if len(rec) > 10:
                    self.record_recommendation(rec, turn_index)
                    break

    def render(self, max_chars: Optional[int] = None) -> str:
        """Render ledger as XML block for context injection."""
        if not self._entries:
            return ""

        max_chars = max_chars or self._config.max_render_chars

        # Group entries by category
        groups: Dict[str, List[LedgerEntry]] = {}
        for entry in self._entries:
            groups.setdefault(entry.category, []).append(entry)

        parts = ["<SESSION_STATE>"]
        category_labels = {
            "file_read": "Files Read",
            "file_modified": "Files Modified",
            "decision": "Decisions Made",
            "recommendation": "Recommendations",
            "pending_action": "Pending Actions",
        }

        for cat, label in category_labels.items():
            entries = groups.get(cat, [])
            if not entries:
                continue
            parts.append(f"  <{cat}s>  <!-- {label} -->")
            for e in entries:
                resolved_tag = ' resolved="true"' if e.resolved else ""
                parts.append(f'    <entry key="{e.key}"{resolved_tag}>{e.summary}</entry>')
            parts.append(f"  </{cat}s>")

        parts.append("</SESSION_STATE>")
        rendered = "\n".join(parts)

        # Truncate if over budget
        if len(rendered) > max_chars:
            rendered = rendered[: max_chars - 20] + "\n... (truncated)\n</SESSION_STATE>"

        return rendered

    def get_recent_actionable_items(self, limit: int = 5) -> List[LedgerEntry]:
        """Get recent decisions, recommendations, and unresolved pending actions."""
        actionable_categories = {"decision", "recommendation", "pending_action"}
        items = [
            e
            for e in reversed(self._entries)
            if e.category in actionable_categories and not e.resolved
        ]
        return items[:limit]

    def get_files_read(self) -> Dict[str, str]:
        return dict(self._files_read)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries": [
                {
                    "timestamp": e.timestamp,
                    "category": e.category,
                    "key": e.key,
                    "summary": e.summary,
                    "turn_index": e.turn_index,
                    "resolved": e.resolved,
                }
                for e in self._entries
            ],
            "files_read": dict(self._files_read),
            "config": {
                "max_entries": self._config.max_entries,
                "max_render_chars": self._config.max_render_chars,
                "file_summary_max_len": self._config.file_summary_max_len,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionLedger":
        config_data = data.get("config", {})
        config = SessionLedgerConfig(
            max_entries=config_data.get("max_entries", SESSION_LEDGER_CONFIG.max_entries),
            max_render_chars=config_data.get(
                "max_render_chars", SESSION_LEDGER_CONFIG.max_render_chars
            ),
            file_summary_max_len=config_data.get(
                "file_summary_max_len", SESSION_LEDGER_CONFIG.file_summary_max_len
            ),
        )
        ledger = cls(config=config)
        ledger._files_read = dict(data.get("files_read", {}))
        for entry_data in data.get("entries", []):
            ledger._entries.append(
                LedgerEntry(
                    timestamp=entry_data["timestamp"],
                    category=entry_data["category"],
                    key=entry_data["key"],
                    summary=entry_data["summary"],
                    turn_index=entry_data["turn_index"],
                    resolved=entry_data.get("resolved", False),
                )
            )
        return ledger
