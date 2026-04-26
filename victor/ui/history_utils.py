"""Utilities for loading and sanitizing interactive input history."""

from __future__ import annotations

import datetime as _dt
import json
import sqlite3
from pathlib import Path

from victor.agent.conversation.history_metadata import is_hidden_from_interactive_history

_INTERNAL_HISTORY_PREFIXES = (
    "[SYSTEM-REMINDER:",
    "[GROUNDING-FEEDBACK:",
    "[SYSTEM:",
    "[TASK-HINT:",
)

_INTERNAL_HISTORY_MESSAGE_PREFIXES = (
    "Continue. Use appropriate tools if needed.",
    "Continue your analysis. Use tools like ",
    "Continue with the implementation. Use tools like ",
    "Please provide a summary of your findings/work so far.",
    "CRITICAL: Provide your FINAL ANALYSIS NOW.",
    "You mentioned using ",
    "You are unable to make tool calls.",
    "The previous action did not complete.",
)

_INTERNAL_HISTORY_SUBSTRINGS = (
    "Call tools sequentially, waiting for results",
    "GROUNDING:",
    "Use ONLY content in <TOOL_OUTPUT>",
    "TOOL RULES:",
)


def is_interactive_history_entry(content: str, max_length: int = 4000) -> bool:
    """Return True when content should appear in user-facing input history."""
    text = content.strip()
    if not text or len(text) > max_length:
        return False

    if text.startswith("<TOOL_OUTPUT") or text.startswith("<") or text.startswith("{"):
        return False

    if any(text.startswith(prefix) for prefix in _INTERNAL_HISTORY_PREFIXES):
        return False

    if any(text.startswith(prefix) for prefix in _INTERNAL_HISTORY_MESSAGE_PREFIXES):
        return False

    if any(fragment in text for fragment in _INTERNAL_HISTORY_SUBSTRINGS):
        return False

    return True


def _is_interactive_history_metadata(metadata_json: str | None) -> bool:
    """Return True when stored metadata allows a message in input history."""
    if not metadata_json:
        return True

    try:
        metadata = json.loads(metadata_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return True

    if not isinstance(metadata, dict):
        return True

    return not is_hidden_from_interactive_history(metadata)


def load_prompt_toolkit_history_entries(history_file: Path) -> list[str]:
    """Load prompt_toolkit FileHistory entries in file order (oldest first)."""
    if not history_file.exists():
        return []

    entries: list[str] = []
    lines: list[str] = []
    with history_file.open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            if raw_line.startswith("+"):
                lines.append(raw_line[1:])
                continue

            if lines:
                entries.append("".join(lines)[:-1])
                lines = []

        if lines:
            entries.append("".join(lines)[:-1])

    return entries


def write_prompt_toolkit_history_entries(history_file: Path, entries: list[str]) -> None:
    """Rewrite a prompt_toolkit FileHistory file from normalized entries."""
    history_file.parent.mkdir(parents=True, exist_ok=True)

    with history_file.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(f"\n# {_dt.datetime.now()}\n")
            for line in entry.split("\n"):
                handle.write(f"+{line}\n")


def sanitize_prompt_toolkit_history_file(history_file: Path, max_entries: int = 250) -> int:
    """Remove invalid history items and trim to the newest complete entries."""
    entries = load_prompt_toolkit_history_entries(history_file)
    if not entries:
        return 0

    filtered = [entry for entry in entries if is_interactive_history_entry(entry)]
    if max_entries > 0:
        filtered = filtered[-max_entries:]

    if filtered != entries:
        write_prompt_toolkit_history_entries(history_file, filtered)

    return len(filtered)


def count_prompt_toolkit_history_entries(history_file: Path) -> int:
    """Return the number of complete prompt_toolkit history entries."""
    return len(load_prompt_toolkit_history_entries(history_file))


def load_input_history_from_db(db_path: Path, limit: int = 100) -> list[str]:
    """Load recent unique user input from the conversation database."""
    if limit <= 0 or not db_path.exists():
        return []

    fetch_limit = max(limit * 10, 200)
    messages: list[str] = []
    seen: set[str] = set()

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT content, metadata
            FROM messages
            WHERE role = 'user'
              AND content IS NOT NULL
              AND content != ''
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (fetch_limit,),
        )

        for content, metadata_json in cursor.fetchall():
            if content in seen:
                continue
            if not _is_interactive_history_metadata(metadata_json):
                continue
            if not is_interactive_history_entry(content):
                continue
            seen.add(content)
            messages.append(content)
            if len(messages) >= limit:
                break

    messages.reverse()
    return messages
