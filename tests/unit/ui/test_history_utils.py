"""Tests for interactive history filtering and sanitization."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from victor.ui.history_utils import (
    is_interactive_history_entry,
    load_input_history_from_db,
    load_prompt_toolkit_history_entries,
    sanitize_prompt_toolkit_history_file,
    write_prompt_toolkit_history_entries,
)


def test_is_interactive_history_entry_filters_internal_prompts():
    assert is_interactive_history_entry("fix the failing test")
    assert not is_interactive_history_entry("[SYSTEM-REMINDER: use tools]")
    assert not is_interactive_history_entry("[GROUNDING-FEEDBACK: correction required]")
    assert not is_interactive_history_entry("Continue. Use appropriate tools if needed.")
    assert not is_interactive_history_entry(
        "- Call tools sequentially, waiting for results\n\nGROUNDING:\n- Use ONLY content in <TOOL_OUTPUT> markers."
    )
    assert not is_interactive_history_entry(
        "Please provide a summary of your findings/work so far. Conclude your response."
    )


def test_load_input_history_from_db_filters_internal_and_keeps_recent_unique(tmp_path: Path):
    db_path = tmp_path / "project.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE messages (
                role TEXT,
                content TEXT,
                timestamp TEXT
            )
            """)
        conn.executemany(
            "INSERT INTO messages(role, content, timestamp) VALUES (?, ?, ?)",
            [
                ("user", "first real prompt", "2026-04-26 10:00:00"),
                ("user", "[SYSTEM-REMINDER: use tools]", "2026-04-26 10:01:00"),
                ("user", "first real prompt", "2026-04-26 10:02:00"),
                ("user", "Continue. Use appropriate tools if needed.", "2026-04-26 10:03:00"),
                ("user", "second real prompt", "2026-04-26 10:04:00"),
                ("user", '<TOOL_OUTPUT tool="read">x</TOOL_OUTPUT>', "2026-04-26 10:05:00"),
                ("assistant", "ignored assistant", "2026-04-26 10:06:00"),
                ("user", "multiline\nreal prompt", "2026-04-26 10:07:00"),
            ],
        )

    assert load_input_history_from_db(db_path, limit=10) == [
        "first real prompt",
        "second real prompt",
        "multiline\nreal prompt",
    ]


def test_sanitize_prompt_toolkit_history_file_preserves_complete_multiline_entries(
    tmp_path: Path,
):
    history_file = tmp_path / "chat_history"
    history_file.write_text(
        "+[SYSTEM-REMINDER: leaked orphan entry]\n"
        "\n# 2026-04-26 10:00:00\n"
        "+first real prompt\n"
        "\n# 2026-04-26 10:01:00\n"
        "+Continue. Use appropriate tools if needed.\n"
        "\n# 2026-04-26 10:02:00\n"
        "+multi line 1\n"
        "+multi line 2\n",
        encoding="utf-8",
    )

    remaining = sanitize_prompt_toolkit_history_file(history_file, max_entries=10)

    assert remaining == 2
    assert load_prompt_toolkit_history_entries(history_file) == [
        "first real prompt",
        "multi line 1\nmulti line 2",
    ]


def test_sanitize_prompt_toolkit_history_file_trims_by_entries_not_lines(tmp_path: Path):
    history_file = tmp_path / "chat_history"
    write_prompt_toolkit_history_entries(
        history_file,
        [
            "one line",
            "two line a\ntwo line b",
            "three line a\nthree line b\nthree line c",
        ],
    )

    sanitize_prompt_toolkit_history_file(history_file, max_entries=2)

    assert load_prompt_toolkit_history_entries(history_file) == [
        "two line a\ntwo line b",
        "three line a\nthree line b\nthree line c",
    ]
