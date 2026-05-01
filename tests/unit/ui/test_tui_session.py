"""Unit tests for TUI session persistence and export."""

from __future__ import annotations

import sqlite3
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from victor.agent.message_history import MessageHistory
from victor.ui.tui.app import VictorTUI
from victor.ui.tui.session import Message, SessionManager


class _FakeDatabaseManager:
    """Minimal database manager stub for TUI session tests."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)

    def get_connection(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        self._conn.close()


@pytest.fixture
def temp_session_manager(tmp_path: Path):
    """Create a SessionManager bound to a temporary SQLite database."""
    fake_db = _FakeDatabaseManager(tmp_path / "tui-session.db")
    with patch("victor.ui.tui.session.get_database", return_value=fake_db):
        manager = SessionManager()
    try:
        yield manager
    finally:
        fake_db.close()


def test_session_manager_roundtrips_preview_metadata(
    temp_session_manager: SessionManager,
) -> None:
    """Saved TUI sessions should preserve preview metadata on load."""
    session = temp_session_manager.create_session(provider="zai", model="glm-5.1", name="Preview")
    session.messages = [
        Message(
            role="system",
            content="File preview: /tmp/test.py",
            metadata={
                "preview_body": "print('hello')",
                "preview_kind": "file",
                "preview_language": "py",
                "preview_path": "/tmp/test.py",
            },
        )
    ]

    temp_session_manager.save(session)
    loaded = temp_session_manager.load(session.id)

    assert loaded is not None
    assert loaded.messages[0].content == "File preview: /tmp/test.py"
    assert loaded.messages[0].metadata == {
        "preview_body": "print('hello')",
        "preview_kind": "file",
        "preview_language": "py",
        "preview_path": "/tmp/test.py",
    }


def test_message_history_roundtrips_preview_sidecar() -> None:
    """Project conversation serialization should preserve preview sidecar messages."""
    history = MessageHistory(system_prompt="system")
    history.add_message("user", "hi")
    history.add_preview_message(
        "system",
        "File preview: /tmp/test.py",
        {
            "preview_body": "print('hello')",
            "preview_kind": "file",
            "preview_language": "py",
            "preview_path": "/tmp/test.py",
        },
    )

    data = history.to_dict()
    restored = MessageHistory.from_dict(data)

    assert data["preview_messages"] == [
        {
            "role": "system",
            "content": "File preview: /tmp/test.py",
            "metadata": {
                "preview_body": "print('hello')",
                "preview_kind": "file",
                "preview_language": "py",
                "preview_path": "/tmp/test.py",
            },
            "after_message_index": 1,
        }
    ]
    assert restored.preview_messages == data["preview_messages"]


def test_message_history_clear_resets_preview_sidecar() -> None:
    """Clearing history should also clear replay-only preview sidecar state."""
    history = MessageHistory(system_prompt="system")
    history.add_message("user", "hi")
    history.add_preview_message(
        "system",
        "File preview: /tmp/test.py",
        {
            "preview_body": "print('hello')",
            "preview_kind": "file",
            "preview_language": "py",
            "preview_path": "/tmp/test.py",
        },
    )

    history.clear()

    assert history.messages == []
    assert history.preview_messages == []


def test_session_manager_export_markdown_includes_preview_code_block(
    temp_session_manager: SessionManager,
    tmp_path: Path,
) -> None:
    """Markdown export should include stored preview bodies as fenced code blocks."""
    session = temp_session_manager.create_session(provider="zai", model="glm-5.1", name="Preview")
    session.messages = [
        Message(
            role="system",
            content="Edit preview: /tmp/test.py",
            metadata={
                "preview_body": "-old\n+new",
                "preview_kind": "edit",
                "preview_language": "diff",
                "preview_path": "/tmp/test.py",
            },
        )
    ]
    temp_session_manager.save(session)

    output_path = tmp_path / "session.md"

    assert temp_session_manager.export_markdown(session.id, output_path) is True
    markdown = output_path.read_text()

    assert "### System" in markdown
    assert "Edit preview: /tmp/test.py" in markdown
    assert "```diff" in markdown
    assert "-old\n+new" in markdown


def test_tui_save_and_load_roundtrip_preview_metadata(
    temp_session_manager: SessionManager,
) -> None:
    """VictorTUI save/load flow should replay preview metadata as a code block."""
    app = VictorTUI(provider="zai", model="glm-5.1")
    app._session_messages = [
        Message(
            role="system",
            content="File preview: /tmp/test.py",
            metadata={
                "preview_body": "print('hello')",
                "preview_kind": "file",
                "preview_language": "py",
                "preview_path": "/tmp/test.py",
            },
        )
    ]
    app._add_system_message = MagicMock()
    app._add_error_message = MagicMock()
    app._restore_agent_conversation = MagicMock()
    app._set_status = MagicMock()
    app._conversation_log = MagicMock()
    app.batch_update = MagicMock(return_value=nullcontext())

    with patch("victor.ui.tui.session.SessionManager", return_value=temp_session_manager):
        app.action_save_session()
        saved_session = temp_session_manager.get_latest()
        assert saved_session is not None

        app._load_session(saved_session.id)

    assert saved_session is not None
    assert saved_session.messages[0].metadata["preview_body"] == "print('hello')"
    assert saved_session.messages[0].metadata["preview_language"] == "py"
    app._conversation_log.add_history_message.assert_any_call(
        "system",
        "File preview: /tmp/test.py",
    )
    app._conversation_log.add_history_code_block.assert_called_once_with("print('hello')", "py")
    assert app._session_messages[0].metadata["preview_body"] == "print('hello')"


def test_tui_export_session_writes_preview_aware_markdown(tmp_path: Path) -> None:
    """VictorTUI export should include preview blocks in the emitted markdown file."""
    app = VictorTUI(provider="zai", model="glm-5.1")
    app._session_messages = [
        Message(
            role="system",
            content="Edit preview: /tmp/test.py",
            metadata={
                "preview_body": "-old\n+new",
                "preview_kind": "edit",
                "preview_language": "diff",
                "preview_path": "/tmp/test.py",
            },
        )
    ]
    app._add_system_message = MagicMock()
    app._add_error_message = MagicMock()
    output_path = tmp_path / "victor_session_export.md"

    def _fake_named_tempfile(*args, **kwargs):
        return output_path.open(mode=kwargs.get("mode", "w"))

    with patch("tempfile.NamedTemporaryFile", side_effect=_fake_named_tempfile):
        app.action_export_session()

    app._add_error_message.assert_not_called()
    assert app._add_system_message.call_args_list == [
        call(f"Session exported to: {output_path}"),
        call("Message count: 1"),
    ]
    markdown = output_path.read_text()
    assert "Edit preview: /tmp/test.py" in markdown
    assert "```diff" in markdown
    assert "-old\n+new" in markdown
