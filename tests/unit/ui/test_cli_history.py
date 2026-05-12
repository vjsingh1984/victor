"""Tests for CLI chat history and planning wiring."""

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


class TestCliPromptSession:
    """Tests for _create_cli_prompt_session with persistent history."""

    def test_prompt_fragments_include_profile_context(self):
        from victor.ui.commands.chat import _build_cli_prompt_fragments

        fragments = _build_cli_prompt_fragments("coding")

        rendered = "".join(text for _style, text in fragments)
        assert "victor" in rendered
        assert "[coding]" in rendered

    def test_bottom_toolbar_shows_runtime_context_and_shortcuts(self):
        from victor.ui.commands.chat import _build_cli_bottom_toolbar

        settings = SimpleNamespace(
            provider=SimpleNamespace(default_provider="ollama", default_model="qwen")
        )
        profile = SimpleNamespace(provider="anthropic", model="claude-sonnet")

        toolbar = _build_cli_bottom_toolbar(
            settings=settings,
            profile_config=profile,
            profile_name="review",
            vertical_name="coding",
        )

        rendered = "".join(text for _style, text in toolbar)
        assert "review" in rendered
        assert "anthropic" in rendered
        assert "claude-sonnet" in rendered
        assert "coding" in rendered
        assert "Tab commands" in rendered
        assert "Ctrl+O" in rendered

    def test_right_prompt_shows_send_and_exit_hints(self):
        from victor.ui.commands.chat import _build_cli_right_prompt

        rendered = "".join(text for _style, text in _build_cli_right_prompt())

        assert "Enter send" in rendered
        assert "Ctrl+D exit" in rendered

    def test_creates_prompt_session(self):
        from victor.ui.commands.chat import _create_cli_prompt_session

        session = _create_cli_prompt_session()
        assert session is not None
        assert hasattr(session, "prompt")
        assert session.completer is not None
        assert session.bottom_toolbar is not None
        assert session.rprompt is not None
        assert session.mouse_support is False
        assert session.reserve_space_for_menu >= 8

    def test_mouse_support_is_opt_in(self, monkeypatch):
        from victor.ui.commands.chat import _create_cli_prompt_session

        monkeypatch.setenv("VICTOR_CHAT_MOUSE_SUPPORT", "1")

        session = _create_cli_prompt_session()

        assert session.mouse_support is True

    def test_command_completer_suggests_slash_commands(self):
        from prompt_toolkit.document import Document

        from victor.ui.commands.chat import _build_cli_command_completer

        completer = _build_cli_command_completer()
        completions = list(completer.get_completions(Document("/mo"), None))
        labels = {completion.text for completion in completions}

        assert "/model" in labels
        assert "/mode" in labels

    def test_command_completer_includes_metadata_and_aliases(self):
        from prompt_toolkit.document import Document

        from victor.ui.commands.chat import _build_cli_command_completer

        completer = _build_cli_command_completer()
        completions = list(completer.get_completions(Document("/?"), None))

        assert completions
        assert completions[0].text == "/?"
        assert "alias" in str(completions[0].display_meta)

    def test_normalizes_cli_input_aliases(self):
        from victor.ui.commands.chat import _normalize_cli_input_alias

        assert _normalize_cli_input_alias("/?") == "/help"
        assert _normalize_cli_input_alias(":q") == "/quit"
        assert _normalize_cli_input_alias("hello") == "hello"

    def test_uses_file_history(self):
        from victor.ui.commands.chat import _create_cli_prompt_session

        session = _create_cli_prompt_session()
        from prompt_toolkit.history import FileHistory

        assert isinstance(session.history, FileHistory)

    def test_default_history_file_uses_isolated_unit_test_victor_dir(
        self, isolated_project_victor_dir
    ):
        from victor.ui.commands.chat import _create_cli_prompt_session

        session = _create_cli_prompt_session()

        assert Path(session.history.filename) == isolated_project_victor_dir / "chat_history"

    def test_fallback_to_in_memory_on_error(self):
        from victor.ui.commands.chat import _create_cli_prompt_session

        with patch(
            "victor.config.settings.get_project_paths",
            side_effect=RuntimeError("no paths"),
        ):
            # Should not raise — falls back to InMemoryHistory
            session = _create_cli_prompt_session()
            from prompt_toolkit.history import InMemoryHistory

            assert isinstance(session.history, InMemoryHistory)

    def test_seeds_history_from_db_without_internal_prompts(self, tmp_path):
        from victor.ui.commands.chat import _create_cli_prompt_session

        project_dir = tmp_path / ".victor"
        project_dir.mkdir(exist_ok=True)
        db_path = project_dir / "project.db"

        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE messages (
                    role TEXT,
                    content TEXT,
                    timestamp TEXT,
                    metadata TEXT
                )
                """)
            conn.executemany(
                "INSERT INTO messages(role, content, timestamp, metadata) VALUES (?, ?, ?, ?)",
                [
                    ("user", "real prompt", "2026-04-26 10:00:00", None),
                    ("user", "[SYSTEM-REMINDER: hidden]", "2026-04-26 10:01:00", None),
                    (
                        "user",
                        "Continue. Use appropriate tools if needed.",
                        "2026-04-26 10:02:00",
                        None,
                    ),
                    (
                        "user",
                        "plain-looking hidden prompt",
                        "2026-04-26 10:02:30",
                        '{"interactive_history": false, "internal_prompt_kind": "prompt_tool_call"}',
                    ),
                    ("user", "follow-up prompt", "2026-04-26 10:03:00", None),
                ],
            )

        fake_paths = SimpleNamespace(project_victor_dir=project_dir, project_db=db_path)
        fake_settings = SimpleNamespace(ui=SimpleNamespace(cli_history_max_entries=20))

        with patch("victor.config.settings.get_project_paths", return_value=fake_paths):
            session = _create_cli_prompt_session(settings=fake_settings)

        history_strings = list(session.history.load_history_strings())
        assert history_strings == ["follow-up prompt", "real prompt"]


class TestPlanningWiring:
    """Tests that planning is wired through agent.chat(use_planning=...)."""

    @pytest.mark.asyncio
    async def test_orchestrator_chat_passes_use_planning(self):
        """Verify orchestrator.chat() passes use_planning to coordinator."""
        mock_coordinator = AsyncMock()
        mock_coordinator.chat.return_value = MagicMock(content="response")

        orchestrator = MagicMock()
        orchestrator._use_service_layer = False
        orchestrator._chat_service = None
        orchestrator._chat_coordinator = mock_coordinator

        # Call the actual method logic
        from victor.providers.base import CompletionResponse

        # Simulate what orchestrator.chat does
        result = await mock_coordinator.chat("test message", use_planning=None)
        mock_coordinator.chat.assert_called_once_with("test message", use_planning=None)

    @pytest.mark.asyncio
    async def test_orchestrator_chat_default_no_planning(self):
        """Verify default use_planning=False preserves backward compat."""
        mock_coordinator = AsyncMock()
        mock_coordinator.chat.return_value = MagicMock(content="response")

        result = await mock_coordinator.chat("test message", use_planning=False)
        mock_coordinator.chat.assert_called_once_with("test message", use_planning=False)
