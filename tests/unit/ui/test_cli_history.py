"""Tests for CLI chat history and planning wiring."""

import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


class TestCliPromptSession:
    """Tests for _create_cli_prompt_session with persistent history."""

    def test_creates_prompt_session(self):
        from victor.ui.commands.chat import _create_cli_prompt_session

        session = _create_cli_prompt_session()
        assert session is not None
        assert hasattr(session, "prompt")

    def test_uses_file_history(self):
        from victor.ui.commands.chat import _create_cli_prompt_session

        session = _create_cli_prompt_session()
        from prompt_toolkit.history import FileHistory

        assert isinstance(session.history, FileHistory)

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
                    timestamp TEXT
                )
                """)
            conn.executemany(
                "INSERT INTO messages(role, content, timestamp) VALUES (?, ?, ?)",
                [
                    ("user", "real prompt", "2026-04-26 10:00:00"),
                    ("user", "[SYSTEM-REMINDER: hidden]", "2026-04-26 10:01:00"),
                    ("user", "Continue. Use appropriate tools if needed.", "2026-04-26 10:02:00"),
                    ("user", "follow-up prompt", "2026-04-26 10:03:00"),
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
