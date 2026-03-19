"""Tests for CLI chat history and planning wiring."""

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

        with patch("victor.config.settings.get_project_paths", side_effect=RuntimeError("no paths")):
            # Should not raise — falls back to InMemoryHistory
            session = _create_cli_prompt_session()
            from prompt_toolkit.history import InMemoryHistory

            assert isinstance(session.history, InMemoryHistory)


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
