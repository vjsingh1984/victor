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

"""Integration tests for victor sessions command with real database."""

from __future__ import annotations

import pytest
from pathlib import Path
import tempfile
import shutil

from victor.agent.sqlite_session_persistence import SQLiteSessionPersistence
from victor.config.settings import get_project_paths


@pytest.fixture
def project_db_path():
    """Get project database path."""
    return get_project_paths().project_root / ".victor" / "project.db"


@pytest.fixture
def backup_db(project_db_path):
    """Backup existing database before tests and provide clean state."""
    backup_path = project_db_path.with_suffix(".db.backup")

    # Backup existing database
    if project_db_path.exists():
        shutil.copy(project_db_path, backup_path)

    # Clear the database for clean test state
    if project_db_path.exists():
        project_db_path.unlink()

    yield

    # Restore backup
    if backup_path.exists():
        shutil.copy(backup_path, project_db_path)
        backup_path.unlink()


class TestSessionsCLIIntegration:
    """Integration tests for sessions CLI command."""

    @pytest.mark.integration
    def test_sessions_list_command(self, backup_db):
        """Test victor sessions list command with real database."""
        from typer.testing import CliRunner
        from victor.ui.commands.sessions import sessions_app

        runner = CliRunner()

        # Create sample sessions
        persistence = SQLiteSessionPersistence()
        persistence.save_session(
            conversation={"messages": [{"role": "user", "content": "Test"}]},
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            profile="default",
            title="Integration Test Session",
        )

        # Test list command
        result = runner.invoke(sessions_app, ["list"])
        assert result.exit_code == 0
        # Title is displayed across multiple lines in table, check for key parts
        assert "Integrati" in result.stdout or "Integration" in result.stdout
        assert "Test" in result.stdout
        assert "Session" in result.stdout

    @pytest.mark.integration
    def test_sessions_show_command(self, backup_db):
        """Test victor sessions show command."""
        from typer.testing import CliRunner
        from victor.ui.commands.sessions import sessions_app

        runner = CliRunner()

        # Create sample session
        persistence = SQLiteSessionPersistence()
        session_id = persistence.save_session(
            conversation={"messages": [{"role": "user", "content": "Test"}]},
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            profile="default",
            title="Show Test",
        )

        # Test show command
        result = runner.invoke(sessions_app, ["show", session_id])
        assert result.exit_code == 0
        assert "Show Test" in result.stdout
        assert session_id in result.stdout

    @pytest.mark.integration
    def test_parallel_sessions_workflow(self, backup_db):
        """Test parallel session creation and retrieval."""
        persistence = SQLiteSessionPersistence()

        # Create session 1
        session_id_1 = persistence.save_session(
            conversation={"messages": [{"role": "user", "content": "DevOps task"}]},
            model="ollama:qwen2.5-coder:7b",
            provider="ollama",
            profile="default",
            title="CI/CD Pipeline",
        )

        # Small delay to ensure different timestamp for unique ID
        import time
        time.sleep(0.01)

        # Create session 2
        session_id_2 = persistence.save_session(
            conversation={"messages": [{"role": "user", "content": "Testing task"}]},
            model="gpt-4",
            provider="openai",
            profile="default",
            title="Unit Tests",
        )

        # Verify both sessions exist and have different IDs
        assert session_id_1 != session_id_2, f"Session IDs should be unique but got: {session_id_1} == {session_id_2}"

        # Load both sessions
        session_1 = persistence.load_session(session_id_1)
        session_2 = persistence.load_session(session_id_2)

        assert session_1 is not None
        assert session_2 is not None
        assert session_1["metadata"]["title"] == "CI/CD Pipeline"
        assert session_2["metadata"]["title"] == "Unit Tests"

    @pytest.mark.integration
    def test_session_id_parsing(self):
        """Test session ID parsing utility."""
        from victor.agent.session_id import parse_session_id, validate_session_id

        # Valid session ID
        session_id = "myproj-9Kx7Z2"
        assert validate_session_id(session_id) is True

        parts = parse_session_id(session_id)
        assert parts["project_root"] == "myproj"
        assert parts["base62_timestamp"] == "9Kx7Z2"
        assert isinstance(parts["timestamp_ms"], int)
        assert isinstance(parts["timestamp_iso"], str)

        # Invalid session ID
        assert validate_session_id("invalid") is False
        assert validate_session_id("no-separator") is False


class TestChatSessionFlagsIntegration:
    """Integration tests for victor chat --sessions and --sessionid flags."""

    @pytest.mark.integration
    def test_chat_sessions_flag(self, backup_db):
        """Test victor chat --sessions lists sessions."""
        # This requires full orchestrator setup
        # Testing with subprocess would be more appropriate
        pass

    @pytest.mark.integration
    def test_chat_sessionid_flag(self, backup_db):
        """Test victor chat --sessionid restores session."""
        # This requires full orchestrator setup
        # Testing with subprocess would be more appropriate
        pass
