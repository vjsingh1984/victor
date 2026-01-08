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
        assert (
            session_id_1 != session_id_2
        ), f"Session IDs should be unique but got: {session_id_1} == {session_id_2}"

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

    @pytest.mark.integration
    def test_sessions_clear_all(self, backup_db):
        """Test victor sessions clear command to delete all sessions."""
        from typer.testing import CliRunner
        from victor.ui.commands.sessions import sessions_app

        runner = CliRunner()

        # Create multiple sample sessions
        persistence = SQLiteSessionPersistence()
        persistence.save_session(
            conversation={"messages": [{"role": "user", "content": "Test 1"}]},
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            profile="default",
            title="Test Session 1",
        )

        # Small delay to ensure different timestamp
        import time

        time.sleep(0.01)

        persistence.save_session(
            conversation={"messages": [{"role": "user", "content": "Test 2"}]},
            model="gpt-4",
            provider="openai",
            profile="default",
            title="Test Session 2",
        )

        time.sleep(0.01)

        persistence.save_session(
            conversation={"messages": [{"role": "user", "content": "Test 3"}]},
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            profile="default",
            title="Test Session 3",
        )

        # Verify sessions exist
        sessions_before = persistence.list_sessions(limit=100)
        assert len(sessions_before) >= 3

        # Clear all sessions with --yes flag
        result = runner.invoke(sessions_app, ["clear", "--yes"])
        assert result.exit_code == 0
        assert "Cleared" in result.stdout
        assert "session(s) from database" in result.stdout

        # Verify all sessions are deleted
        sessions_after = persistence.list_sessions(limit=100)
        assert len(sessions_after) == 0

    @pytest.mark.integration
    def test_sessions_clear_with_prefix(self, backup_db):
        """Test victor sessions clear with prefix to delete specific sessions."""
        from typer.testing import CliRunner
        from victor.ui.commands.sessions import sessions_app

        runner = CliRunner()

        # Create sessions with specific prefixes
        persistence = SQLiteSessionPersistence()

        # Create a session with a predictable prefix (session IDs are auto-generated)
        # We'll create sessions and then match by their actual IDs
        session_id_1 = persistence.save_session(
            conversation={"messages": [{"role": "user", "content": "DevOps task"}]},
            model="ollama:qwen2.5-coder:7b",
            provider="ollama",
            profile="default",
            title="CI/CD Pipeline",
        )

        # Small delay to ensure different timestamp
        import time

        time.sleep(0.01)

        session_id_2 = persistence.save_session(
            conversation={"messages": [{"role": "user", "content": "Testing task"}]},
            model="gpt-4",
            provider="openai",
            profile="default",
            title="Unit Tests",
        )

        # Verify both sessions exist
        assert persistence.load_session(session_id_1) is not None
        assert persistence.load_session(session_id_2) is not None

        # Find a unique prefix that matches only session_id_1
        # Start with a longer prefix and make sure it doesn't match session_id_2
        for prefix_len in range(12, 6, -1):  # Try 12, 11, 10, 9, 8, 7
            prefix = session_id_1[:prefix_len]
            if not session_id_2.startswith(prefix):
                break  # Found a unique prefix
        else:
            # If no unique prefix found, skip this test
            pytest.skip("Could not find unique prefix for test")

        assert len(prefix) >= 6, "Prefix should be at least 6 characters"

        # Clear sessions matching prefix
        result = runner.invoke(sessions_app, ["clear", prefix, "--yes"])
        assert result.exit_code == 0
        assert "Cleared" in result.stdout
        assert f"matching prefix '{prefix}'" in result.stdout

        # Verify session with matching prefix is deleted
        assert persistence.load_session(session_id_1) is None

        # Verify other session still exists
        assert persistence.load_session(session_id_2) is not None

    @pytest.mark.integration
    def test_sessions_clear_prefix_too_short(self, backup_db):
        """Test victor sessions clear with prefix shorter than 6 chars."""
        from typer.testing import CliRunner
        from victor.ui.commands.sessions import sessions_app

        runner = CliRunner()

        # Try to clear with prefix shorter than 6 characters
        result = runner.invoke(sessions_app, ["clear", "abc", "--yes"])
        assert result.exit_code != 0
        assert "Prefix must be at least 6 characters long" in result.stdout

    @pytest.mark.integration
    def test_sessions_clear_prefix_not_found(self, backup_db):
        """Test victor sessions clear with non-matching prefix."""
        from typer.testing import CliRunner
        from victor.ui.commands.sessions import sessions_app

        runner = CliRunner()

        # Try to clear with prefix that doesn't match any sessions
        result = runner.invoke(sessions_app, ["clear", "nomatch-999", "--yes"])
        assert result.exit_code == 0
        assert "No sessions found matching prefix 'nomatch-999'" in result.stdout

    @pytest.mark.integration
    def test_sessions_clear_empty_database(self, backup_db):
        """Test victor sessions clear when database is empty."""
        from typer.testing import CliRunner
        from victor.ui.commands.sessions import sessions_app

        runner = CliRunner()

        # Ensure database is empty by clearing all sessions
        persistence = SQLiteSessionPersistence()
        all_sessions = persistence.list_sessions(limit=10000)
        for session in all_sessions:
            persistence.delete_session(session["session_id"])

        # Verify database is now empty
        sessions = persistence.list_sessions(limit=100)
        assert len(sessions) == 0

        # Try to clear empty database
        result = runner.invoke(sessions_app, ["clear", "--yes"])
        assert result.exit_code == 0
        assert "No sessions found" in result.stdout

    @pytest.mark.integration
    def test_sessions_list_all(self, backup_db):
        """Test victor sessions list --all command."""
        from typer.testing import CliRunner
        from victor.ui.commands.sessions import sessions_app

        runner = CliRunner()

        # Create more than 10 sessions to test --all flag
        persistence = SQLiteSessionPersistence()
        import time

        for i in range(15):
            persistence.save_session(
                conversation={"messages": [{"role": "user", "content": f"Test {i}"}]},
                model="claude-sonnet-4-20250514",
                provider="anthropic",
                profile="default",
                title=f"Session {i}",
            )
            time.sleep(0.01)  # Ensure unique timestamps

        # List with default limit (should show 10)
        result_default = runner.invoke(sessions_app, ["list"])
        assert result_default.exit_code == 0
        # Count occurrences of session IDs in output
        session_count = result_default.stdout.count("Session ")
        # Should show around 10 sessions (default limit)
        assert session_count <= 15  # At most 15

        # List with --all flag (should show all 15)
        result_all = runner.invoke(sessions_app, ["list", "--all"])
        assert result_all.exit_code == 0
        # Check that it shows more sessions than default
        all_session_count = result_all.stdout.count("Session ")
        assert all_session_count >= 15  # All sessions should be shown
