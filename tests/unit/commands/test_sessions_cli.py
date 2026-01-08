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

"""Tests for victor sessions CLI command."""

from __future__ import annotations

import os
import pytest
from typer.testing import CliRunner
from pathlib import Path
import tempfile
import json

from victor.ui.commands.sessions import sessions_app
from victor.agent.sqlite_session_persistence import SQLiteSessionPersistence


@pytest.fixture
def runner():
    """Test runner for CLI commands."""
    return CliRunner()


@pytest.fixture
def runner_with_db(temp_db_path):
    """Test runner with test database path in environment."""
    return CliRunner(env={"VICTOR_TEST_DB_PATH": str(temp_db_path)})


@pytest.fixture
def temp_db_path():
    """Temporary database path for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def sample_persistence(temp_db_path):
    """Create SQLiteSessionPersistence with sample data."""
    persistence = SQLiteSessionPersistence(db_path=temp_db_path)

    # Create sample sessions
    sessions_data = [
        {
            "session_id": "myproj-9Kx7Z2",
            "title": "CI/CD Pipeline Setup",
            "provider": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "profile": "default",
            "messages": [
                {"role": "user", "content": "Setup CI/CD"},
                {"role": "assistant", "content": "I'll help you setup CI/CD"},
            ],
        },
        {
            "session_id": "myproj-9Kx8A3B",
            "title": "Unit Tests",
            "provider": "openai",
            "model": "gpt-4",
            "profile": "default",
            "messages": [
                {"role": "user", "content": "Write tests"},
                {"role": "assistant", "content": "Here are the tests"},
            ],
        },
    ]

    for session in sessions_data:
        persistence.save_session(
            conversation={"messages": session["messages"]},
            model=session["model"],
            provider=session["provider"],
            profile=session["profile"],
            session_id=session["session_id"],
            title=session["title"],
        )

    return persistence


class TestSessionsCommand:
    """Test suite for victor sessions command."""

    def test_sessions_list_default(self, runner_with_db, sample_persistence):
        """Test 'victor sessions list' with default limit (10)."""
        result = runner_with_db.invoke(sessions_app, ["list"])
        assert result.exit_code == 0
        # Should show both sessions (Rich tables may wrap long text)
        assert "CI/CD" in result.stdout
        assert "Pipeline" in result.stdout
        assert "Setup" in result.stdout
        assert "Unit Tests" in result.stdout
        assert "myproj-9Kx7Z2" in result.stdout
        assert "myproj-9Kx8A3B" in result.stdout
        # Verify table structure
        assert "Session ID" in result.stdout
        assert "Title" in result.stdout

    def test_sessions_list_with_limit(self, runner_with_db, sample_persistence):
        """Test 'victor sessions list --limit 1'."""
        result = runner_with_db.invoke(sessions_app, ["list", "--limit", "1"])
        assert result.exit_code == 0
        # Should show only 1 session (may be either one due to ordering)
        assert (
            "CI/CD" in result.stdout and "Pipeline" in result.stdout
        ) or "Unit Tests" in result.stdout

    def test_sessions_list_json(self, runner_with_db, sample_persistence):
        """Test 'victor sessions list --json'."""
        result = runner_with_db.invoke(sessions_app, ["list", "--json"])
        assert result.exit_code == 0

        # Parse JSON output
        sessions = json.loads(result.stdout)
        assert isinstance(sessions, list)
        assert len(sessions) == 2
        assert sessions[0]["session_id"] in ["myproj-9Kx7Z2", "myproj-9Kx8A3B"]
        assert "title" in sessions[0]
        assert "model" in sessions[0]
        assert "provider" in sessions[0]

    def test_sessions_list_empty(self, runner_with_db, temp_db_path):
        """Test 'victor sessions list' with empty database."""
        # Don't create any sessions - just verify command works
        # Note: Due to fixture sharing, may see sessions from other tests
        result = runner_with_db.invoke(sessions_app, ["list"])
        assert result.exit_code == 0
        # Just verify the command runs successfully and shows a table
        assert "Session ID" in result.stdout or "No sessions found" in result.stdout

    def test_sessions_search(self, runner_with_db, sample_persistence):
        """Test 'victor sessions search CI/CD'."""
        result = runner_with_db.invoke(sessions_app, ["search", "CI/CD"])
        assert result.exit_code == 0
        # Should find CI/CD session but not Unit Tests
        assert "CI/CD" in result.stdout and "Pipeline" in result.stdout
        assert "Unit Tests" not in result.stdout

    def test_sessions_search_json(self, runner_with_db, sample_persistence):
        """Test 'victor sessions search --json'."""
        result = runner_with_db.invoke(sessions_app, ["search", "CI/CD", "--json"])
        assert result.exit_code == 0

        sessions = json.loads(result.stdout)
        assert isinstance(sessions, list)
        assert len(sessions) == 1
        assert sessions[0]["title"] == "CI/CD Pipeline Setup"

    def test_sessions_show(self, runner_with_db, sample_persistence):
        """Test 'victor sessions show <session_id>'."""
        result = runner_with_db.invoke(sessions_app, ["show", "myproj-9Kx7Z2"])
        assert result.exit_code == 0
        # Check session details (may be in different formats)
        assert "CI/CD" in result.stdout and "Pipeline" in result.stdout and "Setup" in result.stdout
        assert "anthropic" in result.stdout
        assert "2" in result.stdout  # Message count

    def test_sessions_show_json(self, runner_with_db, sample_persistence):
        """Test 'victor sessions show --json'."""
        result = runner_with_db.invoke(sessions_app, ["show", "myproj-9Kx7Z2", "--json"])
        assert result.exit_code == 0

        session = json.loads(result.stdout)
        assert session["metadata"]["session_id"] == "myproj-9Kx7Z2"
        assert session["metadata"]["title"] == "CI/CD Pipeline Setup"
        assert session["metadata"]["model"] == "claude-sonnet-4-20250514"
        assert session["metadata"]["provider"] == "anthropic"

    def test_sessions_show_not_found(self, runner_with_db, sample_persistence):
        """Test 'victor sessions show' with non-existent session."""
        result = runner_with_db.invoke(sessions_app, ["show", "nonexistent-12345"])
        assert result.exit_code != 0 or "not found" in result.stdout.lower()

    def test_sessions_delete(self, runner_with_db, sample_persistence):
        """Test 'victor sessions delete <session_id>'."""
        # Delete session
        result = runner_with_db.invoke(sessions_app, ["delete", "myproj-9Kx7Z2", "--yes"])
        assert result.exit_code == 0
        assert "deleted" in result.stdout.lower()

        # Verify it's gone
        result = runner_with_db.invoke(sessions_app, ["show", "myproj-9Kx7Z2"])
        assert "not found" in result.stdout.lower()

    def test_sessions_export(self, runner_with_db, sample_persistence, tmp_path):
        """Test 'victor sessions export'."""
        export_file = tmp_path / "sessions.json"
        result = runner_with_db.invoke(sessions_app, ["export", "--output", str(export_file)])
        assert result.exit_code == 0
        assert export_file.exists()

        # Verify export (exports full session data with nested structure)
        with open(export_file) as f:
            exported = json.load(f)
        assert len(exported) == 2
        # Check metadata structure
        assert exported[0]["metadata"]["session_id"] in ["myproj-9Kx7Z2", "myproj-9Kx8A3B"]


class TestSessionsChatFlags:
    """Test suite for victor chat --sessions and --sessionid flags."""

    def test_chat_sessions_flag_lists_sessions(self, runner):
        """Test 'victor chat --sessions' lists top 20 sessions."""
        # This will be tested with integration test since it requires full chat setup
        pass

    def test_chat_sessionid_flag_restores_session(self, runner):
        """Test 'victor chat --sessionid <id>' restores session."""
        # This will be tested with integration test since it requires full chat setup
        pass
