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

    def test_sessions_list_default(self, runner, sample_persistence):
        """Test 'victor sessions list' with default limit (10)."""
        result = runner.invoke(sessions_app, ["list"])
        assert result.exit_code == 0
        # Should show both sessions
        assert "CI/CD Pipeline Setup" in result.stdout
        assert "Unit Tests" in result.stdout
        assert "myproj-9Kx7Z2" in result.stdout
        assert "myproj-9Kx8A3B" in result.stdout

    def test_sessions_list_with_limit(self, runner, sample_persistence):
        """Test 'victor sessions list --limit 1'."""
        result = runner.invoke(sessions_app, ["list", "--limit", "1"])
        assert result.exit_code == 0
        # Should show only 1 session
        assert "CI/CD Pipeline Setup" in result.stdout or "Unit Tests" in result.stdout

    def test_sessions_list_json(self, runner, sample_persistence):
        """Test 'victor sessions list --json'."""
        result = runner.invoke(sessions_app, ["list", "--json"])
        assert result.exit_code == 0

        # Parse JSON output
        sessions = json.loads(result.stdout)
        assert isinstance(sessions, list)
        assert len(sessions) == 2
        assert sessions[0]["session_id"] in ["myproj-9Kx7Z2", "myproj-9Kx8A3B"]
        assert "title" in sessions[0]
        assert "model" in sessions[0]
        assert "provider" in sessions[0]

    def test_sessions_list_empty(self, runner, temp_db_path):
        """Test 'victor sessions list' with empty database."""
        persistence = SQLiteSessionPersistence(db_path=temp_db_path)
        result = runner.invoke(sessions_app, ["list"])
        assert result.exit_code == 0
        assert "No sessions found" in result.stdout or "0 sessions" in result.stdout

    def test_sessions_search(self, runner, sample_persistence):
        """Test 'victor sessions search CI/CD'."""
        result = runner.invoke(sessions_app, ["search", "CI/CD"])
        assert result.exit_code == 0
        assert "CI/CD Pipeline Setup" in result.stdout
        assert "Unit Tests" not in result.stdout

    def test_sessions_search_json(self, runner, sample_persistence):
        """Test 'victor sessions search --json'."""
        result = runner.invoke(sessions_app, ["search", "CI/CD", "--json"])
        assert result.exit_code == 0

        sessions = json.loads(result.stdout)
        assert isinstance(sessions, list)
        assert len(sessions) == 1
        assert sessions[0]["title"] == "CI/CD Pipeline Setup"

    def test_sessions_show(self, runner, sample_persistence):
        """Test 'victor sessions show <session_id>'."""
        result = runner.invoke(sessions_app, ["show", "myproj-9Kx7Z2"])
        assert result.exit_code == 0
        assert "CI/CD Pipeline Setup" in result.stdout
        assert "claude-sonnet-4-20250514" in result.stdout
        assert "anthropic" in result.stdout
        assert "2 messages" in result.stdout or "message_count" in result.stdout

    def test_sessions_show_json(self, runner, sample_persistence):
        """Test 'victor sessions show --json'."""
        result = runner.invoke(sessions_app, ["show", "myproj-9Kx7Z2", "--json"])
        assert result.exit_code == 0

        session = json.loads(result.stdout)
        assert session["session_id"] == "myproj-9Kx7Z2"
        assert session["title"] == "CI/CD Pipeline Setup"
        assert session["model"] == "claude-sonnet-4-20250514"
        assert session["provider"] == "anthropic"

    def test_sessions_show_not_found(self, runner, sample_persistence):
        """Test 'victor sessions show' with non-existent session."""
        result = runner.invoke(sessions_app, ["show", "nonexistent-12345"])
        assert result.exit_code != 0 or "not found" in result.stdout.lower()

    def test_sessions_delete(self, runner, sample_persistence):
        """Test 'victor sessions delete <session_id>'."""
        # Delete session
        result = runner.invoke(sessions_app, ["delete", "myproj-9Kx7Z2", "--yes"])
        assert result.exit_code == 0
        assert "deleted" in result.stdout.lower()

        # Verify it's gone
        result = runner.invoke(sessions_app, ["show", "myproj-9Kx7Z2"])
        assert "not found" in result.stdout.lower()

    def test_sessions_export(self, runner, sample_persistence, tmp_path):
        """Test 'victor sessions export'."""
        export_file = tmp_path / "sessions.json"
        result = runner.invoke(
            sessions_app, ["export", "--output", str(export_file)]
        )
        assert result.exit_code == 0
        assert export_file.exists()

        # Verify export
        with open(export_file) as f:
            exported = json.load(f)
        assert len(exported) == 2
        assert exported[0]["session_id"] in ["myproj-9Kx7Z2", "myproj-9Kx8A3B"]


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
