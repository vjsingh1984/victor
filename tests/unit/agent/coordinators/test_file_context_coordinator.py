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

"""Tests for FileContextCoordinator."""

import pytest

from victor.agent.coordinators.file_context_coordinator import (
    FileContextCoordinator,
    create_file_context_coordinator,
)


class MockTaskAnalyzer:
    """Mock TaskAnalyzer for testing."""

    def extract_required_files_from_prompt(self, prompt: str) -> list[str]:
        """Extract file paths from prompt."""
        # Simple extraction - find .py, .md, .txt files
        import re

        return re.findall(r"[\w/]+\.(?:py|md|txt)", prompt)

    def extract_required_outputs_from_prompt(self, prompt: str) -> list[str]:
        """Extract output requirements from prompt."""
        # Simple extraction - find quoted phrases
        import re

        return re.findall(r'"([^"]+)"', prompt)


@pytest.fixture
def task_analyzer():
    """Fixture for TaskAnalyzer."""
    return MockTaskAnalyzer()


@pytest.fixture
def coordinator(task_analyzer):
    """Fixture for FileContextCoordinator."""
    return FileContextCoordinator(task_analyzer=task_analyzer)


class TestFileContextCoordinator:
    """Test suite for FileContextCoordinator."""

    def test_initialization(self, coordinator, task_analyzer):
        """Test coordinator initialization."""
        assert coordinator._task_analyzer is task_analyzer
        assert coordinator.get_required_files() == []
        assert coordinator.get_required_outputs() == []
        assert coordinator.get_observed_files() == set()
        assert coordinator.get_read_session() == set()
        assert coordinator.get_nudge_sent() is False

    def test_extract_requirements(self, coordinator):
        """Test requirement extraction from prompt."""
        prompt = 'Read main.py and utils.py, create "test report" and "summary table"'
        coordinator.extract_requirements(prompt)

        files = coordinator.get_required_files()
        outputs = coordinator.get_required_outputs()

        assert "main.py" in files
        assert "utils.py" in files
        assert "test report" in outputs
        assert "summary table" in outputs

    def test_extract_empty_requirements(self, coordinator):
        """Test extraction with no requirements."""
        coordinator.extract_requirements("Just a simple prompt")

        assert coordinator.get_required_files() == []
        assert coordinator.get_required_outputs() == []

    def test_mark_observed_single(self, coordinator):
        """Test marking single file as observed."""
        coordinator.mark_observed("main.py")
        observed = coordinator.get_observed_files()

        assert "main.py" in observed
        assert len(observed) == 1

    def test_mark_observed_multiple(self, coordinator):
        """Test marking multiple files as observed."""
        coordinator.mark_many_observed(["main.py", "utils.py", "config.py"])
        observed = coordinator.get_observed_files()

        assert len(observed) == 3
        assert "main.py" in observed
        assert "utils.py" in observed
        assert "config.py" in observed

    def test_set_observed_files(self, coordinator):
        """Test setting observed files."""
        coordinator.set_observed_files({"main.py", "utils.py"})
        observed = coordinator.get_observed_files()

        assert len(observed) == 2
        assert "main.py" in observed

    def test_required_files_get_set(self, coordinator):
        """Test getting and setting required files."""
        files = ["main.py", "utils.py", "config.py"]
        coordinator.set_required_files(files)

        assert coordinator.get_required_files() == files

    def test_required_outputs_get_set(self, coordinator):
        """Test getting and setting required outputs."""
        outputs = ["test report", "summary table", "documentation"]
        coordinator.set_required_outputs(outputs)

        assert coordinator.get_required_outputs() == outputs

    def test_read_session_operations(self, coordinator):
        """Test read session operations."""
        # Add files to session
        coordinator.add_to_read_session("main.py")
        coordinator.add_to_read_session("utils.py")

        session = coordinator.get_read_session()
        assert len(session) == 2
        assert "main.py" in session

        # Set session
        coordinator.set_read_session({"config.py"})
        session = coordinator.get_read_session()
        assert len(session) == 1
        assert "config.py" in session

        # Clear session
        coordinator.clear_read_session()
        session = coordinator.get_read_session()
        assert len(session) == 0

    def test_nudge_sent_get_set(self, coordinator):
        """Test getting and setting nudge sent status."""
        assert coordinator.get_nudge_sent() is False

        coordinator.set_nudge_sent(True)
        assert coordinator.get_nudge_sent() is True

        coordinator.set_nudge_sent(False)
        assert coordinator.get_nudge_sent() is False

    def test_are_all_required_files_observed(self, coordinator):
        """Test checking if all required files are observed."""
        coordinator.set_required_files(["main.py", "utils.py", "config.py"])

        # None observed
        assert coordinator.are_all_required_files_observed() is False

        # Some observed
        coordinator.mark_observed("main.py")
        assert coordinator.are_all_required_files_observed() is False

        # All observed
        coordinator.mark_many_observed(["utils.py", "config.py"])
        assert coordinator.are_all_required_files_observed() is True

    def test_are_all_required_files_observed_empty(self, coordinator):
        """Test file observation check with no required files."""
        assert coordinator.are_all_required_files_observed() is False

    def test_get_missing_files(self, coordinator):
        """Test getting missing files."""
        coordinator.set_required_files(["main.py", "utils.py", "config.py"])

        # None observed
        missing = coordinator.get_missing_files()
        assert len(missing) == 3

        # Some observed
        coordinator.mark_observed("main.py")
        missing = coordinator.get_missing_files()
        assert len(missing) == 2
        assert "main.py" not in missing

        # All observed
        coordinator.mark_many_observed(["utils.py", "config.py"])
        missing = coordinator.get_missing_files()
        assert len(missing) == 0

    def test_get_file_observation_progress(self, coordinator):
        """Test getting file observation progress."""
        coordinator.set_required_files(["main.py", "utils.py", "config.py", "test.py"])

        # Initial progress
        progress = coordinator.get_file_observation_progress()
        assert progress["total_required"] == 4
        assert progress["total_observed"] == 0
        assert progress["progress_percent"] == 0.0
        assert len(progress["missing_files"]) == 4

        # After marking one
        coordinator.mark_observed("main.py")
        progress = coordinator.get_file_observation_progress()
        assert progress["total_observed"] == 1
        assert progress["progress_percent"] == 25.0
        assert len(progress["missing_files"]) == 3

        # After marking all
        coordinator.mark_many_observed(["utils.py", "config.py", "test.py"])
        progress = coordinator.get_file_observation_progress()
        assert progress["total_observed"] == 4
        assert progress["progress_percent"] == 100.0
        assert len(progress["missing_files"]) == 0

    def test_get_file_observation_progress_empty(self, coordinator):
        """Test progress with no required files."""
        progress = coordinator.get_file_observation_progress()
        assert progress["total_required"] == 0
        assert progress["total_observed"] == 0
        assert progress["progress_percent"] == 0.0
        assert len(progress["missing_files"]) == 0

    def test_get_state(self, coordinator):
        """Test getting coordinator state."""
        coordinator.extract_requirements('Read main.py, create "report"')
        coordinator.mark_observed("main.py")
        coordinator.add_to_read_session("main.py")
        coordinator.set_nudge_sent(True)

        state = coordinator.get_state()

        assert state["required_files"] == ["main.py"]
        assert state["required_outputs"] == ["report"]
        assert "main.py" in state["observed_files"]
        assert "main.py" in state["read_files_session"]
        assert state["all_files_read_nudge_sent"] is True

    def test_set_state(self, coordinator):
        """Test setting coordinator state."""
        state = {
            "required_files": ["main.py", "utils.py"],
            "required_outputs": ["report", "summary"],
            "observed_files": ["main.py"],
            "read_files_session": ["main.py", "config.py"],
            "all_files_read_nudge_sent": True,
        }

        coordinator.set_state(state)

        assert coordinator.get_required_files() == ["main.py", "utils.py"]
        assert coordinator.get_required_outputs() == ["report", "summary"]
        assert coordinator.get_observed_files() == {"main.py"}
        assert coordinator.get_read_session() == {"main.py", "config.py"}
        assert coordinator.get_nudge_sent() is True

    def test_set_state_partial(self, coordinator):
        """Test setting partial state."""
        state = {
            "required_files": ["main.py"],
            # missing other keys
        }

        coordinator.set_state(state)

        assert coordinator.get_required_files() == ["main.py"]
        assert coordinator.get_required_outputs() == []
        assert coordinator.get_observed_files() == set()

    def test_reset(self, coordinator):
        """Test resetting coordinator."""
        coordinator.extract_requirements('Read main.py, create "report"')
        coordinator.mark_observed("main.py")
        coordinator.add_to_read_session("main.py")
        coordinator.set_nudge_sent(True)

        coordinator.reset()

        assert coordinator.get_required_files() == []
        assert coordinator.get_required_outputs() == []
        assert coordinator.get_observed_files() == set()
        assert coordinator.get_read_session() == set()
        assert coordinator.get_nudge_sent() is False


class TestCreateFileContextCoordinator:
    """Test suite for create_file_context_coordinator factory."""

    def test_factory_function(self, task_analyzer):
        """Test factory function creates coordinator."""
        coordinator = create_file_context_coordinator(task_analyzer=task_analyzer)

        assert isinstance(coordinator, FileContextCoordinator)
        assert coordinator._task_analyzer is task_analyzer
