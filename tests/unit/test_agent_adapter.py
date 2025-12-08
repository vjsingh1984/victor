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

"""Tests for VictorAgentAdapter."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from victor.evaluation.agent_adapter import (
    AdapterConfig,
    VictorAgentAdapter,
    create_victor_agent_callback,
)
from victor.evaluation.agentic_harness import (
    AgenticExecutionTrace,
    ToolCall,
    FileEdit,
)
from victor.evaluation.protocol import BenchmarkTask, BenchmarkType


class TestAdapterConfig:
    """Tests for AdapterConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AdapterConfig()
        assert config.max_turns == 20
        assert config.tool_budget == 50
        assert config.timeout_per_turn == 120
        assert config.track_file_edits is True
        assert config.track_diffs is True
        assert config.working_dir is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AdapterConfig(
            max_turns=10,
            tool_budget=25,
            timeout_per_turn=60,
            track_file_edits=False,
            working_dir=Path("/tmp"),
        )
        assert config.max_turns == 10
        assert config.tool_budget == 25
        assert config.timeout_per_turn == 60
        assert config.track_file_edits is False
        assert config.working_dir == Path("/tmp")


class TestVictorAgentAdapter:
    """Tests for VictorAgentAdapter class."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        orchestrator = MagicMock()
        orchestrator._on_tool_start_callback = None
        orchestrator._on_tool_complete_callback = None
        orchestrator.reset_conversation = MagicMock()
        orchestrator.chat = AsyncMock(return_value=MagicMock(content="TASK COMPLETE"))
        return orchestrator

    @pytest.fixture
    def adapter(self, mock_orchestrator):
        """Create an adapter with mock orchestrator."""
        return VictorAgentAdapter(mock_orchestrator)

    def test_init_hooks_callbacks(self, mock_orchestrator):
        """Test that adapter hooks into orchestrator callbacks."""
        adapter = VictorAgentAdapter(mock_orchestrator)

        # Verify callbacks are hooked
        assert mock_orchestrator._on_tool_start_callback is not None
        assert mock_orchestrator._on_tool_complete_callback is not None
        assert mock_orchestrator._on_tool_start_callback == adapter._on_tool_start
        assert mock_orchestrator._on_tool_complete_callback == adapter._on_tool_complete

    def test_init_with_config(self, mock_orchestrator):
        """Test initialization with custom config."""
        config = AdapterConfig(max_turns=5, tool_budget=10)
        adapter = VictorAgentAdapter(mock_orchestrator, config)

        assert adapter.config.max_turns == 5
        assert adapter.config.tool_budget == 10

    def test_reset(self, adapter):
        """Test reset clears state."""
        # Simulate some state
        adapter._tool_calls = [ToolCall(name="test", arguments={}, timestamp=0)]
        adapter._file_edits = [FileEdit(path="test.py", action="create")]
        adapter._messages = [{"role": "user", "content": "test"}]
        adapter._turns = 5
        adapter._file_snapshots = {"test.py": "content"}

        adapter.reset()

        assert len(adapter._tool_calls) == 0
        assert len(adapter._file_edits) == 0
        assert len(adapter._messages) == 0
        assert adapter._turns == 0
        assert len(adapter._file_snapshots) == 0
        adapter.orchestrator.reset_conversation.assert_called_once()

    def test_on_tool_start_records_call(self, adapter):
        """Test that tool start callback records tool call."""
        adapter._on_tool_start("file_read", {"path": "test.py"})

        assert len(adapter._tool_calls) == 1
        assert adapter._tool_calls[0].name == "file_read"
        assert adapter._tool_calls[0].arguments == {"path": "test.py"}

    def test_on_tool_start_with_file_edit_snapshots(self, adapter):
        """Test that file edits trigger snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter.config.working_dir = Path(tmpdir)

            # Create a test file
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("original content")

            adapter._on_tool_start("file_edit", {"path": "test.py"})

            assert "test.py" in adapter._file_snapshots
            assert adapter._file_snapshots["test.py"] == "original content"

    def test_on_tool_complete_updates_result(self, adapter):
        """Test that tool complete updates the last tool call."""
        # First record a tool call
        adapter._on_tool_start("test_tool", {})

        # Then complete it
        result = MagicMock()
        result.success = True
        result.result = "test output"
        adapter._on_tool_complete(result)

        assert adapter._tool_calls[0].success is True
        assert adapter._tool_calls[0].result == "test output"

    def test_is_task_complete_detects_completion(self, adapter):
        """Test task completion detection."""
        assert adapter._is_task_complete("The TASK COMPLETE.") is True
        assert adapter._is_task_complete("I have completed the task.") is True
        assert adapter._is_task_complete("Implementation complete.") is True
        assert adapter._is_task_complete("Still working on it...") is False

    def test_build_task_prompt(self, adapter):
        """Test task prompt building."""
        task = BenchmarkTask(
            task_id="test/1",
            benchmark=BenchmarkType.CUSTOM,
            description="Test task",
            prompt="Fix the bug in main.py",
            context_code="def main():\n    pass",
        )
        workspace = Path("/tmp/workspace")

        prompt = adapter._build_task_prompt(task, workspace)

        assert "test/1" in prompt
        assert "Fix the bug in main.py" in prompt
        assert "/tmp/workspace" in prompt
        assert "def main():" in prompt
        assert "TASK COMPLETE" in prompt

    def test_generate_combined_patch(self, adapter):
        """Test combined patch generation."""
        adapter._file_edits = [
            FileEdit(
                path="a.py", action="modify", diff="--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new"
            ),
            FileEdit(
                path="b.py",
                action="create",
                diff="--- /dev/null\n+++ b/b.py\n@@ -0,0 +1 @@\n+new file",
            ),
        ]

        patch = adapter._generate_combined_patch()

        assert "a.py" in patch
        assert "b.py" in patch
        assert "-old" in patch
        assert "+new" in patch

    @pytest.mark.asyncio
    async def test_execute_task_basic(self, adapter, mock_orchestrator):
        """Test basic task execution."""
        task = BenchmarkTask(
            task_id="test/basic",
            benchmark=BenchmarkType.CUSTOM,
            description="Basic test",
            prompt="Create a hello world function",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            trace = await adapter.execute_task(task, workspace)

            assert trace.task_id == "test/basic"
            assert trace.turns >= 1
            assert trace.start_time > 0
            assert trace.end_time >= trace.start_time
            mock_orchestrator.chat.assert_called()

    @pytest.mark.asyncio
    async def test_execute_task_max_turns_limit(self, adapter, mock_orchestrator):
        """Test that max turns limit is enforced."""
        adapter.config.max_turns = 2
        # Make orchestrator never signal completion
        mock_orchestrator.chat.return_value = MagicMock(content="Still working...")

        task = BenchmarkTask(
            task_id="test/limit",
            benchmark=BenchmarkType.CUSTOM,
            description="Max turns test",
            prompt="Do something",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trace = await adapter.execute_task(task, Path(tmpdir))

            assert trace.turns == 2  # Should stop at max_turns


class TestCreateVictorAgentCallback:
    """Tests for create_victor_agent_callback function."""

    def test_creates_callable(self):
        """Test that factory creates a callable."""
        mock_orchestrator = MagicMock()
        mock_orchestrator._on_tool_start_callback = None
        mock_orchestrator._on_tool_complete_callback = None

        adapter = VictorAgentAdapter(mock_orchestrator)
        callback = create_victor_agent_callback(adapter)

        assert callable(callback)

    @pytest.mark.asyncio
    async def test_callback_delegates_to_adapter(self):
        """Test that callback delegates to adapter.execute_task."""
        mock_orchestrator = MagicMock()
        mock_orchestrator._on_tool_start_callback = None
        mock_orchestrator._on_tool_complete_callback = None
        mock_orchestrator.reset_conversation = MagicMock()
        mock_orchestrator.chat = AsyncMock(return_value=MagicMock(content="TASK COMPLETE"))

        adapter = VictorAgentAdapter(mock_orchestrator)
        callback = create_victor_agent_callback(adapter)

        task = BenchmarkTask(
            task_id="test/callback",
            benchmark=BenchmarkType.CUSTOM,
            description="Callback test",
            prompt="Test task",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = await callback(task, Path(tmpdir))

            assert isinstance(result, AgenticExecutionTrace)
            assert result.task_id == "test/callback"


class TestFileEditCapture:
    """Tests for file edit capture functionality."""

    @pytest.fixture
    def adapter_with_workspace(self):
        """Create adapter with temp workspace."""
        mock_orchestrator = MagicMock()
        mock_orchestrator._on_tool_start_callback = None
        mock_orchestrator._on_tool_complete_callback = None

        config = AdapterConfig(track_file_edits=True, track_diffs=True)
        adapter = VictorAgentAdapter(mock_orchestrator, config)
        return adapter

    def test_capture_file_create(self, adapter_with_workspace):
        """Test capturing file creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = adapter_with_workspace
            adapter.config.working_dir = Path(tmpdir)

            # Create a new file
            new_file = Path(tmpdir) / "new.py"
            new_file.write_text("print('hello')")

            adapter._capture_file_edit("new.py", "file_write")

            assert len(adapter._file_edits) == 1
            assert adapter._file_edits[0].path == "new.py"
            assert adapter._file_edits[0].action == "create"
            assert "print('hello')" in adapter._file_edits[0].after_content

    def test_capture_file_modify(self, adapter_with_workspace):
        """Test capturing file modification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = adapter_with_workspace
            adapter.config.working_dir = Path(tmpdir)

            # Create original file
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("original")

            # Snapshot before edit
            adapter._file_snapshots["test.py"] = "original"

            # Modify file
            test_file.write_text("modified")

            adapter._capture_file_edit("test.py", "file_edit")

            assert len(adapter._file_edits) == 1
            assert adapter._file_edits[0].action == "modify"
            assert "original" in adapter._file_edits[0].before_content
            assert "modified" in adapter._file_edits[0].after_content
            assert adapter._file_edits[0].diff != ""  # Should have diff

    def test_diff_generation(self, adapter_with_workspace):
        """Test that unified diff is generated correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = adapter_with_workspace
            adapter.config.working_dir = Path(tmpdir)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("line1\nline2\nline3")
            adapter._file_snapshots["test.py"] = "line1\nold_line\nline3"

            # Write modified content
            test_file.write_text("line1\nline2\nline3")

            adapter._capture_file_edit("test.py", "file_edit")

            diff = adapter._file_edits[0].diff
            assert "---" in diff
            assert "+++" in diff
            assert "-old_line" in diff
            assert "+line2" in diff
