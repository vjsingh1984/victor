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

import os
import time
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

import victor.ui.commands.graph as graph_cmd


def test_graph_watch_lock_file_uses_project_victor_dir(tmp_path):
    """Graph watch startup lock files should live under the project .victor dir."""
    project_root = tmp_path / "repo"
    project_root.mkdir()

    with patch(
        "victor.ui.commands.graph.get_project_paths",
        return_value=type("Paths", (), {"project_victor_dir": project_root / ".victor"})(),
    ):
        lock_file = graph_cmd._default_graph_watch_lock_file(project_root)

    assert lock_file.parent == project_root / ".victor"
    assert lock_file.name == "graph-watch.lock"


def test_graph_watch_startup_lock_recovers_stale_lock_file(tmp_path):
    """Acquiring the graph watch startup lock should recover stale lock files."""
    project_root = tmp_path / "repo"
    project_root.mkdir()
    lock_file = tmp_path / "graph-watch.lock"
    lock_file.write_text("stale", encoding="utf-8")
    stale_time = time.time() - 60
    os.utime(lock_file, (stale_time, stale_time))

    with graph_cmd._acquire_graph_watch_startup_lock(
        project_root,
        lock_file=lock_file,
        timeout_seconds=0.1,
        poll_interval=0.0,
        stale_after_seconds=0.01,
    ):
        assert lock_file.exists()

    assert not lock_file.exists()


def test_graph_watch_start_daemon_is_idempotent_when_already_running(tmp_path):
    """Starting an already-running graph watcher should be a no-op."""
    project_root = tmp_path / "repo"
    project_root.mkdir()
    pid_file = tmp_path / "graph-watch.pid"
    pid_file.write_text("123", encoding="utf-8")
    record_console = Console(record=True, force_terminal=False, width=160)

    with (
        patch.object(graph_cmd, "console", record_console),
        patch.object(graph_cmd.os, "kill") as mock_kill,
        patch.object(graph_cmd, "_fork_watch_daemon") as mock_fork,
    ):
        graph_cmd.graph_watch_start(
            path=str(project_root),
            enable_ccg=True,
            daemon=True,
            pid_file=pid_file,
            poll_interval=1.0,
            debounce_seconds=0.3,
            build_now=False,
        )

    mock_kill.assert_called_once_with(123, 0)
    mock_fork.assert_not_called()
    assert "already running" in record_console.export_text().lower()


def test_graph_watch_start_daemon_replaces_stale_pid_file(tmp_path):
    """Starting the watcher should recover from a stale PID file."""
    project_root = tmp_path / "repo"
    project_root.mkdir()
    pid_file = tmp_path / "graph-watch.pid"
    pid_file.write_text("123", encoding="utf-8")
    record_console = Console(record=True, force_terminal=False, width=160)

    with (
        patch.object(graph_cmd, "console", record_console),
        patch.object(graph_cmd.os, "kill", side_effect=ProcessLookupError),
        patch.object(graph_cmd, "_fork_watch_daemon", return_value=456) as mock_fork,
    ):
        graph_cmd.graph_watch_start(
            path=str(project_root),
            enable_ccg=True,
            daemon=True,
            pid_file=pid_file,
            poll_interval=1.0,
            debounce_seconds=0.3,
            build_now=False,
        )

    mock_fork.assert_called_once_with(
        pid_file,
        str(project_root.resolve()),
        True,
        1.0,
        0.3,
        False,
    )
    assert not pid_file.exists()
    output = record_console.export_text().lower()
    assert "recovered stale pid file" in output
    assert "started (pid 456)" in output


def test_graph_watch_stop_removes_stale_pid_file(tmp_path):
    """Stopping the watcher should clean up stale PID files."""
    project_root = tmp_path / "repo"
    project_root.mkdir()
    pid_file = tmp_path / "graph-watch.pid"
    pid_file.write_text("123", encoding="utf-8")
    record_console = Console(record=True, force_terminal=False, width=160)

    with (
        patch.object(graph_cmd, "console", record_console),
        patch.object(graph_cmd.os, "kill", side_effect=ProcessLookupError),
    ):
        graph_cmd.graph_watch_stop(path=str(project_root), pid_file=pid_file)

    assert not pid_file.exists()
    assert "stale pid file removed" in record_console.export_text().lower()


def test_graph_watch_status_reports_stale_pid_file(tmp_path):
    """Status should distinguish a stale PID file from a healthy daemon."""
    project_root = tmp_path / "repo"
    project_root.mkdir()
    pid_file = tmp_path / "graph-watch.pid"
    pid_file.write_text("123", encoding="utf-8")
    record_console = Console(record=True, force_terminal=False, width=160)

    with (
        patch.object(graph_cmd, "console", record_console),
        patch.object(graph_cmd.os, "kill", side_effect=ProcessLookupError),
    ):
        graph_cmd.graph_watch_status(path=str(project_root), pid_file=pid_file)

    output = record_console.export_text().lower()
    assert "stale pid file" in output
