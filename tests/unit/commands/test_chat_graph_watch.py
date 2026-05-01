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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import victor.ui.commands.chat as chat_cmd
import victor.ui.commands.graph as graph_cmd


def test_chat_graph_watch_ensures_project_singleton_by_default(tmp_path):
    """Interactive chat should ensure a project-wide graph watch daemon exists."""
    project_root = tmp_path / "repo"
    project_root.mkdir()

    paths = SimpleNamespace(
        project_root=project_root,
        ensure_project_dirs=MagicMock(),
    )
    state = graph_cmd.GraphWatchDaemonState(
        pid_file=project_root / ".victor" / "graph-watch.pid",
        running=True,
        pid=321,
        started=True,
    )

    with (
        patch.object(chat_cmd, "get_project_paths", return_value=paths),
        patch("victor.ui.commands.graph.ensure_graph_watch_daemon", return_value=state) as mock_ensure,
    ):
        messages = chat_cmd._ensure_graph_watch_for_chat(enabled=True)

    paths.ensure_project_dirs.assert_called_once_with()
    mock_ensure.assert_called_once_with(
        project_root,
        enable_ccg=True,
        build_now=True,
    )
    assert any("graph watch" in message.lower() for message in messages)


def test_chat_graph_watch_can_be_disabled(tmp_path):
    """Interactive chat should skip graph watch setup when explicitly disabled."""
    project_root = tmp_path / "repo"
    project_root.mkdir()

    paths = SimpleNamespace(
        project_root=project_root,
        ensure_project_dirs=MagicMock(),
    )

    with (
        patch.object(chat_cmd, "get_project_paths", return_value=paths),
        patch("victor.ui.commands.graph.ensure_graph_watch_daemon") as mock_ensure,
    ):
        messages = chat_cmd._ensure_graph_watch_for_chat(enabled=False)

    paths.ensure_project_dirs.assert_not_called()
    mock_ensure.assert_not_called()
    assert messages == []
