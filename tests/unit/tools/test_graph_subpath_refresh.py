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

"""Graph background-refresh must target the project root, not a scoped subpath.

Regression for the stray ``.victor`` dirs created when ``graph(mode=overview,
path='src/network')`` ran: the watcher/incremental-refresh was subscribed for the
subpath, indexing a stray subdirectory DB. It must run against the canonical
project root so the repo's single ``.victor/project.db`` is refreshed instead.
"""

from pathlib import Path

import pytest

from victor.config.settings import reset_project_paths, set_project_root
from victor.core.database import get_project_database, reset_project_database
from victor.tools import graph_tool


def _seed_project_graph(root: Path) -> None:
    db = get_project_database(root)
    conn = db.get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS graph_node (
            node_id TEXT PRIMARY KEY, type TEXT NOT NULL, name TEXT NOT NULL,
            file TEXT NOT NULL, line INTEGER
        );
        CREATE TABLE IF NOT EXISTS graph_edge (
            src TEXT NOT NULL, dst TEXT NOT NULL, type TEXT NOT NULL,
            PRIMARY KEY (src, dst, type)
        );
        """)
    conn.execute(
        "INSERT INTO graph_node (node_id, type, name, file, line) VALUES (?,?,?,?,?)",
        ("NetA", "class", "NetA", "src/network/a.rs", 1),
    )
    conn.commit()


@pytest.mark.asyncio
async def test_overview_subpath_refreshes_project_root_and_creates_no_stray_victor(
    tmp_path, monkeypatch
):
    root = (tmp_path / "repo").resolve()
    (root / ".git").mkdir(parents=True)
    (root / ".victor").mkdir()
    sub = root / "src" / "network"
    sub.mkdir(parents=True)
    try:
        _seed_project_graph(root)
        reset_project_paths()
        set_project_root(root)

        recorded = {}

        class _StubManager:
            async def ensure_background_refresh(self, refresh_root, **_kwargs):
                recorded["root"] = Path(refresh_root)
                return None

        from victor.core.indexing import graph_manager as graph_manager_module

        monkeypatch.setattr(
            graph_manager_module.GraphManager,
            "get_instance",
            classmethod(lambda cls: _StubManager()),
        )
        # Force the subscribe branch to run (no live daemon).
        monkeypatch.setattr(graph_tool, "_project_graph_watch_daemon_active", lambda _r: False)

        result = await graph_tool._graph_impl(mode="overview", path=str(sub), reindex=False)

        assert result.get("success") is True
        # The refresh must target the project root, never the scoped subpath.
        assert recorded.get("root") == root
        # And no stray .victor/ is left under the subdirectory.
        assert not (sub / ".victor").exists()
    finally:
        reset_project_database(root)
        reset_project_paths()
