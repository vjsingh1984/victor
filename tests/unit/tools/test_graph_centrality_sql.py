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

"""Regression: degree-centrality SQL binding for multi-type edge groups.

The ``patterns``/``centrality`` fast path interpolates the edge-type filter
twice (outgoing + incoming degree subqueries). Previously the bound parameters
were supplied only once and in the wrong order, so a multi-type ``edge_group``
(e.g. ``type_hierarchy`` → INHERITS/IMPLEMENTS/IS_A) raised
``sqlite3.ProgrammingError: Incorrect number of bindings supplied``.
"""

from pathlib import Path

from victor.core.database import get_project_database, reset_project_database
from victor.tools.graph_tool import (
    _build_cheap_overview_from_project_store,
    _build_degree_centrality_from_project_store,
)


def _seed_graph(root: Path) -> None:
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
    nodes = [
        ("Base", "class", "Base", "src/lib.rs", 1),
        ("Mid", "class", "Mid", "src/mid.rs", 2),
        ("Leaf", "class", "Leaf", "src/leaf.rs", 3),
        ("helper", "function", "helper", "src/util.rs", 4),
    ]
    conn.executemany(
        "INSERT INTO graph_node (node_id, type, name, file, line) VALUES (?, ?, ?, ?, ?)",
        nodes,
    )
    edges = [
        ("Mid", "Base", "INHERITS"),
        ("Leaf", "Mid", "INHERITS"),
        ("Leaf", "Base", "IMPLEMENTS"),
        ("Mid", "Base", "IS_A"),
        ("helper", "Leaf", "CALLS"),
    ]
    conn.executemany("INSERT INTO graph_edge (src, dst, type) VALUES (?, ?, ?)", edges)
    conn.commit()


def test_degree_centrality_multi_edge_and_node_types(tmp_path: Path) -> None:
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    (root / ".victor").mkdir()
    try:
        _seed_graph(root)

        # type_hierarchy = 3 edge types; class/function = 2 node types. The old
        # code supplied 3 + 2 + 1 = 6 bindings for a query needing 2*3 + 2 + 1 = 9.
        result = _build_degree_centrality_from_project_store(
            root,
            top_k=10,
            edge_types=["INHERITS", "IMPLEMENTS", "IS_A"],
            node_types={"class", "function"},
        )

        nodes = result["nodes"]
        assert nodes, "expected ranked nodes, not an empty/raising result"
        by_name = {n["name"]: n for n in nodes}
        # Base is the inheritance hub: 3 incoming hierarchy edges, 0 outgoing.
        assert by_name["Base"]["in_degree"] == 3
        assert by_name["Base"]["out_degree"] == 0
        # CALLS is excluded by the edge-type filter, so helper has degree 0.
        assert by_name["helper"]["degree"] == 0
    finally:
        reset_project_database(root)


def _seed_two_subtrees(root: Path) -> None:
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
    nodes = [
        ("NetA", "class", "NetA", "src/network/a.rs", 1),
        ("NetB", "class", "NetB", "src/network/b.rs", 2),
        ("StoreA", "class", "StoreA", "src/storage/a.rs", 3),
    ]
    conn.executemany(
        "INSERT INTO graph_node (node_id, type, name, file, line) VALUES (?, ?, ?, ?, ?)",
        nodes,
    )
    conn.executemany(
        "INSERT INTO graph_edge (src, dst, type) VALUES (?, ?, ?)",
        [("NetA", "NetB", "CALLS"), ("StoreA", "NetA", "CALLS")],
    )
    conn.commit()


def test_overview_scopes_to_requested_subdirectory(tmp_path: Path) -> None:
    root = (tmp_path / "repo3").resolve()
    root.mkdir()
    (root / ".victor").mkdir()
    (root / "src" / "network").mkdir(parents=True)
    try:
        _seed_two_subtrees(root)

        scoped = _build_cheap_overview_from_project_store(root / "src" / "network", top_k=25)
        files = {m["file"] for m in scoped["important_modules"]}
        assert files == {"src/network/a.rs", "src/network/b.rs"}
        assert all("storage" not in f for f in files)

        # Whole-repo request still sees everything.
        full = _build_cheap_overview_from_project_store(root, top_k=25)
        full_files = {m["file"] for m in full["important_modules"]}
        assert "src/storage/a.rs" in full_files
    finally:
        reset_project_database(root)


def test_degree_centrality_no_filters(tmp_path: Path) -> None:
    """Unfiltered path still binds correctly (only the LIMIT placeholder)."""
    root = (tmp_path / "repo2").resolve()
    root.mkdir()
    (root / ".victor").mkdir()
    try:
        _seed_graph(root)
        result = _build_degree_centrality_from_project_store(
            root, top_k=10, edge_types=None, node_types=None
        )
        assert result["nodes"]
    finally:
        reset_project_database(root)
