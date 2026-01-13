# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from victor.storage.graph.protocol import GraphEdge, GraphNode
from victor.storage.graph.sqlite_store import SqliteGraphStore


@pytest.mark.asyncio
async def test_sqlite_graph_store_upsert_and_query(tmp_path):
    db_path = tmp_path / "graph.db"
    store = SqliteGraphStore(db_path)

    nodes = [
        GraphNode(node_id="file:main.py", type="file", name="main.py", file="main.py"),
        GraphNode(
            node_id="symbol:main.py:foo",
            type="function",
            name="foo",
            file="main.py",
            line=1,
            metadata={"signature": "foo()"},
        ),
    ]
    edges = [GraphEdge(src="file:main.py", dst="symbol:main.py:foo", type="CONTAINS")]

    await store.upsert_nodes(nodes)
    await store.upsert_edges(edges)

    stats = await store.stats()
    assert stats["nodes"] == 2
    assert stats["edges"] == 1

    found = await store.find_nodes(name="foo")
    assert len(found) == 1
    assert found[0].node_id == "symbol:main.py:foo"

    neighbors = await store.get_neighbors("file:main.py")
    assert len(neighbors) == 1
    assert neighbors[0].dst == "symbol:main.py:foo"


@pytest.mark.asyncio
async def test_sqlite_graph_store_weight_type_conversion(tmp_path):
    """Test that weights stored as strings are properly converted to floats.

    This test defends against the type comparison error: "'<' not supported between
    instances of 'int' and 'str'" that can occur when SQLite returns weights as
    strings due to flexible typing.

    See: https://github.com/anthropics/victor/issues/XXX
    """
    import sqlite3

    db_path = tmp_path / "graph.db"
    store = SqliteGraphStore(db_path)

    # Add some nodes
    nodes = [
        GraphNode(node_id="node_a", type="function", name="a", file="a.py"),
        GraphNode(node_id="node_b", type="function", name="b", file="b.py"),
    ]
    await store.upsert_nodes(nodes)

    # Manually insert an edge with a string weight (simulating old/corrupted data)
    # SQLite allows this due to flexible typing
    async with store._lock:
        conn = store._connect()
        # Insert weight as a string to simulate data corruption or legacy data
        conn.execute(
            "INSERT INTO graph_edge (src, dst, type, weight, metadata) VALUES (?, ?, ?, ?, ?)",
            ("node_a", "node_b", "CALLS", "2.5", "{}"),  # Weight as string
        )
        conn.commit()

    # Verify that get_all_edges properly converts weight to float
    edges = await store.get_all_edges()
    assert len(edges) == 1
    assert edges[0].weight == 2.5
    assert isinstance(edges[0].weight, float)
    assert not isinstance(edges[0].weight, str)

    # Verify that get_neighbors also converts weight properly
    neighbors = await store.get_neighbors("node_a")
    assert len(neighbors) == 1
    assert neighbors[0].weight == 2.5
    assert isinstance(neighbors[0].weight, float)

    # Verify that the weight can be used in numeric operations without error
    # This would fail with the original bug if weight was a string
    result = neighbors[0].weight * 2
    assert result == 5.0
