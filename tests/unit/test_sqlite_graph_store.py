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
