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

import asyncio
from pathlib import Path

import pytest

from victor.codebase.graph.sqlite_store import SqliteGraphStore
from victor.codebase.indexer import CodebaseIndex


@pytest.mark.asyncio
async def test_indexer_writes_graph(tmp_path):
    # Create minimal repo
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text(
        "import math\n\ndef foo():\n    return math.sqrt(4)\n\nclass Bar:\n    def method(self):\n        return foo()\n\nclass Baz(Bar):\n    pass\n",
        encoding="utf-8",
    )
    (repo / "utils.py").write_text(
        "def helper():\n    return 42\n\n\ndef caller():\n    return helper()\n",
        encoding="utf-8",
    )

    graph_path = repo / ".victor" / "graph" / "graph.db"
    store = SqliteGraphStore(graph_path)

    indexer = CodebaseIndex(
        root_path=str(repo),
        use_embeddings=False,
        enable_watcher=False,
        graph_store=store,
    )

    await indexer.index_codebase(force=True)

    stats = await store.stats()
    assert stats["nodes"] >= 2
    assert stats["edges"] >= 1

    # Symbol node should be present
    symbols = await store.find_nodes(type="function")
    assert symbols, "expected function symbols in graph store"

    neighbors = await store.get_neighbors("file:main.py")
    assert neighbors, "expected CONTAINS edges from file node"

    # CALLS edge should link method -> foo
    call_edges = await store.get_neighbors("symbol:main.py:Bar.method")
    assert any(edge.type == "CALLS" and edge.dst.endswith("foo") for edge in call_edges)

    # IMPORTS edge should point to module node
    import_edges = await store.get_neighbors("file:main.py")
    assert any(edge.type == "IMPORTS" for edge in import_edges)

    # Cross-file CALLS resolution (caller -> helper)
    cross_calls = await store.get_neighbors("symbol:utils.py:caller")
    assert any(edge.type == "CALLS" and "helper" in edge.dst for edge in cross_calls)

    # REFERENCES edge from file to helper symbol
    file_refs = await store.get_neighbors("file:utils.py")
    assert any(edge.type == "REFERENCES" and "helper" in edge.dst for edge in file_refs)

    # INHERITS edge from Baz -> Bar
    inherits_edges = await store.get_neighbors("symbol:main.py:Baz")
    assert any(edge.type == "INHERITS" and "Bar" in edge.dst for edge in inherits_edges)
