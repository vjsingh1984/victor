# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Generic graph storage module for Victor framework.

This module provides a plugin-based architecture for graph storage with
multiple backend options:

- **SqliteGraphStore**: Production-ready, embedded SQLite
- **MemoryGraphStore**: Fast in-memory store for testing
- **DuckDBGraphStore**: Analytics-optimized (optional)
- **LanceDBGraphStore**: Vector-enhanced graphs (optional)
- **Neo4jGraphStore**: Enterprise graph database (optional)

Example:
    ```python
    from victor.storage.graph import create_graph_store, GraphNode, GraphEdge

    # Create a SQLite-backed graph store
    store = create_graph_store("sqlite", "/path/to/graph.db")

    # Add nodes and edges
    nodes = [
        GraphNode(node_id="entity:1", type="person", name="Alice", file="data.json"),
        GraphNode(node_id="entity:2", type="org", name="Acme", file="data.json"),
    ]
    edges = [
        GraphEdge(src="entity:1", dst="entity:2", type="WORKS_AT")
    ]

    await store.upsert_nodes(nodes)
    await store.upsert_edges(edges)
    ```

This is a core framework module available to all verticals, not just coding.
"""

from victor.storage.graph.protocol import GraphNode, GraphEdge, GraphStoreProtocol
from victor.storage.graph.registry import create_graph_store
from victor.storage.graph.sqlite_store import SqliteGraphStore
from victor.storage.graph.memory_store import MemoryGraphStore

__all__ = [
    # Protocol
    "GraphNode",
    "GraphEdge",
    "GraphStoreProtocol",
    # Factory
    "create_graph_store",
    # Stores
    "SqliteGraphStore",
    "MemoryGraphStore",
]
