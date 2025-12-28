# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Graph storage module for codebase analysis.

This module re-exports from the core victor.graph module for backward
compatibility. The graph storage infrastructure is now part of victor-core
and available to all verticals.

For new code, prefer importing directly from victor.graph:
    from victor.graph import GraphNode, GraphEdge, create_graph_store
"""

# Re-export from victor-core
from victor.graph import (
    GraphNode,
    GraphEdge,
    GraphStoreProtocol,
    create_graph_store,
    SqliteGraphStore,
    MemoryGraphStore,
)

__all__ = [
    "GraphNode",
    "GraphEdge",
    "GraphStoreProtocol",
    "create_graph_store",
    "SqliteGraphStore",
    "MemoryGraphStore",
]
