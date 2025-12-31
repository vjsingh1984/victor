# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Generic graph storage module for Victor framework.

This module has moved to victor.storage.graph.
Import from victor.storage.graph instead for new code.

This module provides backward-compatible re-exports.
"""

# Re-export from new location for backward compatibility
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
