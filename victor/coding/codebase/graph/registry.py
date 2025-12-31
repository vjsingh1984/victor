# Pluggable graph store factory.
# Re-exports from victor.storage.graph for backward compatibility.
from victor.storage.graph.registry import create_graph_store
from victor.storage.graph.sqlite_store import SqliteGraphStore
from victor.storage.graph.memory_store import MemoryGraphStore
from victor.storage.graph.protocol import GraphStoreProtocol

__all__ = [
    "create_graph_store",
    "SqliteGraphStore",
    "MemoryGraphStore",
    "GraphStoreProtocol",
]
