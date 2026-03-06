"""Loader for graph algorithm implementations.

Tries to use Rust-accelerated versions, falls back to pure Python.
"""

try:
    from victor_native import (
        pagerank,
        weighted_pagerank,
        betweenness_centrality,
        connected_components,
        detect_cycles,
    )

    GRAPH_ALGO_BACKEND = "rust"
except ImportError:
    from victor.native.python.graph_algo import (
        pagerank,
        weighted_pagerank,
        betweenness_centrality,
        connected_components,
        detect_cycles,
    )

    GRAPH_ALGO_BACKEND = "python"

__all__ = [
    "pagerank",
    "weighted_pagerank",
    "betweenness_centrality",
    "connected_components",
    "detect_cycles",
    "GRAPH_ALGO_BACKEND",
]
