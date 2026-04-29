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

"""Graph-based query tools for code intelligence.

This module provides tools for querying the code graph using
natural language and for performing impact analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from victor.tools.base import AccessMode, DangerLevel, Priority
from victor.tools.decorators import tool

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Input Models
# =============================================================================

class GraphSemanticSearchInput(BaseModel):
    """Input for graph_semantic_search tool."""

    query: str = Field(
        description="Natural language query about the codebase"
    )
    path: str = Field(
        default=".",
        description="Path to search within (default: current directory)"
    )
    mode: str = Field(
        default="semantic",
        description="Query mode: semantic, structural, or hybrid"
    )
    max_hops: int = Field(
        default=2,
        description="Maximum hops for graph traversal (1-3)"
    )
    max_results: int = Field(
        default=10,
        description="Maximum number of results to return"
    )


class ImpactAnalysisInput(BaseModel):
    """Input for impact_analysis tool."""

    target: str = Field(
        description="Target symbol or file:line (e.g., 'my_function' or 'src/main.py:42')"
    )
    analysis_type: str = Field(
        default="forward",
        description="Analysis type: forward (downstream) or backward (upstream)"
    )
    max_depth: int = Field(
        default=3,
        description="Maximum depth for impact analysis (1-5)"
    )
    include_test_impact: bool = Field(
        default=True,
        description="Whether to include test impact in analysis"
    )
    path: str = Field(
        default=".",
        description="Path to search within (default: current directory)"
    )


# =============================================================================
# Tool Implementations
# =============================================================================

@tool(
    name="graph_semantic_search",
    category="analysis",
    keywords=["graph", "semantic", "dependency", "code structure", "impact", "rag"],
    use_cases=[
        "Understanding code dependencies",
        "Finding related code",
        "Analyzing change impact",
        "Tracing function calls",
        "Discovering code relationships"
    ],
    priority=Priority.MEDIUM,
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    task_types=["analysis", "search"],
    execution_category="read_only",
    timeout=60.0,  # Graph queries can take longer
)
async def graph_semantic_search(
    query: str,
    path: str = ".",
    mode: str = "semantic",
    max_hops: int = 2,
    max_results: int = 10,
) -> Dict[str, Any]:
    """Query codebase graph using natural language.

    This tool performs intelligent code search by combining:
    - Semantic search (vector similarity)
    - Graph traversal (multi-hop)
    - Structural analysis (code relationships)

    Args:
        query: Natural language query about the codebase
        path: Path to search within (default: current directory)
        mode: Query mode - semantic (meaning-based), structural (pattern-based),
              or hybrid (both)
        max_hops: Maximum hops for graph traversal (1-3, default: 2)
        max_results: Maximum number of results to return

    Returns:
        Dictionary with:
        - results: List of matching symbols with context
        - query: Original query
        - execution_time_ms: Time taken
        - metadata: Additional metadata

    Example:
        >>> await graph_query("How to handle authentication?",
        ...                    mode="semantic", max_hops=2)
    """
    import time
    from victor.core.feature_flags import get_feature_flag_manager, FeatureFlag

    start_time = time.time()

    # Check feature flag
    flag_manager = get_feature_flag_manager()
    if not flag_manager.is_enabled(FeatureFlag.USE_GRAPH_QUERY_TOOL):
        logger.warning("Graph query tool is not enabled")
        return {
            "results": [],
            "query": query,
            "error": "Graph query tool is not enabled. "
                    "Set VICTOR_USE_GRAPH_QUERY_TOOL=true to enable.",
            "execution_time_ms": (time.time() - start_time) * 1000,
        }

    logger.info(f"Graph query: {query} (mode={mode}, hops={max_hops})")

    try:
        # Import graph RAG components
        from victor.storage.graph import create_graph_store
        from victor.core.graph_rag import MultiHopRetriever, RetrievalConfig

        # Create graph store
        project_path = Path(path).resolve()
        graph_store = create_graph_store(name="sqlite", project_path=project_path)
        await graph_store.initialize()

        # Create retriever with config
        config = RetrievalConfig(
            seed_count=max_results,
            max_hops=max_hops,
            top_k=max_results,
        )
        retriever = MultiHopRetriever(graph_store, config)

        # Execute retrieval
        result = await retriever.retrieve(query, config)

        # Format results
        formatted_results = []
        for node in result.nodes[:max_results]:
            formatted_results.append({
                "name": node.name,
                "type": node.type,
                "file": node.file,
                "line": node.line,
                "signature": node.signature,
                "docstring": node.docstring,
                "relevance_score": result.scores.get(node.node_id, 0.0),
                "hop_distance": result.hop_distances.get(node.node_id, 0),
            })

        return {
            "results": formatted_results,
            "query": query,
            "execution_time_ms": result.execution_time_ms,
            "metadata": {
                "mode": mode,
                "max_hops": max_hops,
                "seed_count": len(result.seed_nodes),
            },
        }

    except Exception as e:
        logger.error(f"Error in graph_query: {e}")
        return {
            "results": [],
            "query": query,
            "error": str(e),
            "execution_time_ms": (time.time() - start_time) * 1000,
        }


@tool(
    name="impact_analysis",
    category="analysis",
    keywords=["impact", "change", "break", "affect", "dependency", "refactor"],
    use_cases=[
        "Analyzing change impact before refactoring",
        "Finding what depends on a function",
        "Understanding upstream dependencies",
        "Checking if code is safe to modify",
        "Identifying downstream effects"
    ],
    priority=Priority.HIGH,
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    task_types=["analysis"],
    execution_category="read_only",
    timeout=45.0,
)
async def impact_analysis(
    target: str,
    analysis_type: str = "forward",
    max_depth: int = 3,
    include_test_impact: bool = True,
    path: str = ".",
) -> Dict[str, Any]:
    """Analyze impact of code changes using Code Context Graph.

    This tool traces dependencies through:
    - Control Flow Graph (CFG): Execution paths
    - Control Dependence Graph (CDG): Control dependencies
    - Data Dependence Graph (DDG): Variable definitions and uses

    Args:
        target: Target symbol or file:line (e.g., 'my_function' or 'src/main.py:42')
        analysis_type: forward (downstream impact) or backward (upstream dependencies)
        max_depth: Maximum depth for impact analysis (1-5, default: 3)
        include_test_impact: Whether to include test impact in analysis
        path: Path to search within (default: current directory)

    Returns:
        Dictionary with:
        - target: Analyzed target
        - impacted_symbols: List of impacted symbols
        - impact_paths: Paths from target to impacted symbols
        - test_impact: Test files that may be affected
        - execution_time_ms: Time taken
        - metadata: Additional metadata

    Example:
        >>> await impact_analysis("authenticate_user",
        ...                       analysis_type="forward", max_depth=3)
    """
    import time
    from victor.core.feature_flags import get_feature_flag_manager, FeatureFlag

    start_time = time.time()

    # Check feature flag
    flag_manager = get_feature_flag_manager()
    if not flag_manager.is_enabled(FeatureFlag.USE_GRAPH_QUERY_TOOL):
        logger.warning("Graph query tool is not enabled")
        return {
            "target": target,
            "impacted_symbols": [],
            "error": "Graph query tool is not enabled. "
                    "Set VICTOR_USE_GRAPH_QUERY_TOOL=true to enable.",
            "execution_time_ms": (time.time() - start_time) * 1000,
        }

    logger.info(f"Impact analysis: {target} (type={analysis_type}, depth={max_depth})")

    try:
        from victor.storage.graph import create_graph_store
        from victor.storage.graph.edge_types import EdgeType

        # Create graph store with project path
        project_path = Path(path).resolve()
        graph_store = create_graph_store(name="sqlite", project_path=project_path)
        await graph_store.initialize()

        # Parse target
        target_node_id = await _resolve_target(target, graph_store)
        if target_node_id is None:
            return {
                "target": target,
                "impacted_symbols": [],
                "error": f"Could not resolve target: {target}",
                "execution_time_ms": (time.time() - start_time) * 1000,
            }

        # Traverse for impact
        # Forward: what depends on target (incoming edges)
        # Backward: what target depends on (outgoing edges)
        direction = "in" if analysis_type == "forward" else "out"
        impacted_edges = await graph_store.get_neighbors(
            target_node_id,
            direction=direction,
            max_depth=max_depth,
        )

        # Collect impacted nodes
        # For forward (incoming edges): we want the source nodes (what calls target)
        # For backward (outgoing edges): we want the destination nodes (what target calls)
        impacted_node_ids: set[str] = set()
        for edge in impacted_edges:
            impacted_node_ids.add(edge.src if direction == "in" else edge.dst)

        # Get node details
        impacted_symbols = []
        for node_id in impacted_node_ids:
            node = await graph_store.get_node_by_id(node_id)
            if node:
                impacted_symbols.append({
                    "name": node.name,
                    "type": node.type,
                    "file": node.file,
                    "line": node.line,
                    "relationship": _get_relationship_type(node, analysis_type),
                })

        # Find test impact if requested
        test_impact = []
        if include_test_impact:
            test_impact = await _find_test_impact(
                {target_node_id} | impacted_node_ids,
                graph_store,
            )

        # Build impact paths
        impact_paths = _build_impact_paths(
            target_node_id,
            impacted_edges,
            direction,
        )

        return {
            "target": target,
            "target_node_id": target_node_id,
            "analysis_type": analysis_type,
            "impacted_symbols": impacted_symbols,
            "impact_count": len(impacted_symbols),
            "impact_paths": impact_paths[:10],  # Limit paths
            "test_impact": test_impact,
            "execution_time_ms": (time.time() - start_time) * 1000,
            "metadata": {
                "max_depth": max_depth,
                "total_edges_analyzed": len(impacted_edges),
            },
        }

    except Exception as e:
        logger.error(f"Error in impact_analysis: {e}")
        return {
            "target": target,
            "impacted_symbols": [],
            "error": str(e),
            "execution_time_ms": (time.time() - start_time) * 1000,
        }


# =============================================================================
# Helper Functions
# =============================================================================

async def _resolve_target(
    target: str,
    graph_store: Any,
) -> str | None:
    """Resolve a target string to a node ID.

    Args:
        target: Target string (symbol name or file:line)
        graph_store: Graph store instance

    Returns:
        Node ID or None if not found
    """
    # Check if it's file:line format
    if ":" in target:
        file_path, line_str = target.rsplit(":", 1)
        try:
            line = int(line_str)
            # Find nodes at this location
            nodes = await graph_store.get_nodes_by_file(file_path)
            for node in nodes:
                if node.line and node.line <= line <= (node.end_line or line):
                    return node.node_id
        except ValueError:
            pass

    # Search by name
    nodes = await graph_store.find_nodes(name=target)
    if nodes:
        return nodes[0].node_id

    return None


def _get_relationship_type(node: Any, analysis_type: str) -> str:
    """Get relationship type description.

    Args:
        node: Graph node
        analysis_type: Analysis type (forward/backward)

    Returns:
        Relationship description
    """
    if analysis_type == "forward":
        return "impacted" if node.type in {"function", "method"} else "used_by"
    else:
        return "depends_on" if node.type in {"function", "method"} else "uses"


async def _find_test_impact(
    node_ids: set[str],
    graph_store: Any,
) -> List[Dict[str, Any]]:
    """Find test files that may be impacted.

    Args:
        node_ids: Set of node IDs to check
        graph_store: Graph store instance

    Returns:
        List of potentially impacted test files
    """
    from victor.storage.graph.edge_types import EdgeType

    test_impact: List[Dict[str, Any]] = []

    # Find test nodes linked via TESTS edge
    for node_id in node_ids:
        edges = await graph_store.get_neighbors(
            node_id,
            edge_types={EdgeType.TESTS, EdgeType.COVERS},
            direction="in",
        )

        for edge in edges:
            test_node = await graph_store.get_node_by_id(edge.src)
            if test_node and "test" in test_node.file.lower():
                test_impact.append({
                    "file": test_node.file,
                    "name": test_node.name,
                    "type": test_node.type,
                })

    return test_impact


def _build_impact_paths(
    start_id: str,
    edges: List[Any],
    direction: str,
) -> List[Dict[str, Any]]:
    """Build impact paths from edges.

    Args:
        start_id: Starting node ID
        edges: Traversed edges
        direction: Traversal direction

    Returns:
        List of impact paths
    """
    # Build adjacency
    adjacency: Dict[str, List[str]] = {}
    for edge in edges:
        src, dst = edge.src, edge.dst
        key = src if direction == "out" else dst
        value = dst if direction == "out" else src

        if key not in adjacency:
            adjacency[key] = []
        adjacency[key].append(value)

    # Simple BFS to find paths
    paths: List[Dict[str, Any]] = []
    visited: set[str] = set()

    def dfs(current: str, path: List[str], depth: int) -> None:
        if depth > 3 or current in visited:
            return
        visited.add(current)

        for neighbor in adjacency.get(current, []):
            new_path = path + [neighbor]
            paths.append({
                "path": new_path,
                "length": len(new_path),
                "edge_type": edges[0].type if edges else "unknown",
            })
            dfs(neighbor, new_path, depth + 1)

    dfs(start_id, [start_id], 0)

    return paths[:20]  # Limit paths


__all__ = [
    "graph_semantic_search",
    "impact_analysis",
    "GraphSemanticSearchInput",
    "ImpactAnalysisInput",
]
