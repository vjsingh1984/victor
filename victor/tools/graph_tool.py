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

"""
Unified Code Graph Tool - Rich graph traversal and analysis for LLMs.

Exposes the codebase knowledge graph to enable algorithms like:
- PageRank (most important symbols)
- Neighbors (direct connections)
- Shortest path (relationship chains)
- Centrality (most connected nodes)
- Subgraph extraction
- Impact analysis (what changes if X changes)
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from victor.codebase.graph.protocol import GraphEdge, GraphNode, GraphStoreProtocol
from victor.codebase.graph.registry import create_graph_store
from victor.tools.base import AccessMode, CostTier, DangerLevel, Priority, ExecutionCategory
from victor.tools.decorators import tool

logger = logging.getLogger(__name__)

# =============================================================================
# Graph Analysis Modes
# =============================================================================

GraphMode = Literal[
    "find",  # Find symbols by name/pattern, optionally expand via graph
    "neighbors",  # Get direct connections (callers/callees)
    "pagerank",  # Find most important symbols
    "centrality",  # Find most connected symbols
    "path",  # Find shortest path between symbols
    "impact",  # What would be affected if symbol changes
    "clusters",  # Find tightly coupled symbol groups
    "stats",  # Graph statistics
    "subgraph",  # Extract subgraph around a symbol
    "file_deps",  # Get file-level dependencies
    "patterns",  # Detect design patterns via graph structure
]

EdgeType = Literal[
    "CALLS",  # Function calls another function
    "REFERENCES",  # Symbol references another
    "CONTAINS",  # File/class contains symbol
    "INHERITS",  # Class inherits from another
    "IMPLEMENTS",  # Class implements interface
    "COMPOSED_OF",  # Class has composition relationship
    "IMPORTS",  # File imports module
]

ALL_EDGE_TYPES = [
    "CALLS",
    "REFERENCES",
    "CONTAINS",
    "INHERITS",
    "IMPLEMENTS",
    "COMPOSED_OF",
    "IMPORTS",
]


@dataclass
class GraphAnalyzer:
    """In-memory graph analyzer with common algorithms."""

    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    # Adjacency lists: node_id -> [(target_id, edge_type, weight)]
    outgoing: Dict[str, List[Tuple[str, str, float]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    incoming: Dict[str, List[Tuple[str, str, float]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        weight = edge.weight or 1.0
        self.outgoing[edge.src].append((edge.dst, edge.type, weight))
        self.incoming[edge.dst].append((edge.src, edge.type, weight))

    def get_neighbors(
        self,
        node_id: str,
        direction: Literal["in", "out", "both"] = "both",
        edge_types: Optional[List[str]] = None,
        max_depth: int = 1,
    ) -> Dict[str, Any]:
        """Get neighbors up to max_depth hops away."""
        visited: Set[str] = set()
        result: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        queue: deque[Tuple[str, int]] = deque([(node_id, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth or current in visited:
                continue
            visited.add(current)

            edges_to_check = []
            if direction in ("out", "both"):
                edges_to_check.extend(
                    [(t, et, w, "out") for t, et, w in self.outgoing.get(current, [])]
                )
            if direction in ("in", "both"):
                edges_to_check.extend(
                    [(t, et, w, "in") for t, et, w in self.incoming.get(current, [])]
                )

            for target, edge_type, weight, dir_label in edges_to_check:
                if edge_types and edge_type not in edge_types:
                    continue
                if target not in visited:
                    node = self.nodes.get(target)
                    result[depth + 1].append(
                        {
                            "node_id": target,
                            "name": node.name if node else target,
                            "type": node.type if node else "unknown",
                            "file": node.file if node else "",
                            "edge_type": edge_type,
                            "direction": dir_label,
                            "weight": weight,
                        }
                    )
                    queue.append((target, depth + 1))

        return {
            "source": node_id,
            "source_name": self.nodes.get(node_id, GraphNode(node_id, "", node_id, "")).name,
            "neighbors_by_depth": dict(result),
            "total_neighbors": sum(len(v) for v in result.values()),
        }

    def pagerank(
        self,
        damping: float = 0.85,
        iterations: int = 100,
        edge_types: Optional[List[str]] = None,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Compute PageRank to find most important symbols."""
        if not self.nodes:
            return []

        # Initialize scores
        n = len(self.nodes)
        scores: Dict[str, float] = dict.fromkeys(self.nodes, 1.0 / n)

        for _ in range(iterations):
            new_scores: Dict[str, float] = {}
            for node_id in self.nodes:
                rank_sum = 0.0
                for src, edge_type, _weight in self.incoming.get(node_id, []):
                    if edge_types and edge_type not in edge_types:
                        continue
                    out_degree = len(
                        [
                            e
                            for e in self.outgoing.get(src, [])
                            if not edge_types or e[1] in edge_types
                        ]
                    )
                    if out_degree > 0:
                        rank_sum += scores.get(src, 0) / out_degree
                new_scores[node_id] = (1 - damping) / n + damping * rank_sum
            scores = new_scores

        # Return top-k
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [
            {
                "rank": i + 1,
                "node_id": nid,
                "name": self.nodes.get(nid, GraphNode(nid, "", nid, "")).name,
                "type": self.nodes.get(nid, GraphNode(nid, "", nid, "")).type,
                "file": self.nodes.get(nid, GraphNode(nid, "", nid, "")).file,
                "score": round(score, 6),
                "in_degree": len(self.incoming.get(nid, [])),
                "out_degree": len(self.outgoing.get(nid, [])),
            }
            for i, (nid, score) in enumerate(ranked)
        ]

    def degree_centrality(
        self,
        edge_types: Optional[List[str]] = None,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Find most connected nodes by degree centrality."""
        centrality: Dict[str, int] = {}

        for node_id in self.nodes:
            in_edges = [
                e for e in self.incoming.get(node_id, []) if not edge_types or e[1] in edge_types
            ]
            out_edges = [
                e for e in self.outgoing.get(node_id, []) if not edge_types or e[1] in edge_types
            ]
            centrality[node_id] = len(in_edges) + len(out_edges)

        ranked = sorted(centrality.items(), key=lambda x: -x[1])[:top_k]
        return [
            {
                "rank": i + 1,
                "node_id": nid,
                "name": self.nodes.get(nid, GraphNode(nid, "", nid, "")).name,
                "type": self.nodes.get(nid, GraphNode(nid, "", nid, "")).type,
                "file": self.nodes.get(nid, GraphNode(nid, "", nid, "")).file,
                "degree": degree,
                "in_degree": len(
                    [e for e in self.incoming.get(nid, []) if not edge_types or e[1] in edge_types]
                ),
                "out_degree": len(
                    [e for e in self.outgoing.get(nid, []) if not edge_types or e[1] in edge_types]
                ),
            }
            for i, (nid, degree) in enumerate(ranked)
        ]

    def shortest_path(
        self,
        source: str,
        target: str,
        edge_types: Optional[List[str]] = None,
        max_depth: int = 10,
    ) -> Dict[str, Any]:
        """Find shortest path between two symbols using BFS."""
        if source not in self.nodes:
            return {"error": f"Source node '{source}' not found"}
        if target not in self.nodes:
            return {"error": f"Target node '{target}' not found"}

        visited: Set[str] = {source}
        parent: Dict[str, Tuple[str, str]] = {}  # child -> (parent, edge_type)
        queue: deque[Tuple[str, int]] = deque([(source, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue

            if current == target:
                # Reconstruct path
                path = []
                node = target
                while node != source:
                    p, et = parent[node]
                    path.append(
                        {
                            "from": p,
                            "to": node,
                            "edge_type": et,
                            "from_name": self.nodes.get(p, GraphNode(p, "", p, "")).name,
                            "to_name": self.nodes.get(node, GraphNode(node, "", node, "")).name,
                        }
                    )
                    node = p
                path.reverse()
                return {
                    "found": True,
                    "source": source,
                    "target": target,
                    "length": len(path),
                    "path": path,
                }

            for neighbor, edge_type, _ in self.outgoing.get(current, []):
                if edge_types and edge_type not in edge_types:
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = (current, edge_type)
                    queue.append((neighbor, depth + 1))

        return {
            "found": False,
            "source": source,
            "target": target,
            "message": f"No path found within {max_depth} hops",
        }

    def impact_analysis(
        self,
        node_id: str,
        edge_types: Optional[List[str]] = None,
        max_depth: int = 3,
    ) -> Dict[str, Any]:
        """Analyze what would be affected if a symbol changes."""
        if node_id not in self.nodes:
            return {"error": f"Node '{node_id}' not found"}

        # Find all nodes that depend on this one (incoming edges)
        affected: Dict[int, Set[str]] = defaultdict(set)
        visited: Set[str] = {node_id}
        queue: deque[Tuple[str, int]] = deque([(node_id, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue

            # Find what depends on current (incoming edges = dependents)
            for src, edge_type, _ in self.incoming.get(current, []):
                if edge_types and edge_type not in edge_types:
                    continue
                if src not in visited:
                    visited.add(src)
                    affected[depth + 1].add(src)
                    queue.append((src, depth + 1))

        # Group by file for easier navigation
        by_file: Dict[str, List[str]] = defaultdict(list)
        for depth_nodes in affected.values():
            for nid in depth_nodes:
                node = self.nodes.get(nid)
                if node:
                    by_file[node.file].append(node.name)

        return {
            "node_id": node_id,
            "node_name": self.nodes.get(node_id, GraphNode(node_id, "", node_id, "")).name,
            "total_affected": sum(len(s) for s in affected.values()),
            "affected_by_depth": {
                k: [
                    {
                        "node_id": nid,
                        "name": self.nodes.get(nid, GraphNode(nid, "", nid, "")).name,
                        "type": self.nodes.get(nid, GraphNode(nid, "", nid, "")).type,
                        "file": self.nodes.get(nid, GraphNode(nid, "", nid, "")).file,
                    }
                    for nid in v
                ]
                for k, v in affected.items()
            },
            "affected_files": dict(by_file),
            "files_count": len(by_file),
        }

    def find_symbols(
        self,
        query: str,
        node_type: Optional[str] = None,
        file_pattern: Optional[str] = None,
        expand_neighbors: bool = False,
        edge_types: Optional[List[str]] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Find symbols by name pattern, optionally expand via graph."""
        import fnmatch
        import re

        query_lower = query.lower()
        matches: List[Dict[str, Any]] = []

        for node_id, node in self.nodes.items():
            # Skip if type filter doesn't match
            if node_type and node.type != node_type:
                continue

            # Skip if file pattern doesn't match
            if file_pattern and not fnmatch.fnmatch(node.file, file_pattern):
                continue

            # Match by name (case-insensitive contains)
            if query_lower in node.name.lower():
                match_info = {
                    "node_id": node_id,
                    "name": node.name,
                    "type": node.type,
                    "file": node.file,
                    "line": node.line,
                    "in_degree": len(self.incoming.get(node_id, [])),
                    "out_degree": len(self.outgoing.get(node_id, [])),
                }

                # Optionally expand neighbors
                if expand_neighbors:
                    neighbors = []
                    for target, et, _ in self.outgoing.get(node_id, [])[:5]:
                        if not edge_types or et in edge_types:
                            n = self.nodes.get(target)
                            if n:
                                neighbors.append(
                                    {"name": n.name, "type": n.type, "edge": et, "direction": "out"}
                                )
                    for source, et, _ in self.incoming.get(node_id, [])[:5]:
                        if not edge_types or et in edge_types:
                            n = self.nodes.get(source)
                            if n:
                                neighbors.append(
                                    {"name": n.name, "type": n.type, "edge": et, "direction": "in"}
                                )
                    match_info["neighbors"] = neighbors

                matches.append(match_info)

                if len(matches) >= limit:
                    break

        return {
            "query": query,
            "filters": {"node_type": node_type, "file_pattern": file_pattern},
            "total_matches": len(matches),
            "matches": matches,
        }

    def get_file_dependencies(
        self,
        file_path: str,
        direction: Literal["imports", "imported_by", "both"] = "both",
    ) -> Dict[str, Any]:
        """Get file-level dependencies."""
        # Find nodes in the file
        file_nodes = [
            n for n in self.nodes.values() if n.file == file_path or n.file.endswith(file_path)
        ]

        if not file_nodes:
            return {"error": f"No symbols found in file: {file_path}"}

        imports: Set[str] = set()
        imported_by: Set[str] = set()

        for node in file_nodes:
            node_id = node.node_id
            # Files this file imports (outgoing IMPORTS edges)
            if direction in ("imports", "both"):
                for target, et, _ in self.outgoing.get(node_id, []):
                    if et == "IMPORTS":
                        target_node = self.nodes.get(target)
                        if target_node and target_node.file != file_path:
                            imports.add(target_node.file)

            # Files that import this file (incoming IMPORTS edges)
            if direction in ("imported_by", "both"):
                for source, et, _ in self.incoming.get(node_id, []):
                    if et == "IMPORTS":
                        source_node = self.nodes.get(source)
                        if source_node and source_node.file != file_path:
                            imported_by.add(source_node.file)

        return {
            "file": file_path,
            "imports": sorted(imports),
            "imports_count": len(imports),
            "imported_by": sorted(imported_by),
            "imported_by_count": len(imported_by),
            "symbols_in_file": [{"name": n.name, "type": n.type} for n in file_nodes[:20]],
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        edge_type_counts: Counter = Counter()
        node_type_counts: Counter = Counter()

        for node in self.nodes.values():
            node_type_counts[node.type] += 1

        for edges in self.outgoing.values():
            for _, edge_type, _ in edges:
                edge_type_counts[edge_type] += 1

        return {
            "total_nodes": len(self.nodes),
            "total_edges": sum(len(e) for e in self.outgoing.values()),
            "node_types": dict(node_type_counts),
            "edge_types": dict(edge_type_counts),
            "avg_in_degree": round(
                sum(len(e) for e in self.incoming.values()) / max(len(self.nodes), 1), 2
            ),
            "avg_out_degree": round(
                sum(len(e) for e in self.outgoing.values()) / max(len(self.nodes), 1), 2
            ),
        }

    def detect_patterns(self) -> Dict[str, Any]:
        """Detect design patterns via graph structure analysis.

        Detects patterns like:
        - Factory: Classes with many outgoing CALLS to constructors
        - Facade: High in-degree classes that delegate to many others
        - Hub/God classes: Very high degree centrality
        - Inheritance hierarchies: INHERITS chains
        - Strategy/Provider: Base class with multiple implementations
        - Composition: COMPOSED_OF relationships
        """
        patterns: List[Dict[str, Any]] = []

        # Pattern 1: Provider/Strategy Pattern - Base class with multiple INHERITS
        inheritance_counts: Dict[str, List[str]] = defaultdict(list)
        for node_id in self.nodes:
            for src, edge_type, _ in self.incoming.get(node_id, []):
                if edge_type == "INHERITS":
                    inheritance_counts[node_id].append(src)

        for base_id, children in inheritance_counts.items():
            if len(children) >= 2:
                base_node = self.nodes.get(base_id)
                patterns.append(
                    {
                        "pattern": "provider_strategy",
                        "name": "Provider/Strategy Pattern",
                        "base_class": base_node.name if base_node else base_id,
                        "implementations": [
                            self.nodes.get(c, GraphNode(c, "", c, "")).name for c in children
                        ],
                        "count": len(children),
                        "file": base_node.file if base_node else "",
                        "confidence": min(0.5 + len(children) * 0.1, 0.95),
                    }
                )

        # Pattern 2: Facade - High in-degree, high out-degree (orchestration)
        for node_id, node in self.nodes.items():
            in_calls = len([e for e in self.incoming.get(node_id, []) if e[1] == "CALLS"])
            out_calls = len([e for e in self.outgoing.get(node_id, []) if e[1] == "CALLS"])

            # Facade: many callers (in) and calls many others (out)
            if in_calls >= 3 and out_calls >= 5:
                patterns.append(
                    {
                        "pattern": "facade",
                        "name": "Facade/Orchestrator",
                        "class": node.name,
                        "file": node.file,
                        "incoming_calls": in_calls,
                        "outgoing_calls": out_calls,
                        "confidence": min(0.6 + (in_calls + out_calls) * 0.02, 0.95),
                    }
                )

        # Pattern 3: Hub/God Class - Very high total degree
        degree_threshold = 15
        for node_id, node in self.nodes.items():
            if node.type != "class":
                continue
            total_degree = len(self.incoming.get(node_id, [])) + len(self.outgoing.get(node_id, []))
            if total_degree >= degree_threshold:
                patterns.append(
                    {
                        "pattern": "hub_god_class",
                        "name": "Hub/God Class (potential smell)",
                        "class": node.name,
                        "file": node.file,
                        "total_connections": total_degree,
                        "recommendation": "Consider decomposition",
                        "confidence": min(0.7 + total_degree * 0.01, 0.95),
                    }
                )

        # Pattern 4: Factory - Class/function with many outgoing "creation" calls
        for node_id, node in self.nodes.items():
            if "factory" in node.name.lower() or "create" in node.name.lower():
                out_calls = len(self.outgoing.get(node_id, []))
                if out_calls >= 2:
                    patterns.append(
                        {
                            "pattern": "factory",
                            "name": "Factory Pattern",
                            "class": node.name,
                            "file": node.file,
                            "creates": out_calls,
                            "confidence": 0.8,
                        }
                    )

        # Pattern 5: Composition - COMPOSED_OF relationships
        composition_holders: Dict[str, List[str]] = defaultdict(list)
        for node_id in self.nodes:
            for target, edge_type, _ in self.outgoing.get(node_id, []):
                if edge_type == "COMPOSED_OF":
                    composition_holders[node_id].append(target)

        for holder_id, composed in composition_holders.items():
            if len(composed) >= 2:
                holder_node = self.nodes.get(holder_id)
                patterns.append(
                    {
                        "pattern": "composition",
                        "name": "Composition Pattern",
                        "class": holder_node.name if holder_node else holder_id,
                        "file": holder_node.file if holder_node else "",
                        "composed_of": [
                            self.nodes.get(c, GraphNode(c, "", c, "")).name for c in composed
                        ],
                        "count": len(composed),
                        "confidence": min(0.7 + len(composed) * 0.05, 0.95),
                    }
                )

        # Pattern 6: Dependency Injection - Classes receiving many external dependencies
        for node_id, node in self.nodes.items():
            if node.type != "class":
                continue
            # Look for classes with many incoming REFERENCES from different files
            external_refs = set()
            for src, edge_type, _ in self.incoming.get(node_id, []):
                if edge_type == "REFERENCES":
                    src_node = self.nodes.get(src)
                    if src_node and src_node.file != node.file:
                        external_refs.add(src_node.file)

            if len(external_refs) >= 4:
                patterns.append(
                    {
                        "pattern": "dependency_injection_target",
                        "name": "Dependency Injection Target",
                        "class": node.name,
                        "file": node.file,
                        "injected_into_files": len(external_refs),
                        "confidence": min(0.6 + len(external_refs) * 0.05, 0.9),
                    }
                )

        # Sort by confidence
        patterns.sort(key=lambda x: -x.get("confidence", 0))

        # Summary statistics
        pattern_summary = Counter(p["pattern"] for p in patterns)

        return {
            "total_patterns_found": len(patterns),
            "pattern_summary": dict(pattern_summary),
            "patterns": patterns[:30],  # Limit output
        }

    def extract_subgraph(
        self,
        center_node: str,
        max_depth: int = 2,
        edge_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Extract a subgraph centered on a node."""
        if center_node not in self.nodes:
            return {"error": f"Node '{center_node}' not found"}

        visited: Set[str] = set()
        nodes_in_subgraph: List[Dict[str, Any]] = []
        edges_in_subgraph: List[Dict[str, Any]] = []
        queue: deque[Tuple[str, int]] = deque([(center_node, 0)])

        while queue:
            current, depth = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            node = self.nodes.get(current)
            if node:
                nodes_in_subgraph.append(
                    {
                        "node_id": current,
                        "name": node.name,
                        "type": node.type,
                        "file": node.file,
                        "depth_from_center": depth,
                    }
                )

            if depth >= max_depth:
                continue

            # Add outgoing and incoming edges
            for target, edge_type, weight in self.outgoing.get(current, []):
                if edge_types and edge_type not in edge_types:
                    continue
                edges_in_subgraph.append(
                    {
                        "from": current,
                        "to": target,
                        "type": edge_type,
                        "weight": weight,
                    }
                )
                if target not in visited:
                    queue.append((target, depth + 1))

            for source, edge_type, weight in self.incoming.get(current, []):
                if edge_types and edge_type not in edge_types:
                    continue
                edges_in_subgraph.append(
                    {
                        "from": source,
                        "to": current,
                        "type": edge_type,
                        "weight": weight,
                    }
                )
                if source not in visited:
                    queue.append((source, depth + 1))

        return {
            "center": center_node,
            "center_name": self.nodes.get(
                center_node, GraphNode(center_node, "", center_node, "")
            ).name,
            "nodes_count": len(nodes_in_subgraph),
            "edges_count": len(edges_in_subgraph),
            "nodes": nodes_in_subgraph,
            "edges": edges_in_subgraph,
        }


async def _load_graph(graph_store: GraphStoreProtocol) -> GraphAnalyzer:
    """Load graph from store into analyzer."""
    analyzer = GraphAnalyzer()

    # Load all nodes
    nodes = await graph_store.find_nodes()
    for node in nodes:
        analyzer.add_node(node)

    # Load all edges in one query (much faster than per-node queries)
    if hasattr(graph_store, "get_all_edges"):
        edges = await graph_store.get_all_edges()
        for edge in edges:
            analyzer.add_edge(edge)
    else:
        # Fallback for stores that don't have get_all_edges
        for node_id in analyzer.nodes:
            edges = await graph_store.get_neighbors(node_id, edge_types=None, max_depth=1)
            for edge in edges:
                analyzer.add_edge(edge)

    return analyzer


# =============================================================================
# Main Tool
# =============================================================================


@tool(
    category="code_intelligence",
    priority=Priority.HIGH,
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    execution_category=ExecutionCategory.READ_ONLY,
    keywords=[
        # Graph operations
        "graph",
        "traverse",
        "analyze",
        "network",
        # Specific algorithms
        "pagerank",
        "important",
        "centrality",
        "connected",
        "hub",
        "neighbors",
        "callers",
        "callees",
        "dependencies",
        "path",
        "relationship",
        "chain",
        "impact",
        "affected",
        "ripple",
        "changes",
        # Natural language
        "most used",
        "most called",
        "most important",
        "what calls",
        "what uses",
        "who depends",
        "how to get from",
        "path between",
        "what happens if",
        "impact of changing",
    ],
    stages=["analysis", "reading"],
)
async def graph(
    mode: GraphMode,
    node: Optional[str] = None,
    target: Optional[str] = None,
    query: Optional[str] = None,
    file: Optional[str] = None,
    node_type: Optional[str] = None,
    edge_types: Optional[List[str]] = None,
    depth: int = 2,
    top_k: int = 20,
    direction: Literal["in", "out", "both"] = "both",
    expand: bool = False,
) -> Dict[str, Any]:
    """[GRAPH] Query the codebase knowledge graph for rich structural analysis.

    This tool exposes the code graph built from AST analysis, enabling powerful
    traversal and analysis algorithms to understand code structure.

    Args:
        mode: Analysis mode - determines what algorithm to run:
            - "find": Search symbols by name pattern (combines search + graph)
            - "neighbors": Get direct callers/callees of a symbol
            - "pagerank": Find most important symbols (PageRank algorithm)
            - "centrality": Find most connected symbols (degree centrality)
            - "path": Find shortest path between two symbols
            - "impact": What would be affected if symbol changes
            - "subgraph": Extract neighborhood around a symbol
            - "file_deps": Get file-level import dependencies
            - "patterns": Detect design patterns (Factory, Facade, Strategy, etc.)
            - "stats": Get graph statistics

        node: Symbol name or node ID (required for most modes except stats/pagerank/find)
        target: Target symbol for "path" mode
        query: Search query for "find" mode (pattern to match symbol names)
        file: File path for "file_deps" mode, or filter for "find" mode
        node_type: Filter by node type ("function", "class", "module") for "find" mode
        edge_types: Filter by edge types. Options:
            - "CALLS": Function call relationships
            - "REFERENCES": Symbol references
            - "INHERITS": Class inheritance
            - "IMPLEMENTS": Interface implementation
            - "IMPORTS": Module imports
            Default: all types
        depth: Max traversal depth (default: 2)
        top_k: Number of results for ranking/find modes (default: 20)
        direction: Edge direction for neighbors/file_deps - "in", "out", or "both"
        expand: For "find" mode - also return immediate graph neighbors

    Returns:
        Results vary by mode:
        - find: {query, matches: [{name, type, file, in_degree, out_degree, neighbors?}]}
        - neighbors: {source, neighbors_by_depth, total_neighbors}
        - pagerank: [{rank, node_id, name, type, file, score, in_degree, out_degree}]
        - centrality: [{rank, node_id, name, degree, in_degree, out_degree}]
        - path: {found, source, target, length, path}
        - impact: {node_name, total_affected, affected_by_depth, affected_files}
        - subgraph: {center, nodes, edges, nodes_count, edges_count}
        - file_deps: {file, imports, imported_by, symbols_in_file}
        - stats: {total_nodes, total_edges, node_types, edge_types, avg_degree}

    Examples:
        # Find symbols matching a pattern with graph context
        graph(mode="find", query="process", node_type="function", expand=True)

        # Find most important symbols in codebase
        graph(mode="pagerank", top_k=10)

        # What functions call "process_request"?
        graph(mode="neighbors", node="process_request", direction="in", edge_types=["CALLS"])

        # What would be affected if I change "UserModel"?
        graph(mode="impact", node="UserModel", depth=3)

        # How is "main" connected to "database_connect"?
        graph(mode="path", node="main", target="database_connect")

        # Get file dependencies
        graph(mode="file_deps", file="orchestrator.py")

        # Get neighborhood around a symbol
        graph(mode="subgraph", node="AgentOrchestrator", depth=2)
    """
    try:
        # Get graph store
        graph_path = Path(".victor/graph")
        graph_path.mkdir(parents=True, exist_ok=True)
        store = create_graph_store("sqlite", graph_path)

        # Load into analyzer
        analyzer = await _load_graph(store)

        if not analyzer.nodes:
            return {
                "error": "Graph is empty. Run 'victor index' to build the code graph first.",
                "hint": "The graph is populated during codebase indexing.",
            }

        # Resolve node name to node_id if needed
        resolved_node = None
        if node:
            # First try exact match on node_id
            if node in analyzer.nodes:
                resolved_node = node
            else:
                # Search by name
                for nid, n in analyzer.nodes.items():
                    if n.name == node:
                        resolved_node = nid
                        break

            if mode not in ("pagerank", "centrality", "stats") and not resolved_node:
                # Try fuzzy match
                matches = [n for nid, n in analyzer.nodes.items() if node.lower() in n.name.lower()]
                if matches:
                    return {
                        "error": f"Node '{node}' not found exactly",
                        "suggestions": [
                            {"name": m.name, "type": m.type, "file": m.file} for m in matches[:5]
                        ],
                    }
                return {"error": f"Node '{node}' not found in graph"}

        resolved_target = None
        if target:
            if target in analyzer.nodes:
                resolved_target = target
            else:
                for nid, n in analyzer.nodes.items():
                    if n.name == target:
                        resolved_target = nid
                        break
            if mode == "path" and not resolved_target:
                return {"error": f"Target node '{target}' not found"}

        # Execute mode
        if mode == "neighbors":
            if not resolved_node:
                return {"error": "node parameter required for 'neighbors' mode"}
            return analyzer.get_neighbors(resolved_node, direction, edge_types, depth)

        elif mode == "pagerank":
            return {
                "mode": "pagerank",
                "results": analyzer.pagerank(edge_types=edge_types, top_k=top_k),
            }

        elif mode == "centrality":
            return {
                "mode": "centrality",
                "results": analyzer.degree_centrality(edge_types=edge_types, top_k=top_k),
            }

        elif mode == "path":
            if not resolved_node or not resolved_target:
                return {"error": "Both 'node' and 'target' required for 'path' mode"}
            return analyzer.shortest_path(resolved_node, resolved_target, edge_types, depth)

        elif mode == "impact":
            if not resolved_node:
                return {"error": "node parameter required for 'impact' mode"}
            return analyzer.impact_analysis(resolved_node, edge_types, depth)

        elif mode == "subgraph":
            if not resolved_node:
                return {"error": "node parameter required for 'subgraph' mode"}
            return analyzer.extract_subgraph(resolved_node, depth, edge_types)

        elif mode == "stats":
            return analyzer.get_stats()

        elif mode == "find":
            if not query:
                return {"error": "query parameter required for 'find' mode"}
            return analyzer.find_symbols(
                query=query,
                node_type=node_type,
                file_pattern=file,
                expand_neighbors=expand,
                edge_types=edge_types,
                limit=top_k,
            )

        elif mode == "file_deps":
            if not file:
                return {"error": "file parameter required for 'file_deps' mode"}
            file_direction = (
                "imports"
                if direction == "out"
                else ("imported_by" if direction == "in" else "both")
            )
            return analyzer.get_file_dependencies(file, direction=file_direction)

        elif mode == "patterns":
            return analyzer.detect_patterns()

        else:
            return {"error": f"Unknown mode: {mode}"}

    except Exception as e:
        logger.exception(f"Graph tool error: {e}")
        return {"error": str(e)}
