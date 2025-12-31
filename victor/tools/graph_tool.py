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

from victor.coding.codebase.graph.protocol import GraphEdge, GraphNode, GraphStoreProtocol
from victor.coding.codebase.graph.registry import create_graph_store
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
    # Module-level analysis (architectural importance)
    "module_pagerank",  # PageRank at file/package level
    "module_centrality",  # Most connected modules
    "call_flow",  # Inter-module call flow analysis
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

    # =========================================================================
    # Module-Level Analysis Methods
    # =========================================================================

    def _aggregate_to_modules(
        self, granularity: Literal["file", "package"] = "file"
    ) -> Dict[str, Set[str]]:
        """Aggregate symbols to modules (files or packages).

        Args:
            granularity: "file" for per-file, "package" for per-directory

        Returns:
            Dict mapping module path to set of node IDs in that module
        """
        module_to_nodes: Dict[str, Set[str]] = defaultdict(set)
        for node_id, node in self.nodes.items():
            if not node.file:
                continue
            if granularity == "file":
                module_key = node.file
            else:  # package
                # e.g., "victor/tools/bash.py" -> "victor/tools"
                module_key = str(Path(node.file).parent)
                if not module_key or module_key == ".":
                    module_key = "root"
            module_to_nodes[module_key].add(node_id)
        return dict(module_to_nodes)

    def _build_module_graph(
        self,
        granularity: Literal["file", "package"] = "file",
        edge_types: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
        """Build a module-level graph from symbol-level edges.

        Returns:
            Tuple of (module_to_nodes, outgoing_module_edges, incoming_module_edges)
            where edge dicts are: module -> {target_module -> edge_count}
        """
        module_to_nodes = self._aggregate_to_modules(granularity)

        # Build module-level adjacency with edge counts
        outgoing: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        incoming: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Map node_id -> module for quick lookup
        node_to_module: Dict[str, str] = {}
        for module, nodes in module_to_nodes.items():
            for node_id in nodes:
                node_to_module[node_id] = module

        # Aggregate symbol edges to module edges
        for src_node_id in self.nodes:
            src_module = node_to_module.get(src_node_id)
            if not src_module:
                continue

            for target_id, edge_type, _ in self.outgoing.get(src_node_id, []):
                if edge_types and edge_type not in edge_types:
                    continue
                target_module = node_to_module.get(target_id)
                if target_module and target_module != src_module:
                    # Cross-module edge
                    outgoing[src_module][target_module] += 1
                    incoming[target_module][src_module] += 1

        return module_to_nodes, dict(outgoing), dict(incoming)

    def module_pagerank(
        self,
        granularity: Literal["file", "package"] = "file",
        edge_types: Optional[List[str]] = None,
        damping: float = 0.85,
        iterations: int = 100,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Compute PageRank at module/file level for architectural importance.

        Unlike symbol-level PageRank which favors utility functions (__init__, copy, etc.),
        module-level PageRank reveals architecturally significant files:
        - Central coordinators (orchestrators, facades)
        - Core abstractions (base classes, protocols)
        - Key business logic modules

        Args:
            granularity: "file" for per-file analysis, "package" for per-directory
            edge_types: Filter by edge types (CALLS, IMPORTS, REFERENCES, etc.)
            damping: PageRank damping factor (default 0.85)
            iterations: Number of iterations (default 100)
            top_k: Number of results to return

        Returns:
            List of modules ranked by architectural importance
        """
        module_to_nodes, outgoing, incoming = self._build_module_graph(granularity, edge_types)

        if not module_to_nodes:
            return []

        # Initialize scores
        modules = list(module_to_nodes.keys())
        n = len(modules)
        scores: Dict[str, float] = dict.fromkeys(modules, 1.0 / n)

        # PageRank iteration
        for _ in range(iterations):
            new_scores: Dict[str, float] = {}
            for module in modules:
                rank_sum = 0.0
                # Sum contributions from modules that link to this one
                for src_module, count in incoming.get(module, {}).items():
                    # out_degree = total edges from src to all targets
                    out_degree = sum(outgoing.get(src_module, {}).values())
                    if out_degree > 0:
                        # Weight by edge count
                        rank_sum += scores.get(src_module, 0) * count / out_degree
                new_scores[module] = (1 - damping) / n + damping * rank_sum
            scores = new_scores

        # Calculate additional metrics
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        results = []
        for i, (module, score) in enumerate(ranked):
            in_edges = sum(incoming.get(module, {}).values())
            out_edges = sum(outgoing.get(module, {}).values())
            in_modules = len(incoming.get(module, {}))
            out_modules = len(outgoing.get(module, {}))

            # Classify module role based on in/out ratio
            if in_edges > 0 and out_edges > 0:
                ratio = in_edges / out_edges
                if ratio > 2:
                    role = "service"  # Many callers, few dependencies
                elif ratio < 0.5:
                    role = "orchestrator"  # Few callers, many dependencies
                else:
                    role = "intermediary"
            elif in_edges > 0:
                role = "leaf_service"  # Only incoming, no outgoing
            elif out_edges > 0:
                role = "entry_point"  # Only outgoing, no incoming
            else:
                role = "isolated"

            results.append(
                {
                    "rank": i + 1,
                    "module": module,
                    "score": round(score, 6),
                    "role": role,
                    "in_edges": in_edges,
                    "out_edges": out_edges,
                    "imports_from": in_modules,  # Modules that call/import this
                    "depends_on": out_modules,  # Modules this calls/imports
                    "symbols_count": len(module_to_nodes.get(module, set())),
                }
            )

        return results

    def module_centrality(
        self,
        granularity: Literal["file", "package"] = "file",
        edge_types: Optional[List[str]] = None,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Find most connected modules by degree centrality.

        Identifies modules that are hubs in the architecture - either because
        they are widely used (high in-degree) or because they integrate many
        components (high out-degree).

        Args:
            granularity: "file" for per-file, "package" for per-directory
            edge_types: Filter by edge types
            top_k: Number of results

        Returns:
            List of modules ranked by total connectivity
        """
        module_to_nodes, outgoing, incoming = self._build_module_graph(granularity, edge_types)

        if not module_to_nodes:
            return []

        # Calculate centrality for each module
        centrality: List[Tuple[str, int, int, int]] = []
        for module in module_to_nodes:
            in_edges = sum(incoming.get(module, {}).values())
            out_edges = sum(outgoing.get(module, {}).values())
            total = in_edges + out_edges
            centrality.append((module, total, in_edges, out_edges))

        # Sort by total degree
        centrality.sort(key=lambda x: -x[1])

        results = []
        for i, (module, total, in_deg, out_deg) in enumerate(centrality[:top_k]):
            # Identify coupling pattern
            if in_deg > out_deg * 2:
                coupling = "high_fan_in"  # Many dependents
            elif out_deg > in_deg * 2:
                coupling = "high_fan_out"  # Many dependencies
            elif total > 20:
                coupling = "hub"  # Central hub
            else:
                coupling = "normal"

            results.append(
                {
                    "rank": i + 1,
                    "module": module,
                    "total_degree": total,
                    "in_degree": in_deg,
                    "out_degree": out_deg,
                    "coupling_pattern": coupling,
                    "unique_importers": len(incoming.get(module, {})),
                    "unique_dependencies": len(outgoing.get(module, {})),
                    "symbols_count": len(module_to_nodes.get(module, set())),
                }
            )

        return results

    def call_flow(
        self,
        source_module: str,
        target_module: Optional[str] = None,
        granularity: Literal["file", "package"] = "file",
        edge_types: Optional[List[str]] = None,
        depth: int = 3,
    ) -> Dict[str, Any]:
        """Analyze call flow between modules.

        Shows how control/data flows from one module to another, including
        intermediate hops. Useful for understanding:
        - How a request flows through the system
        - Dependencies between architectural layers
        - Potential circular dependencies

        Args:
            source_module: Starting module (file path or package)
            target_module: Optional target module (if None, shows all flows from source)
            granularity: "file" or "package"
            edge_types: Filter by edge types
            depth: Maximum depth to traverse

        Returns:
            Dict with flow analysis including paths, intermediate modules, and edge counts
        """
        module_to_nodes, outgoing, incoming = self._build_module_graph(granularity, edge_types)

        # Find source module (fuzzy match)
        resolved_source = None
        for mod in module_to_nodes:
            if source_module in mod or mod.endswith(source_module):
                resolved_source = mod
                break

        if not resolved_source:
            return {
                "error": f"Source module '{source_module}' not found",
                "available_modules": sorted(module_to_nodes.keys())[:20],
            }

        # If target specified, find shortest path
        if target_module:
            resolved_target = None
            for mod in module_to_nodes:
                if target_module in mod or mod.endswith(target_module):
                    resolved_target = mod
                    break

            if not resolved_target:
                return {
                    "error": f"Target module '{target_module}' not found",
                    "available_modules": sorted(module_to_nodes.keys())[:20],
                }

            # BFS for shortest path
            visited: Set[str] = {resolved_source}
            parent: Dict[str, Tuple[str, int]] = {}  # child -> (parent, edge_count)
            queue: deque[Tuple[str, int]] = deque([(resolved_source, 0)])

            while queue:
                current, d = queue.popleft()
                if d >= depth:
                    continue

                if current == resolved_target:
                    # Reconstruct path
                    path = []
                    node = resolved_target
                    while node != resolved_source:
                        p, count = parent[node]
                        path.append(
                            {
                                "from": p,
                                "to": node,
                                "edge_count": count,
                            }
                        )
                        node = p
                    path.reverse()
                    return {
                        "found": True,
                        "source": resolved_source,
                        "target": resolved_target,
                        "length": len(path),
                        "path": path,
                        "total_edges": sum(p["edge_count"] for p in path),
                    }

                for next_mod, count in outgoing.get(current, {}).items():
                    if next_mod not in visited:
                        visited.add(next_mod)
                        parent[next_mod] = (current, count)
                        queue.append((next_mod, d + 1))

            return {
                "found": False,
                "source": resolved_source,
                "target": resolved_target,
                "message": f"No path found within {depth} hops",
            }

        # No target - show all outgoing flows from source
        direct_deps = []
        for target_mod, count in sorted(
            outgoing.get(resolved_source, {}).items(), key=lambda x: -x[1]
        ):
            direct_deps.append(
                {
                    "module": target_mod,
                    "edge_count": count,
                    "symbols_in_target": len(module_to_nodes.get(target_mod, set())),
                }
            )

        direct_importers = []
        for src_mod, count in sorted(
            incoming.get(resolved_source, {}).items(), key=lambda x: -x[1]
        ):
            direct_importers.append(
                {
                    "module": src_mod,
                    "edge_count": count,
                    "symbols_in_source": len(module_to_nodes.get(src_mod, set())),
                }
            )

        # Find transitive dependencies (2-hop)
        transitive_deps: Dict[str, int] = defaultdict(int)
        for direct in outgoing.get(resolved_source, {}):
            for transitive, count in outgoing.get(direct, {}).items():
                if transitive != resolved_source and transitive not in outgoing.get(
                    resolved_source, {}
                ):
                    transitive_deps[transitive] += count

        return {
            "module": resolved_source,
            "symbols_count": len(module_to_nodes.get(resolved_source, set())),
            "direct_dependencies": direct_deps,
            "direct_dependencies_count": len(direct_deps),
            "imported_by": direct_importers,
            "imported_by_count": len(direct_importers),
            "transitive_dependencies": [
                {"module": m, "paths": c}
                for m, c in sorted(transitive_deps.items(), key=lambda x: -x[1])[:15]
            ],
            "transitive_count": len(transitive_deps),
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
    stages=["initial", "planning", "reading", "analysis"],  # Rapid discovery in early stages
    mandatory_keywords=[
        "analyze codebase",
        "codebase analysis",
        "architecture",
        "analyze dependencies",
        "dependency graph",
    ],  # From MANDATORY_TOOL_KEYWORDS
)
async def graph(
    mode: GraphMode,
    node: Optional[str] = None,
    source: Optional[str] = None,
    target: Optional[str] = None,
    query: Optional[str] = None,
    file: Optional[str] = None,
    node_type: Optional[str] = None,
    edge_types: Optional[List[str]] = None,
    depth: int = 2,
    top_k: int = 20,
    direction: Literal["in", "out", "both"] = "both",
    expand: bool = False,
    exclude_paths: Optional[List[str]] = None,
    only_runtime: bool = True,
    runtime_weighted: bool = True,
    modules_only: bool = False,
    include_callsites: bool = True,
    max_callsites: int = 3,
    structured: bool = True,
    include_symbols: bool = True,
    include_modules: bool = True,
    include_calls: bool = True,
    include_refs: bool = False,
    include_callsites_modules: bool = True,
    max_callsites_modules: int = 3,
    module_edge_weight_bias: bool = True,
    include_neighbors: bool = False,
    neighbors_edge_types: Optional[List[str]] = None,
    neighbors_limit: int = 3,
    granularity: Literal["file", "package"] = "file",
) -> Dict[str, Any]:
    """[GRAPH] Query codebase STRUCTURE for relationships, impact, and importance.

    Uses the code graph built from AST analysis for structural queries like
    "what calls X", "what depends on Y", "most important symbols", etc.

    DIFFERS FROM:
    - symbol(): Gets actual CODE of a definition. Use when you need source code.
    - refs(): Finds all USAGES (file:line locations). Use for "where is X used".
    - search(): Finds by TEXT/CONCEPT. Use when you don't know exact symbol names.

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

            MODULE-LEVEL ANALYSIS (architectural importance, avoids utility function bias):
            - "module_pagerank": PageRank at file/package level - finds architecturally
              significant modules (orchestrators, facades, core abstractions)
            - "module_centrality": Most connected modules - identifies hubs and
              coupling patterns (high fan-in, high fan-out)
            - "call_flow": Inter-module call flow analysis - shows how control flows
              between modules, transitive dependencies, circular dependency detection

        node: Symbol name or node ID (required for most modes except stats/pagerank/find)
        target: Target symbol for "path" mode
        query: Search query for "find" mode (pattern to match symbol names)
            file: File path for "file_deps" mode, or filter for "find" mode
            source: Alias for "file" in call_flow mode (source module)
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
        exclude_paths: Optional list of path substrings to exclude (e.g., ["tests/", "node_modules/"])
        only_runtime: If True (default), automatically down-rank/exclude tests/build/venv/output paths
        runtime_weighted: If True (default), prefer CALLS/INHERITS/IMPLEMENTS edges over REFERENCES for rankings
        modules_only: If True, module-level modes/hotspots only (no symbol-level ranks)
        include_callsites: If True, return sample callsites (file:line) for symbol-level results where available
        max_callsites: Limit for returned callsites when include_callsites is True
        structured: If True (default), return separate module vs symbol sections with edge-type breakdowns
        include_symbols: Include symbol-level ranks (when structured)
        include_modules: Include module-level ranks (when structured)
        include_calls: Include CALLS edges in edge-type breakdowns (default: True)
        include_refs: Include REFERENCES edges in breakdowns (default: False to reduce noise)

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
        - module_pagerank: [{rank, module, score, role, in_edges, out_edges, symbols_count}]
        - module_centrality: [{rank, module, total_degree, coupling_pattern, symbols_count}]
        - call_flow: {module, direct_dependencies, imported_by, transitive_dependencies}

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

        # MODULE-LEVEL ANALYSIS EXAMPLES:

        # Find architecturally important modules (avoids utility function bias)
        graph(mode="module_pagerank", top_k=15)

        # Find most connected modules (hub detection)
        graph(mode="module_centrality", top_k=10)

        # Analyze call flow from orchestrator module
        graph(mode="call_flow", file="orchestrator.py")

        # Find path between two modules
        graph(mode="call_flow", file="cli.py", node="base.py")

        # Package-level analysis (use file="package" for granularity)
        graph(mode="module_pagerank", file="package", top_k=10)
    """
    try:
        # Get graph store
        graph_dir = Path(".victor/graph")
        graph_dir.mkdir(parents=True, exist_ok=True)
        graph_path = graph_dir / "graph.db"
        store = create_graph_store("sqlite", graph_path)

        # Load into analyzer
        analyzer = await _load_graph(store)

        # Lazy indexing: if graph is empty, trigger automatic indexing
        if not analyzer.nodes:
            logger.info("Graph is empty, triggering lazy indexing...")
            try:
                from victor.coding.codebase.indexer import CodebaseIndexer

                # Get project root (current working directory or from context)
                project_root = Path.cwd()
                indexer = CodebaseIndexer(project_root)

                # Check if we should do full or incremental index
                if not indexer._is_indexed:
                    logger.info("Building initial code graph index (this may take a moment)...")
                    await indexer.index_codebase()
                else:
                    # Index exists but graph is empty - rebuild graph from index
                    logger.info("Rebuilding graph from existing index...")
                    await indexer.incremental_reindex()

                # Reload the graph after indexing
                analyzer = await _load_graph(store)
                logger.info(f"Graph indexed: {len(analyzer.nodes)} nodes loaded")

            except Exception as index_err:
                logger.warning(f"Lazy indexing failed: {index_err}")
                return {
                    "error": "Graph is empty and automatic indexing failed.",
                    "details": str(index_err),
                    "hint": "Run 'victor index' manually to build the code graph.",
                }

        default_excludes = [
            "tests/",
            "test/",
            "node_modules/",
            "build/",
            "dist/",
            "out/",
            "venv/",
            ".venv/",
            "env/",
            "__pycache__",
            "htmlcov",
            "coverage",
            "vscode-victor/out/",
            "web/ui/node_modules/",
        ]
        effective_excludes = set(default_excludes)
        if exclude_paths:
            effective_excludes.update(exclude_paths)

        def _skip_path(path: Optional[str]) -> bool:
            if not path:
                return False
            if only_runtime:
                return any(excl in path for excl in effective_excludes)
            return any(excl in path for excl in effective_excludes)

        def _add_edge_counts(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            enriched: List[Dict[str, Any]] = []
            for item in items:
                nid = item.get("node_id")
                if not nid:
                    enriched.append(item)
                    continue
                in_edges = analyzer.incoming.get(nid, [])
                out_edges = analyzer.outgoing.get(nid, [])

                def edge_count(et: str, edges: list) -> int:
                    return len([e for e in edges if e[1] == et])

                item["edge_counts"] = {
                    "calls_in": edge_count("CALLS", in_edges) if include_calls else 0,
                    "calls_out": edge_count("CALLS", out_edges) if include_calls else 0,
                    "refs_in": edge_count("REFERENCES", in_edges) if include_refs else 0,
                    "refs_out": edge_count("REFERENCES", out_edges) if include_refs else 0,
                    "imports": edge_count("IMPORTS", out_edges),
                    "inherits": edge_count("INHERITS", out_edges),
                    "implements": edge_count("IMPLEMENTS", out_edges),
                    "composed_of": edge_count("COMPOSED_OF", out_edges),
                }
                enriched.append(item)
            return enriched

        def _add_callsites(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            if not include_callsites:
                return items
            enriched: List[Dict[str, Any]] = []
            for item in items:
                nid = item.get("node_id")
                if not nid:
                    enriched.append(item)
                    continue
                calls = analyzer.get_neighbors(
                    node_id=nid,
                    direction="in",
                    edge_types=["CALLS"],
                    max_depth=1,
                )
                neighbor_calls = []
                for _depth, neighs in calls.get("neighbors_by_depth", {}).items():
                    for n in neighs:
                        if _skip_path(n.get("file")):
                            continue
                        line_val = None
                        target_node = analyzer.nodes.get(n.get("node_id", ""))
                        if target_node:
                            line_val = target_node.line
                        neighbor_calls.append(
                            {"name": n.get("name"), "file": n.get("file"), "line": line_val}
                        )
                item["callsites"] = neighbor_calls[:max_callsites]
                enriched.append(item)
            return enriched

        if not analyzer.nodes:
            return {
                "error": "Graph is empty. Run 'victor index' to build the code graph first.",
                "hint": "The graph is populated during codebase indexing.",
            }

        # Resolve node name to node_id if needed
        resolved_node = None
        if node:
            # Normalize bare file paths to file: nodes
            candidate_node = node
            if not node.startswith("file:") and any(
                node.endswith(ext) for ext in (".py", ".ts", ".js", ".rs", ".go", ".java", ".cpp")
            ):
                candidate_node = f"file:{node}"

            # First try exact match on node_id
            if candidate_node in analyzer.nodes:
                resolved_node = candidate_node
            else:
                # Search by exact name
                for nid, n in analyzer.nodes.items():
                    if n.name == node:
                        resolved_node = nid
                        break

            if (
                mode
                not in ("pagerank", "centrality", "stats", "module_pagerank", "module_centrality")
                and not resolved_node
            ):
                # Enhanced fuzzy matching with multiple strategies (SOLID: Strategy Pattern)
                node_lower = node.lower()

                # Strategy 1: Substring match in name
                name_matches = [
                    (nid, n) for nid, n in analyzer.nodes.items() if node_lower in n.name.lower()
                ]

                # Strategy 2: Match against file path (e.g., "DatabaseSchema" matches database_schema.py)
                # Normalize search term: "DatabaseSchema" -> "database_schema" or "database-schema"
                normalized_search = ""
                for i, c in enumerate(node):
                    if i > 0 and c.isupper():
                        normalized_search += "_"
                    normalized_search += c.lower()

                file_matches = [
                    (nid, n)
                    for nid, n in analyzer.nodes.items()
                    if n.file
                    and (
                        normalized_search in n.file.lower()
                        or node_lower.replace("_", "") in n.file.lower().replace("_", "")
                    )
                ]

                # Strategy 3: Partial word match for CamelCase or snake_case variations
                snake_case_search = normalized_search  # Already converted above
                partial_matches = [
                    (nid, n)
                    for nid, n in analyzer.nodes.items()
                    if snake_case_search in n.name.lower().replace("_", "")
                ]

                # Combine and deduplicate (prioritize name matches)
                all_matches: Dict[str, GraphNode] = {}
                for nid, n in name_matches + file_matches + partial_matches:
                    if nid not in all_matches:
                        all_matches[nid] = n

                if all_matches:
                    # If only one match and it's a file match, auto-resolve to help LLM
                    # This handles "DatabaseSchema" -> nodes in database_schema.py
                    unique_files = set(n.file for n in all_matches.values() if n.file)
                    if len(all_matches) == 1:
                        # Single match - use it directly
                        resolved_node = list(all_matches.keys())[0]
                        logger.debug(f"[GraphTool] Auto-resolved '{node}' to '{resolved_node}'")
                    elif len(unique_files) == 1 and len(all_matches) <= 10:
                        # All matches from same file - suggest them but also provide hint
                        return {
                            "error": f"Node '{node}' not found exactly. Did you mean one of these symbols from {list(unique_files)[0]}?",
                            "suggestions": [
                                {"name": m.name, "type": m.type, "file": m.file}
                                for m in list(all_matches.values())[:10]
                            ],
                            "hint": f"Use the exact symbol name from suggestions. For file-level analysis, try: graph(mode='file_deps', file='{list(unique_files)[0]}')",
                        }
                    else:
                        return {
                            "error": f"Node '{node}' not found exactly",
                            "suggestions": [
                                {"name": m.name, "type": m.type, "file": m.file}
                                for m in list(all_matches.values())[:10]
                            ],
                        }

                if not resolved_node:
                    return {
                        "error": f"Node '{node}' not found in graph",
                        "hint": "Try using graph(mode='find', query='your_search_term') to discover available symbols, or graph(mode='file_deps', file='filename.py') for file-level analysis.",
                    }

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
            weighted_edges = edge_types
            if runtime_weighted and not edge_types:
                weighted_edges = ["CALLS", "INHERITS", "IMPLEMENTS", "COMPOSED_OF", "IMPORTS"]
            # Request more results to account for filtering, then trim to top_k
            results = analyzer.pagerank(edge_types=weighted_edges, top_k=top_k * 3)
            results = [r for r in results if not _skip_path(r.get("file"))][:top_k]
            # Re-assign ranks after filtering (1-indexed)
            for i, r in enumerate(results):
                r["rank"] = i + 1
            if structured:
                results = _add_callsites(_add_edge_counts(results))
                return {
                    "mode": "pagerank",
                    "modules": [] if include_modules else [],
                    "symbols": results if include_symbols else [],
                }
            return {
                "mode": "pagerank",
                "results": results,
            }

        elif mode == "centrality":
            weighted_edges = edge_types
            if runtime_weighted and not edge_types:
                weighted_edges = ["CALLS", "INHERITS", "IMPLEMENTS", "COMPOSED_OF", "IMPORTS"]
            # Request more results to account for filtering, then trim to top_k
            results = analyzer.degree_centrality(edge_types=weighted_edges, top_k=top_k * 3)
            results = [r for r in results if not _skip_path(r.get("file"))][:top_k]
            # Re-assign ranks after filtering (1-indexed)
            for i, r in enumerate(results):
                r["rank"] = i + 1
            if structured:
                results = _add_callsites(_add_edge_counts(results))
                return {
                    "mode": "centrality",
                    "modules": [] if include_modules else [],
                    "symbols": results if include_symbols else [],
                }
            return {
                "mode": "centrality",
                "results": results,
            }

        elif mode == "path":
            if not resolved_node or not resolved_target:
                return {"error": "Both 'node' and 'target' required for 'path' mode"}
            return analyzer.shortest_path(resolved_node, resolved_target, edge_types, depth)

        elif mode == "impact":
            if not resolved_node:
                return {"error": "node parameter required for 'impact' mode"}
            impact = analyzer.impact_analysis(resolved_node, edge_types, depth)
            if include_callsites and isinstance(impact, dict):
                # Gather incoming CALLS for the node to provide callsites
                calls = analyzer.get_neighbors(
                    node_id=resolved_node,
                    direction="in",
                    edge_types=["CALLS"],
                    max_depth=1,
                )
                neighbor_calls = []
                for depth, neighs in calls.get("neighbors_by_depth", {}).items():
                    for n in neighs:
                        if _skip_path(n.get("file")):
                            continue
                        neighbor_calls.append({"name": n.get("name"), "file": n.get("file")})
                impact["callsites"] = neighbor_calls[:max_callsites]
                if include_neighbors:
                    impact["neighbors"] = {
                        "edge_types": neighbors_edge_types or ["CALLS"],
                        "limit": neighbors_limit,
                    }
            return impact

        elif mode == "subgraph":
            if not resolved_node:
                return {"error": "node parameter required for 'subgraph' mode"}
            return analyzer.extract_subgraph(resolved_node, depth, edge_types)

        elif mode == "stats":
            return analyzer.get_stats()

        elif mode == "find":
            if not query:
                return {"error": "query parameter required for 'find' mode"}
            found = analyzer.find_symbols(
                query=query,
                node_type=node_type,
                file_pattern=file,
                expand_neighbors=expand,
                edge_types=edge_types,
                limit=top_k,
            )
            if isinstance(found, dict) and "matches" in found:
                filtered = []
                for m in found["matches"]:
                    if _skip_path(m.get("file")):
                        continue
                    if include_callsites and m.get("name"):
                        # Add nearby callsites if available via neighbors
                        calls = analyzer.get_neighbors(
                            node_id=m["node_id"] if "node_id" in m else m.get("name", ""),
                            direction="in",
                            edge_types=["CALLS"],
                            max_depth=1,
                        )
                        neighbor_calls = []
                        for depth, neighs in calls.get("neighbors_by_depth", {}).items():
                            for n in neighs:
                                if _skip_path(n.get("file")):
                                    continue
                                neighbor_calls.append(
                                    {"name": n.get("name"), "file": n.get("file")}
                                )
                        m["callsites"] = neighbor_calls[:max_callsites]
                    filtered.append(m)
                found["matches"] = filtered
            return found

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

        # Module-level analysis modes
        elif mode == "module_pagerank":
            # Use granularity parameter, but also support legacy file="package" approach
            effective_granularity: Literal["file", "package"] = granularity
            if file and file in ("package", "packages", "directory", "dir"):
                effective_granularity = "package"
            weighted_edges = edge_types
            if runtime_weighted and not edge_types:
                weighted_edges = ["CALLS", "INHERITS", "IMPLEMENTS", "COMPOSED_OF", "IMPORTS"]
            module_results = []
            module_to_nodes, outgoing_modules, incoming_modules = analyzer._build_module_graph(
                effective_granularity, weighted_edges
            )
            # Build node->module map
            node_to_module: Dict[str, str] = {}
            for mod, nodes in module_to_nodes.items():
                for nid in nodes:
                    node_to_module[nid] = mod

            # Aggregate edge counts per module by edge type
            module_edge_counts: Dict[str, Dict[str, int]] = defaultdict(
                lambda: {
                    "calls_in": 0,
                    "calls_out": 0,
                    "refs_in": 0,
                    "refs_out": 0,
                    "imports": 0,
                    "inherits": 0,
                    "implements": 0,
                    "composed_of": 0,
                }
            )
            for src_id, edges in analyzer.outgoing.items():
                src_mod = node_to_module.get(src_id)
                if not src_mod or _skip_path(src_mod):
                    continue
                for dst_id, etype, _w in edges:
                    dst_mod = node_to_module.get(dst_id)
                    if not dst_mod or src_mod == dst_mod or _skip_path(dst_mod):
                        continue
                    if etype == "CALLS" and include_calls:
                        module_edge_counts[src_mod]["calls_out"] += 1
                        module_edge_counts[dst_mod]["calls_in"] += 1
                    elif etype == "REFERENCES" and include_refs:
                        module_edge_counts[src_mod]["refs_out"] += 1
                        module_edge_counts[dst_mod]["refs_in"] += 1
                    elif etype == "IMPORTS":
                        module_edge_counts[src_mod]["imports"] += 1
                    elif etype == "INHERITS":
                        module_edge_counts[src_mod]["inherits"] += 1
                    elif etype == "IMPLEMENTS":
                        module_edge_counts[src_mod]["implements"] += 1
                    elif etype == "COMPOSED_OF":
                        module_edge_counts[src_mod]["composed_of"] += 1
            for r in analyzer.module_pagerank(
                granularity=effective_granularity,
                edge_types=weighted_edges,
                top_k=top_k,
            ):
                module_name = r.get("module")
                if _skip_path(module_name):
                    continue
                _in_edges = sum(incoming_modules.get(module_name, {}).values())  # noqa: F841
                out_edges = sum(outgoing_modules.get(module_name, {}).values())
                # Merge aggregated counts with aggregate totals
                edge_counts = module_edge_counts.get(module_name, {}).copy()
                edge_counts["imports"] = out_edges  # keep imports total
                edge_counts.setdefault("calls_in", 0)
                edge_counts.setdefault("calls_out", 0)
                edge_counts.setdefault("refs_in", 0)
                edge_counts.setdefault("refs_out", 0)
                edge_counts.setdefault("inherits", 0)
                edge_counts.setdefault("implements", 0)
                edge_counts.setdefault("composed_of", 0)
                # Collect callsites (representative nodes in module)
                callsites = []
                if include_callsites_modules and module_name in module_to_nodes:
                    for node_id in list(module_to_nodes[module_name])[
                        : int(max_callsites_modules) * 2
                    ]:
                        for src, et, _w in analyzer.incoming.get(node_id, []):
                            if et == "CALLS":
                                src_node = analyzer.nodes.get(src)
                                if src_node and not _skip_path(src_node.file):
                                    callsites.append({"name": src_node.name, "file": src_node.file})
                        if len(callsites) >= int(max_callsites_modules):
                            break

                r["edge_counts"] = edge_counts
                if callsites:
                    r["callsites"] = callsites[:max_callsites_modules]
                module_results.append(r)
            if structured:
                return {
                    "mode": "module_pagerank",
                    "granularity": effective_granularity,
                    "description": "Architectural importance at module level (avoids utility function bias)",
                    "modules": module_results if include_modules else [],
                    "symbols": [] if include_symbols else [],
                }
            return {
                "mode": "module_pagerank",
                "granularity": effective_granularity,
                "description": "Architectural importance at module level (avoids utility function bias)",
                "results": module_results,
            }

        elif mode == "module_centrality":
            # Use granularity parameter, but also support legacy file="package" approach
            effective_granularity_c: Literal["file", "package"] = granularity
            if file and file in ("package", "packages", "directory", "dir"):
                effective_granularity_c = "package"
            weighted_edges = edge_types
            if runtime_weighted and not edge_types:
                weighted_edges = ["CALLS", "INHERITS", "IMPLEMENTS", "COMPOSED_OF", "IMPORTS"]
            module_results = []
            module_to_nodes, outgoing_modules, incoming_modules = analyzer._build_module_graph(
                effective_granularity_c, weighted_edges
            )
            node_to_module: Dict[str, str] = {}
            for mod, nodes in module_to_nodes.items():
                for nid in nodes:
                    node_to_module[nid] = mod
            module_edge_counts: Dict[str, Dict[str, int]] = defaultdict(
                lambda: {
                    "calls_in": 0,
                    "calls_out": 0,
                    "refs_in": 0,
                    "refs_out": 0,
                    "imports": 0,
                    "inherits": 0,
                    "implements": 0,
                    "composed_of": 0,
                }
            )
            for src_id, edges in analyzer.outgoing.items():
                src_mod = node_to_module.get(src_id)
                if not src_mod or _skip_path(src_mod):
                    continue
                for dst_id, etype, _w in edges:
                    dst_mod = node_to_module.get(dst_id)
                    if not dst_mod or src_mod == dst_mod or _skip_path(dst_mod):
                        continue
                    if etype == "CALLS" and include_calls:
                        module_edge_counts[src_mod]["calls_out"] += 1
                        module_edge_counts[dst_mod]["calls_in"] += 1
                    elif etype == "REFERENCES" and include_refs:
                        module_edge_counts[src_mod]["refs_out"] += 1
                        module_edge_counts[dst_mod]["refs_in"] += 1
                    elif etype == "IMPORTS":
                        module_edge_counts[src_mod]["imports"] += 1
                    elif etype == "INHERITS":
                        module_edge_counts[src_mod]["inherits"] += 1
                    elif etype == "IMPLEMENTS":
                        module_edge_counts[src_mod]["implements"] += 1
                    elif etype == "COMPOSED_OF":
                        module_edge_counts[src_mod]["composed_of"] += 1
            for r in analyzer.module_centrality(
                granularity=effective_granularity_c,
                edge_types=weighted_edges,
                top_k=top_k,
            ):
                module_name = r.get("module")
                if _skip_path(module_name):
                    continue
                _in_edges = sum(incoming_modules.get(module_name, {}).values())  # noqa: F841
                out_edges = sum(outgoing_modules.get(module_name, {}).values())
                edge_counts = module_edge_counts.get(module_name, {}).copy()
                edge_counts["imports"] = out_edges
                edge_counts.setdefault("calls_in", 0)
                edge_counts.setdefault("calls_out", 0)
                edge_counts.setdefault("refs_in", 0)
                edge_counts.setdefault("refs_out", 0)
                edge_counts.setdefault("inherits", 0)
                edge_counts.setdefault("implements", 0)
                edge_counts.setdefault("composed_of", 0)
                callsites = []
                if include_callsites_modules and module_name in module_to_nodes:
                    for node_id in list(module_to_nodes[module_name])[
                        : int(max_callsites_modules) * 2
                    ]:
                        for src, et, _w in analyzer.incoming.get(node_id, []):
                            if et == "CALLS":
                                src_node = analyzer.nodes.get(src)
                                if src_node and not _skip_path(src_node.file):
                                    callsites.append(
                                        {
                                            "name": src_node.name,
                                            "file": src_node.file,
                                            "line": src_node.line,
                                        }
                                    )
                        if len(callsites) >= int(max_callsites_modules):
                            break
                r["edge_counts"] = edge_counts
                if callsites:
                    r["callsites"] = callsites[:max_callsites_modules]
                module_results.append(r)
            if structured:
                return {
                    "mode": "module_centrality",
                    "granularity": effective_granularity_c,
                    "description": "Most connected modules (hub detection)",
                    "modules": module_results if include_modules else [],
                    "symbols": [] if include_symbols else [],
                }
            return {
                "mode": "module_centrality",
                "granularity": effective_granularity_c,
                "description": "Most connected modules (hub detection)",
                "results": module_results,
            }

        elif mode == "call_flow":
            source_module = file or source
            if not source_module:
                return {"error": "file parameter required for 'call_flow' mode (source module)"}
            # Use node as target if provided
            return analyzer.call_flow(
                source_module=source_module,
                target_module=node,
                granularity=granularity,
                edge_types=edge_types,
                depth=depth,
            )

        else:
            return {"error": f"Unknown mode: {mode}"}

    except Exception as e:
        logger.exception(f"Graph tool error: {e}")
        return {"error": str(e)}
