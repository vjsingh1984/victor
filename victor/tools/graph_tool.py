from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Literal, Optional, Set

from victor.config.settings import get_project_paths, load_settings
from victor.native.python.graph_algo import connected_components, pagerank, weighted_pagerank
from victor.storage.graph.protocol import GraphEdge, GraphNode, GraphStoreProtocol
from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority
from victor.tools.code_search_tool import _get_or_build_index
from victor.tools.context import ToolExecutionContext
from victor.tools.decorators import tool

logger = logging.getLogger(__name__)

# =============================================================================
# Enums for Graph Operations
# =============================================================================


class GraphDirection(str, Enum):
    """Direction for graph traversal.

    Determines which edges to follow:
    - OUT: Follow outgoing edges (forward)
    - IN: Follow incoming edges (backward)
    - BOTH: Follow both directions (undirected)
    """

    OUT = "out"  # Follow outgoing edges
    IN = "in"  # Follow incoming edges
    BOTH = "both"  # Follow both directions


class GraphMode(str, Enum):
    """Mode for graph analysis operations.

    Determines what type of analysis to perform:
    - find: Find nodes by query
    - neighbors: Get neighboring nodes
    - pagerank: Calculate PageRank scores
    - centrality: Calculate centrality measures
    - path: Find paths between nodes
    - impact: Analyze impact/dependencies
    - clusters: Find connected components
    - stats: Calculate graph statistics
    - subgraph: Extract subgraph
    - file_deps: File dependency analysis
    - patterns: Find code patterns
    - module_pagerank: Module-level PageRank
    - module_centrality: Module-level centrality
    - call_flow: Call flow analysis
    - callers: Find callers of a node
    - callees: Find callees of a node
    - trace: Trace execution paths
    """

    FIND = "find"  # Find nodes by query
    NEIGHBORS = "neighbors"  # Get neighboring nodes
    PAGERANK = "pagerank"  # Calculate PageRank scores
    CENTRALITY = "centrality"  # Calculate centrality measures
    PATH = "path"  # Find paths between nodes
    IMPACT = "impact"  # Analyze impact/dependencies
    CLUSTERS = "clusters"  # Find connected components
    STATS = "stats"  # Calculate graph statistics
    SUBGRAPH = "subgraph"  # Extract subgraph
    FILE_DEPS = "file_deps"  # File dependency analysis
    PATTERNS = "patterns"  # Find code patterns
    MODULE_PAGERANK = "module_pagerank"  # Module-level PageRank
    MODULE_CENTRALITY = "module_centrality"  # Module-level centrality
    CALL_FLOW = "call_flow"  # Call flow analysis
    CALLERS = "callers"  # Find callers of a node
    CALLEES = "callees"  # Find callees of a node
    TRACE = "trace"  # Trace execution paths


ALL_EDGE_TYPES = [
    "CALLS",
    "REFERENCES",
    "CONTAINS",
    "INHERITS",
    "IMPLEMENTS",
    "COMPOSED_OF",
    "IMPORTS",
]

_RUNTIME_EDGE_TYPES = {"CALLS", "REFERENCES", "INHERITS", "IMPLEMENTS", "COMPOSED_OF"}
_SYMBOL_TYPES = {
    "function",
    "method",
    "class",
    "interface",
    "struct",
    "trait",
    "enum",
    "module",
}


def _ctx_value(exec_ctx: Optional[Dict[str, Any]], key: str, default: Any = None) -> Any:
    if exec_ctx is None:
        return default
    if isinstance(exec_ctx, ToolExecutionContext):
        return getattr(exec_ctx, key, default)
    return exec_ctx.get(key, default)


def _normalize_relpath(file_path: str) -> str:
    return file_path.replace("\\", "/").strip()


def _module_name_from_file(file_path: str) -> str:
    normalized = _normalize_relpath(file_path)
    path = Path(normalized)
    if path.name == "__init__.py":
        parts = path.parent.parts
    else:
        parts = path.with_suffix("").parts
    return ".".join(part for part in parts if part and part != ".")


def _module_name_for_node(node: GraphNode) -> Optional[str]:
    if node.type == "module":
        return node.name
    if node.type == "stdlib_module":
        return None
    if node.file:
        return _module_name_from_file(node.file)
    return None


def _node_payload(node: GraphNode, **extra: Any) -> Dict[str, Any]:
    payload = {
        "node_id": node.node_id,
        "name": node.name,
        "type": node.type,
        "file": node.file,
        "line": node.line,
        "end_line": node.end_line,
        "lang": node.lang,
    }
    payload.update(extra)
    return payload


@dataclass
class LoadedGraph:
    root_path: Path
    index: Any
    graph_store: GraphStoreProtocol
    analyzer: "GraphAnalyzer"
    rebuilt: bool


class GraphAnalyzer:
    """In-memory graph analytics helper backed by GraphNode/GraphEdge objects."""

    def __init__(self) -> None:
        self.nodes: Dict[str, GraphNode] = {}
        self.outgoing: DefaultDict[str, List[GraphEdge]] = defaultdict(list)
        self.incoming: DefaultDict[str, List[GraphEdge]] = defaultdict(list)
        self._name_index: DefaultDict[str, List[str]] = defaultdict(list)
        self._file_index: DefaultDict[str, List[str]] = defaultdict(list)

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.node_id] = node
        self._name_index[node.name.lower()].append(node.node_id)
        self._file_index[_normalize_relpath(node.file)].append(node.node_id)

    def add_edge(self, edge: GraphEdge) -> None:
        self.outgoing[edge.src].append(edge)
        self.incoming[edge.dst].append(edge)

    def resolve_node_id(
        self,
        reference: str,
        *,
        preferred_types: Optional[Set[str]] = None,
    ) -> Optional[str]:
        if reference in self.nodes:
            return reference

        normalized = reference.strip().strip("`'\"")
        lowered = normalized.lower()
        candidates = list(self._name_index.get(lowered, ()))

        normalized_file = _normalize_relpath(normalized)
        if not candidates:
            candidates.extend(self._file_index.get(normalized_file, ()))
            if not candidates:
                for file_path, node_ids in self._file_index.items():
                    if file_path.endswith(normalized_file):
                        candidates.extend(node_ids)

        if not candidates:
            for node in self.nodes.values():
                if lowered in node.name.lower() or normalized_file in _normalize_relpath(node.file):
                    candidates.append(node.node_id)

        if not candidates:
            return None

        def _score(node_id: str) -> tuple[int, int, str]:
            node = self.nodes[node_id]
            preferred = 1 if preferred_types and node.type in preferred_types else 0
            exact_name = 1 if node.name.lower() == lowered else 0
            return (-preferred, -exact_name, node.node_id)

        candidates = sorted(dict.fromkeys(candidates), key=_score)
        return candidates[0]

    def search(
        self,
        query: str,
        *,
        node_types: Optional[Set[str]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        lowered = query.lower()
        results: List[Dict[str, Any]] = []
        for node in self.nodes.values():
            if node_types is not None and node.type not in node_types:
                continue
            score = 0
            if node.name.lower() == lowered:
                score += 5
            elif lowered in node.name.lower():
                score += 3
            if lowered in _normalize_relpath(node.file):
                score += 2
            if score == 0:
                continue
            results.append(_node_payload(node, score=score))
        results.sort(key=lambda item: (-item["score"], item["file"], item["line"] or 0))
        return results[:limit]

    def _iter_edges(
        self,
        node_id: str,
        *,
        direction: GraphDirection,
        edge_types: Optional[Set[str]],
    ) -> Iterable[tuple[GraphEdge, str, GraphDirection]]:
        if direction in {"out", "both"}:
            for edge in self.outgoing.get(node_id, ()):
                if edge_types is None or edge.type in edge_types:
                    yield edge, edge.dst, "out"
        if direction in {"in", "both"}:
            for edge in self.incoming.get(node_id, ()):
                if edge_types is None or edge.type in edge_types:
                    yield edge, edge.src, "in"

    def get_neighbors(
        self,
        node_id: str,
        *,
        direction: GraphDirection = GraphDirection.OUT,
        edge_types: Optional[Iterable[str]] = None,
        max_depth: int = 1,
        node_types: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        if node_id not in self.nodes:
            return {
                "source": node_id,
                "direction": direction,
                "max_depth": max_depth,
                "total_neighbors": 0,
                "neighbors_by_depth": {},
                "message": f"Node '{node_id}' not found",
            }

        allowed_edges = set(edge_types) if edge_types else None
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        visited = {node_id}
        neighbors_by_depth: Dict[int, List[Dict[str, Any]]] = {}

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue

            for edge, neighbor_id, edge_direction in self._iter_edges(
                current,
                direction=direction,
                edge_types=allowed_edges,
            ):
                if neighbor_id in visited:
                    continue
                neighbor = self.nodes.get(neighbor_id)
                if neighbor is None:
                    continue
                if node_types is not None and neighbor.type not in node_types:
                    continue
                visited.add(neighbor_id)
                queue.append((neighbor_id, depth + 1))
                neighbors_by_depth.setdefault(depth + 1, []).append(
                    _node_payload(
                        neighbor,
                        edge_type=edge.type,
                        direction=edge_direction,
                        via=current,
                    )
                )

        return {
            "source": node_id,
            "direction": direction,
            "max_depth": max_depth,
            "total_neighbors": sum(len(items) for items in neighbors_by_depth.values()),
            "neighbors_by_depth": neighbors_by_depth,
        }

    def pagerank(
        self,
        *,
        edge_types: Optional[Iterable[str]] = None,
        iterations: int = 20,
        top_k: int = 10,
        node_types: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.nodes:
            return []

        allowed_edges = set(edge_types) if edge_types else None
        allowed_nodes = {
            node_id
            for node_id, node in self.nodes.items()
            if node_types is None or node.type in node_types
        }
        if not allowed_nodes:
            return []

        adjacency = {node_id: [] for node_id in allowed_nodes}
        for node_id in allowed_nodes:
            for edge in self.outgoing.get(node_id, ()):
                if allowed_edges is not None and edge.type not in allowed_edges:
                    continue
                if edge.dst in allowed_nodes:
                    adjacency[node_id].append(edge.dst)

        scores = pagerank(adjacency, iterations=iterations)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        results: List[Dict[str, Any]] = []
        for rank, (node_id, score) in enumerate(ranked, start=1):
            node = self.nodes[node_id]
            results.append(_node_payload(node, rank=rank, score=score))
        return results

    def degree_centrality(
        self,
        *,
        top_k: int = 10,
        edge_types: Optional[Iterable[str]] = None,
        node_types: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.nodes:
            return []

        allowed_edges = set(edge_types) if edge_types else None
        allowed_nodes = {
            node_id
            for node_id, node in self.nodes.items()
            if node_types is None or node.type in node_types
        }
        if not allowed_nodes:
            return []

        results: List[Dict[str, Any]] = []
        for node_id in allowed_nodes:
            node = self.nodes[node_id]
            out_degree = sum(
                1
                for edge in self.outgoing.get(node_id, ())
                if (allowed_edges is None or edge.type in allowed_edges)
                and edge.dst in allowed_nodes
            )
            in_degree = sum(
                1
                for edge in self.incoming.get(node_id, ())
                if (allowed_edges is None or edge.type in allowed_edges)
                and edge.src in allowed_nodes
            )
            results.append(
                _node_payload(
                    node,
                    degree=in_degree + out_degree,
                    in_degree=in_degree,
                    out_degree=out_degree,
                )
            )

        results.sort(key=lambda item: (-item["degree"], item["node_id"]))
        for rank, item in enumerate(results[:top_k], start=1):
            item["rank"] = rank
        return results[:top_k]

    def shortest_path(
        self,
        source: str,
        target: str,
        *,
        edge_types: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        if source not in self.nodes or target not in self.nodes:
            return {
                "found": False,
                "source": source,
                "target": target,
                "message": "Source or target node not found",
            }

        allowed_edges = set(edge_types) if edge_types else None
        queue: deque[str] = deque([source])
        previous: Dict[str, Optional[str]] = {source: None}

        while queue:
            current = queue.popleft()
            if current == target:
                break
            for edge in self.outgoing.get(current, ()):
                if allowed_edges is not None and edge.type not in allowed_edges:
                    continue
                if edge.dst in previous:
                    continue
                previous[edge.dst] = current
                queue.append(edge.dst)

        if target not in previous:
            return {
                "found": False,
                "source": source,
                "target": target,
                "message": f"No path from '{source}' to '{target}'",
            }

        path_ids: List[str] = []
        current: Optional[str] = target
        while current is not None:
            path_ids.append(current)
            current = previous[current]
        path_ids.reverse()

        return {
            "found": True,
            "source": source,
            "target": target,
            "path": [_node_payload(self.nodes[node_id]) for node_id in path_ids],
        }


def _resolve_root_path(path: str) -> Path:
    search_root = path
    if not search_root or search_root == ".":
        search_root = str(get_project_paths().project_root)
    root_path = Path(search_root).resolve()
    if root_path.is_file():
        return root_path.parent
    return root_path


async def _load_graph(
    path: str = ".",
    *,
    reindex: bool = False,
    exec_ctx: Optional[Dict[str, Any]] = None,
) -> LoadedGraph:
    root_path = _resolve_root_path(path)
    settings = _ctx_value(exec_ctx, "settings")
    if settings is None:
        settings = load_settings()

    index, rebuilt = await _get_or_build_index(
        root_path,
        settings,
        force_reindex=reindex,
        exec_ctx=exec_ctx,
    )
    graph_store = getattr(index, "graph_store", None)
    if graph_store is None:
        raise ValueError("Index does not provide graph_store support")

    get_all_nodes = getattr(graph_store, "get_all_nodes", None)
    if callable(get_all_nodes):
        nodes = await get_all_nodes()
    else:
        nodes = await graph_store.find_nodes()
    edges = await graph_store.get_all_edges()

    analyzer = GraphAnalyzer()
    for node in nodes:
        analyzer.add_node(node)
    for edge in edges:
        analyzer.add_edge(edge)

    return LoadedGraph(
        root_path=root_path,
        index=index,
        graph_store=graph_store,
        analyzer=analyzer,
        rebuilt=rebuilt,
    )


def _default_edge_types(mode: GraphMode, *, only_runtime: bool) -> Optional[List[str]]:
    if mode in {"callers", "callees", "trace", "call_flow"}:
        return ["CALLS"]
    if only_runtime:
        return sorted(_RUNTIME_EDGE_TYPES)
    return None


def _node_type_filter(
    mode: GraphMode,
    *,
    files_only: bool,
    modules_only: bool,
) -> Optional[Set[str]]:
    if files_only:
        return {"file"}
    if modules_only:
        return {"module"}
    if mode in {"pagerank", "centrality"}:
        return set(_SYMBOL_TYPES)
    return None


def _flatten_neighbors(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for neighbors in result.get("neighbors_by_depth", {}).values():
        items.extend(neighbors)
    return items


def _build_structured_neighbors(
    analyzer: GraphAnalyzer,
    focus_id: str,
    base_result: Dict[str, Any],
    *,
    include_modules: bool,
    include_symbols: bool,
    include_calls: bool,
    include_refs: bool,
    include_callsites: bool,
    max_callsites: int,
) -> Dict[str, Any]:
    focus = analyzer.nodes[focus_id]
    flattened = _flatten_neighbors(base_result)

    structured = {
        "focus": _node_payload(focus),
        "neighborhood": base_result,
    }
    if include_modules:
        modules = {
            module_name
            for item in [structured["focus"], *flattened]
            for module_name in [_module_name_from_file(item["file"])]
            if module_name
        }
        structured["modules"] = sorted(modules)
    if include_symbols:
        structured["symbols"] = [
            item for item in flattened if item["type"] not in {"file", "module", "stdlib_module"}
        ]
    if include_calls or include_callsites:
        structured["calls"] = [item for item in flattened if item["edge_type"] == "CALLS"][
            :max_callsites
        ]
    if include_refs:
        structured["references"] = [
            item for item in flattened if item["edge_type"] == "REFERENCES"
        ][:max_callsites]
    return structured


def _project_module_adjacency(
    analyzer: GraphAnalyzer,
    *,
    only_runtime: bool,
    include_callsites: bool,
    max_callsites: int,
) -> Dict[str, Any]:
    adjacency: Dict[str, Dict[str, int]] = {}
    callsites: Dict[tuple[str, str], List[Dict[str, Any]]] = {}

    for node in analyzer.nodes.values():
        module_name = _module_name_for_node(node)
        if module_name:
            adjacency.setdefault(module_name, {})

    for edges in analyzer.outgoing.values():
        for edge in edges:
            if only_runtime and edge.type not in _RUNTIME_EDGE_TYPES:
                continue
            src_node = analyzer.nodes.get(edge.src)
            dst_node = analyzer.nodes.get(edge.dst)
            if src_node is None or dst_node is None:
                continue
            src_module = _module_name_for_node(src_node)
            dst_module = _module_name_for_node(dst_node)
            if not src_module or not dst_module or src_module == dst_module:
                continue
            adjacency.setdefault(src_module, {})
            adjacency.setdefault(dst_module, {})
            adjacency[src_module][dst_module] = adjacency[src_module].get(dst_module, 0) + 1
            if include_callsites:
                callsites.setdefault((src_module, dst_module), []).append(
                    {
                        "src_node": edge.src,
                        "dst_node": edge.dst,
                        "edge_type": edge.type,
                        "src_file": src_node.file,
                        "dst_file": dst_node.file,
                    }
                )

    projected = {"adjacency": adjacency}
    if include_callsites:
        projected["callsites"] = {
            f"{src}->{dst}": samples[:max_callsites] for (src, dst), samples in callsites.items()
        }
    return projected


def _rank_projected_modules(
    adjacency: Dict[str, Dict[str, int]],
    *,
    mode: Literal["pagerank", "centrality"],
    top_k: int,
) -> List[Dict[str, Any]]:
    if not adjacency:
        return []

    if mode == "pagerank":
        scores = weighted_pagerank(adjacency, iterations=25)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [
            {"rank": rank, "module": module, "score": score}
            for rank, (module, score) in enumerate(ranked, start=1)
        ]

    degree_rows: List[Dict[str, Any]] = []
    incoming: Dict[str, int] = dict.fromkeys(adjacency, 0)
    for src, neighbors in adjacency.items():
        for dst, weight in neighbors.items():
            incoming[dst] = incoming.get(dst, 0) + weight
            incoming.setdefault(src, incoming.get(src, 0))

    for module, neighbors in adjacency.items():
        out_degree = sum(neighbors.values())
        in_degree = incoming.get(module, 0)
        degree_rows.append(
            {
                "module": module,
                "degree": in_degree + out_degree,
                "in_degree": in_degree,
                "out_degree": out_degree,
            }
        )

    degree_rows.sort(key=lambda item: (-item["degree"], item["module"]))
    for rank, row in enumerate(degree_rows[:top_k], start=1):
        row["rank"] = rank
    return degree_rows[:top_k]


def _resolve_file_metadata(index: Any, file_path: str) -> tuple[str, Optional[Any]]:
    normalized = _normalize_relpath(file_path)
    files = getattr(index, "files", None) or {}
    if normalized in files:
        return normalized, files[normalized]

    for candidate, metadata in files.items():
        candidate_norm = _normalize_relpath(candidate)
        if candidate_norm == normalized or candidate_norm.endswith(normalized):
            return candidate_norm, metadata
    return normalized, None


def _build_file_dependency_result(
    loaded: LoadedGraph,
    file_path: str,
    *,
    direction: GraphDirection,
    structured: bool,
    include_modules: bool,
) -> Dict[str, Any]:
    normalized, metadata = _resolve_file_metadata(loaded.index, file_path)
    dependencies: List[str] = []
    dependents: List[str] = []

    if metadata is not None:
        dependencies = sorted(dict.fromkeys(getattr(metadata, "dependencies", []) or []))
        files = getattr(loaded.index, "files", None) or {}
        dependents = sorted(
            candidate
            for candidate, candidate_meta in files.items()
            if normalized in (getattr(candidate_meta, "dependencies", []) or [])
        )
    else:
        file_node_id = loaded.analyzer.resolve_node_id(normalized, preferred_types={"file"})
        if file_node_id is not None:
            file_neighbors = loaded.analyzer.get_neighbors(
                file_node_id,
                direction="out",
                edge_types=["IMPORTS"],
                max_depth=1,
                node_types={"module"},
            )
            dependencies = sorted(item["name"] for item in _flatten_neighbors(file_neighbors))

    result: Dict[str, Any] = {"file": normalized}
    if direction in {"out", "both"}:
        result["dependencies"] = dependencies
    if direction in {"in", "both"}:
        result["dependents"] = dependents
    if structured:
        summary = {"focus_module": _module_name_from_file(normalized)}
        if include_modules:
            summary["modules"] = sorted(
                {
                    _module_name_from_file(normalized),
                    *[
                        _module_name_from_file(dep)
                        for dep in dependencies
                        if "." in dep or "/" in dep
                    ],
                    *[
                        _module_name_from_file(dep)
                        for dep in dependents
                        if "." in dep or "/" in dep
                    ],
                }
            )
        result["summary"] = summary
    return result


def _build_stats(loaded: LoadedGraph) -> Dict[str, Any]:
    node_types: Dict[str, int] = {}
    edge_types: Dict[str, int] = {}
    for node in loaded.analyzer.nodes.values():
        node_types[node.type] = node_types.get(node.type, 0) + 1
    for edges in loaded.analyzer.outgoing.values():
        for edge in edges:
            edge_types[edge.type] = edge_types.get(edge.type, 0) + 1

    return {
        "nodes": len(loaded.analyzer.nodes),
        "edges": sum(len(edges) for edges in loaded.analyzer.outgoing.values()),
        "node_types": node_types,
        "edge_types": edge_types,
        "root_path": str(loaded.root_path),
        "rebuilt": loaded.rebuilt,
    }


@tool(
    category="search",
    priority=Priority.HIGH,
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    execution_category=ExecutionCategory.READ_ONLY,
    keywords=[
        "graph",
        "callers",
        "callees",
        "trace",
        "dependencies",
        "impact",
        "pagerank",
        "centrality",
        "architecture",
        "neighbors",
    ],
    aliases=["graph_tool"],
    timeout=60.0,
)
async def graph(
    mode: GraphMode = GraphMode.NEIGHBORS,
    path: str = ".",
    node: Optional[str] = None,
    source: Optional[str] = None,
    target: Optional[str] = None,
    file: Optional[str] = None,
    query: Optional[str] = None,
    depth: int = 2,
    top_k: int = 10,
    direction: GraphDirection = "out",
    edge_types: Optional[List[str]] = None,
    reindex: bool = False,
    only_runtime: bool = False,
    files_only: bool = False,
    modules_only: bool = False,
    structured: bool = False,
    include_modules: bool = False,
    include_symbols: bool = False,
    include_calls: bool = False,
    include_refs: bool = False,
    include_callsites: bool = False,
    max_callsites: int = 3,
    _exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Analyze the indexed code graph using the active SQLite + LanceDB code index.

    Enhanced with file watching for automatic cache invalidation.
    """
    # Subscribe to file watcher for automatic cache invalidation (only once per root)
    from pathlib import Path as PathlibPath
    from victor.core.indexing.file_watcher import FileWatcherRegistry
    from victor.core.indexing.graph_manager import GraphManager

    root_path = _resolve_root_path(path)

    # Subscribe GraphManager to file watcher for this root
    if not reindex:
        try:
            manager = GraphManager.get_instance()
            await manager._ensure_file_watcher(root_path, _exec_ctx)
        except Exception as e:
            # Non-fatal: log warning but continue
            logger.warning(f"[graph] Failed to subscribe to file watcher: {e}")

    try:
        loaded = await _load_graph(path, reindex=reindex, exec_ctx=_exec_ctx)
        default_edge_types = _default_edge_types(mode, only_runtime=only_runtime)
        effective_edge_types = edge_types or default_edge_types
        node_types = _node_type_filter(mode, files_only=files_only, modules_only=modules_only)

        if mode == "stats":
            result = _build_stats(loaded)
        elif mode == "search":
            # Alias for 'find' mode - more intuitive for users
            search_query = query or node or file
            if not search_query:
                raise ValueError("search mode requires query, node, or file")
            result = {
                "matches": loaded.analyzer.search(search_query, node_types=node_types, limit=top_k)
            }
        elif mode == "find":
            search_query = query or node or file
            if not search_query:
                raise ValueError("find mode requires query, node, or file")
            result = {
                "matches": loaded.analyzer.search(search_query, node_types=node_types, limit=top_k)
            }
        elif mode in {
            "callers",
            "callees",
            "trace",
            "call_flow",
            "neighbors",
            "impact",
            "subgraph",
        }:
            target_ref = node or source
            if not target_ref:
                raise ValueError(f"{mode} mode requires node")
            preferred_types = (
                {"file"} if files_only else {"module"} if modules_only else _SYMBOL_TYPES
            )
            resolved_id = loaded.analyzer.resolve_node_id(
                target_ref, preferred_types=preferred_types
            )
            if resolved_id is None:
                raise ValueError(f"Could not resolve graph node '{target_ref}'")

            effective_direction = direction
            if mode == "callers":
                effective_direction = "in"
            elif mode in {"callees", "trace", "call_flow"}:
                effective_direction = "out"
            elif mode in {"impact", "subgraph"}:
                effective_direction = "both"

            base_result = loaded.analyzer.get_neighbors(
                resolved_id,
                direction=effective_direction,
                edge_types=effective_edge_types,
                max_depth=max(1, depth),
                node_types=node_types,
            )
            if structured:
                result = _build_structured_neighbors(
                    loaded.analyzer,
                    resolved_id,
                    base_result,
                    include_modules=include_modules,
                    include_symbols=include_symbols,
                    include_calls=include_calls,
                    include_refs=include_refs,
                    include_callsites=include_callsites,
                    max_callsites=max_callsites,
                )
            else:
                result = base_result
        elif mode == "path":
            if not source or not target:
                raise ValueError("path mode requires source and target")
            resolved_source = loaded.analyzer.resolve_node_id(source, preferred_types=_SYMBOL_TYPES)
            resolved_target = loaded.analyzer.resolve_node_id(target, preferred_types=_SYMBOL_TYPES)
            if resolved_source is None or resolved_target is None:
                raise ValueError("Could not resolve source or target node")
            result = loaded.analyzer.shortest_path(
                resolved_source,
                resolved_target,
                edge_types=effective_edge_types,
            )
        elif mode in {"pagerank", "centrality"}:
            if mode == "pagerank":
                result = loaded.analyzer.pagerank(
                    edge_types=effective_edge_types,
                    top_k=top_k,
                    node_types=node_types,
                )
            else:
                result = loaded.analyzer.degree_centrality(
                    top_k=top_k,
                    edge_types=effective_edge_types,
                    node_types=node_types,
                )
        elif mode in {"module_pagerank", "module_centrality"}:
            projected = _project_module_adjacency(
                loaded.analyzer,
                only_runtime=only_runtime,
                include_callsites=include_callsites,
                max_callsites=max_callsites,
            )
            result = {
                "modules": _rank_projected_modules(
                    projected["adjacency"],
                    mode="pagerank" if mode == "module_pagerank" else "centrality",
                    top_k=top_k,
                )
            }
            if include_callsites and "callsites" in projected:
                result["callsites"] = projected["callsites"]
        elif mode == "file_deps":
            if not file:
                raise ValueError("file_deps mode requires file")
            result = _build_file_dependency_result(
                loaded,
                file,
                direction=direction,
                structured=structured,
                include_modules=include_modules,
            )
        elif mode == "clusters":
            adjacency = {
                node_id: [
                    edge.dst
                    for edge in edges
                    if effective_edge_types is None or edge.type in set(effective_edge_types)
                ]
                for node_id, edges in loaded.analyzer.outgoing.items()
            }
            components = connected_components(adjacency)
            ranked = sorted(components, key=lambda component: (-len(component), component))
            result = {
                "components": [
                    {"size": len(component), "nodes": component} for component in ranked[:top_k]
                ]
            }
        elif mode == "patterns":
            projected = _project_module_adjacency(
                loaded.analyzer,
                only_runtime=only_runtime,
                include_callsites=False,
                max_callsites=max_callsites,
            )
            result = {
                "important_symbols": loaded.analyzer.pagerank(
                    edge_types=effective_edge_types,
                    top_k=min(top_k, 5),
                    node_types=set(_SYMBOL_TYPES),
                ),
                "hub_symbols": loaded.analyzer.degree_centrality(
                    top_k=min(top_k, 5),
                    edge_types=effective_edge_types,
                    node_types=set(_SYMBOL_TYPES),
                ),
                "important_modules": _rank_projected_modules(
                    projected["adjacency"],
                    mode="pagerank",
                    top_k=min(top_k, 5),
                ),
            }
        else:
            raise ValueError(f"Unsupported graph mode: {mode}")

        graph_result = {
            "success": True,
            "mode": mode,
            "root_path": str(loaded.root_path),
            "rebuilt": loaded.rebuilt,
            "result": result,
        }

        return graph_result
    except (ImportError, RuntimeError) as exc:
        # CodebaseIndex provider not installed or not bootstrapped —
        # return helpful guidance instead of a confusing error the LLM retries.
        return {
            "success": False,
            "mode": mode,
            "error": str(exc),
            "suggestion": (
                "The graph tool requires a codebase indexing provider. "
                "Use code_search, overview, ls, or read tools for code exploration instead. "
                "To enable graph analysis: pip install victor-coding"
            ),
            "unavailable": True,
        }
    except Exception as exc:
        error_msg = str(exc)
        # Detect CodebaseIndex-related errors even if not ImportError
        if "CodebaseIndex" in error_msg or "codebase indexing" in error_msg:
            return {
                "success": False,
                "mode": mode,
                "error": error_msg,
                "suggestion": (
                    "The graph tool requires a codebase indexing provider. "
                    "Use code_search, overview, ls, or read tools instead. "
                    "To enable graph analysis: pip install victor-coding"
                ),
                "unavailable": True,
            }
        return {
            "success": False,
            "mode": mode,
            "error": error_msg,
        }


# =============================================================================
# Focused Graph Tools (Split from monolithic graph function)
# =============================================================================

@tool(
    category="search",
    priority=Priority.HIGH,
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    execution_category=ExecutionCategory.READ_ONLY,
    keywords=["graph", "search", "find", "query"],
    timeout=60.0,
)
async def graph_search(
    query: str,
    path: str = ".",
    top_k: int = 10,
    _exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Find nodes by query in code graph.

    Fast search for symbols, files, and modules by name or pattern.
    Use for: Quick lookups when you know what you're looking for.

    Args:
        query: Search query (symbol name, file path, or pattern)
        path: Path to codebase root (default: ".")
        top_k: Maximum number of results to return

    Returns:
        Dict with matches list containing node IDs and metadata
    """
    return await graph(
        mode="search",
        query=query,
        path=path,
        top_k=top_k,
        _exec_ctx=_exec_ctx,
    )


@tool(
    category="search",
    priority=Priority.HIGH,
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    execution_category=ExecutionCategory.READ_ONLY,
    keywords=["graph", "neighbors", "callers", "callees", "relationships"],
    timeout=60.0,
)
async def graph_neighbors(
    node: str,
    path: str = ".",
    depth: int = 2,
    direction: GraphDirection = "out",
    edge_types: Optional[List[str]] = None,
    include_callsites: int = 3,
    _exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Get neighboring nodes in code graph.

    Explore relationships: callers, callees, references, etc.
    Use for: Understanding what calls or is called by a symbol.

    Args:
        node: Node identifier (symbol, file, or module)
        path: Path to codebase root (default: ".")
        depth: How many hops to explore (default: 2)
        direction: "out" (callees), "in" (callers), or "both"
        edge_types: Filter to specific edge types (None = all)
        include_callsites: Number of call sites to include (default: 3)

    Returns:
        Dict with neighbors list and relationship details
    """
    return await graph(
        mode="neighbors",
        node=node,
        path=path,
        depth=depth,
        direction=direction,
        edge_types=edge_types,
        include_callsites=include_callsites,
        _exec_ctx=_exec_ctx,
    )


@tool(
    category="analysis",
    priority=Priority.MEDIUM,
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    execution_category=ExecutionCategory.READ_ONLY,
    keywords=["graph", "analytics", "metrics", "pagerank", "centrality", "stats"],
    timeout=60.0,
)
async def graph_analytics(
    path: str = ".",
    reindex: bool = False,
    only_runtime: bool = False,
    files_only: bool = False,
    modules_only: bool = False,
    top_k: int = 10,
    _exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Calculate graph metrics and analytics.

    PageRank, centrality, stats, clusters, etc.
    Use for: Understanding codebase structure and importance.

    Args:
        path: Path to codebase root (default: ".")
        reindex: Force rebuild of graph index
        only_runtime: Only include runtime dependencies
        files_only: Analyze files only (not symbols)
        modules_only: Analyze modules only (not files/symbols)
        top_k: Top K results to return

    Returns:
        Dict with metrics: pagerank, centrality, stats, etc.
    """
    return await graph(
        mode="stats",
        path=path,
        reindex=reindex,
        only_runtime=only_runtime,
        files_only=files_only,
        modules_only=modules_only,
        top_k=top_k,
        _exec_ctx=_exec_ctx,
    )


@tool(
    category="search",
    priority=Priority.MEDIUM,
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    execution_category=ExecutionCategory.READ_ONLY,
    keywords=["graph", "path", "trace", "flow", "dependencies"],
    timeout=60.0,
)
async def graph_path(
    source: str,
    target: str,
    path: str = ".",
    max_depth: int = 5,
    _exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Find paths between nodes in code graph.

    Discover how code connects: call chains, data flow, etc.
    Use for: Tracing execution paths or data flow.

    Args:
        source: Source node identifier
        target: Target node identifier
        path: Path to codebase root (default: ".")
        max_depth: Maximum path length to search

    Returns:
        Dict with paths list and path details
    """
    return await graph(
        mode="path",
        source=source,
        target=target,
        path=path,
        depth=max_depth,
        _exec_ctx=_exec_ctx,
    )


@tool(
    category="analysis",
    priority=Priority.MEDIUM,
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    execution_category=ExecutionCategory.READ_ONLY,
    keywords=["graph", "dependencies", "file", "imports", "impact"],
    timeout=60.0,
)
async def graph_dependencies(
    path: str = ".",
    reindex: bool = False,
    top_k: int = 10,
    _exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Analyze file dependencies in codebase.

    Understand import relationships and file dependencies.
    Use for: Impact analysis, dependency mapping.

    Args:
        path: Path to codebase root (default: ".")
        reindex: Force rebuild of graph index
        top_k: Top K files to return

    Returns:
        Dict with dependency information for each file
    """
    return await graph(
        mode="file_deps",
        path=path,
        reindex=reindex,
        top_k=top_k,
        _exec_ctx=_exec_ctx,
    )


@tool(
    category="search",
    priority=Priority.LOW,
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    execution_category=ExecutionCategory.READ_ONLY,
    keywords=["graph", "patterns", "anti-patterns", "structure"],
    timeout=60.0,
)
async def graph_patterns(
    query: str,
    path: str = ".",
    top_k: int = 10,
    _exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Find code patterns in graph.

    Discover structural patterns, anti-patterns, and code smells.
    Use for: Finding design patterns, architectural patterns.

    Args:
        query: Pattern description or query
        path: Path to codebase root (default: ".")
        top_k: Top K patterns to return

    Returns:
        Dict with pattern matches and analysis
    """
    return await graph(
        mode="patterns",
        query=query,
        path=path,
        top_k=top_k,
        _exec_ctx=_exec_ctx,
    )


# Legacy alias for backward compatibility
graph_tool = graph

__all__ = [
    "ALL_EDGE_TYPES",
    "GraphAnalyzer",
    "GraphMode",
    "_load_graph",
    "graph",
    "graph_search",
    "graph_neighbors",
    "graph_analytics",
    "graph_path",
    "graph_dependencies",
    "graph_patterns",
    "graph_tool",
]
