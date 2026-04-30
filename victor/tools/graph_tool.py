from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, DefaultDict, Dict, Iterable, List, Literal, Optional, Set

from victor.config.settings import get_project_paths, load_settings
from victor.core.indexing.graph_enrichment import ensure_project_graph_enriched
from victor.native.python.graph_algo import connected_components, pagerank, weighted_pagerank
from victor.storage.graph.protocol import GraphEdge, GraphNode, GraphStoreProtocol
from victor.storage.unified.protocol import UnifiedId
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
    SEARCH = "search"  # Alias for find
    OVERVIEW = "overview"  # High-level graph summary
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
    SEMANTIC = "semantic"  # Semantic relationship discovery
    QUERY = "query"  # Direct SQL query mode


_GRAPH_MODE_ALIAS_NOTES = {
    "hub_analysis": "overview",
    "top_k": "search (when query/node/file is provided) or pagerank",
}

_FILE_FALLBACK_DIRECTIONS: Dict[str, GraphDirection] = {
    "callers": "in",
    "callees": "out",
    "trace": "out",
    "call_flow": "out",
    "neighbors": "both",
    "impact": "both",
    "subgraph": "both",
}

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
_SYMBOL_IDENTITY_BASIS = "unique node_id (repo-relative path-qualified symbols)"


def _ctx_value(exec_ctx: Optional[Dict[str, Any]], key: str, default: Any = None) -> Any:
    if exec_ctx is None:
        return default
    if isinstance(exec_ctx, ToolExecutionContext):
        return getattr(exec_ctx, key, default)
    return exec_ctx.get(key, default)


def _normalize_relpath(file_path: str) -> str:
    return file_path.replace("\\", "/").strip()


def _graph_mode_value(mode: str | GraphMode) -> str:
    """Return the string value for a graph mode enum or raw string."""
    if isinstance(mode, GraphMode):
        return mode.value
    return str(mode).strip()


def _normalize_graph_mode_alias(
    mode: str | GraphMode,
    *,
    node: Optional[str] = None,
    source: Optional[str] = None,
    target: Optional[str] = None,
    file: Optional[str] = None,
    query: Optional[str] = None,
) -> str:
    """Map common model-invented graph mode aliases to canonical modes."""
    raw_mode = _graph_mode_value(mode).strip().lower()
    if not raw_mode:
        return GraphMode.NEIGHBORS.value

    if "|" in raw_mode:
        normalized_parts = [
            _normalize_graph_mode_alias(
                part,
                node=node,
                source=source,
                target=target,
                file=file,
                query=query,
            )
            for part in raw_mode.split("|")
            if part.strip()
        ]
        return "|".join(normalized_parts)

    if raw_mode == "hub_analysis":
        return GraphMode.OVERVIEW.value
    if raw_mode == "top_k":
        if query or node or file:
            return GraphMode.SEARCH.value
        return GraphMode.PAGERANK.value

    return raw_mode


def _format_tool_command_value(value: Any) -> str:
    """Format a tool argument value for a follow-up command string."""
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, str):
        return json.dumps(value)
    return repr(value)


def _build_tool_follow_up_command(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Build a compact follow-up command string."""
    serialized_args = ", ".join(
        f"{key}={_format_tool_command_value(value)}"
        for key, value in arguments.items()
        if value is not None
    )
    return f"{tool_name}({serialized_args})"


def _build_follow_up_suggestion(
    tool_name: str,
    arguments: Dict[str, Any],
    description: str,
) -> Dict[str, Any]:
    """Create a normalized follow-up suggestion payload."""
    return {
        "tool": tool_name,
        "command": _build_tool_follow_up_command(tool_name, arguments),
        "arguments": arguments,
        "description": description,
        "reason": description,
    }


def _build_graph_error_follow_up_suggestions(
    *,
    path: str,
    requested_mode: str,
    normalized_mode: str,
    node: Optional[str],
    source: Optional[str],
    target: Optional[str],
    file: Optional[str],
    query: Optional[str],
    depth: int,
    top_k: int,
    unavailable: bool = False,
    unresolved_node: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build structured follow-up suggestions for recoverable graph failures."""
    suggestions: List[Dict[str, Any]] = []
    search_seed = unresolved_node or query or node or source or target or file
    effective_top_k = max(1, min(top_k, 10))

    if unavailable:
        suggestions.append(
            _build_follow_up_suggestion(
                "project_overview",
                {"path": path, "max_depth": 2},
                "Inspect the project structure without relying on the graph index.",
            )
        )
        if search_seed:
            suggestions.append(
                _build_follow_up_suggestion(
                    "code_search",
                    {
                        "query": search_seed,
                        "path": path,
                        "mode": "text",
                        "k": effective_top_k,
                    },
                    f'Search code textually for "{search_seed}" while graph indexing is unavailable.',
                )
            )
        return suggestions

    if normalized_mode != requested_mode:
        canonical_args: Dict[str, Any] = {"mode": normalized_mode, "path": path}
        if normalized_mode in {"search", "find"}:
            canonical_args["query"] = search_seed
            canonical_args["top_k"] = effective_top_k
        elif normalized_mode in {
            "neighbors",
            "callers",
            "callees",
            "trace",
            "call_flow",
            "impact",
            "subgraph",
        }:
            canonical_args["node"] = node or source or target
            canonical_args["depth"] = depth
        elif normalized_mode == "file_deps":
            canonical_args["file"] = file
        elif normalized_mode not in {"stats"}:
            canonical_args["top_k"] = effective_top_k
        suggestions.append(
            _build_follow_up_suggestion(
                "graph",
                canonical_args,
                f'Use the supported graph mode "{normalized_mode}" instead of "{requested_mode}".',
            )
        )

    if search_seed:
        suggestions.append(
            _build_follow_up_suggestion(
                "graph",
                {"mode": "search", "query": search_seed, "path": path, "top_k": effective_top_k},
                f'Search the graph index for matches to "{search_seed}".',
            )
        )
    elif not suggestions:
        suggestions.append(
            _build_follow_up_suggestion(
                "graph",
                {"mode": "overview", "path": path, "top_k": effective_top_k},
                "Start with a supported graph overview of the codebase.",
            )
        )

    return suggestions[:2]


def _graph_error_response(
    *,
    requested_mode: str,
    mode: str,
    error: str,
    suggestions: Optional[List[Dict[str, Any]]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a structured graph error response."""
    response: Dict[str, Any] = {
        "success": False,
        "mode": mode,
        "requested_mode": requested_mode,
        "error": error,
    }
    if suggestions:
        response["metadata"] = {"follow_up_suggestions": suggestions}
    if extra:
        response.update(extra)
    return response


def _unsupported_graph_mode_error(mode: str) -> str:
    """Build a user/model-facing unsupported mode error with recovery hints."""
    supported_modes = ", ".join(sorted(graph_mode.value for graph_mode in GraphMode))
    alias_notes = "; ".join(
        f"{alias} -> {target}" for alias, target in sorted(_GRAPH_MODE_ALIAS_NOTES.items())
    )
    return (
        f"Unsupported graph mode: {mode}. "
        f"Supported modes: {supported_modes}. "
        f"Common aliases: {alias_notes}."
    )


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


def _qualified_name_for_node(node: GraphNode) -> str:
    metadata_name = node.metadata.get("qualified_name")
    if isinstance(metadata_name, str) and metadata_name:
        return metadata_name

    try:
        unified_id = UnifiedId.from_string(node.node_id)
    except ValueError:
        unified_id = None

    if unified_id is not None:
        if unified_id.type == "file":
            return unified_id.path
        if unified_id.name:
            return f"{unified_id.path}:{unified_id.name}"
        if unified_id.path:
            return unified_id.path

    module_name = _module_name_for_node(node)
    if node.type in {"file", "module"}:
        return node.file or module_name or node.name
    if node.file:
        return f"{node.file}:{node.name}"
    return module_name or node.name


def _node_payload(node: GraphNode, **extra: Any) -> Dict[str, Any]:
    payload = {
        "node_id": node.node_id,
        "name": node.name,
        "qualified_name": _qualified_name_for_node(node),
        "type": node.type,
        "file": node.file,
        "module": _module_name_for_node(node),
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
        compound_matches = self._resolve_compound_reference(
            normalized,
            lowered=lowered,
            preferred_types=preferred_types,
        )
        if compound_matches:
            return compound_matches[0]
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

    def _resolve_compound_reference(
        self,
        normalized: str,
        *,
        lowered: str,
        preferred_types: Optional[Set[str]] = None,
    ) -> List[str]:
        """Resolve references like ``path/to/file.py:SymbolName``."""
        if ":" not in normalized:
            return []

        file_ref, _, symbol_ref = normalized.rpartition(":")
        if not file_ref or not symbol_ref:
            return []

        normalized_file = _normalize_relpath(file_ref)
        symbol_lower = symbol_ref.strip().lower()
        if not symbol_lower:
            return []

        matches: List[str] = []
        for node in self.nodes.values():
            if node.name.lower() != symbol_lower:
                continue
            node_file = _normalize_relpath(node.file)
            if node_file == normalized_file or node_file.endswith(normalized_file):
                matches.append(node.node_id)

        if preferred_types:
            matches.sort(
                key=lambda node_id: (
                    0 if self.nodes[node_id].type in preferred_types else 1,
                    node_id,
                )
            )
        else:
            matches.sort()
        return matches

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


def _has_enhanced_codebase_index_provider() -> bool:
    from victor.core.capability_registry import CapabilityRegistry
    from victor.framework.vertical_protocols import CodebaseIndexFactoryProtocol

    registry = CapabilityRegistry.get_instance()
    try:
        registry.ensure_bootstrapped()
    except Exception:
        logger.debug("[graph] Capability bootstrap failed during availability check", exc_info=True)

    factory = registry.get(CodebaseIndexFactoryProtocol)
    return factory is not None and registry.is_enhanced(CodebaseIndexFactoryProtocol)


def _project_graph_has_data(root_path: Path) -> bool:
    from victor.core.database import get_project_database

    project_db = get_project_database(root_path)
    if not project_db.table_exists("graph_node") or not project_db.table_exists("graph_edge"):
        return False

    node_row = project_db.query_one("SELECT COUNT(*) FROM graph_node")
    edge_row = project_db.query_one("SELECT COUNT(*) FROM graph_edge")
    node_count = int(node_row[0]) if node_row is not None else 0
    edge_count = int(edge_row[0]) if edge_row is not None else 0
    return node_count > 0 or edge_count > 0


def _ensure_project_graph_ready(root_path: Path) -> None:
    try:
        ensure_project_graph_enriched(root_path)
    except Exception as exc:  # pragma: no cover - defensive logging only
        logger.warning(
            "[graph] Failed to enrich persisted project graph for %s: %s", root_path, exc
        )


def _graph_tool_is_available() -> bool:
    try:
        if _has_enhanced_codebase_index_provider():
            return True
    except Exception:
        logger.debug("[graph] Provider availability check failed", exc_info=True)

    try:
        return _project_graph_has_data(_resolve_root_path("."))
    except Exception:
        return False


def _resolve_requested_subject_path(path: str) -> Path:
    requested = Path(path)
    if requested.is_absolute():
        return requested.resolve()
    return (Path(get_project_paths().project_root) / requested).resolve()


def _recover_file_deps_without_file(
    *,
    loaded: LoadedGraph,
    requested_path: str,
    requested_mode: str,
    direction: GraphDirection,
    structured: bool,
    include_modules: bool,
    top_k: int,
    effective_edge_types: Optional[List[str]],
    only_runtime: bool,
    include_callsites: bool,
    max_callsites: int,
) -> tuple[str, Dict[str, Any]]:
    """Recover file dependency requests when models overload `path` as the subject."""
    if not requested_path or requested_path == ".":
        raise ValueError("file_deps mode requires file")

    requested_location = _resolve_requested_subject_path(requested_path)
    requested_ref = Path(requested_path)
    if requested_ref.suffix:
        file_ref = requested_path
        if requested_location.is_relative_to(loaded.root_path):
            file_ref = requested_location.relative_to(loaded.root_path).as_posix()
        else:
            parent_ref = requested_ref.parent.as_posix()
            if parent_ref and loaded.root_path.as_posix().endswith(parent_ref):
                file_ref = requested_ref.name

        result = _build_file_dependency_result(
            loaded,
            file_ref,
            direction=direction,
            structured=structured,
            include_modules=include_modules,
        )
        result["recovered_from_mode"] = requested_mode
        result["recovered_from_path"] = requested_path
        return "file_deps", result

    result = _build_overview(
        loaded,
        top_k=top_k,
        effective_edge_types=effective_edge_types,
        only_runtime=only_runtime,
        include_callsites=include_callsites,
        max_callsites=max_callsites,
    )
    result["recovered_from_mode"] = requested_mode
    result["recovered_from_path"] = requested_path
    return "overview", result


async def _materialize_loaded_graph(
    root_path: Path,
    *,
    index: Any,
    graph_store: GraphStoreProtocol,
    rebuilt: bool,
) -> LoadedGraph:
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


async def _load_graph_from_project_store(root_path: Path) -> LoadedGraph:
    from victor.core.database import get_project_database
    from victor.storage.graph.sqlite_store import SqliteGraphStore

    project_db = get_project_database(root_path)
    if not project_db.table_exists("graph_node") or not project_db.table_exists("graph_edge"):
        raise RuntimeError("Project graph tables are unavailable")

    node_row = project_db.query_one("SELECT COUNT(*) FROM graph_node")
    edge_row = project_db.query_one("SELECT COUNT(*) FROM graph_edge")
    node_count = int(node_row[0]) if node_row is not None else 0
    edge_count = int(edge_row[0]) if edge_row is not None else 0
    if node_count == 0 and edge_count == 0:
        raise RuntimeError("Project graph database is empty")

    _ensure_project_graph_ready(root_path)

    graph_store = SqliteGraphStore(root_path)
    fallback_index = SimpleNamespace(graph_store=graph_store, files={})
    return await _materialize_loaded_graph(
        root_path,
        index=fallback_index,
        graph_store=graph_store,
        rebuilt=False,
    )


async def _load_graph(
    path: str = ".",
    *,
    reindex: bool = False,
    exec_ctx: Optional[Dict[str, Any]] = None,
) -> LoadedGraph:
    """Load graph data with fast-path: use project database immediately, rebuild index asynchronously.

    Priority order:
    1. Project graph database (fast, non-blocking)
    2. CodebaseIndex with graph_store (if available)
    3. Fallback to project database (if index unavailable)

    Index rebuilds happen asynchronously in the background and don't block queries.
    """
    root_path = _resolve_root_path(path)
    settings = _ctx_value(exec_ctx, "settings")
    if settings is None:
        settings = load_settings()

    # Fast path: Check if project graph database exists and has data
    # This is non-blocking and serves from existing structural bridge data
    if not reindex and _project_graph_has_data(root_path):
        try:
            logger.info("[graph] Using existing project graph database (fast path)")
            return await _load_graph_from_project_store(root_path)
        except Exception as exc:
            logger.debug("[graph] Project database load failed, falling back to index: %s", exc)

    # Secondary path: Try CodebaseIndex if available
    # Only blocks if reindex=True or project database doesn't exist
    try:
        index, rebuilt = await _get_or_build_index(
            root_path,
            settings,
            force_reindex=reindex,
            exec_ctx=exec_ctx,
        )
        graph_store = getattr(index, "graph_store", None)
        if graph_store is not None:
            return await _materialize_loaded_graph(
                root_path,
                index=index,
                graph_store=graph_store,
                rebuilt=rebuilt,
            )
    except (ImportError, RuntimeError, ValueError) as exc:
        logger.debug("[graph] CodebaseIndex unavailable, trying project database: %s", exc)

    # Fallback: Project database (even if empty, better than nothing)
    try:
        loaded = await _load_graph_from_project_store(root_path)
        logger.info("[graph] Loaded persisted project graph for %s (fallback)", root_path)
        return loaded
    except Exception as exc:
        raise ValueError(
            f"Graph data unavailable. CodebaseIndex failed and project database is empty: {exc}"
        ) from exc


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


async def _run_graph_sql_query(
    loaded: LoadedGraph,
    sql: str,
) -> Dict[str, Any]:
    """Execute a raw SQL query against the graph database.

    Enables 'Big Picture' metrics by leveraging SQLite's aggregation capabilities.
    For safety, only SELECT statements are allowed.

    Args:
        loaded: LoadedGraph instance
        sql: SQL SELECT query to execute

    Returns:
        Dict with query results and metadata
    """
    return await _run_graph_sql_query_for_root(loaded.root_path, sql)


async def _run_graph_sql_query_for_root(
    root_path: Path,
    sql: str,
) -> Dict[str, Any]:
    """Execute a raw SQL query against the persisted project graph database."""
    from victor.core.database import get_project_database

    # Security: strictly enforce read-only SELECT queries
    # Strip whitespace and check prefix
    clean_sql = sql.strip()
    if not clean_sql.upper().startswith("SELECT"):
        return {
            "error": "Only SELECT queries are allowed for security reasons.",
            "success": False,
        }

    # Block common malicious patterns
    forbidden = ["DELETE", "UPDATE", "INSERT", "DROP", "ALTER", "CREATE", "REPLACE", "ATTACH"]
    for word in forbidden:
        if f" {word} " in clean_sql.upper() or clean_sql.upper().startswith(f"{word} "):
            return {
                "error": f"Forbidden keyword detected in query: {word}",
                "success": False,
            }

    try:
        _ensure_project_graph_ready(root_path)
        project_db = get_project_database(root_path)

        # Execute query
        start_time = time.perf_counter()
        rows = project_db.query(clean_sql)
        duration = time.perf_counter() - start_time

        # Convert Row objects to serializable dicts
        results = [dict(row) for row in rows]

        return {
            "results": results,
            "row_count": len(results),
            "execution_time_sec": round(duration, 4),
            "query": clean_sql,
            "success": True,
        }
    except Exception as e:
        error_str = str(e)
        logger.warning(f"[graph] SQL query failed: {e}")

        # Add helpful context for common column errors
        available_columns = "node_id, type, name, file, line, end_line, lang, signature, docstring, parent_id, embedding_ref, metadata"
        if "no such column" in error_str.lower() or "does not exist" in error_str.lower():
            return {
                "error": f"SQL execution failed: {error_str}\n\nAvailable columns in graph_node: {available_columns}\nAvailable columns in graph_edge: src, dst, type, file, line",
                "success": False,
                "available_columns": {
                    "graph_node": available_columns,
                    "graph_edge": "src, dst, type, file, line",
                },
            }
        return {
            "error": f"SQL execution failed: {error_str}",
            "success": False,
        }


def _build_stats_from_project_store(root_path: Path) -> Dict[str, Any]:
    """Build graph stats directly from persisted SQLite graph tables."""
    from victor.core.database import get_project_database

    _ensure_project_graph_ready(root_path)
    project_db = get_project_database(root_path)
    if not project_db.table_exists("graph_node") or not project_db.table_exists("graph_edge"):
        raise RuntimeError("Project graph tables are unavailable")

    node_row = project_db.query_one("SELECT COUNT(*) FROM graph_node")
    edge_row = project_db.query_one("SELECT COUNT(*) FROM graph_edge")
    node_types = {
        str(row["type"]): int(row["count"])
        for row in project_db.query(
            "SELECT type, COUNT(*) AS count FROM graph_node GROUP BY type ORDER BY type"
        )
    }
    edge_types = {
        str(row["type"]): int(row["count"])
        for row in project_db.query(
            "SELECT type, COUNT(*) AS count FROM graph_edge GROUP BY type ORDER BY type"
        )
    }

    return {
        "nodes": int(node_row[0]) if node_row is not None else 0,
        "edges": int(edge_row[0]) if edge_row is not None else 0,
        "node_types": node_types,
        "edge_types": edge_types,
        "root_path": str(root_path),
        "rebuilt": False,
    }


async def _find_semantic_relationships(
    loaded: LoadedGraph,
    node_id: str,
    *,
    threshold: float = 0.5,
    limit: int = 10,
    node_types: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Find components semantically similar to a given node.

    Uses vector embeddings to discover 'Potential Architectural Relationships'
    between components that lack explicit code-level links.

    Args:
        loaded: LoadedGraph instance with index and analyzer
        node_id: Target node identifier
        threshold: Minimum similarity score (0.0 to 1.0)
        limit: Maximum number of relationships to return
        node_types: Optional set of node types to filter results

    Returns:
        Dict with discovered potential relationships
    """
    target_node = loaded.analyzer.nodes.get(node_id)
    if not target_node:
        return {"error": f"Node '{node_id}' not found in graph analyzer"}

    # 1. Create semantic anchor for the node
    # Combine name, type, and docstring to capture the node's intent
    parts = [target_node.name, f"type:{target_node.type}"]
    if target_node.signature:
        parts.append(target_node.signature)
    if target_node.docstring:
        # Take first 250 chars of docstring to focus on intent, not details
        parts.append(target_node.docstring[:250])

    anchor_text = " ".join(parts)
    logger.info(f"[graph] Searching for semantic matches for '{target_node.name}'")

    # 2. Search for similar nodes using CodebaseIndex
    semantic_search = getattr(loaded.index, "semantic_search", None)
    if not callable(semantic_search):
        return {
            "focus_node": _node_payload(target_node),
            "potential_relationships": [],
            "threshold_used": threshold,
            "total_discovered": 0,
            "semantic_search_available": False,
        }

    try:
        # Use existing semantic search infrastructure
        search_results = await semantic_search(
            query=anchor_text,
            max_results=limit * 3,  # Over-fetch for filtering
            similarity_threshold=threshold,
        )
    except Exception as e:
        logger.warning(f"[graph] Semantic search failed during discovery: {e}")
        return {"error": f"Semantic search failed: {str(e)}"}

    # 3. Filter and rank results
    # Get existing structural neighbors (OUT only for dependency-like relationships)
    neighbors = loaded.analyzer.get_neighbors(node_id, direction=GraphDirection.OUT, max_depth=1)
    existing_neighbor_ids = {
        item["node_id"]
        for depth_items in neighbors["neighbors_by_depth"].values()
        for item in depth_items
    }
    # Also ignore self
    existing_neighbor_ids.add(node_id)

    relationships: List[Dict[str, Any]] = []
    for result in search_results:
        # Match search result back to a graph node if possible
        # Result typically contains 'file_path', 'symbol_name'
        res_file = result.get("file_path")
        res_name = result.get("symbol_name")

        if not res_file or not res_name:
            continue

        # Find corresponding node ID in graph analyzer
        match_id = loaded.analyzer.resolve_node_id(
            res_name, preferred_types={result.get("symbol_type") or "function"}
        )
        if not match_id or match_id in existing_neighbor_ids:
            continue

        # Double check it's the right file to avoid collisions with same-named symbols
        match_node = loaded.analyzer.nodes[match_id]
        if _normalize_relpath(match_node.file) != _normalize_relpath(res_file):
            continue

        if node_types and match_node.type not in node_types:
            continue

        relationships.append(
            _node_payload(
                match_node,
                similarity=round(float(result.get("score") or 0.0), 3),
                discovery_reason="semantic_similarity",
            )
        )

        if len(relationships) >= limit:
            break

    return {
        "focus_node": _node_payload(target_node),
        "potential_relationships": sorted(
            relationships, key=lambda x: x["similarity"], reverse=True
        ),
        "threshold_used": threshold,
        "total_discovered": len(relationships),
        "semantic_search_available": True,
    }


def _find_similar_node_names(
    analyzer: GraphAnalyzer, node_ref: str, max_suggestions: int = 5
) -> List[str]:
    """Find similar node names for error suggestions.

    Args:
        analyzer: GraphAnalyzer instance
        node_ref: The node reference that failed to resolve
        max_suggestions: Maximum number of suggestions to return

    Returns:
        List of similar node names
    """
    # Use the analyzer's search method to find similar nodes
    results = analyzer.search(node_ref, limit=max_suggestions)
    return [r["name"] for r in results]


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


def _build_overview(
    loaded: LoadedGraph,
    *,
    top_k: int,
    effective_edge_types: Optional[List[str]],
    only_runtime: bool,
    include_callsites: bool,
    max_callsites: int,
) -> Dict[str, Any]:
    """Build a high-level graph overview for exploratory requests."""
    projected = _project_module_adjacency(
        loaded.analyzer,
        only_runtime=only_runtime,
        include_callsites=include_callsites,
        max_callsites=max_callsites,
    )
    return {
        "stats": _build_stats(loaded),
        "symbol_identity_basis": _SYMBOL_IDENTITY_BASIS,
        "important_symbols": loaded.analyzer.pagerank(
            edge_types=effective_edge_types,
            top_k=min(top_k, 10),
            node_types=set(_SYMBOL_TYPES),
        ),
        "hub_symbols": loaded.analyzer.degree_centrality(
            top_k=min(top_k, 10),
            edge_types=effective_edge_types,
            node_types=set(_SYMBOL_TYPES),
        ),
        "important_modules": _rank_projected_modules(
            projected["adjacency"],
            mode="pagerank",
            top_k=min(top_k, 10),
        ),
        "hub_modules": _rank_projected_modules(
            projected["adjacency"],
            mode="centrality",
            top_k=min(top_k, 10),
        ),
    }


async def _handle_multi_mode(
    loaded: Any,
    mode_str: str,
    path: str,
    node: Optional[str],
    source: Optional[str],
    target: Optional[str],
    file: Optional[str],
    query: Optional[str],
    depth: int,
    top_k: int,
    direction: GraphDirection,
    edge_types: Optional[List[str]],
    only_runtime: bool,
    files_only: bool,
    modules_only: bool,
    structured: bool,
    include_modules: bool,
    include_symbols: bool,
    include_calls: bool,
    include_refs: bool,
    include_callsites: bool,
    max_callsites: int,
    _exec_ctx: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Handle pipe-separated modes by expanding and combining results.

    Example: mode="callers|callees" executes both modes and merges results.

    Returns:
        Combined results with mode-specific sections
    """
    modes = [m.strip() for m in mode_str.split("|")]
    logger.info(f"[graph] Expanding multi-mode request: {mode_str} -> {modes}")

    combined_results = {}

    for single_mode in modes:
        try:
            # Execute each mode directly without recursion
            # We replicate the graph function logic here for single modes
            default_edge_types = _default_edge_types(single_mode, only_runtime=only_runtime)
            effective_edge_types = edge_types or default_edge_types
            node_types = _node_type_filter(
                single_mode, files_only=files_only, modules_only=modules_only
            )

            if single_mode in {
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
                    raise ValueError(f"{single_mode} mode requires node")

                preferred_types = (
                    {"file"} if files_only else {"module"} if modules_only else _SYMBOL_TYPES
                )
                resolved_id = loaded.analyzer.resolve_node_id(
                    target_ref, preferred_types=preferred_types
                )
                if resolved_id is None:
                    suggestions = _find_similar_node_names(loaded.analyzer, target_ref)
                    error_msg = f"Could not resolve graph node '{target_ref}'"
                    if suggestions:
                        error_msg += "\n\nDid you mean one of these?\n  - " + "\n  - ".join(
                            suggestions[:5]
                        )
                    raise ValueError(error_msg)

                effective_direction = direction
                if single_mode == "callers":
                    effective_direction = "in"
                elif single_mode in {"callees", "trace", "call_flow"}:
                    effective_direction = "out"
                elif single_mode in {"impact", "subgraph"}:
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

                combined_results[single_mode] = result

            else:
                # For other modes, set error (not implemented in multi-mode yet)
                combined_results[single_mode] = {
                    "error": f"Mode '{single_mode}' not supported in multi-mode queries",
                    "mode": single_mode,
                }

        except Exception as e:
            logger.warning(f"[graph] Mode {single_mode} failed: {e}")
            combined_results[single_mode] = {
                "error": str(e),
                "mode": single_mode,
            }

    # Return combined results with metadata
    return {
        "success": True,
        "mode": mode_str,
        "modes_expanded": modes,
        "results": combined_results,
        "node": node,
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
    availability_check=_graph_tool_is_available,
    timeout=180.0,
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
    threshold: float = 0.5,
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

    Valid modes include: overview, stats, search/find, neighbors, callers, callees,
    trace, path, pagerank, centrality, patterns, clusters, semantic, query, and
    module/file dependency variants.

    Common alias recovery is supported for model-generated variants such as:
    - hub_analysis -> overview
    - top_k -> search (when a query/node/file is supplied) or pagerank

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

    requested_mode = _graph_mode_value(mode)
    normalized_mode = _normalize_graph_mode_alias(
        mode,
        node=node,
        source=source,
        target=target,
        file=file,
        query=query,
    )
    try:
        if not reindex and normalized_mode == "stats" and _project_graph_has_data(root_path):
            return {
                "success": True,
                "mode": "stats",
                "requested_mode": requested_mode,
                "root_path": str(root_path),
                "rebuilt": False,
                "result": _build_stats_from_project_store(root_path),
            }

        if (
            not reindex
            and normalized_mode == GraphMode.QUERY
            and _project_graph_has_data(root_path)
        ):
            if not query:
                raise ValueError("query mode requires a SQL SELECT statement")
            return {
                "success": True,
                "mode": GraphMode.QUERY,
                "requested_mode": requested_mode,
                "root_path": str(root_path),
                "rebuilt": False,
                "result": await _run_graph_sql_query_for_root(root_path, query),
            }

        loaded = await _load_graph(path, reindex=reindex, exec_ctx=_exec_ctx)

        # Handle pipe-separated modes (e.g., "callers|callees")
        # Expand into multiple results combined
        if "|" in normalized_mode:
            return await _handle_multi_mode(
                loaded=loaded,
                mode_str=normalized_mode,
                path=path,
                node=node,
                source=source,
                target=target,
                file=file,
                query=query,
                depth=depth,
                top_k=top_k,
                direction=direction,
                edge_types=edge_types,
                only_runtime=only_runtime,
                files_only=files_only,
                modules_only=modules_only,
                structured=structured,
                include_modules=include_modules,
                include_symbols=include_symbols,
                include_calls=include_calls,
                include_refs=include_refs,
                include_callsites=include_callsites,
                max_callsites=max_callsites,
                _exec_ctx=_exec_ctx,
            )

        mode = normalized_mode

        default_edge_types = _default_edge_types(mode, only_runtime=only_runtime)
        effective_edge_types = edge_types or default_edge_types
        node_types = _node_type_filter(mode, files_only=files_only, modules_only=modules_only)

        if mode == "overview":
            result = _build_overview(
                loaded,
                top_k=top_k,
                effective_edge_types=effective_edge_types,
                only_runtime=only_runtime,
                include_callsites=include_callsites,
                max_callsites=max_callsites,
            )
        elif mode == "stats":
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
                if file:
                    mode = "file_deps"
                    fallback_direction = _FILE_FALLBACK_DIRECTIONS.get(
                        requested_mode,
                        direction if isinstance(direction, str) else direction.value,
                    )
                    result = _build_file_dependency_result(
                        loaded,
                        file,
                        direction=fallback_direction,
                        structured=structured,
                        include_modules=include_modules,
                    )
                    result["recovered_from_mode"] = requested_mode
                else:
                    raise ValueError(f"{mode} mode requires node")
            else:
                preferred_types = (
                    {"file"} if files_only else {"module"} if modules_only else _SYMBOL_TYPES
                )
                resolved_id = loaded.analyzer.resolve_node_id(
                    target_ref, preferred_types=preferred_types
                )
                if resolved_id is None:
                    # Suggest similar node names
                    suggestions = _find_similar_node_names(loaded.analyzer, target_ref)
                    error_msg = f"Could not resolve graph node '{target_ref}'"
                    if suggestions:
                        error_msg += "\n\nDid you mean one of these?\n  - " + "\n  - ".join(
                            suggestions[:5]
                        )
                    raise ValueError(error_msg)

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
                # Determine which node failed and provide suggestions
                error_parts = []
                if resolved_source is None:
                    source_suggestions = _find_similar_node_names(loaded.analyzer, source)
                    error_parts.append(f"Source '{source}' not found")
                    if source_suggestions:
                        error_parts.append("  Did you mean?")
                        error_parts.extend(f"    - {s}" for s in source_suggestions[:3])
                if resolved_target is None:
                    target_suggestions = _find_similar_node_names(loaded.analyzer, target)
                    error_parts.append(f"Target '{target}' not found")
                    if target_suggestions:
                        error_parts.append("  Did you mean?")
                        error_parts.extend(f"    - {s}" for s in target_suggestions[:3])
                raise ValueError("\n".join(error_parts))
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
                mode, result = _recover_file_deps_without_file(
                    loaded=loaded,
                    requested_path=path,
                    requested_mode=requested_mode,
                    direction=direction,
                    structured=structured,
                    include_modules=include_modules,
                    top_k=top_k,
                    effective_edge_types=effective_edge_types,
                    only_runtime=only_runtime,
                    include_callsites=include_callsites,
                    max_callsites=max_callsites,
                )
            else:
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
                "symbol_identity_basis": _SYMBOL_IDENTITY_BASIS,
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
                "hub_modules": _rank_projected_modules(
                    projected["adjacency"],
                    mode="centrality",
                    top_k=min(top_k, 5),
                ),
            }
        elif mode == GraphMode.SEMANTIC:
            target_ref = node or source
            if not target_ref:
                raise ValueError("semantic mode requires node")

            preferred_types = (
                {"file"} if files_only else {"module"} if modules_only else _SYMBOL_TYPES
            )
            resolved_id = loaded.analyzer.resolve_node_id(
                target_ref, preferred_types=preferred_types
            )
            if resolved_id is None:
                suggestions = _find_similar_node_names(loaded.analyzer, target_ref)
                error_msg = f"Could not resolve graph node '{target_ref}'"
                if suggestions:
                    error_msg += "\n\nDid you mean one of these?\n  - " + "\n  - ".join(
                        suggestions[:5]
                    )
                raise ValueError(error_msg)

            result = await _find_semantic_relationships(
                loaded,
                resolved_id,
                threshold=threshold,
                limit=top_k,
                node_types=node_types,
            )
        elif mode == GraphMode.QUERY:
            if not query:
                raise ValueError("query mode requires a SQL SELECT statement")
            result = await _run_graph_sql_query(loaded, query)
        else:
            raise ValueError(_unsupported_graph_mode_error(str(mode)))

        graph_result = {
            "success": True,
            "mode": mode,
            "requested_mode": requested_mode,
            "root_path": str(loaded.root_path),
            "rebuilt": loaded.rebuilt,
            "result": result,
        }

        return graph_result
    except (ImportError, RuntimeError) as exc:
        # CodebaseIndex provider not installed or not bootstrapped —
        # return helpful guidance instead of a confusing error the LLM retries.
        return _graph_error_response(
            requested_mode=requested_mode,
            mode=normalized_mode,
            error=str(exc),
            suggestions=_build_graph_error_follow_up_suggestions(
                path=path,
                requested_mode=requested_mode,
                normalized_mode=normalized_mode,
                node=node,
                source=source,
                target=target,
                file=file,
                query=query,
                depth=depth,
                top_k=top_k,
                unavailable=True,
            ),
            extra={
                "suggestion": (
                    "The graph tool requires a codebase indexing provider. "
                    "Use code_search, overview, ls, or read tools for code exploration instead. "
                    "To enable graph analysis: pip install victor-coding"
                ),
                "unavailable": True,
            },
        )
    except Exception as exc:
        error_msg = str(exc)
        # Detect CodebaseIndex-related errors even if not ImportError
        if "CodebaseIndex" in error_msg or "codebase indexing" in error_msg:
            return _graph_error_response(
                requested_mode=requested_mode,
                mode=normalized_mode,
                error=error_msg,
                suggestions=_build_graph_error_follow_up_suggestions(
                    path=path,
                    requested_mode=requested_mode,
                    normalized_mode=normalized_mode,
                    node=node,
                    source=source,
                    target=target,
                    file=file,
                    query=query,
                    depth=depth,
                    top_k=top_k,
                    unavailable=True,
                ),
                extra={
                    "suggestion": (
                        "The graph tool requires a codebase indexing provider. "
                        "Use code_search, overview, ls, or read tools instead. "
                        "To enable graph analysis: pip install victor-coding"
                    ),
                    "unavailable": True,
                },
            )

        unresolved_node = None
        if error_msg.startswith("Could not resolve graph node '"):
            unresolved_node = error_msg.split("Could not resolve graph node '", 1)[1].split(
                "'",
                1,
            )[0]

        return _graph_error_response(
            requested_mode=requested_mode,
            mode=normalized_mode,
            error=error_msg,
            suggestions=_build_graph_error_follow_up_suggestions(
                path=path,
                requested_mode=requested_mode,
                normalized_mode=normalized_mode,
                node=node,
                source=source,
                target=target,
                file=file,
                query=query,
                depth=depth,
                top_k=top_k,
                unresolved_node=unresolved_node,
            ),
        )


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
    timeout=180.0,
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
    timeout=180.0,
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
    timeout=180.0,
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
    timeout=180.0,
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
    priority=Priority.HIGH,
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    execution_category=ExecutionCategory.READ_ONLY,
    keywords=["graph", "semantic", "discovery", "similarity", "architectural patterns"],
    timeout=180.0,
)
async def graph_semantic(
    node: str,
    path: str = ".",
    threshold: float = 0.5,
    top_k: int = 5,
    _exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Discover potential architectural relationships based on semantic similarity.

    Identifies components that are semantically related but lack explicit code-level links.
    Use for: Finding members of a registry, related service implementations, or pattern participants.

    Args:
        node: Node identifier (symbol, file, or module) to analyze
        path: Path to codebase root (default: ".")
        threshold: Minimum similarity score (default: 0.5)
        top_k: Maximum number of results to return (default: 5)

    Returns:
        Dict with potential relationships and similarity scores
    """
    return await graph(
        mode="semantic",
        node=node,
        path=path,
        top_k=top_k,
        threshold=threshold,
        _exec_ctx=_exec_ctx,
    )


@tool(
    category="analysis",
    priority=Priority.MEDIUM,
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    execution_category=ExecutionCategory.READ_ONLY,
    keywords=["graph", "sql", "query", "aggregate", "metrics", "coupling"],
    timeout=30.0,
)
async def graph_query(
    query: str,
    path: str = ".",
    _exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a raw SQL SELECT query against the code graph database.

    Enables 'Big Picture' metrics by leveraging SQLite's aggregation capabilities.
    For security, only SELECT statements are allowed.

    Tables available:
    - graph_node (node_id, type, name, file, line, end_line, lang, signature, docstring, parent_id, metadata)
    - graph_edge (src, dst, type, weight, metadata)
    - graph_file_mtime (file, mtime, indexed_at)

    IMPORTANT — column names are exact. Common mistakes:
    - There is NO 'module' column. graph_analytics returns a computed 'module' field that
      does NOT exist as a SQL column. To extract the top-level package name (e.g. 'victor'
      from 'victor/core/foo.py') use: SUBSTR(file, 1, INSTR(file || '/', '/') - 1)
      Note: this returns the first path component, not the full dotted module path.
      Top-level files with no slash (e.g. 'setup.py') return the full filename.
    - graph_edge has NO 'file' column — joins must go through graph_node on src/dst = node_id.

    Examples:
    - Find most imported modules: "SELECT count(*) as count, dst FROM graph_edge WHERE type='IMPORTS' GROUP BY dst ORDER BY count DESC LIMIT 10"
    - Count symbols by type: "SELECT type, count(*) as count FROM graph_node GROUP BY type ORDER BY count DESC"
    - Find files with most symbols: "SELECT file, count(*) as count FROM graph_node GROUP BY file ORDER BY count DESC LIMIT 10"
    - Group by top-level package: "SELECT SUBSTR(file, 1, INSTR(file || '/', '/') - 1) as pkg, count(*) as cnt FROM graph_node GROUP BY pkg ORDER BY cnt DESC LIMIT 20"
    - Cross-package coupling (imports between packages): "SELECT SUBSTR(n1.file, 1, INSTR(n1.file || '/', '/') - 1) as src_pkg, SUBSTR(n2.file, 1, INSTR(n2.file || '/', '/') - 1) as dst_pkg, count(*) as edges FROM graph_edge e JOIN graph_node n1 ON e.src = n1.node_id JOIN graph_node n2 ON e.dst = n2.node_id WHERE e.type='IMPORTS' GROUP BY src_pkg, dst_pkg ORDER BY edges DESC LIMIT 20"

    Args:
        query: SQL SELECT query to execute
        path: Path to codebase root (default: ".")

    Returns:
        Dict with query results and execution metadata
    """
    return await graph(
        mode="query",
        query=query,
        path=path,
        _exec_ctx=_exec_ctx,
    )


@tool(
    category="analysis",
    priority=Priority.MEDIUM,
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    execution_category=ExecutionCategory.READ_ONLY,
    keywords=["graph", "dependencies", "file", "imports", "impact"],
    timeout=180.0,
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
    timeout=180.0,
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
