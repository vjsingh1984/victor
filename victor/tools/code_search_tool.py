import asyncio
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from victor.tools.base import AccessMode, DangerLevel, Priority
from victor.tools.common import EXCLUDE_DIRS, DEFAULT_CODE_EXTENSIONS, latest_mtime
from victor.tools.decorators import tool

if TYPE_CHECKING:
    from victor.tools.cache_manager import CacheNamespace

logger = logging.getLogger(__name__)


# Legacy cache for semantic indexes (use _get_index_cache() for DI support)
_INDEX_CACHE: Dict[str, Dict[str, Any]] = {}


def clear_index_cache() -> None:
    """Clear all cached indexes. Call between benchmark tasks for isolation."""
    _INDEX_CACHE.clear()


async def _probe_index_integrity(index: Any, timeout: float = 5.0) -> bool:
    """Validate persistent index integrity with a test query.

    Returns True if index was corrupt and rebuilt, False if healthy.
    """
    try:
        await asyncio.wait_for(
            index.semantic_search(query="test", max_results=1),
            timeout=timeout,
        )
        return False  # Healthy — no rebuild needed
    except Exception as e:
        logger.warning("Persistent index corrupt (%s), forcing rebuild", e)
        if hasattr(index, "_is_indexed"):
            index._is_indexed = False
        try:
            await index.index_codebase()
        except Exception as rebuild_err:
            logger.warning("Index rebuild failed: %s", rebuild_err)
        return True  # Rebuilt (or attempted)


def _get_index_cache(exec_ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get the index cache, preferring DI-injected cache if available.

    Args:
        exec_ctx: Execution context that may contain cache_manager

    Returns:
        Cache dict-like object for index storage
    """
    # Try to get from execution context (DI pattern)
    if exec_ctx is not None:
        # Check for ToolExecutionContext with cache_manager
        from victor.tools.context import ToolExecutionContext

        if isinstance(exec_ctx, ToolExecutionContext):
            if exec_ctx.cache_manager is not None:
                # Return the namespace's internal dict-like interface
                return exec_ctx.index_cache
        elif isinstance(exec_ctx, dict):
            # Legacy dict context - check for cache_manager
            cache_manager = exec_ctx.get("cache_manager")
            if cache_manager is not None:
                return cache_manager.index_cache

    # Fallback to global cache for backward compatibility
    return _INDEX_CACHE


# Directories that indicate non-core code (lower importance)
NON_CORE_DIRS = {
    "test",
    "tests",
    "testing",
    "spec",
    "specs",
    "demo",
    "demos",
    "example",
    "examples",
    "sample",
    "samples",
    "client",
    "clients",
    "sdk",
    "benchmark",
    "benchmarks",
    "bench",
    "doc",
    "docs",
    "documentation",
    "script",
    "scripts",
    "tool",
    "tools",
    "util",
    "utils",
    "mock",
    "mocks",
    "fixture",
    "fixtures",
    "stub",
    "stubs",
}

# Directories that indicate core code (higher importance)
CORE_DIRS = {
    "src",
    "lib",
    "core",
    "pkg",
    "internal",
    "main",
    "engine",
    "engines",
    "service",
    "services",
    "storage",
    "index",
    "compute",
    "network",
    "api",
}

GRAPH_FOLLOW_UP_SYMBOL_TYPES = {"function", "method"}
ENTRYPOINT_SYMBOL_NAMES = {"main", "run", "start", "serve", "cli", "bootstrap"}


def _calculate_importance_score(file_path: str, symbol_type: Optional[str] = None) -> float:
    """Calculate importance score for a search result.

    Higher scores = more architecturally important.

    Scoring factors:
    - Core directories: +0.3
    - Non-core directories: -0.4
    - Class/struct definitions: +0.2
    - Test files: -0.3
    """
    score = 1.0
    path_lower = file_path.lower()
    parts = set(path_lower.replace("\\", "/").split("/"))

    # Check for core directories
    if parts & CORE_DIRS:
        score += 0.3

    # Check for non-core directories
    if parts & NON_CORE_DIRS:
        score -= 0.4

    # Explicit test file patterns
    if any(p in path_lower for p in ["test_", "_test.", ".test.", "/test/", "/tests/"]):
        score -= 0.3

    # Symbol type bonus
    if symbol_type in ("class", "struct", "trait", "interface", "impl"):
        score += 0.2

    # Depth penalty - deeper nesting = less core
    depth = len([p for p in parts if p and p not in {".", ".."}])
    if depth > 5:
        score -= 0.1

    return max(0.1, score)  # Floor at 0.1


def _latest_mtime(root: Path) -> float:
    """Find latest modification time under root, respecting EXCLUDE_DIRS."""
    return latest_mtime(root)


def _normalize_extensions(exts: Optional[List[str]]) -> Set[str]:
    """Normalize extensions to ensure they have leading dots.

    Handles:
    - None -> returns DEFAULT_CODE_EXTENSIONS
    - String 'py' -> {'.py'}
    - List ['py', 'js'] -> {'.py', '.js'}
    - Already dotted ['.py'] -> {'.py'}
    """
    if exts is None:
        return DEFAULT_CODE_EXTENSIONS

    # Handle string input (convert to list)
    if isinstance(exts, str):
        exts = [exts]

    # Normalize each extension to have leading dot
    normalized = set()
    for ext in exts:
        ext = ext.strip()
        if not ext.startswith("."):
            ext = "." + ext
        normalized.add(ext)

    return normalized


def _gather_files(root: str, exts: Optional[List[str]], max_files: int) -> List[str]:
    """Gather files from directory tree."""
    ext_set: Set[str] = _normalize_extensions(exts)
    files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")]
        for fname in filenames:
            if os.path.splitext(fname)[1] in ext_set:
                files.append(os.path.join(dirpath, fname))
                if len(files) >= max_files:
                    return files
    return files


def _keyword_score(text: str, query: str) -> int:
    q = query.lower().split()
    t = text.lower()
    return sum(t.count(word) for word in q)


def _normalize_result_dict(result: Any) -> Dict[str, Any]:
    """Normalize search results from models or dict-like providers."""
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if isinstance(result, dict):
        return dict(result)
    return dict(result)


def _prepare_ranked_results(
    results: List[Any],
    search_mode: str,
    max_content_chars: int = 500,
) -> List[Dict[str, Any]]:
    """Normalize, truncate, and importance-rank search results."""
    ranked_results: List[Dict[str, Any]] = []
    for result in results:
        result_dict = _normalize_result_dict(result)
        result_dict.setdefault("search_mode", search_mode)

        content = result_dict.get("content", "")
        if isinstance(content, str) and len(content) > max_content_chars:
            result_dict["content"] = (
                content[:max_content_chars] + f"... [truncated, {len(content)} chars total]"
            )
            result_dict["content_truncated"] = True

        file_path = result_dict.get("file_path", result_dict.get("path", ""))
        symbol_type = result_dict.get("symbol_type")
        importance = _calculate_importance_score(file_path, symbol_type)
        result_dict["importance_score"] = round(importance, 2)

        raw_score = result_dict.get("score", result_dict.get("similarity"))
        if raw_score is None:
            raw_score = result_dict.get("combined_score", 0.5)
        try:
            numeric_score = float(raw_score)
        except (TypeError, ValueError):
            numeric_score = 0.5
        result_dict["combined_score"] = round(numeric_score * importance, 3)

        ranked_results.append(result_dict)

    ranked_results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
    return ranked_results


def _extract_graph_follow_up_symbol(result: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Extract a symbol candidate suitable for graph follow-up suggestions."""
    metadata = result.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}

    symbol_type = (
        result.get("symbol_type")
        or result.get("type")
        or metadata_dict.get("symbol_type")
        or metadata_dict.get("type")
    )
    symbol_name = (
        result.get("name")
        or result.get("symbol_name")
        or metadata_dict.get("name")
        or metadata_dict.get("symbol_name")
    )

    if not isinstance(symbol_type, str) or not isinstance(symbol_name, str):
        return None

    normalized_type = symbol_type.strip().lower()
    normalized_name = symbol_name.strip()
    if normalized_type not in GRAPH_FOLLOW_UP_SYMBOL_TYPES or not normalized_name:
        return None

    return {"name": normalized_name, "symbol_type": normalized_type}


def _build_graph_follow_up_suggestions(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build graph-tool follow-up suggestions from ranked search results."""
    for result in results:
        symbol = _extract_graph_follow_up_symbol(result)
        if symbol is None:
            continue

        symbol_name = symbol["name"]
        symbol_name_lower = symbol_name.lower()
        suggestions: List[Dict[str, Any]] = []

        if symbol_name_lower in ENTRYPOINT_SYMBOL_NAMES:
            trace_args = {"mode": "trace", "node": symbol_name, "depth": 3}
            suggestions.append(
                {
                    "tool": "graph",
                    "command": f'graph(mode="trace", node="{symbol_name}", depth=3)',
                    "arguments": trace_args,
                    "reason": f"Trace execution starting from {symbol_name}.",
                }
            )

        for mode, reason in (
            ("callers", f"Find who calls {symbol_name}."),
            ("callees", f"Find what {symbol_name} calls."),
        ):
            depth = 2
            suggestions.append(
                {
                    "tool": "graph",
                    "command": f'graph(mode="{mode}", node="{symbol_name}", depth={depth})',
                    "arguments": {"mode": mode, "node": symbol_name, "depth": depth},
                    "reason": reason,
                }
            )

        return suggestions

    return []


def _build_search_response(
    *,
    results: List[Dict[str, Any]],
    mode: str,
    rebuilt: bool,
    root_path: Path,
    exec_ctx: Optional[Dict[str, Any]],
    filters_applied: List[str],
    ranking_note: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
    follow_up_suggestions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a consistent search tool response envelope."""
    cache_entry = _get_index_cache(exec_ctx).get(str(root_path), {})
    hint = (
        "Use read_file with offset/limit based on line_number/end_line for precise reads. "
        "Example: read_file(path, offset=line_number-1, limit=end_line-line_number+5)"
    )
    if follow_up_suggestions:
        hint += f" Graph follow-up: {follow_up_suggestions[0]['command']}"

    metadata = {
        "rebuilt": rebuilt,
        "root": str(root_path),
        "indexed_at": cache_entry.get("indexed_at"),
        "filters_applied": filters_applied if filters_applied else None,
        "chunking_strategy": "BODY_AWARE",
        "importance_weighted": True,
        "available_filters": [
            "file_path",
            "symbol_type",
            "visibility",
            "language",
            "is_test_file",
            "has_docstring",
        ],
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    if follow_up_suggestions:
        metadata["follow_up_suggestions"] = follow_up_suggestions

    return {
        "success": True,
        "results": results,
        "count": len(results),
        "mode": mode,
        "hint": hint,
        "ranking_note": ranking_note,
        "metadata": metadata,
    }


async def _get_or_build_index(
    root: Path,
    settings: Any,
    force_reindex: bool = False,
    exec_ctx: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, bool]:
    """Return cached CodebaseIndex or build/update one. Returns (index, rebuilt?).

    Uses intelligent caching:
    1. In-memory cache for same session (DI-aware)
    2. Persistent disk storage in {root}/.victor/embeddings/
    3. Incremental updates for changed files only (not full rebuild)

    Args:
        root: Root path for the codebase
        settings: Application settings
        force_reindex: Force full re-index
        exec_ctx: Execution context for DI-based cache access
    """
    from victor.core.capability_registry import CapabilityRegistry
    from victor.framework.vertical_protocols import CodebaseIndexFactoryProtocol

    _index_factory = CapabilityRegistry.get_instance().get(CodebaseIndexFactoryProtocol)
    if _index_factory is None or not CapabilityRegistry.get_instance().is_enhanced(
        CodebaseIndexFactoryProtocol
    ):
        raise ImportError(
            "Codebase indexing requires victor-coding package. "
            "Install with: pip install victor-coding"
        )

    # Get cache using DI-aware accessor
    index_cache = _get_index_cache(exec_ctx)

    cache_entry = index_cache.get(str(root))
    cached_index = cache_entry["index"] if cache_entry else None
    last_mtime = cache_entry["latest_mtime"] if cache_entry else 0.0

    latest = _latest_mtime(root)

    # Check if we have a valid cached index
    if cached_index and not force_reindex:
        if latest <= last_mtime:
            # No files changed, use cache directly
            return cached_index, False
        else:
            # Files changed - do incremental update instead of full rebuild
            await cached_index.incremental_reindex()
            index_cache[str(root)]["latest_mtime"] = latest
            return cached_index, False  # Not a full rebuild

    # Default persist directory is {root}/.victor/embeddings/ for project-local storage
    from victor.config.settings import get_project_paths

    default_persist_dir = str(get_project_paths(root).embeddings_dir)

    embedding_config = {
        "vector_store": getattr(settings, "codebase_vector_store", "lancedb"),
        "embedding_model_type": getattr(
            settings, "codebase_embedding_provider", "sentence-transformers"
        ),
        "embedding_model_name": getattr(
            settings,
            "codebase_embedding_model",
            getattr(settings, "unified_embedding_model", "all-MiniLM-L12-v2"),
        ),
        "persist_directory": getattr(settings, "codebase_persist_directory", None)
        or default_persist_dir,
        "extra_config": {},
    }

    graph_store_name = getattr(settings, "codebase_graph_store", "sqlite")
    graph_path = getattr(settings, "codebase_graph_path", None)

    # Create new index - it will load from disk if available
    index = _index_factory.create(
        root_path=str(root),
        use_embeddings=True,
        embedding_config=embedding_config,
        graph_store_name=graph_store_name,
        graph_path=Path(graph_path) if graph_path else None,
    )

    # Only do full index if forced or no persistent data exists
    persist_path = Path(default_persist_dir)
    has_persistent_data = persist_path.exists() and any(persist_path.iterdir())
    if force_reindex or not has_persistent_data:
        # First time or forced - full index
        await index.index_codebase()
        rebuilt = True
    else:
        # Persistent embeddings exist on disk (LanceDB tables).
        # Mark as indexed so semantic_search() works directly against
        # the persisted data without triggering a full rebuild.
        if hasattr(index, "_is_indexed"):
            index._is_indexed = True
        logger.info("Using persistent embeddings from %s (skip full rebuild)", persist_path)
        # Validate integrity — corrupt LanceDB data will fail silently
        rebuilt = await _probe_index_integrity(index)

    index_cache[str(root)] = {
        "index": index,
        "latest_mtime": latest,
        "indexed_at": time.time(),
    }
    return index, rebuilt


async def _literal_search(
    query: str,
    path: str = ".",
    k: int = 5,
    exts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Literal/keyword search using ripgrep (rg) or grep subprocess.

    Uses native search tools instead of Python file-by-file scanning for
    speed and correctness — no file count limits, handles binary detection,
    and searches the full directory tree efficiently.

    Args:
        query: Search terms
        path: Directory to search
        k: Max results
        exts: File extensions ([".py", ".js"])
    """
    import shutil
    import subprocess

    try:
        # Resolve '.' to the framework's project root
        search_path = path
        if search_path == ".":
            try:
                from victor.config.settings import get_project_paths

                search_path = str(get_project_paths().project_root)
            except Exception:
                pass

        # Normalize dotted notation: "Class.method" → search for "def method"
        # in files containing "class Class". This handles the common pattern
        # where models search for "SQLCompiler.get_order_by" but the source has
        # "class SQLCompiler:" and "def get_order_by(self):" on separate lines.
        search_query = query
        dotted_class = None
        if "." in query and not query.startswith(".") and " " not in query:
            parts = query.rsplit(".", 1)
            if len(parts) == 2 and parts[0][0].isupper():
                dotted_class, method = parts
                search_query = f"def {method}"
                logger.info(
                    "Dotted notation detected: %s → searching for '%s' in files with 'class %s'",
                    query,
                    search_query,
                    dotted_class,
                )

        logger.info(
            f"Literal search: query={query!r}, path={path!r}, "
            f"resolved={search_path!r}, exts={exts}"
        )

        # Build command: prefer ripgrep, fall back to grep
        use_rg = shutil.which("rg") is not None
        if use_rg:
            cmd = [
                "rg",
                "--no-heading",
                "--line-number",
                "--color=never",
                "--max-count=5",  # max matches per file
                "--max-columns=200",
            ]
            if exts:
                for ext in exts:
                    cmd.extend(["--glob", f"*{ext}"])
            else:
                # Default: code files only
                cmd.extend(
                    [
                        "--type=py",
                        "--type=js",
                        "--type=ts",
                        "--type=go",
                        "--type=java",
                        "--type=c",
                        "--type=cpp",
                        "--type=rust",
                        "--type=yaml",
                    ]
                )
            cmd.extend(["--", search_query, search_path])
        else:
            cmd = ["grep", "-rn", "--include=*.py", "--", search_query, search_path]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Parse results: group by file, take top k files
        file_matches: Dict[str, List[str]] = {}
        for line in result.stdout.splitlines():
            # Format: path:line_number:content
            parts = line.split(":", 2)
            if len(parts) >= 3:
                fpath = parts[0]
                if fpath not in file_matches:
                    file_matches[fpath] = []
                file_matches[fpath].append(line)

        # For dotted notation (Class.method), filter to files containing the class
        if dotted_class and file_matches:
            filtered = {}
            for fpath, matches in file_matches.items():
                try:
                    content = Path(fpath).read_text(errors="ignore")
                    if f"class {dotted_class}" in content:
                        filtered[fpath] = matches
                except Exception:
                    pass
            if filtered:
                file_matches = filtered
                logger.info(
                    "Dotted notation filter: %d files contain 'class %s'",
                    len(filtered),
                    dotted_class,
                )

        # Sort by number of matches (most matches = most relevant)
        ranked = sorted(file_matches.items(), key=lambda x: len(x[1]), reverse=True)
        top = ranked[:k]

        results = []
        for fpath, matches in top:
            snippet = "\n".join(matches[:5])  # first 5 matching lines
            results.append(
                {
                    "path": fpath,
                    "score": len(matches),
                    "snippet": snippet,
                }
            )

        logger.info(
            f"Literal search: found {len(file_matches)} files matching "
            f"{query!r} in {search_path} ({'rg' if use_rg else 'grep'})"
        )
        return {
            "success": True,
            "results": results,
            "count": len(results),
            "mode": "literal",
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Search timed out after 30s"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@tool(
    category="search",
    priority=Priority.CRITICAL,  # Always available for code exploration
    access_mode=AccessMode.READONLY,  # Only reads files for search
    danger_level=DangerLevel.SAFE,  # No side effects
    # Registry-driven metadata for tool selection and loop detection
    progress_params=[
        "query",
        "path",
        "mode",
    ],  # Different queries/paths = exploration not loop
    stages=[
        "initial",
        "planning",
        "reading",
        "analysis",
    ],  # Relevant for exploration stages
    task_types=["search", "analysis"],  # Classification-aware selection
    execution_category="read_only",  # Safe for parallel execution
    keywords=[
        "search",
        "semantic",
        "embedding",
        "concept",
        "pattern",
        "similar",
        "bug",
        "bugs",
        "regression",
        "crash",
        "failure",
        "find",
        "grep",
        "literal",
        "code search",
    ],
    mandatory_keywords=[
        "search code",
        "find in code",
        "search for",
        "code search",
        # Analysis-related from MANDATORY_TOOL_KEYWORDS
        "analyze",
        "analyze codebase",
        "codebase analysis",
        "understand",
        "explore codebase",
        "architecture",
        "search",
    ],  # Force inclusion
    aliases=["search"],  # Backward compatibility alias
    timeout=60.0,  # Embedding search on large repos can be slow
)
async def code_search(
    query: str,
    path: str = ".",
    k: int = 10,
    mode: str = "semantic",
    reindex: bool = False,
    file: Optional[str] = None,
    symbol: Optional[str] = None,
    lang: Optional[str] = None,
    test: Optional[bool] = None,
    exts: Optional[List[str]] = None,
    _exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Find code by CONCEPT or TEXT when you DON'T know exact location/name.

    Use this tool for exploration when you need to discover where relevant code
    lives. Returns file snippets ranked by relevance.

    DIFFERS FROM:
    - symbol(): Gets FULL CODE of a known symbol. Use when you know file + name.
    - refs(): Finds USAGE locations of a known symbol. Use for "where is X called".
    - graph(): Shows RELATIONSHIPS between symbols. Use for dependencies/impact.

    Modes:
    - "semantic": Embedding-based search. Best for concepts, patterns, inheritance.
    - "literal": Keyword matching (like grep). Best for exact text/identifiers.
    - "bugs": Similar bug search with graph context when supported by the provider.

    Args:
        query: Search query (semantic concepts or literal text)
        path: Directory to search
        k: Max results
        mode: Search mode - "semantic" (default), "literal", or "bugs"
        reindex: Force re-index (semantic mode only)
        file: Filter by file path (semantic mode only)
        symbol: Filter by type (class/function/method) (semantic mode only)
        lang: Filter by language (python/rust/js) (semantic and bugs modes)
        test: Filter test files (true/false) (semantic mode only)
        exts: File extensions for literal mode (e.g., [".py", ".js"])
        _exec_ctx: Framework execution context (contains settings, etc.)

    Example:
        search(query="error handling in providers")  # Semantic: find related concepts
        search(query="BaseProvider", mode="literal")  # Literal: grep-like text match
        search(query="json parsing crash on empty payload", mode="bugs")
    """
    # Route to literal search if mode is "literal"
    if mode == "literal":
        result = await _literal_search(query, path, k, exts)
        if result.get("count", 0) > 0:
            return result
        # Auto-escalate: literal returned 0 results, try semantic if available
        settings = _exec_ctx.get("settings") if _exec_ctx else None
        disable_embeddings = (_exec_ctx or {}).get("disable_embeddings", False)
        if settings and not disable_embeddings:
            logger.info(
                "Literal search returned 0 results for '%s', auto-escalating to semantic",
                query,
            )
            mode = "semantic"  # Fall through to semantic path below
        else:
            return result
    search_root = path
    try:
        root_path = Path(search_root).resolve()
        # If path is a file, use its parent directory (model passed file path instead of directory)
        if root_path.is_file():
            root_path = root_path.parent
            logger.debug(f"Path '{search_root}' is a file, using parent: {root_path}")
        # If path doesn't exist, try parent directory (model may have passed path with extra segment)
        elif not root_path.exists():
            parent_path = root_path.parent
            if parent_path.exists() and parent_path.is_dir():
                root_path = parent_path
                logger.debug(f"Path '{search_root}' not found, using parent: {root_path}")
            else:
                # Path and parent don't exist - return error with helpful message
                return {
                    "success": False,
                    "error": f"Search root '{search_root}' not found. Please provide a valid directory path.",
                }

        settings = _exec_ctx.get("settings") if _exec_ctx else None
        if settings is None:
            return {
                "success": False,
                "error": "Settings not available in tool context.",
            }

        # Check if embeddings are disabled for this agent (workflow-level service mode)
        disable_embeddings = _exec_ctx.get("disable_embeddings", False) if _exec_ctx else False
        if disable_embeddings:
            logger.info("Embeddings disabled for this agent, falling back to literal search")
            return await _literal_search(query, path, k, exts)

        # Build metadata filter from optional parameters
        filter_metadata: Optional[Dict[str, Any]] = None
        filters_applied = []

        if any([file, symbol, lang, test is not None]):
            filter_metadata = {}
            if file:
                filter_metadata["file_path"] = file
                filters_applied.append(f"file={file}")
            if symbol:
                filter_metadata["symbol_type"] = symbol
                filters_applied.append(f"symbol={symbol}")
            if lang:
                filter_metadata["language"] = lang
                filters_applied.append(f"lang={lang}")
            if test is not None:
                filter_metadata["is_test_file"] = test
                filters_applied.append(f"test={test}")

        try:
            index, rebuilt = await asyncio.wait_for(
                _get_or_build_index(root_path, settings, force_reindex=reindex, exec_ctx=_exec_ctx),
                timeout=30.0,
            )
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning("Semantic index build failed (%s), falling back to literal search", exc)
            result = await _literal_search(query, path, k, exts)
            result["fallback"] = "semantic_index_timeout"
            return result

        if mode == "bugs":
            ignored_filters = [
                name
                for name, value in (("file", file), ("symbol", symbol), ("test", test))
                if value is not None
            ]
            try:
                bug_results = await index.find_similar_bugs(
                    bug_description=query,
                    language=lang,
                    top_k=k,
                    include_graph_context=True,
                    context_limit=min(max(1, k), 3),
                )
                ranked_results = _prepare_ranked_results(
                    bug_results,
                    search_mode="bug_similarity",
                )
                extra_metadata = {
                    "provider_capability": "find_similar_bugs",
                }
                if ignored_filters:
                    extra_metadata["ignored_filters"] = ignored_filters
                follow_up_suggestions = _build_graph_follow_up_suggestions(ranked_results)
                return _build_search_response(
                    results=ranked_results,
                    mode="bugs",
                    rebuilt=rebuilt,
                    root_path=root_path,
                    exec_ctx=_exec_ctx,
                    filters_applied=filters_applied,
                    ranking_note="Results ranked by combined_score (bug_similarity × importance). Graph context included when available.",
                    extra_metadata=extra_metadata,
                    follow_up_suggestions=follow_up_suggestions,
                )
            except NotImplementedError as exc:
                logger.info(
                    "Bug similarity mode is unsupported by %s; falling back to semantic search",
                    type(index).__name__,
                )
                filters_applied.append("mode_fallback=semantic")
                fallback_metadata = {
                    "requested_mode": "bugs",
                    "fallback_mode": "semantic",
                    "fallback_reason": str(exc),
                }
            else:
                fallback_metadata = {}
        else:
            fallback_metadata = {}

        # Get semantic search configuration from settings
        # Default threshold lowered from 0.5 to 0.25 for better recall on technical queries
        similarity_threshold = getattr(settings, "semantic_similarity_threshold", 0.25)
        expand_query = getattr(settings, "semantic_query_expansion_enabled", True)
        enable_hybrid = getattr(settings, "enable_hybrid_search", False)

        # Strip filter fields not in the index schema to avoid LanceDB errors.
        # The index has: file_path, symbol_name, symbol_type, line_number, end_line.
        # Fields like "language", "is_test_file" may not exist in all index backends.
        _INDEX_FILTER_FIELDS = {
            "file_path",
            "symbol_name",
            "symbol_type",
            "line_number",
            "end_line",
        }
        safe_filter = None
        if filter_metadata:
            safe_filter = {k: v for k, v in filter_metadata.items() if k in _INDEX_FILTER_FIELDS}
            dropped = set(filter_metadata) - set(safe_filter)
            if dropped:
                logger.debug("Dropped unsupported filter fields: %s", dropped)
            if not safe_filter:
                safe_filter = None

        # Perform semantic search with timeout and literal fallback
        try:
            results = await asyncio.wait_for(
                index.semantic_search(
                    query=query,
                    max_results=k * 2 if enable_hybrid else k,
                    filter_metadata=safe_filter,
                    similarity_threshold=similarity_threshold,
                    expand_query=expand_query,
                ),
                timeout=15.0,
            )
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning("Semantic search failed (%s), falling back to literal search", exc)
            result = await _literal_search(query, path, k, exts)
            result["fallback"] = "semantic_search_timeout"
            return result

        # Record outcome for RL threshold learning if enabled
        if getattr(settings, "enable_semantic_threshold_rl_learning", False):
            try:
                from victor.framework.rl.coordinator import get_rl_coordinator
                from victor.framework.rl.base import RLOutcome

                coordinator = get_rl_coordinator()

                # Get embedding model from settings
                embedding_model = getattr(
                    settings,
                    "codebase_embedding_model",
                    getattr(settings, "unified_embedding_model", "all-MiniLM-L12-v2"),
                )

                # Get task type from execution context (default to "search")
                task_type = _exec_ctx.get("task_type", "search") if _exec_ctx else "search"

                # Create outcome with semantic search metadata
                outcome = RLOutcome(
                    provider=embedding_model,  # Using embedding model as "provider"
                    model="code_search",  # Tool name as "model"
                    task_type=task_type,
                    success=(len(results) > 0),
                    quality_score=0.5,  # Neutral score, would need user feedback for better estimate
                    metadata={
                        "embedding_model": embedding_model,
                        "tool_name": "code_search",
                        "query": query,
                        "results_count": len(results),
                        "threshold_used": similarity_threshold,
                        "false_negatives": (len(results) == 0),
                        "false_positives": False,  # Would need user feedback
                    },
                    vertical="coding",
                )

                # Record outcome
                coordinator.record_outcome("semantic_threshold", outcome, "coding")

                # Check if we have a learned threshold recommendation
                recommendation = coordinator.get_recommendation(
                    "semantic_threshold",
                    embedding_model,  # provider param
                    "code_search",  # model param (tool name)
                    task_type,
                )

                if recommendation and recommendation.value is not None:
                    logger.debug(
                        f"RL: Learned threshold {recommendation.value:.2f} "
                        f"(current: {similarity_threshold:.2f}, "
                        f"confidence={recommendation.confidence:.2f}) for "
                        f"{embedding_model}:{task_type}:code_search"
                    )

            except Exception as e:
                logger.debug(f"Failed to record threshold learning outcome: {e}")

        # Optionally combine with keyword search using hybrid RRF
        if enable_hybrid and results:
            try:
                from victor.framework.search import create_hybrid_search_engine

                # Get keyword search results
                keyword_results = await _literal_search(query, str(root_path), k * 2, exts=None)

                if keyword_results.get("success"):
                    # Convert semantic results to dict format for hybrid engine
                    semantic_dicts = [
                        {
                            "file_path": r.get("file_path", ""),
                            "content": r.get("content", ""),
                            "score": r.get("score", 0.5),
                            "line_number": r.get("line_number", 0),
                            "metadata": r.get("metadata", {}),
                        }
                        for r in results
                    ]

                    # Create hybrid search engine with configured weights
                    semantic_weight = getattr(settings, "hybrid_search_semantic_weight", 0.6)
                    keyword_weight = getattr(settings, "hybrid_search_keyword_weight", 0.4)
                    engine = create_hybrid_search_engine(semantic_weight, keyword_weight)

                    # Combine results using RRF
                    hybrid_results = engine.combine_results(
                        semantic_dicts,
                        keyword_results.get("results", []),
                        max_results=k,
                    )

                    # Convert back to dict format
                    results = [
                        {
                            "file_path": hr.file_path,
                            "content": hr.content,
                            "score": hr.combined_score,
                            "semantic_score": hr.semantic_score,
                            "keyword_score": hr.keyword_score,
                            "line_number": hr.line_number,
                            "metadata": hr.metadata,
                            "search_mode": "hybrid",
                        }
                        for hr in hybrid_results
                    ]

                    logger.info(
                        f"Hybrid search combined semantic + keyword → {len(results)} results"
                    )

            except Exception as e:
                logger.warning(f"Hybrid search failed, falling back to semantic: {e}")
                # Fall back to semantic-only results (already have them)

        ranked_results = _prepare_ranked_results(results, search_mode="semantic")
        follow_up_suggestions = _build_graph_follow_up_suggestions(ranked_results)

        return _build_search_response(
            results=ranked_results,
            mode="semantic",
            rebuilt=rebuilt,
            root_path=root_path,
            exec_ctx=_exec_ctx,
            filters_applied=filters_applied,
            ranking_note="Results ranked by combined_score (semantic_similarity × importance). Core src/ code ranked higher than test/demo files.",
            extra_metadata=fallback_metadata,
            follow_up_suggestions=follow_up_suggestions,
        )
    except ImportError as exc:
        return {
            "success": False,
            "error": f"Semantic search dependencies missing: {exc}. Install lancedb/chromadb + sentence-transformers.",
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}
