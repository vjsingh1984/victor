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
) -> Dict[str, Any]:
    """Build a consistent search tool response envelope."""
    cache_entry = _get_index_cache(exec_ctx).get(str(root_path), {})
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

    return {
        "success": True,
        "results": results,
        "count": len(results),
        "mode": mode,
        "hint": "Use read_file with offset/limit based on line_number/end_line for precise reads. Example: read_file(path, offset=line_number-1, limit=end_line-line_number+5)",
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
    if force_reindex or not persist_path.exists() or not any(persist_path.iterdir()):
        # First time or forced - full index
        await index.index_codebase()
        rebuilt = True
    else:
        # Persistent data exists - just ensure indexed (incremental)
        await index.ensure_indexed(auto_reindex=True)
        rebuilt = False

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
    """Internal literal/keyword search implementation.

    Args:
        query: Search terms
        path: Directory to search
        k: Max results
        exts: File extensions ([".py", ".js"])
    """
    try:
        files = _gather_files(path, exts, 200)
        scores: List[Dict[str, Any]] = []
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read(50000)
                score = _keyword_score(text, query)
                if score > 0:
                    scores.append({"path": file_path, "score": score, "snippet": text[:800]})
            except Exception:
                continue

        scores.sort(key=lambda x: x["score"], reverse=True)
        top = scores[: max(1, min(k, len(scores)))]
        return {"success": True, "results": top, "count": len(top), "mode": "literal"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@tool(
    category="search",
    priority=Priority.CRITICAL,  # Always available for code exploration
    access_mode=AccessMode.READONLY,  # Only reads files for search
    danger_level=DangerLevel.SAFE,  # No side effects
    # Registry-driven metadata for tool selection and loop detection
    progress_params=["query", "path", "mode"],  # Different queries/paths = exploration not loop
    stages=["initial", "planning", "reading", "analysis"],  # Relevant for exploration stages
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
        return await _literal_search(query, path, k, exts)
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
            return {"success": False, "error": "Settings not available in tool context."}

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

        index, rebuilt = await _get_or_build_index(
            root_path, settings, force_reindex=reindex, exec_ctx=_exec_ctx
        )

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
                return _build_search_response(
                    results=ranked_results,
                    mode="bugs",
                    rebuilt=rebuilt,
                    root_path=root_path,
                    exec_ctx=_exec_ctx,
                    filters_applied=filters_applied,
                    ranking_note="Results ranked by combined_score (bug_similarity × importance). Graph context included when available.",
                    extra_metadata=extra_metadata,
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

        # Perform semantic search
        results = await index.semantic_search(
            query=query,
            max_results=k * 2 if enable_hybrid else k,  # Get more for hybrid combining
            filter_metadata=filter_metadata,
            similarity_threshold=similarity_threshold,
            expand_query=expand_query,
        )

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
                        semantic_dicts, keyword_results.get("results", []), max_results=k
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

        return _build_search_response(
            results=ranked_results,
            mode="semantic",
            rebuilt=rebuilt,
            root_path=root_path,
            exec_ctx=_exec_ctx,
            filters_applied=filters_applied,
            ranking_note="Results ranked by combined_score (semantic_similarity × importance). Core src/ code ranked higher than test/demo files.",
            extra_metadata=fallback_metadata,
        )
    except ImportError as exc:
        return {
            "success": False,
            "error": f"Semantic search dependencies missing: {exc}. Install lancedb/chromadb + sentence-transformers.",
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}
