import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from victor.tools.common import EXCLUDE_DIRS, DEFAULT_CODE_EXTENSIONS, latest_mtime
from victor.tools.decorators import tool

# Alias for backward compatibility
DEFAULT_EXTS = DEFAULT_CODE_EXTENSIONS

# Cache for semantic indexes to avoid re-embedding on every call
_INDEX_CACHE: Dict[str, Dict[str, Any]] = {}

# Directories that indicate non-core code (lower importance)
NON_CORE_DIRS = {
    "test", "tests", "testing", "spec", "specs",
    "demo", "demos", "example", "examples", "sample", "samples",
    "client", "clients", "sdk",
    "benchmark", "benchmarks", "bench",
    "doc", "docs", "documentation",
    "script", "scripts", "tool", "tools", "util", "utils",
    "mock", "mocks", "fixture", "fixtures", "stub", "stubs",
}

# Directories that indicate core code (higher importance)
CORE_DIRS = {
    "src", "lib", "core", "pkg", "internal", "main",
    "engine", "engines", "service", "services",
    "storage", "index", "compute", "network", "api",
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
    - None -> returns DEFAULT_EXTS
    - String 'py' -> {'.py'}
    - List ['py', 'js'] -> {'.py', '.js'}
    - Already dotted ['.py'] -> {'.py'}
    """
    if exts is None:
        return DEFAULT_EXTS

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


async def _get_or_build_index(
    root: Path, settings: Any, force_reindex: bool = False
) -> Tuple[Any, bool]:
    """Return cached CodebaseIndex or build/update one. Returns (index, rebuilt?).

    Uses intelligent caching:
    1. In-memory cache for same session
    2. Persistent disk storage in {root}/.victor/embeddings/
    3. Incremental updates for changed files only (not full rebuild)
    """
    from victor.codebase.indexer import CodebaseIndex

    cache_entry = _INDEX_CACHE.get(str(root))
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
            _INDEX_CACHE[str(root)]["latest_mtime"] = latest
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

    # Create new index - it will load from disk if available
    index = CodebaseIndex(
        root_path=str(root),
        use_embeddings=True,
        embedding_config=embedding_config,
    )

    # Only do full index if forced or no persistent data exists
    persist_path = Path(default_persist_dir)
    if force_reindex or not persist_path.exists() or not any(persist_path.iterdir()):
        # First time or forced - full index
        await index.index_codebase(force=force_reindex)
        rebuilt = True
    else:
        # Persistent data exists - just ensure indexed (incremental)
        await index.ensure_indexed(auto_reindex=True)
        rebuilt = False

    _INDEX_CACHE[str(root)] = {
        "index": index,
        "latest_mtime": latest,
        "indexed_at": time.time(),
    }
    return index, rebuilt


@tool
async def code_search(
    query: str,
    root: str = ".",
    k: int = 5,
    max_files: int = 200,
    max_chars: int = 50000,
    exts: Optional[List[str]] = None,
    path: Optional[str] = None,  # Alias for 'root' - models often use 'path'
) -> Dict[str, Any]:
    """
    Simple keyword-based code search for exact text matches.

    Use this ONLY for simple, literal keyword searches (e.g., "def foo", "import xyz").
    For conceptual queries like "find classes that inherit", "find error handling",
    or "find all implementations of X", use semantic_code_search instead.

    Scans files and scores by keyword match frequency. Fast but misses semantic meaning.

    Args:
        query: Search query string
        root: Root directory to search (default: current directory)
        path: Alias for 'root' - accepts either parameter name
    """
    # Support 'path' as alias for 'root' (models often use 'path' intuitively)
    search_root = path if path is not None else root
    try:
        files = _gather_files(search_root, exts, max_files)
        scores: List[Dict[str, Any]] = []
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read(max_chars)
                score = _keyword_score(text, query)
                if score > 0:
                    scores.append({"path": file_path, "score": score, "snippet": text[:800]})
            except Exception:
                continue

        scores.sort(key=lambda x: x["score"], reverse=True)
        top = scores[: max(1, min(k, len(scores)))]
        return {"success": True, "results": top, "count": len(top)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@tool
async def semantic_code_search(
    query: str,
    root: str = ".",
    k: int = 10,
    force_reindex: bool = False,
    filter_file_path: Optional[str] = None,
    filter_symbol_type: Optional[str] = None,
    filter_visibility: Optional[str] = None,
    filter_language: Optional[str] = None,
    filter_is_test_file: Optional[bool] = None,
    filter_has_docstring: Optional[bool] = None,
    context: Optional[Dict[str, Any]] = None,
    path: Optional[str] = None,  # Alias for 'root' - models often use 'path'
) -> Dict[str, Any]:
    """AI-powered semantic code search using embeddings.

    PREFERRED over code_search for conceptual queries like "find authentication",
    "error handling patterns", or "classes inheriting from X".

    Filters: filter_file_path, filter_symbol_type (class/function/method),
    filter_visibility (public/private), filter_language, filter_is_test_file.

    Args:
        query: Semantic search query
        root: Root directory to search (default: current directory)
        path: Alias for 'root' - accepts either parameter name
    """
    # Support 'path' as alias for 'root' (models often use 'path' intuitively)
    search_root = path if path is not None else root
    try:
        root_path = Path(search_root).resolve()
        if not root_path.exists():
            return {"success": False, "error": f"Root not found: {search_root}"}

        settings = context.get("settings") if context else None
        if settings is None:
            return {"success": False, "error": "Settings not available in tool context."}

        # Build metadata filter from optional parameters
        filter_metadata: Optional[Dict[str, Any]] = None
        filters_applied = []

        if any(
            [
                filter_file_path,
                filter_symbol_type,
                filter_visibility,
                filter_language,
                filter_is_test_file is not None,
                filter_has_docstring is not None,
            ]
        ):
            filter_metadata = {}
            if filter_file_path:
                filter_metadata["file_path"] = filter_file_path
                filters_applied.append(f"file_path={filter_file_path}")
            if filter_symbol_type:
                filter_metadata["symbol_type"] = filter_symbol_type
                filters_applied.append(f"symbol_type={filter_symbol_type}")
            if filter_visibility:
                filter_metadata["visibility"] = filter_visibility
                filters_applied.append(f"visibility={filter_visibility}")
            if filter_language:
                filter_metadata["language"] = filter_language
                filters_applied.append(f"language={filter_language}")
            if filter_is_test_file is not None:
                filter_metadata["is_test_file"] = filter_is_test_file
                filters_applied.append(f"is_test_file={filter_is_test_file}")
            if filter_has_docstring is not None:
                filter_metadata["has_docstring"] = filter_has_docstring
                filters_applied.append(f"has_docstring={filter_has_docstring}")

        index, rebuilt = await _get_or_build_index(root_path, settings, force_reindex=force_reindex)
        results = await index.semantic_search(
            query=query,
            max_results=k,
            filter_metadata=filter_metadata,
        )

        # Truncate content in results to prevent context overflow
        # Keep enough for context but not entire function bodies
        MAX_CONTENT_CHARS = 500
        truncated_results = []
        for result in results:
            # Convert to dict for serialization
            result_dict = result.model_dump() if hasattr(result, "model_dump") else dict(result)
            content = result_dict.get("content", "")
            if len(content) > MAX_CONTENT_CHARS:
                result_dict["content"] = (
                    content[:MAX_CONTENT_CHARS] + f"... [truncated, {len(content)} chars total]"
                )
                result_dict["content_truncated"] = True

            # Calculate importance score for ranking
            file_path = result_dict.get("file_path", "")
            symbol_type = result_dict.get("symbol_type")
            importance = _calculate_importance_score(file_path, symbol_type)
            result_dict["importance_score"] = round(importance, 2)

            # Combine semantic similarity with importance for final ranking
            semantic_score = result_dict.get("score", result_dict.get("similarity", 0.5))
            result_dict["combined_score"] = round(semantic_score * importance, 3)

            truncated_results.append(result_dict)

        # Re-sort by combined score (importance-weighted semantic relevance)
        truncated_results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)

        return {
            "success": True,
            "results": truncated_results,
            "count": len(truncated_results),
            "hint": "Use read_file with line_start/line_end to see full content of specific results.",
            "ranking_note": "Results ranked by combined_score (semantic_similarity × importance). Core src/ code ranked higher than test/demo files.",
            "metadata": {
                "rebuilt": rebuilt,
                "root": str(root_path),
                "indexed_at": _INDEX_CACHE[str(root_path)]["indexed_at"],
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
            },
        }
    except ImportError as exc:
        return {
            "success": False,
            "error": f"Semantic search dependencies missing: {exc}. Install lancedb/chromadb + sentence-transformers.",
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}
