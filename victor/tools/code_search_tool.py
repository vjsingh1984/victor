import asyncio
import importlib.util
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from victor.framework.search import (
    CODEBASE_INDEX_MANIFEST_NAME,
    DEFAULT_CODEBASE_CHUNKING_STRATEGY,
    DEFAULT_CODEBASE_CHUNK_OVERLAP,
    DEFAULT_CODEBASE_CHUNK_SIZE,
    STRUCTURAL_CODEBASE_VECTOR_STORE,
    build_codebase_index_manifest,
    enable_structural_codebase_embeddings,
    enrich_code_search_results,
    has_compatible_codebase_index_manifest,
    has_persisted_codebase_index_data,
    rerank_code_search_results,
    write_codebase_index_manifest,
)
from victor.tools.base import AccessMode, DangerLevel, Priority
from victor.tools.common import EXCLUDE_DIRS, DEFAULT_CODE_EXTENSIONS, latest_mtime
from victor.tools.decorators import tool
from victor.tools.formatters import format_search_results

if TYPE_CHECKING:
    from victor.tools.cache_manager import CacheNamespace

    # File watching types
    from victor.core.indexing.file_watcher import FileChangeEvent

logger = logging.getLogger(__name__)


@dataclass
class SearchFilters:
    """Filters for code search operations.

    Consolidates filter parameters into a single object.
    Reduces parameter count from 5 to 1 for code_search.
    """

    file_pattern: Optional[str] = None  # Search by filename pattern
    symbol: Optional[str] = None  # Search by symbol name
    language: Optional[str] = None  # Filter by programming language
    test_only: Optional[bool] = None  # Only search test files
    extensions: Optional[List[str]] = None  # Filter by file extensions


def extract_skeleton(source: str, language: str = "python") -> str:
    """Extract a program skeleton from source code.

    Returns function/class signatures + docstrings without implementation
    details. Inspired by arXiv:2604.07502 (SE Conventions for Agentic Dev).

    Args:
        source: Source code string
        language: Programming language (currently 'python' supported)

    Returns:
        Skeleton string with signatures and docstrings
    """
    if not source.strip():
        return ""

    if language != "python":
        # Fallback: first 50 lines for unknown languages
        lines = source.split("\n")[:50]
        return "\n".join(lines)

    lines = source.split("\n")
    skeleton_lines: List[str] = []
    in_body = False
    in_docstring = False
    docstring_quote: Optional[str] = None

    for line in lines:
        stripped = line.strip()

        # Import statements — always include
        if stripped.startswith(("import ", "from ")):
            skeleton_lines.append(line)
            in_body = False
            continue

        # Decorators — include
        if stripped.startswith("@"):
            skeleton_lines.append(line)
            in_body = False
            continue

        # Class/function definitions — always include
        if stripped.startswith(("def ", "class ", "async def ")):
            skeleton_lines.append(line)
            in_body = True
            in_docstring = False
            continue

        # Docstrings right after def/class — include
        if in_body and not in_docstring:
            if '"""' in stripped or "'''" in stripped:
                skeleton_lines.append(line)
                quote = '"""' if '"""' in stripped else "'''"
                # Check if single-line docstring
                if stripped.count(quote) >= 2:
                    in_body = True  # Continue looking for nested defs
                    continue
                in_docstring = True
                docstring_quote = quote
                continue
            elif stripped.startswith("#"):
                # Comment right after def — include as pseudo-docstring
                skeleton_lines.append(line)
                continue
            else:
                # First non-docstring line in body — skip body
                in_body = True
                continue

        # Inside docstring — include until closing
        if in_docstring:
            skeleton_lines.append(line)
            if docstring_quote and docstring_quote in stripped:
                in_docstring = False
            continue

        # Module-level assignments/constants (not indented) — include
        if not stripped.startswith(" ") and not stripped.startswith("\t"):
            if "=" in stripped and not stripped.startswith("#"):
                # Module-level constant
                skeleton_lines.append(line)
                in_body = False
                continue

        # Blank lines between definitions — preserve structure
        if not stripped and not in_body:
            skeleton_lines.append(line)

    return "\n".join(skeleton_lines).rstrip()


# Legacy cache for semantic indexes (use _get_index_cache() for DI support)
_INDEX_CACHE: Dict[str, Dict[str, Any]] = {}


def clear_index_cache() -> None:
    """Clear all cached indexes. Call between benchmark tasks for isolation."""
    _INDEX_CACHE.clear()


def _get_instance_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read explicitly assigned instance attributes without triggering mock fallbacks."""

    try:
        instance_dict = vars(obj)
    except TypeError:
        instance_dict = getattr(obj, "__dict__", None)

    if isinstance(instance_dict, dict) and name in instance_dict:
        return instance_dict[name]
    return default


def _get_index_build_failure_cache(exec_ctx: Optional[Dict[str, Any]] = None) -> Any:
    """Resolve the index-build failure cache without triggering mock fallback attrs."""

    cache_manager = None
    if exec_ctx and isinstance(exec_ctx, dict):
        cache_manager = exec_ctx.get("cache_manager")
    elif exec_ctx and hasattr(exec_ctx, "cache_manager"):
        cache_manager = exec_ctx.cache_manager

    if cache_manager:
        return cache_manager.get_namespace("index_build_failures")

    failure_cache = _get_instance_attr(_get_or_build_index, "_failure_cache")
    if failure_cache is None:
        failure_cache = {}
        _get_or_build_index._failure_cache = failure_cache
    return failure_cache


async def _probe_index_integrity(index: Any, timeout: float = 5.0) -> bool:
    """Validate persistent index integrity with a lightweight check.

    Returns True if corruption was detected (rebuild triggered),
    False if index is healthy or currently being initialized.

    ENHANCEMENT: Distinguishes between init-time races and actual corruption.
    """
    probe_task: Optional[asyncio.Task[Any]] = None
    try:
        # NEW: Check if index is currently being built
        if _get_instance_attr(index, "_is_indexing", False) is True:
            logger.debug("Index currently being built, skipping integrity check")
            return False  # Don't trigger rebuild during init

        # Quick check: does the vector store have data?
        store = _get_instance_attr(index, "_vector_store", None)
        if store is None:
            store = _get_instance_attr(index, "vector_store", None)
        if store:
            table = getattr(store, "_table", None)
            if table is not None:
                row_count = table.count_rows() if hasattr(table, "count_rows") else -1
                if isinstance(row_count, (int, float)) and row_count > 0:
                    logger.info("Persistent index healthy: %d rows in vector store", row_count)
                    return False  # Healthy

        # Fallback: try a semantic search with timeout
        probe_task = asyncio.get_running_loop().create_task(
            index.semantic_search(query="test", max_results=1)
        )
        done, pending = await asyncio.wait({probe_task}, timeout=timeout)
        if pending:
            probe_task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            logger.warning(
                "Persistent index probe timed out after %.2fs; rebuilding inline", timeout
            )
            if hasattr(index, "_is_indexed"):
                index._is_indexed = False
            await _background_index_rebuild(index)
            return True
        await probe_task
        return False  # Healthy — no rebuild needed
    except asyncio.CancelledError:
        if probe_task is not None and not probe_task.done():
            probe_task.cancel()
            await asyncio.gather(probe_task, return_exceptions=True)
        raise
    except Exception as e:
        # ENHANCEMENT: Better error classification
        error_msg = str(e) if str(e) else f"{type(e).__name__}"

        # Don't rebuild for common init-time / transient errors
        transient_keywords = [
            "locked",
            "timeout",
            "timed out",
            "not ready",
            "unavailable",
            "initializing",
            "building",
            "in progress",
            "not have attribute",
            "has no attribute",  # missing attr = code bug, not corruption
        ]

        if any(kw in error_msg.lower() for kw in transient_keywords):
            logger.debug("Transient init issue detected (%s), skipping rebuild", error_msg)
            return False  # Don't rebuild for transient issues

        # Log with full error details for debugging
        logger.warning("Persistent index corrupt (%s); rebuilding inline", error_msg)

        if hasattr(index, "_is_indexed"):
            index._is_indexed = False

        # Rebuild inline while the startup path already holds the index lock.
        await _background_index_rebuild(index)
        return True  # Corruption detected, rebuild in flight


def _build_codebase_embedding_extra_config(settings: Any, root: Optional[Path] = None) -> Dict[str, Any]:
    """Build provider-specific embedding config from settings.

    Existing search settings define dimension and batch size but the code_search
    path historically failed to forward them to the backend provider. This
    helper keeps the translation in one place and allows future provider-
    specific extensions without widening the factory call sites.
    """

    extra_config: Dict[str, Any] = {
        "dimension": getattr(settings, "codebase_dimension", 384),
        "batch_size": getattr(settings, "codebase_batch_size", 32),
        "structural_indexing_enabled": getattr(
            settings, "codebase_structural_indexing_enabled", True
        ),
        "code_chunking_strategy": getattr(
            settings, "codebase_chunking_strategy", DEFAULT_CODEBASE_CHUNKING_STRATEGY
        ),
        "chunk_size": getattr(settings, "codebase_chunk_size", DEFAULT_CODEBASE_CHUNK_SIZE),
        "chunk_overlap": getattr(
            settings, "codebase_chunk_overlap", DEFAULT_CODEBASE_CHUNK_OVERLAP
        ),
    }

    custom_extra = getattr(settings, "codebase_embedding_extra_config", None)
    if isinstance(custom_extra, dict):
        extra_config.update(custom_extra)

    if root is not None:
        extra_config.setdefault("workspace_root", str(root))

    return extra_config


def _build_codebase_embedding_config(settings: Any, root: Path) -> Dict[str, Any]:
    """Build the embedding configuration passed to the codebase index factory."""

    from victor.config.settings import get_project_paths

    default_persist_dir = str(get_project_paths(root).embeddings_dir)
    config = {
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
        "extra_config": _build_codebase_embedding_extra_config(settings, root),
    }
    return enable_structural_codebase_embeddings(config)


def _collect_code_search_backend_metadata(index: Any, settings: Any) -> Dict[str, Any]:
    """Collect compact backend metadata for code_search responses."""

    metadata: Dict[str, Any] = {}
    provider = getattr(index, "embedding_provider", None)
    config = getattr(provider, "config", None)

    def _assign_if_present(key: str, value: Any) -> None:
        if value not in (None, "", {}):
            metadata[key] = value

    if config is not None:
        extra_config = getattr(config, "extra_config", None)
        raw_vector_store = getattr(config, "vector_store", None)
        if raw_vector_store == STRUCTURAL_CODEBASE_VECTOR_STORE and isinstance(extra_config, dict):
            _assign_if_present("vector_store", extra_config.get("upstream_vector_store"))
            metadata["indexing_backend"] = "structural_bridge"
            metadata["structural_indexing_enabled"] = True
        else:
            _assign_if_present("vector_store", raw_vector_store)
        _assign_if_present("embedding_provider", getattr(config, "embedding_model_type", None))
        _assign_if_present("embedding_model", getattr(config, "embedding_model_name", None))
        if isinstance(extra_config, dict):
            _assign_if_present("embedding_dimension", extra_config.get("dimension"))
            _assign_if_present("embedding_batch_size", extra_config.get("batch_size"))
            _assign_if_present(
                "chunking_strategy",
                extra_config.get("code_chunking_strategy") or extra_config.get("chunking_strategy"),
            )

    # Fall back to settings when the backend does not expose explicit config.
    metadata.setdefault("vector_store", getattr(settings, "codebase_vector_store", "lancedb"))
    metadata.setdefault(
        "embedding_provider",
        getattr(settings, "codebase_embedding_provider", "sentence-transformers"),
    )
    metadata.setdefault(
        "embedding_model",
        getattr(
            settings,
            "codebase_embedding_model",
            getattr(settings, "unified_embedding_model", "all-MiniLM-L12-v2"),
        ),
    )
    metadata.setdefault("embedding_dimension", getattr(settings, "codebase_dimension", 384))
    metadata.setdefault("embedding_batch_size", getattr(settings, "codebase_batch_size", 32))
    metadata.setdefault(
        "chunking_strategy",
        getattr(settings, "codebase_chunking_strategy", DEFAULT_CODEBASE_CHUNKING_STRATEGY),
    )
    return metadata


async def _finalize_index_storage(index: Any) -> None:
    """Flush provider-level buffered writes before marking an index reusable."""

    provider = getattr(index, "embedding_provider", None)
    config = getattr(provider, "config", None)
    if getattr(config, "vector_store", None) != STRUCTURAL_CODEBASE_VECTOR_STORE:
        return

    get_stats = getattr(provider, "get_stats", None)
    if not callable(get_stats):
        return

    try:
        await get_stats()
    except Exception as exc:
        logger.debug("Failed to finalize structural code_search provider writes: %s", exc)


async def _background_index_rebuild(index: Any, rebuild_timeout: float = 120.0) -> None:
    """Rebuild a corrupt index in the background without blocking callers."""
    import time

    start_time = time.time()
    try:
        index_path = getattr(index, "root", "unknown")
        logger.info(
            "Background index rebuild started (timeout=%ds, index=%s)", rebuild_timeout, index_path
        )
        await asyncio.wait_for(index.index_codebase(), timeout=rebuild_timeout)
        await _finalize_index_storage(index)
        elapsed = time.time() - start_time
        logger.info(
            "Background index rebuild completed successfully in %.2fs (index=%s)",
            elapsed,
            index_path,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Background index rebuild timed out after %ds; index remains stale (index=%s)",
            rebuild_timeout,
            getattr(index, "root", "unknown"),
        )
    except Exception as err:
        error_msg = str(err) if str(err) else f"{type(err).__name__}"
        logger.warning(
            "Background index rebuild failed after %.2fs: %s (index=%s)",
            time.time() - start_time,
            error_msg,
            getattr(index, "root", "unknown"),
        )


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


def _cache_entry_matches_manifest(
    cache_entry: Optional[Dict[str, Any]],
    index_manifest: Dict[str, Any],
) -> bool:
    """Return True when an in-memory cache entry matches the requested index manifest."""

    if not isinstance(cache_entry, dict):
        return False
    cached_manifest = cache_entry.get("index_manifest")
    return isinstance(cached_manifest, dict) and cached_manifest == index_manifest


def _build_index_failure_key(root: Path, index_manifest: Dict[str, Any]) -> str:
    """Build a manifest-scoped failure cache key for code_search indexes."""

    import hashlib
    import json

    root_hash = hashlib.md5(str(root).encode()).hexdigest()
    manifest_hash = hashlib.md5(
        json.dumps(index_manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"{root_hash}:{manifest_hash}_build_failure"


def _failure_cache_get(failure_cache: Any, key: str) -> Any:
    """Get a failure entry from either a namespace cache or a plain dict."""

    getter = getattr(failure_cache, "get", None)
    if callable(getter):
        return getter(key)
    return None


def _failure_cache_set(failure_cache: Any, key: str, value: Any) -> None:
    """Set a failure entry on either a namespace cache or a plain dict."""

    setter = getattr(failure_cache, "set", None)
    if callable(setter):
        setter(key, value)
        return
    if isinstance(failure_cache, dict):
        failure_cache[key] = value


def _failure_cache_delete(failure_cache: Any, key: str) -> None:
    """Delete a failure entry from either a namespace cache or a plain dict."""

    deleter = getattr(failure_cache, "delete", None)
    if callable(deleter):
        deleter(key)
        return
    if isinstance(failure_cache, dict):
        failure_cache.pop(key, None)


def _classify_semantic_fallback(exc: BaseException, *, scope: str) -> str:
    """Return a precise fallback reason for semantic code-search failures."""

    if isinstance(exc, asyncio.TimeoutError):
        return f"{scope}_timeout"
    return f"{scope}_error"


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


async def _subscribe_to_file_watcher(
    root: Path,
    exec_ctx: Optional[Dict[str, Any]] = None,
) -> bool:
    """Subscribe to file watcher for automatic index invalidation.

    Args:
        root: Root path of codebase
        exec_ctx: Execution context
    """
    from victor.core.indexing.file_watcher import FileWatcherRegistry

    try:
        # Get or create file watcher for this path
        watcher_registry = FileWatcherRegistry.get_instance()
        file_watcher = await watcher_registry.get_watcher(root)

        # Subscribe to file changes (synchronous method)
        file_watcher.subscribe(lambda e: _schedule_file_change_refresh(e, root, exec_ctx))

        logger.info(f"[code_search] Subscribed to file watcher for {root}")
        return True
    except Exception as e:
        logger.error(f"[code_search] Failed to subscribe to file watcher: {e}")
        return False


def _schedule_file_change_refresh(
    event: "FileChangeEvent",
    root: Path,
    exec_ctx: Optional[Dict[str, Any]] = None,
) -> asyncio.Task[Any]:
    """Schedule watcher-driven cache refresh work with exception containment."""

    async def _runner() -> None:
        try:
            await _on_file_change(event, root, exec_ctx)
        except Exception:
            logger.exception("[code_search] File change refresh failed for %s", root)

    return asyncio.create_task(_runner())


async def _ensure_file_watcher_subscription(
    cache_entry: Dict[str, Any],
    root: Path,
    exec_ctx: Optional[Dict[str, Any]] = None,
) -> bool:
    """Subscribe a cached index to file watching exactly once per root.

    Concurrent cache-hit callers can reach the watcher subscription path before
    the per-root index lock is acquired. Keep the subscription single-flight so
    duplicate callbacks do not stack up on the same watcher.
    """

    if cache_entry.get("watcher_subscribed", False):
        return True

    pending_task = cache_entry.get("watcher_subscription_task")
    if isinstance(pending_task, asyncio.Task) and pending_task.done():
        try:
            subscribed = bool(pending_task.result())
        except asyncio.CancelledError:
            subscribed = False
        except Exception:
            subscribed = False
        if cache_entry.get("watcher_subscription_task") is pending_task:
            cache_entry.pop("watcher_subscription_task", None)
        cache_entry["watcher_subscribed"] = subscribed
        if subscribed:
            return True
        pending_task = None

    if not isinstance(pending_task, asyncio.Task):
        pending_task = asyncio.create_task(_subscribe_to_file_watcher(root, exec_ctx))
        cache_entry["watcher_subscription_task"] = pending_task

    try:
        subscribed = bool(await asyncio.shield(pending_task))
    except asyncio.CancelledError:
        if cache_entry.get("watcher_subscription_task") is pending_task and pending_task.done():
            cache_entry.pop("watcher_subscription_task", None)
        raise
    finally:
        if cache_entry.get("watcher_subscription_task") is pending_task and pending_task.done():
            cache_entry.pop("watcher_subscription_task", None)

    cache_entry["watcher_subscribed"] = subscribed
    return subscribed


async def _refresh_cached_index_incrementally(
    cache_entry: Dict[str, Any],
    root: Path,
    *,
    exec_ctx: Optional[Dict[str, Any]] = None,
    ensure_watcher: bool = False,
) -> bool:
    """Incrementally refresh a cached index entry in-place.

    Returns True when the cached index supports incremental refresh and was
    updated successfully. Returns False when the index cannot be refreshed
    incrementally, leaving the caller to mark or treat the cache as stale.
    """

    index = cache_entry.get("index")
    if not hasattr(index, "incremental_reindex"):
        logger.warning(
            "[code_search] Index doesn't support incremental updates, marking stale: %s",
            root,
        )
        cache_entry["stale"] = True
        return False

    await index.incremental_reindex()
    await _finalize_index_storage(index)
    cache_entry["latest_mtime"] = _latest_mtime(root)
    cache_entry["stale"] = False

    if ensure_watcher:
        await _ensure_file_watcher_subscription(cache_entry, root, exec_ctx)

    return True


async def _on_file_change(
    event: "FileChangeEvent",
    root: Path,
    exec_ctx: Optional[Dict[str, Any]] = None,
) -> None:
    """Handle file change event for codebase index auto-update.

    Invalidates cache and triggers incremental update if index exists.

    Args:
        event: File change event from FileWatcherService
        root: Root path of codebase
        exec_ctx: Execution context for cache access
    """
    from victor.core.indexing.file_watcher import FileChangeType
    from victor.core.indexing.index_lock import IndexLockRegistry

    index_cache = _get_index_cache(exec_ctx)
    index_key = str(root)
    lock_registry = IndexLockRegistry.get_instance()
    path_lock = await lock_registry.acquire_lock(root)

    async with path_lock:
        cache_entry = index_cache.get(index_key)

        if not cache_entry:
            return  # No index exists, nothing to update

        if event.change_type in (
            FileChangeType.DELETED,
            FileChangeType.MODIFIED,
            FileChangeType.CREATED,
            FileChangeType.RENAMED,
        ):
            logger.info(
                "[code_search] File change detected (%s), triggering incremental update: %s",
                event.change_type.value,
                event.path,
            )

            try:
                refreshed = await _refresh_cached_index_incrementally(cache_entry, root)
                if refreshed:
                    logger.info(f"[code_search] Incremental update complete for {root}")
            except Exception as e:
                cache_entry["stale"] = True
                logger.error(f"[code_search] Incremental update failed: {e}")


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


_SEARCH_RESULT_METADATA_DROP_KEYS = {
    "vector",
    "embedding",
    "embeddings",
    "content",
    "file_path",
    "path",
    "score",
    "similarity",
    "combined_score",
    "semantic_score",
    "keyword_score",
}


def _sanitize_search_metadata(value: Any) -> Any:
    """Remove bulky or duplicate search metadata before tool return.

    Search hits only need compact grounding metadata for follow-up reads.
    Raw embeddings and duplicate payload fields add large token cost without
    helping the LLM act on the result.
    """
    if isinstance(value, dict):
        sanitized: Dict[str, Any] = {}
        for key, item in value.items():
            if key in _SEARCH_RESULT_METADATA_DROP_KEYS:
                continue
            sanitized[key] = _sanitize_search_metadata(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_search_metadata(item) for item in value]
    return value


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
        metadata = result_dict.get("metadata")
        if isinstance(metadata, dict):
            result_dict["metadata"] = _sanitize_search_metadata(metadata)

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
    """Build graph-tool follow-up suggestions from ranked search results.

    Optimized to avoid duplication:
    - Combines multiple modes into single suggestion using pipe separator
    - Truncates long node definitions to save tokens
    - Shows unique variations only (no duplicate node values)
    """
    for result in results:
        symbol = _extract_graph_follow_up_symbol(result)
        if symbol is None:
            continue

        symbol_name = symbol["name"]
        symbol_name_lower = symbol_name.lower()
        suggestions: List[Dict[str, Any]] = []

        # Truncate long definitions to save tokens (keep first 100 chars)
        if len(symbol_name) > 100:
            display_name = symbol_name[:97] + "..."
        else:
            display_name = symbol_name

        if symbol_name_lower in ENTRYPOINT_SYMBOL_NAMES:
            # Entry point gets separate trace suggestion
            suggestions.append(
                {
                    "tool": "graph",
                    "command": f'graph(mode="trace", node="{display_name}", depth=3)',
                    "arguments": {"mode": "trace", "node": symbol_name, "depth": 3},
                    "reason": f"Trace execution starting from {display_name}.",
                }
            )

        # Combine callers and callees into single suggestion with pipe separator
        # Instead of duplicating the entire node definition
        combined_modes = "callers|callees"
        depth = 2
        suggestions.append(
            {
                "tool": "graph",
                "command": f'graph(mode="{combined_modes}", node="{display_name}", depth={depth})',
                "arguments": {"mode": combined_modes, "node": symbol_name, "depth": depth},
                "reason": f"Explore call relationships for {display_name} (who calls it and what it calls).",
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
        "Use read with offset/limit based on line_number/end_line for precise reads. "
        "Example: read(path, offset=line_number-1, limit=end_line-line_number+5)"
    )
    if follow_up_suggestions:
        hint += f" Graph follow-up: {follow_up_suggestions[0]['command']}"

    metadata = {
        "rebuilt": rebuilt,
        "root": str(root_path),
        "indexed_at": cache_entry.get("indexed_at"),
        "filters_applied": filters_applied if filters_applied else None,
        "chunking_strategy": "provider_default",
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

    preview_results: List[Dict[str, Any]] = []
    for result in results:
        content = result.get("content", "")
        snippet = result.get("snippet")
        if not isinstance(snippet, str) or not snippet.strip():
            if isinstance(content, str):
                snippet = " ".join(line.strip() for line in content.splitlines()[:2] if line.strip())
            else:
                snippet = ""

        preview_results.append(
            {
                "path": result.get("file_path", result.get("path", "unknown")),
                "line": result.get("line_number", result.get("line", "?")),
                "score": result.get("score", result.get("combined_score", 0)),
                "snippet": snippet,
            }
        )

    formatted_result = format_search_results(
        {
            "results": preview_results,
            "mode": mode,
        }
    )

    return {
        "success": True,
        "results": results,
        "count": len(results),
        "mode": mode,
        "hint": hint,
        "ranking_note": ranking_note,
        "metadata": metadata,
        "formatted_results": formatted_result.content,
        "contains_markup": formatted_result.contains_markup,
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
    4. Per-path locking to prevent concurrent indexing (NEW)

    Args:
        root: Root path for the codebase
        settings: Application settings
        force_reindex: Force full re-index
        exec_ctx: Execution context for DI-based cache access
    """
    from victor.core.capability_registry import CapabilityRegistry
    from victor.framework.vertical_protocols import CodebaseIndexFactoryProtocol
    from victor.core.indexing.index_lock import IndexLockRegistry

    registry = CapabilityRegistry.get_instance()
    # Ensure plugins are bootstrapped so victor-coding registers its factory
    registry.ensure_bootstrapped()

    _index_factory = registry.get(CodebaseIndexFactoryProtocol)
    if _index_factory is None or not registry.is_enhanced(CodebaseIndexFactoryProtocol):
        # Recovery 1: force plugin capability re-discovery (plugin chain may have
        # been bootstrapped before victor-coding was installed, or an exception
        # was silently swallowed during _auto_register_vertical_capabilities)
        from victor.core.bootstrap import _discover_plugin_capabilities

        _discover_plugin_capabilities(None)
        _index_factory = registry.get(CodebaseIndexFactoryProtocol)

    if _index_factory is None or not registry.is_enhanced(CodebaseIndexFactoryProtocol):
        # Recovery 2: direct import — bypasses plugin→vertical→capability chain entirely.
        # This handles cases where the plugin chain fails silently (DEBUG-level exception
        # swallowing in _auto_register_vertical_capabilities) but the package IS installed.
        try:
            from victor.core.capability_registry import CapabilityStatus
            from victor.core.search.indexer import EnhancedCodebaseIndexFactory

            factory = EnhancedCodebaseIndexFactory()
            registry.register(CodebaseIndexFactoryProtocol, factory, CapabilityStatus.ENHANCED)
            _index_factory = factory
            logger.info(
                "[code_search] Recovered CodebaseIndex factory via direct import "
                "(victor-coding is installed, plugin chain failed)"
            )
        except ImportError:
            pass  # victor-coding genuinely not installed

    if _index_factory is None or not registry.is_enhanced(CodebaseIndexFactoryProtocol):
        # Enhanced error messages with installation guidance
        try:
            if importlib.util.find_spec("victor_coding") is not None:
                # Package is installed but factory not registered - guide user
                raise ImportError(
                    "CodebaseIndex factory not registered. The victor-coding package "
                    "is installed but failed to register its provider. Try:\n"
                    "  1. Restart your Python session\n"
                    "  2. Reinstall: pip install --force-reinstall victor-coding\n"
                    "  3. Check for errors during package import\n\n"
                    "For literal/keyword search only, use mode='literal' in code_search()."
                )
            else:
                # Package not installed - guide installation
                raise ImportError(
                    "CodebaseIndex requires victor-coding package for semantic search. "
                    "To enable semantic code search:\n"
                    "  pip install victor-coding\n\n"
                    "For literal/keyword search only, use mode='literal' in code_search()."
                )
        except ImportError:
            # Re-raise with enhanced message
            raise

    embedding_config = _build_codebase_embedding_config(settings, root)
    index_manifest = build_codebase_index_manifest(embedding_config)

    # Get cache using DI-aware accessor
    index_cache = _get_index_cache(exec_ctx)

    # Get or create failure cache for index build failures
    failure_cache = _get_index_build_failure_cache(exec_ctx)

    # Check for recent build failures before attempting build
    if failure_cache and not force_reindex:
        failure_key = _build_index_failure_key(root, index_manifest)
        failure_entry = _failure_cache_get(failure_cache, failure_key)
        if failure_entry:
            # Check if failure entry is still valid (hasn't expired)
            if hasattr(failure_entry, "is_expired") and not failure_entry.is_expired():
                logger.info(
                    f"[code_search] Skipping index build due to recent failure: {failure_entry.value.get('error', 'Unknown error')}"
                )
                raise ImportError(
                    f"Semantic index build recently failed: {failure_entry.value.get('error', 'Unknown error')}\n"
                    f"Use reindex=True to retry, or mode='literal' for keyword search."
                )
            elif isinstance(failure_entry, dict):
                # Fallback for simple dict cache (no expiry check)
                logger.info(
                    f"[code_search] Skipping index build due to recent failure: {failure_entry.get('error', 'Unknown error')}"
                )
                raise ImportError(
                    f"Semantic index build recently failed: {failure_entry.get('error', 'Unknown error')}\n"
                    f"Use reindex=True to retry, or mode='literal' for keyword search."
                )

    cache_entry = index_cache.get(str(root))
    cached_index = cache_entry["index"] if cache_entry else None
    last_mtime = cache_entry["latest_mtime"] if cache_entry else 0.0

    latest = _latest_mtime(root)

    # Check if we have a valid cached index
    if cached_index and not force_reindex:
        # Check if cache is marked as stale from file watcher
        if cache_entry.get("stale", False):
            logger.info(f"[code_search] Cache marked stale for {root}, will rebuild")
            # Fall through to rebuild below
        elif not _cache_entry_matches_manifest(cache_entry, index_manifest):
            logger.info("[code_search] Cache manifest mismatch for %s, rebuilding", root)
            # Fall through to rebuild below
        elif latest <= last_mtime:
            # No files changed, use cache directly
            # Subscribe to file watcher for auto-invalidation (only once per index)
            await _ensure_file_watcher_subscription(cache_entry, root, exec_ctx)

            return cached_index, False
        else:
            logger.info("[code_search] Cache changed for %s, refreshing under lock", root)

    # Acquire lock for this path to prevent concurrent indexing
    lock_registry = IndexLockRegistry.get_instance()
    path_lock = await lock_registry.acquire_lock(root)

    async with path_lock:
        # Double-check cache inside lock (another task may have built it while we waited)
        cache_entry = index_cache.get(str(root))
        cached_index = cache_entry["index"] if cache_entry else None
        last_mtime = cache_entry["latest_mtime"] if cache_entry else 0.0

        latest = _latest_mtime(root)

        # Check if we have a valid cached index (double-check)
        if cached_index and not force_reindex:
            if cache_entry.get("stale", False):
                logger.info("[code_search] Cache marked stale for %s (inside lock), rebuilding", root)
            elif not _cache_entry_matches_manifest(cache_entry, index_manifest):
                logger.info("[code_search] Cache manifest mismatch for %s (inside lock), rebuilding", root)
            elif latest <= last_mtime:
                # Another task built it while we waited for lock
                logger.info(f"[code_search] Cache hit for {root} (inside lock)")

                # Subscribe to file watcher if not already subscribed
                await _ensure_file_watcher_subscription(cache_entry, root, exec_ctx)

                return cached_index, False
            else:
                refreshed = await _refresh_cached_index_incrementally(
                    cache_entry,
                    root,
                    exec_ctx=exec_ctx,
                    ensure_watcher=True,
                )
                if refreshed:
                    logger.info(f"[code_search] Incremental refresh complete for {root} (inside lock)")
                    return cached_index, False

        # Build index with exclusive access to this path
        logger.info(f"[code_search] Building index for {root} (exclusive lock acquired)")

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

        logger.info(f"[code_search] Index creation complete for {root}")

        # Only do full index if forced or no persistent data exists
        persist_path = Path(embedding_config["persist_directory"])
        has_persistent_data = has_persisted_codebase_index_data(persist_path)
        has_compatible_manifest = has_compatible_codebase_index_manifest(persist_path, index_manifest)
        if has_persistent_data and not has_compatible_manifest:
            logger.info(
                "Persistent code_search index manifest mismatch for %s; rebuilding (%s)",
                persist_path,
                CODEBASE_INDEX_MANIFEST_NAME,
            )

        if force_reindex or not has_persistent_data or not has_compatible_manifest:
            # First time or forced - full index
            await index.index_codebase()
            await _finalize_index_storage(index)
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
            if rebuilt:
                await _finalize_index_storage(index)

        try:
            write_codebase_index_manifest(persist_path, index_manifest)
        except OSError as exc:
            logger.warning("Failed to write code_search index manifest to %s: %s", persist_path, exc)

        index_cache[str(root)] = {
            "index": index,
            "latest_mtime": latest,
            "indexed_at": time.time(),
            "index_manifest": index_manifest,
            "watcher_subscribed": False,  # Will be subscribed on next access
        }

        # Mark lock as used
        lock_registry.mark_lock_used(root)

        logger.info(f"[code_search] Index build complete for {root} (releasing lock)")

    # Clear failure cache on successful build
    try:
        failure_key = _build_index_failure_key(root, index_manifest)

        # Get failure cache (same logic as above)
        failure_cache = _get_index_build_failure_cache(exec_ctx)

        if failure_cache and _failure_cache_get(failure_cache, failure_key):
            _failure_cache_delete(failure_cache, failure_key)
            logger.info("[code_search] Cleared index build failure cache after successful build")
    except Exception as cache_err:
        logger.debug(f"[code_search] Failed to clear index build failure cache: {cache_err}")

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
        # Resolve empty or '.' to the framework's project root
        search_path = path
        if not search_path or search_path == ".":
            try:
                from victor.config.settings import get_project_paths

                search_path = str(get_project_paths().project_root)
            except Exception:
                search_path = "."  # Fallback to CWD

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

        # Filename detection: if query looks like a filename (has extension),
        # use find/rg --files instead of content search
        filename_exts = (
            ".py",
            ".js",
            ".ts",
            ".go",
            ".java",
            ".c",
            ".cpp",
            ".rs",
            ".yaml",
            ".yml",
            ".json",
            ".toml",
            ".md",
            ".txt",
            ".sh",
            ".rb",
            ".php",
            ".swift",
            ".kt",
            ".scala",
            ".r",
            ".sql",
        )
        query_lower = search_query.lower()
        is_filename_query = (
            any(query_lower.endswith(ext) for ext in filename_exts)
            and " " not in search_query
            and len(search_query) < 100
        )

        if is_filename_query:
            import platform

            system = platform.system()
            logger.info(
                f"Filename query detected: using {'dir' if system == 'Windows' else 'find'} "
                f"for {search_query!r} on {system}"
            )

            if system == "Windows":
                # Windows: use dir /s /b for recursive file search
                find_cmd = ["cmd", "/c", "dir", "/s", "/b", f"*{search_query}*"]
                find_cwd = search_path
            else:
                # Linux / macOS: use find
                find_cmd = ["find", search_path, "-type", "f", "-name", f"*{search_query}*"]
                find_cwd = None

            try:
                # Use asyncio subprocess to avoid blocking event loop
                proc = await asyncio.create_subprocess_exec(
                    *find_cmd,
                    cwd=find_cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
                find_result = type(
                    "obj", (object,), {"stdout": stdout.decode("utf-8", errors="ignore")}
                )
                found_files = [f.strip() for f in find_result.stdout.splitlines() if f.strip()]
                if found_files:
                    results = []
                    for fpath in found_files[:k]:
                        results.append(
                            {
                                "path": fpath,
                                "score": 10 if fpath.endswith(search_query) else 5,
                                "snippet": f"[File found: {fpath}]",
                            }
                        )
                    logger.info(
                        f"Filename search: found {len(found_files)} files matching "
                        f"{search_query!r} in {search_path}"
                    )
                    return {
                        "success": True,
                        "results": results,
                        "count": len(results),
                        "mode": "filename",
                    }
            except (subprocess.TimeoutExpired, Exception) as e:
                logger.debug(f"Filename search failed, falling back to content: {e}")
            # Fall through to content search if find returns nothing

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

        # Use asyncio subprocess to avoid blocking event loop
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        result = type("obj", (object,), {"stdout": stdout.decode("utf-8", errors="ignore")})

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
        "find file",
        "locate file",
        "grep",
        "literal",
        "code search",
        "filename",
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
    timeout=300.0,  # Increased to 5 minutes for large codebases (was 60s)
)
async def code_search(
    query: str,
    path: str = ".",
    k: int = 10,
    mode: str = "semantic",
    reindex: bool = False,
    filters: Optional[SearchFilters] = None,
    _exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Find code by CONCEPT, TEXT, or FILENAME when you DON'T know exact location.

    Use this tool for exploration when you need to discover where relevant code
    lives. Returns file snippets ranked by relevance.

    Modes:
    - semantic: Vector search using embeddings (default)
    - text: Literal text search
    - filename: Search by filename pattern

    Args:
        query: Search query (code concept, text, or filename pattern)
        path: Root directory to search (default: ".")
        k: Maximum number of results to return (default: 10)
        mode: Search mode - 'semantic', 'text', or 'filename' (default: 'semantic')
        reindex: Force rebuild of search index (default: False)
        filters: SearchFilters object with file_pattern, symbol, language, test_only, extensions

    Returns:
        Dictionary with search results including matches, scores, file paths

    Example:
        # Simple semantic search
        code_search(query="dataclass validation", path=".")

        # Search with filters
        filters = SearchFilters(
            language="python",
            test_only=True,
            extensions=[".py"]
        )
        code_search(query="pytest fixtures", path=".", filters=filters)

        # Filename search (auto-detected from query)
        code_search(query="*.py", path=".")

    DIFFERS FROM:
    - symbol(): Gets FULL CODE of a known symbol. Use when you know file + name.
    - refs(): Finds USAGE locations of a known symbol. Use for "where is X called".
    - graph(): Shows RELATIONSHIPS between symbols. Use for dependencies/impact.

    Legacy modes:
    - "literal": Keyword matching (like grep). Best for exact text/identifiers.
    - "bugs": Similar bug search with graph context when supported by the provider.
    - "localize": File-level issue localization using semantic seeds plus graph expansion.
    - "impact": Change-impact / blast-radius analysis using graph expansion when available.
    """
    # Auto-detect filename search mode
    if filters and filters.file_pattern and not filters.symbol:
        # If only file_pattern is provided, treat as filename search
        if mode == "semantic":
            mode = "filename"

    # Route to literal search if mode is "literal" or "text" (alias)
    if mode in ("literal", "text"):
        exts = filters.extensions if filters else None
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
    exts: Optional[List[str]] = filters.extensions if filters else None
    search_root = path
    # Resolve empty path to project root (benchmark/agent sets this)
    if not search_root or search_root == ".":
        try:
            from victor.config.settings import get_project_paths

            search_root = str(get_project_paths().project_root)
        except Exception:
            search_root = "."
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

        # Check if path is a subdirectory to provide smart hints
        # This helps avoid duplicate embeddings while supporting cross-repo analysis
        try:
            from victor.config.settings import get_project_paths

            project_root = Path(get_project_paths().project_root).resolve()

            # Check if root_path is within the same git repository as project root
            # If they're in different repos, no warning needed (cross-repo analysis)
            def get_git_repo_root(path: Path) -> Optional[Path]:
                """Find the git repository root for a given path."""
                current = path
                while current != current.parent:  # Stop at filesystem root
                    git_dir = current / ".git"
                    if git_dir.exists():
                        return current
                    current = current.parent
                return None

            root_repo = get_git_repo_root(root_path)
            project_repo = get_git_repo_root(project_root)

            # Only warn if they're in the same repo and root_path is a subdirectory
            if root_repo and project_repo and root_repo == project_repo:
                try:
                    root_path.relative_to(project_root)
                    is_subdirectory = True
                except ValueError:
                    is_subdirectory = False

                if is_subdirectory and root_path != project_root:
                    # Path is a subdirectory within the same repo
                    rel_path = root_path.relative_to(project_root)
                    logger.info(
                        f"[code_search] Searching in subdirectory '{rel_path}' (same repo as project root). "
                        f"Consider using path='.' for full repository coverage if broader context is needed."
                    )
                    # Proceed with subdirectory search - user may have specific intent
            elif root_repo and project_repo and root_repo != project_repo:
                # Different repos - cross-repo analysis, no warning needed
                logger.debug(
                    f"[code_search] Cross-repo analysis: searching in separate repo at {root_repo}"
                )
        except Exception:
            # If we can't determine repo boundaries, proceed with provided path
            pass

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
            exts = filters.extensions if filters else None
            return await _literal_search(query, path, k, exts)

        # Build metadata filter from optional parameters
        filter_metadata: Optional[Dict[str, Any]] = None
        filters_applied = []

        if filters and any(
            [filters.file_pattern, filters.symbol, filters.language, filters.test_only is not None]
        ):
            filter_metadata = {}
            if filters.file_pattern:
                filter_metadata["file_path"] = filters.file_pattern
                filters_applied.append(f"file={filters.file_pattern}")
            if filters.symbol:
                filter_metadata["symbol_type"] = filters.symbol
                filters_applied.append(f"symbol={filters.symbol}")
            if filters.language:
                filter_metadata["language"] = filters.language
                filters_applied.append(f"lang={filters.language}")
            if filters.test_only is not None:
                filter_metadata["is_test_file"] = filters.test_only
                filters_applied.append(f"test={filters.test_only}")

        try:
            index, rebuilt = await asyncio.wait_for(
                _get_or_build_index(root_path, settings, force_reindex=reindex, exec_ctx=_exec_ctx),
                timeout=180.0,  # Increased to 3 minutes for large codebases (was 30s)
            )
        except (asyncio.TimeoutError, Exception) as exc:
            error_msg = str(exc)
            logger.warning("Semantic index build failed (%s), falling back to literal search", exc)

            # Cache the failure for 1 hour to prevent repeated attempts
            try:
                failure_key = _build_index_failure_key(
                    root_path,
                    build_codebase_index_manifest(_build_codebase_embedding_config(settings, root_path)),
                )

                # Get failure cache (same logic as in _get_or_build_index)
                failure_cache = _get_index_build_failure_cache(_exec_ctx)

                if failure_cache:
                    from victor.tools.cache_manager import GenericCacheEntry

                    failure_entry = GenericCacheEntry(
                        value={"error": error_msg, "timestamp": time.time()},
                        ttl=3600,  # 1 hour
                    )
                    _failure_cache_set(failure_cache, failure_key, failure_entry)
                    logger.info("[code_search] Cached index build failure for 1 hour")
            except Exception as cache_err:
                logger.debug(f"[code_search] Failed to cache index build failure: {cache_err}")

            exts = filters.extensions if filters else None
            result = await _literal_search(query, path, k, exts)
            result["fallback"] = _classify_semantic_fallback(exc, scope="semantic_index")
            return result

        backend_metadata = _collect_code_search_backend_metadata(index, settings)

        if mode == "bugs":
            ignored_filters = []
            if filters:
                ignored_filters = [
                    name
                    for name, value in (
                        ("file", filters.file_pattern),
                        ("symbol", filters.symbol),
                        ("test", filters.test_only),
                    )
                    if value is not None
                ]
            try:
                bug_results = await index.find_similar_bugs(
                    bug_description=query,
                    language=filters.language if filters else None,
                    top_k=k,
                    include_graph_context=True,
                    context_limit=min(max(1, k), 3),
                )
                ranked_results = _prepare_ranked_results(
                    bug_results,
                    search_mode="bug_similarity",
                )
                ranked_results, enrichment_metadata = enrich_code_search_results(
                    ranked_results,
                    root_path=root_path,
                )
                extra_metadata = {**backend_metadata, "provider_capability": "find_similar_bugs"}
                extra_metadata.update(enrichment_metadata)
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
        elif mode == "localize":
            ignored_filters = []
            if filters:
                ignored_filters = [
                    name
                    for name, value in (
                        ("file", filters.file_pattern),
                        ("symbol", filters.symbol),
                        ("test", filters.test_only),
                    )
                    if value is not None
                ]
            try:
                localize_issue = getattr(index, "localize_issue", None)
                if localize_issue is None:
                    raise NotImplementedError("localize_issue unsupported")

                localization_results = await localize_issue(
                    issue_description=query,
                    language=filters.language if filters else None,
                    top_k=k,
                    include_graph_context=True,
                    context_limit=min(max(1, k), 3),
                )
                ranked_results = _prepare_ranked_results(
                    localization_results,
                    search_mode="issue_localization",
                )
                ranked_results, enrichment_metadata = enrich_code_search_results(
                    ranked_results,
                    root_path=root_path,
                )
                extra_metadata = {**backend_metadata, "provider_capability": "localize_issue"}
                extra_metadata.update(enrichment_metadata)
                if ignored_filters:
                    extra_metadata["ignored_filters"] = ignored_filters
                follow_up_suggestions = _build_graph_follow_up_suggestions(ranked_results)
                return _build_search_response(
                    results=ranked_results,
                    mode="localize",
                    rebuilt=rebuilt,
                    root_path=root_path,
                    exec_ctx=_exec_ctx,
                    filters_applied=filters_applied,
                    ranking_note="Results ranked by combined_score (issue_localization × importance). Graph-expanded file candidates included when available.",
                    extra_metadata=extra_metadata,
                    follow_up_suggestions=follow_up_suggestions,
                )
            except NotImplementedError as exc:
                logger.info(
                    "Issue localization mode is unsupported by %s; falling back to semantic search",
                    type(index).__name__,
                )
                filters_applied.append("mode_fallback=semantic")
                fallback_metadata = {
                    "requested_mode": "localize",
                    "fallback_mode": "semantic",
                    "fallback_reason": str(exc),
                }
        elif mode == "impact":
            ignored_filters = []
            if filters:
                ignored_filters = [
                    name
                    for name, value in (
                        ("file", filters.file_pattern),
                        ("symbol", filters.symbol),
                        ("test", filters.test_only),
                    )
                    if value is not None
                ]
            try:
                analyze_change_impact = getattr(index, "analyze_change_impact", None)
                if analyze_change_impact is None:
                    raise NotImplementedError("analyze_change_impact unsupported")

                impact_results = await analyze_change_impact(
                    change_description=query,
                    language=filters.language if filters else None,
                    top_k=k,
                    include_graph_context=True,
                    context_limit=min(max(1, k), 3),
                )
                ranked_results = _prepare_ranked_results(
                    impact_results,
                    search_mode="change_impact",
                )
                ranked_results, enrichment_metadata = enrich_code_search_results(
                    ranked_results,
                    root_path=root_path,
                )
                extra_metadata = {**backend_metadata, "provider_capability": "analyze_change_impact"}
                extra_metadata.update(enrichment_metadata)
                if ignored_filters:
                    extra_metadata["ignored_filters"] = ignored_filters
                follow_up_suggestions = _build_graph_follow_up_suggestions(ranked_results)
                return _build_search_response(
                    results=ranked_results,
                    mode="impact",
                    rebuilt=rebuilt,
                    root_path=root_path,
                    exec_ctx=_exec_ctx,
                    filters_applied=filters_applied,
                    ranking_note="Results ranked by combined_score (change_impact × importance). Graph-expanded blast-radius candidates included when available.",
                    extra_metadata=extra_metadata,
                    follow_up_suggestions=follow_up_suggestions,
                )
            except NotImplementedError as exc:
                logger.info(
                    "Impact mode is unsupported by %s; falling back to semantic search",
                    type(index).__name__,
                )
                filters_applied.append("mode_fallback=semantic")
                fallback_metadata = {
                    "requested_mode": "impact",
                    "fallback_mode": "semantic",
                    "fallback_reason": str(exc),
                }
        else:
            fallback_metadata = dict(backend_metadata)

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
            result["fallback"] = _classify_semantic_fallback(exc, scope="semantic_search")
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
        ranked_results, enrichment_metadata = enrich_code_search_results(
            ranked_results,
            root_path=root_path,
        )
        ranked_results, utility_metadata = rerank_code_search_results(
            ranked_results,
            query=query,
        )
        fallback_metadata.update(enrichment_metadata)
        fallback_metadata.update(backend_metadata)
        fallback_metadata["retrieval_utility"] = utility_metadata
        follow_up_suggestions = _build_graph_follow_up_suggestions(ranked_results)

        return _build_search_response(
            results=ranked_results,
            mode="semantic",
            rebuilt=rebuilt,
            root_path=root_path,
            exec_ctx=_exec_ctx,
            filters_applied=filters_applied,
            ranking_note="Results ranked by utility-adjusted score (semantic/hybrid relevance × importance, then bounded file-diversity and duplicate-snippet adjustments).",
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
