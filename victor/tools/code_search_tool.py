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


def _latest_mtime(root: Path) -> float:
    """Find latest modification time under root, respecting EXCLUDE_DIRS."""
    return latest_mtime(root)


def _gather_files(root: str, exts: Optional[List[str]], max_files: int) -> List[str]:
    """Gather files from directory tree."""
    ext_set: Set[str] = set(exts) if exts else DEFAULT_EXTS
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
    """Return cached CodebaseIndex or build a new one. Returns (index, rebuilt?)."""
    cache_entry = _INDEX_CACHE.get(str(root))
    cached_index = cache_entry["index"] if cache_entry else None
    last_mtime = cache_entry["latest_mtime"] if cache_entry else 0.0

    latest = _latest_mtime(root)
    needs_rebuild = force_reindex or not cached_index or latest > last_mtime

    if cached_index and not needs_rebuild:
        return cached_index, False

    from victor.codebase.indexer import CodebaseIndex

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
        "persist_directory": getattr(settings, "codebase_persist_directory", None),
        "extra_config": {},
    }

    index = CodebaseIndex(
        root_path=str(root),
        use_embeddings=True,
        embedding_config=embedding_config,
    )
    await index.index_codebase()
    _INDEX_CACHE[str(root)] = {
        "index": index,
        "latest_mtime": latest,
        "indexed_at": time.time(),
    }
    return index, True


@tool
async def code_search(
    query: str,
    root: str = ".",
    k: int = 5,
    max_files: int = 200,
    max_chars: int = 4000,
    exts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Lightweight code search: find relevant files/chunks for a query.

    Scans files under `root`, scores them by keyword match, and returns top-k with snippets.
    Use this before read_file to pick real targets. Caps files and snippet lengths for speed.
    """
    try:
        files = _gather_files(root, exts, max_files)
        scores: List[Dict[str, Any]] = []
        for path in files:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read(max_chars)
                score = _keyword_score(text, query)
                if score > 0:
                    scores.append({"path": path, "score": score, "snippet": text[:800]})
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
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Semantic code search using the embedding-backed indexer.

    Builds (or reuses) an embedding index for the codebase and returns the top-k
    matches with file paths, scores, and line numbers. Reindexes automatically
    when files change; use force_reindex to rebuild on demand.
    """
    try:
        root_path = Path(root).resolve()
        if not root_path.exists():
            return {"success": False, "error": f"Root not found: {root}"}

        settings = context.get("settings") if context else None
        if settings is None:
            return {"success": False, "error": "Settings not available in tool context."}

        index, rebuilt = await _get_or_build_index(root_path, settings, force_reindex=force_reindex)
        results = await index.semantic_search(query=query, max_results=k)

        return {
            "success": True,
            "results": results,
            "count": len(results),
            "metadata": {
                "rebuilt": rebuilt,
                "root": str(root_path),
                "indexed_at": _INDEX_CACHE[str(root_path)]["indexed_at"],
            },
        }
    except ImportError as exc:
        return {
            "success": False,
            "error": f"Semantic search dependencies missing: {exc}. Install lancedb/chromadb + sentence-transformers.",
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}
