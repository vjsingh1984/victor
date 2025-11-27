import os
from typing import Any, Dict, List, Optional

from victor.tools.decorators import tool

EXCLUDE_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "web/ui/node_modules"}
DEFAULT_EXTS = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml"}


def _gather_files(root: str, exts: Optional[List[str]], max_files: int) -> List[str]:
    exts = set(exts) if exts else DEFAULT_EXTS
    files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")]
        for fname in filenames:
            if os.path.splitext(fname)[1] in exts:
                files.append(os.path.join(dirpath, fname))
                if len(files) >= max_files:
                    return files
    return files


def _keyword_score(text: str, query: str) -> int:
    q = query.lower().split()
    t = text.lower()
    return sum(t.count(word) for word in q)


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
