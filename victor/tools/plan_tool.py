import os
from typing import Any, Dict, List, Optional

from victor.tools.decorators import tool

EXCLUDE_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "web/ui/node_modules"}


def _safe_walk(root: str) -> List[str]:
    files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")]
        for fname in filenames:
            files.append(os.path.join(dirpath, fname))
    return files


@tool
async def plan_files(
    root: str = ".",
    patterns: Optional[List[str]] = None,
    limit: int = 5,
) -> Dict[str, Any]:
    """
    Plan which files to inspect (evidence-focused).

    Finds up to `limit` existing files under `root` matching optional substrings in `patterns`.
    Use this to pick a small set of real files before calling read_file.
    """
    try:
        limit = max(1, min(limit, 10))
        if isinstance(patterns, str):
            patterns = [p.strip() for p in patterns.split(",") if p.strip()]
        patterns = patterns or []
        all_files = _safe_walk(root)
        matches: List[str] = []
        for path in all_files:
            if patterns:
                if any(p.lower() in path.lower() for p in patterns):
                    matches.append(path)
            else:
                matches.append(path)
            if len(matches) >= limit:
                break

        return {
            "success": True,
            "files": matches[:limit],
            "note": "Use these paths with read_file; do not invent other files.",
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}
