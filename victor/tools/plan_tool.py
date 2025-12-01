from typing import Any, Dict, List, Optional

from victor.tools.common import safe_walk
from victor.tools.decorators import tool


def _safe_walk(root: str) -> List[str]:
    """Walk directory tree safely, excluding common non-code directories."""
    return safe_walk(root)


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
