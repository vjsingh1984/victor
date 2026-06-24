# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared search helpers for the unified command tools.

These pure functions back the literal-search surfaces that several unified
dispatchers reach for:

* :func:`grep_search` — literal/regex content search across files. Used by
  ``code grep`` (the canonical literal content search) and as the graceful
  fallback for ``code search`` when the semantic ``victor_coding`` package is
  absent. Also used by the legacy ``search grep`` back-compat shim.

Extracted here (out of ``search_tool.py``) so the helpers survive the
``search`` domain's retirement and can be reused without re-importing a
deprecated tool module.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

# Directories never worth scanning for code search.
_SEARCH_SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules"}


async def grep_search(
    query: str,
    path: str,
    regex: bool = False,
    case_sensitive: bool = False,
) -> List[Dict[str, Any]]:
    """Search file contents and return grep-like match dictionaries.

    Args:
        query: Text or regex pattern to search for.
        path: File or directory to search under.
        regex: When ``True``, treat ``query`` as a regular expression;
            otherwise escape it for a literal substring match.
        case_sensitive: When ``True``, match case-sensitively; otherwise
            case-insensitively (the default, matching grep ``-i``).

    Returns:
        List of ``{"file": str, "line": int, "content": str}`` dicts, one per
        matching line.
    """
    root = Path(path).expanduser()
    targets = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(query if regex else re.escape(query), flags)
    results: List[Dict[str, Any]] = []
    for file_path in targets:
        if any(part in _SEARCH_SKIP_DIRS for part in file_path.parts):
            continue
        try:
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line_no, line in enumerate(lines, start=1):
            if pattern.search(line):
                results.append(
                    {
                        "file": str(file_path),
                        "line": line_no,
                        "content": line,
                    }
                )
    return results


def format_grep_results(results: List[Dict[str, Any]], *, limit: int = 50) -> str:
    """Render grep matches as ``file:line: content`` lines with a truncation hint.

    Args:
        results: Output of :func:`grep_search`.
        limit: Maximum number of matches to render before truncating.

    Returns:
        A human-readable string; ``"No matches found."`` when empty.
    """
    if not isinstance(results, list):
        return str(results)

    out: List[str] = []
    for i, match in enumerate(results):
        if i >= limit:
            break
        file_path = match.get("file", "unknown")
        line_number = match.get("line", "?")
        content = match.get("content", "").strip()
        out.append(f"{file_path}:{line_number}: {content}")

    if len(results) > limit:
        out.append(
            f"\n### 💡 SYSTEM HINT\nToo many matches found ({len(results)}). "
            "Results truncated. Please refine your search query or directory."
        )

    if not out:
        return "No matches found."
    return "\n".join(out)


__all__ = ["grep_search", "format_grep_results"]
