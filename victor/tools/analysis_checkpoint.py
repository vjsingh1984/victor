"""Persistent analysis checkpoint tool for cross-session codebase analysis caching.

Checkpoints are stored in .victor/analysis/{slug}-analysis.md with YAML frontmatter
tracking source file mtimes. Staleness is detected by comparing current mtimes against
recorded mtimes via the graph_file_mtime DB table (filesystem fallback when unavailable).

Token economics: reading a cached checkpoint (~500 tokens) vs re-running a deep analysis
(8,000–15,000 tokens) gives a 15–30x saving when source files haven't changed.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority
from victor.tools.decorators import tool

_ANALYSIS_DIR_REL = ".victor/analysis"
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(topic: str) -> str:
    return _SLUG_RE.sub("-", topic.lower().strip()).strip("-")


def _analysis_dir() -> Path:
    d = Path.cwd() / _ANALYSIS_DIR_REL
    d.mkdir(parents=True, exist_ok=True)
    return d


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Return (frontmatter_dict, body_text). Raises ValueError if malformed."""
    if not text.startswith("---"):
        raise ValueError("No YAML frontmatter")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError("Unclosed YAML frontmatter")
    return yaml.safe_load(parts[1]) or {}, parts[2].lstrip()


def _check_staleness(source_files: List[Dict[str, Any]]) -> List[str]:
    """Return list of source file paths whose current mtime exceeds the recorded mtime."""
    stale: List[str] = []
    try:
        from victor.core.database import get_project_database

        project_db = get_project_database()
        for entry in source_files:
            path, recorded = entry["path"], float(entry["mtime"])
            rows = project_db.query("SELECT mtime FROM graph_file_mtime WHERE file = ?", (path,))
            if rows:
                current = float(dict(rows[0])["mtime"])
            else:
                try:
                    current = Path(path).stat().st_mtime
                except OSError:
                    continue  # file removed — treat as not stale (can't compare)
            if current > recorded:
                stale.append(path)
    except Exception:
        # DB unavailable — fall back to direct filesystem comparison
        for entry in source_files:
            try:
                current = Path(entry["path"]).stat().st_mtime
                if current > float(entry["mtime"]):
                    stale.append(entry["path"])
            except OSError:
                pass
    return stale


@tool(
    category="analysis",
    priority=Priority.HIGH,
    access_mode=AccessMode.MIXED,
    danger_level=DangerLevel.SAFE,
    execution_category=ExecutionCategory.READ_ONLY,
    keywords=["analysis", "checkpoint", "cache", "reuse", "staleness", "manifest", "cached"],
    stages=["initial", "planning", "analysis"],
)
async def analysis_checkpoint(
    mode: str,
    topic: str,
    source_files: Optional[List[str]] = None,
    content: Optional[str] = None,
    _exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Read or write a persistent analysis checkpoint in .victor/analysis/.

    Checkpoints store Markdown analysis output with YAML frontmatter recording which
    source files the analysis covers and their modification times at write time.
    On read, staleness is auto-detected via the graph_file_mtime DB table — if any
    source file has been modified since the checkpoint was written, status is "stale".

    ALWAYS call this with mode="read" before starting any deep structural analysis.
    If status is "current", use the cached content and skip the analysis entirely.

    Modes:
      read  — Check for an existing checkpoint for this topic.
              Returns {"status": "current", "content": <str>} if up to date.
              Returns {"status": "stale", "stale_files": [...]} if source changed.
              Returns {"status": "not_found"} if no checkpoint exists yet.
      write — Persist analysis content with auto-generated frontmatter.
              Requires: source_files (list of relative paths) and content (markdown).
      list  — List all checkpoints with their current staleness status.
              topic is ignored in list mode.

    File naming convention: topic is slugified →
      .victor/analysis/{slug}-analysis.md
    Examples:
      "orchestrator structure"  → orchestrator-structure-analysis.md
      "CLI layering violations" → cli-layering-violations-analysis.md

    Args:
        mode: "read" | "write" | "list"
        topic: Human-readable topic name (e.g. "orchestrator structure")
        source_files: (write only) Relative paths of source files this analysis covers.
                      Used to detect staleness on future reads.
        content: (write only) Full Markdown body of the analysis.

    Returns:
        read:  {"status": "current"|"stale"|"not_found", "content": str|None,
                "stale_files": list[str], "path": str, "created_at": str|None}
        write: {"written": True, "path": str, "topic": str, "source_files": int}
        list:  {"checkpoints": list[dict], "count": int}
    """
    analysis_dir = _analysis_dir()

    if mode == "read":
        slug = _slug(topic)
        path = analysis_dir / f"{slug}-analysis.md"
        if not path.exists():
            return {
                "status": "not_found",
                "content": None,
                "stale_files": [],
                "path": str(path),
                "created_at": None,
            }
        try:
            fm, body = _parse_frontmatter(path.read_text())
        except (ValueError, Exception):
            return {
                "status": "not_found",
                "content": None,
                "stale_files": [],
                "path": str(path),
                "created_at": None,
            }
        stale = _check_staleness(fm.get("source_files", []))
        if stale:
            return {
                "status": "stale",
                "content": None,
                "stale_files": stale,
                "path": str(path),
                "created_at": fm.get("created_at"),
            }
        return {
            "status": "current",
            "content": body,
            "stale_files": [],
            "path": str(path),
            "created_at": fm.get("created_at"),
        }

    elif mode == "write":
        if not source_files or content is None:
            return {"error": "write mode requires both source_files and content"}
        slug = _slug(topic)
        path = analysis_dir / f"{slug}-analysis.md"
        file_entries = []
        for rel_path in source_files:
            try:
                mtime = Path(rel_path).stat().st_mtime
            except OSError:
                mtime = 0.0
            file_entries.append({"path": rel_path, "mtime": mtime})
        fm = {
            "topic": topic,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_files": file_entries,
        }
        frontmatter_text = yaml.dump(fm, default_flow_style=False, allow_unicode=True)
        path.write_text(f"---\n{frontmatter_text}---\n\n{content}")
        return {
            "written": True,
            "path": str(path),
            "topic": topic,
            "source_files": len(file_entries),
        }

    elif mode == "list":
        results = []
        for md_file in sorted(analysis_dir.glob("*-analysis.md")):
            try:
                fm, _ = _parse_frontmatter(md_file.read_text())
                stale = _check_staleness(fm.get("source_files", []))
                results.append(
                    {
                        "topic": fm.get("topic", md_file.stem),
                        "path": str(md_file),
                        "status": "stale" if stale else "current",
                        "created_at": fm.get("created_at"),
                        "source_files": [e["path"] for e in fm.get("source_files", [])],
                        "stale_files": stale,
                    }
                )
            except Exception:
                results.append(
                    {
                        "topic": md_file.stem,
                        "path": str(md_file),
                        "status": "unknown",
                    }
                )
        return {"checkpoints": results, "count": len(results)}

    return {"error": f"Unknown mode: {mode!r}. Use 'read', 'write', or 'list'"}
