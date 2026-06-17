"""Agent identity naming helpers.

Stable IDs are persistence keys. Display names are human-facing labels and can
change without breaking stored state.
"""

from __future__ import annotations

import re
import uuid
from enum import Enum
from typing import Optional

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "all",
    "analyze",
    "by",
    "codebase",
    "create",
    "do",
    "find",
    "for",
    "from",
    "go",
    "implement",
    "in",
    "into",
    "list",
    "make",
    "map",
    "of",
    "on",
    "review",
    "research",
    "the",
    "this",
    "to",
    "update",
    "using",
    "with",
}


def generate_agent_id(role: object, *, prefix: str = "agent") -> str:
    """Generate a stable internal agent ID suitable for persistence keys."""
    return f"{prefix}_{_slug(_role_value(role))}_{uuid.uuid4().hex[:12]}"


def build_display_name(
    role: object,
    *,
    task: Optional[str] = None,
    ordinal: Optional[object] = None,
) -> str:
    """Build a readable display name from role, task subject, and optional ordinal."""
    role_name = _title_words(_role_value(role))
    subject = _task_subject(task)
    base = f"{subject} {role_name}" if subject else role_name
    return f"{base} {ordinal}" if ordinal is not None else base


def build_member_id(
    role: object, *, task: Optional[str] = None, ordinal: Optional[object] = None
) -> str:
    """Build a deterministic member-slot ID from role and optional task context."""
    subject = _slug(_task_subject(task) or "")
    role_slug = _slug(_role_value(role))
    pieces = [
        piece
        for piece in (subject, role_slug, str(ordinal) if ordinal is not None else "")
        if piece
    ]
    return "_".join(pieces) or role_slug or "member"


def _role_value(role: object) -> str:
    if isinstance(role, Enum):
        return str(role.value)
    return str(role)


def _task_subject(task: Optional[str]) -> str:
    if not task:
        return ""
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_+#.-]*", task)
    selected = []
    for token in tokens:
        normalized = token.strip("_-.").lower()
        if not normalized or normalized in _STOP_WORDS:
            continue
        selected.append(_display_token(token))
        if len(selected) == 2:
            break
    return " ".join(selected)


def _display_token(token: str) -> str:
    token = token.strip("_-.")
    if token.isupper() or any(ch.isdigit() for ch in token):
        return token
    if token.lower() in {"api", "orm", "llm", "tdd", "sql", "sqlite", "rust"}:
        return token.upper() if token.lower() != "rust" else "Rust"
    return token.replace("_", " ").title()


def _title_words(value: str) -> str:
    return " ".join(_display_token(part) for part in re.split(r"[_\s-]+", value) if part)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "agent"
