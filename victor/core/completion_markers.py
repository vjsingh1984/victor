"""Shared terminal completion markers for prompt and runtime coordination.

These markers are intentionally rare and line-anchored so normal prose like
"DONE:" or "summary:" does not accidentally terminate an agent run.
"""

from __future__ import annotations

import re
from typing import Dict, Pattern

FILE_DONE_MARKER = "VICTOR_FILE_DONE::"
TASK_DONE_MARKER = "VICTOR_TASK_DONE::"
SUMMARY_MARKER = "VICTOR_SUMMARY::"
BLOCKED_MARKER = "VICTOR_BLOCKED::"

ACTIVE_COMPLETION_MARKERS: tuple[str, ...] = (
    FILE_DONE_MARKER,
    TASK_DONE_MARKER,
    SUMMARY_MARKER,
    BLOCKED_MARKER,
)


def _compile_line_start_pattern(marker: str) -> Pattern[str]:
    """Compile a case-insensitive line-start matcher for a terminal marker."""
    return re.compile(rf"(?:^|\n)\s*{re.escape(marker)}", re.IGNORECASE)


ACTIVE_COMPLETION_MARKER_PATTERNS: Dict[str, Pattern[str]] = {
    marker: _compile_line_start_pattern(marker) for marker in ACTIVE_COMPLETION_MARKERS
}


def detect_active_completion_marker(text: str) -> str | None:
    """Return the first matching active completion marker in text, if any."""
    for marker in ACTIVE_COMPLETION_MARKERS:
        if ACTIVE_COMPLETION_MARKER_PATTERNS[marker].search(text):
            return marker
    return None


def strip_active_completion_markers(text: str) -> str:
    """Remove visible completion marker tokens while preserving their payload text.

    These markers are runtime coordination tokens, not user-facing prose. The
    returned text is suitable for display while still allowing the raw response
    to be used for completion detection elsewhere.
    """
    if not text:
        return text

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        replaced = False
        for marker in ACTIVE_COMPLETION_MARKERS:
            if stripped.upper().startswith(marker):
                payload = stripped[len(marker) :].strip()
                if payload:
                    cleaned_lines.append(payload)
                replaced = True
                break
        if not replaced:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()
