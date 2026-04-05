"""Null task type hinter stub.

Returns empty hints when victor-coding is not installed.
Enhanced by victor-coding when installed.
"""

from __future__ import annotations


class NullTaskTypeHinter:
    """Stub task type hinter that returns empty hints."""

    def get_hint(self, task_type: str) -> str:
        """Return empty string — no hints available."""
        return ""
