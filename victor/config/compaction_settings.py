"""Session compaction configuration for managing context window limits."""

from __future__ import annotations

from pydantic import BaseModel


class CompactionSettings(BaseModel):
    """Session compaction configuration for managing context window limits.

    When conversation history exceeds token thresholds, older messages
    are summarized to free context space while preserving recent history.
    """

    compaction_enabled: bool = True
    compaction_preserve_recent: int = 4
    compaction_max_estimated_tokens: int = 10000
    compaction_auto_compact: bool = False
