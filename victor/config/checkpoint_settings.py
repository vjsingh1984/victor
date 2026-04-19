"""State checkpointing for time-travel debugging."""

from __future__ import annotations

from pydantic import BaseModel


class CheckpointSettings(BaseModel):
    """State checkpointing for time-travel debugging."""

    checkpoint_enabled: bool = True
    checkpoint_auto_interval: int = 5
    checkpoint_max_per_session: int = 50
    checkpoint_compression_enabled: bool = True
    checkpoint_compression_threshold: int = 1024
