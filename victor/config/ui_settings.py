"""User interface display preferences."""

from __future__ import annotations

import os

from pydantic import BaseModel, Field


class UISettings(BaseModel):
    """User interface display preferences."""

    theme: str = "monokai"
    show_token_count: bool = True
    show_cost_metrics: bool = False
    stream_responses: bool = True
    use_emojis: bool = Field(
        default_factory=lambda: not os.getenv("CI", "false").lower() == "true",
    )
    cli_history_max_entries: int = Field(
        default=250,
        ge=10,
        le=1000,
        description="Maximum number of entries in CLI chat history file. "
        "Larger values provide more history but slow down typing.",
    )
