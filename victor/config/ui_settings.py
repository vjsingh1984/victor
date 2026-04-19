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
