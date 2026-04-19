"""Pre/post tool use hook configuration."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class HooksSettings(BaseModel):
    """Pre/post tool use hook configuration.

    Hooks are shell commands that run before and after tool execution.
    They can allow, deny, or warn based on exit codes:
      - Exit 0: Allow (stdout as optional message)
      - Exit 2: Deny (stdout as denial reason)
      - Other: Warn but allow
    """

    hooks_enabled: bool = True
    hooks_pre_tool_use: List[str] = Field(default_factory=list)
    hooks_post_tool_use: List[str] = Field(default_factory=list)
