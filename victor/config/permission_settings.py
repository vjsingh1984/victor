"""Permission hierarchy for tool access control."""

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field


class PermissionSettings(BaseModel):
    """Permission hierarchy for tool access control.

    Three-tier permission model: read-only, workspace-write, danger-full-access.
    Controls which tools agents can use based on session permission level.
    """

    permission_mode: str = "workspace-write"
    permission_prompt_on_escalation: bool = True
    permission_tool_overrides: Dict[str, str] = Field(default_factory=dict)
