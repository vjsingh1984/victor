"""Sandbox/isolation configuration for safe tool execution."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class SandboxSettings(BaseModel):
    """Sandbox/isolation configuration for safe tool execution.

    Controls filesystem isolation, process namespace restrictions,
    and environment variable filtering when executing tools.
    """

    sandbox_enabled: bool = False
    sandbox_filesystem_mode: str = "workspace-only"
    sandbox_namespace_restrictions: bool = True
    sandbox_network_isolation: bool = False
    sandbox_allowed_mounts: List[str] = Field(default_factory=list)
