"""Git automation, headless mode, and provider fallback."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class AutomationSettings(BaseModel):
    """Git automation, headless mode, and provider fallback."""

    auto_commit_enabled: bool = False
    headless_mode: bool = False
    dry_run_mode: bool = False
    auto_approve_safe: bool = False
    one_shot_mode: bool = False
    max_file_changes: Optional[int] = None
    provider_health_checks: bool = True
    provider_auto_fallback: bool = True
    fallback_providers: List[str] = Field(default_factory=list)
