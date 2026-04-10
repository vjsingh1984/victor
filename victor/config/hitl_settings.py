"""Human-in-the-loop workflow interrupt configuration."""

from __future__ import annotations

from pydantic import BaseModel


class HITLSettings(BaseModel):
    """Human-in-the-loop workflow interrupt configuration."""

    hitl_default_timeout: float = 300.0
    hitl_default_fallback: str = "abort"
    hitl_auto_approve_low_risk: bool = False
    hitl_keyboard_shortcuts_enabled: bool = True
