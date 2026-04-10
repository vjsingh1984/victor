"""System prompt enforcement and fallback templates."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class PromptPolicySettings(BaseModel):
    """System prompt enforcement and fallback templates."""

    prompt_policy_enforce_identity: bool = True
    prompt_policy_enforce_guidelines: bool = True
    prompt_policy_enforce_operating_preamble: bool = True
    prompt_policy_enforce_unique_sections: bool = True
    prompt_policy_protected_sections: List[str] = Field(
        default_factory=lambda: ["identity", "guidelines", "operating_mode"]
    )
    prompt_policy_max_section_chars: int = 18000
    prompt_policy_identity: Optional[str] = None
    prompt_policy_guidelines: Optional[str] = None
    prompt_policy_operating_template: Optional[str] = None
    prompt_policy_fallback_template: Optional[str] = None
