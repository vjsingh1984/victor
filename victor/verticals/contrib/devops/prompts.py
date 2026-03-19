"""DevOps Prompt Contributor - Task hints and system prompt extensions for infrastructure."""

from __future__ import annotations

from typing import Dict, Optional

from victor.core.verticals.prompt_adapter import PromptContributorAdapter
from victor.verticals.contrib.devops.prompt_metadata import (
    DEVOPS_GROUNDING_RULES,
    DEVOPS_PROMPT_PRIORITY,
    DEVOPS_SYSTEM_PROMPT_SECTION,
    DEVOPS_TASK_TYPE_HINTS,
)


class DevOpsPromptContributor(PromptContributorAdapter):
    """Contributes DevOps-specific prompts and task hints."""

    def __init__(self) -> None:
        adapter = PromptContributorAdapter.from_dict(
            task_hints=DEVOPS_TASK_TYPE_HINTS,
            system_prompt_section=DEVOPS_SYSTEM_PROMPT_SECTION,
            grounding_rules=DEVOPS_GROUNDING_RULES,
            priority=DEVOPS_PROMPT_PRIORITY,
        )
        super().__init__(
            task_hints=adapter.get_task_type_hints(),
            system_prompt_section=adapter.get_system_prompt_section(),
            grounding_rules=adapter.get_grounding_rules(),
            priority=adapter.get_priority(),
        )

    def get_context_hints(self, task_type: Optional[str] = None) -> Optional[str]:
        """Return contextual hints based on detected task type."""
        if task_type and task_type in DEVOPS_TASK_TYPE_HINTS:
            return str(DEVOPS_TASK_TYPE_HINTS[task_type]["hint"])
        return None
