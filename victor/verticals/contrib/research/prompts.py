"""Research Prompt Contributor - Task hints and system prompt extensions for research."""

from __future__ import annotations

from typing import Optional

from victor.core.verticals.prompt_adapter import PromptContributorAdapter
from victor.verticals.contrib.research.prompt_metadata import (
    RESEARCH_GROUNDING_RULES,
    RESEARCH_PROMPT_PRIORITY,
    RESEARCH_SYSTEM_PROMPT_SECTION,
    RESEARCH_TASK_TYPE_HINTS,
)


class ResearchPromptContributor(PromptContributorAdapter):
    """Contributes research-specific prompts and task hints."""

    def __init__(self) -> None:
        adapter = PromptContributorAdapter.from_dict(
            task_hints=RESEARCH_TASK_TYPE_HINTS,
            system_prompt_section=RESEARCH_SYSTEM_PROMPT_SECTION,
            grounding_rules=RESEARCH_GROUNDING_RULES,
            priority=RESEARCH_PROMPT_PRIORITY,
        )
        super().__init__(
            task_hints=adapter.get_task_type_hints(),
            system_prompt_section=adapter.get_system_prompt_section(),
            grounding_rules=adapter.get_grounding_rules(),
            priority=adapter.get_priority(),
        )

    def get_context_hints(self, task_type: Optional[str] = None) -> Optional[str]:
        """Return contextual hints based on detected task type."""
        if task_type and task_type in RESEARCH_TASK_TYPE_HINTS:
            return str(RESEARCH_TASK_TYPE_HINTS[task_type]["hint"])
        return None
