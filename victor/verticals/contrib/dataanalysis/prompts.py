"""Data Analysis Prompt Contributor - Task hints for data science workflows."""

from __future__ import annotations

from typing import Optional

from victor.core.verticals.prompt_adapter import PromptContributorAdapter
from victor.verticals.contrib.dataanalysis.prompt_metadata import (
    DATA_ANALYSIS_GROUNDING_RULES,
    DATA_ANALYSIS_PROMPT_PRIORITY,
    DATA_ANALYSIS_SYSTEM_PROMPT_SECTION,
    DATA_ANALYSIS_TASK_TYPE_HINTS,
)


class DataAnalysisPromptContributor(PromptContributorAdapter):
    """Contributes data analysis-specific prompts and task hints."""

    def __init__(self) -> None:
        adapter = PromptContributorAdapter.from_dict(
            task_hints=DATA_ANALYSIS_TASK_TYPE_HINTS,
            system_prompt_section=DATA_ANALYSIS_SYSTEM_PROMPT_SECTION,
            grounding_rules=DATA_ANALYSIS_GROUNDING_RULES,
            priority=DATA_ANALYSIS_PROMPT_PRIORITY,
        )
        super().__init__(
            task_hints=adapter.get_task_type_hints(),
            system_prompt_section=adapter.get_system_prompt_section(),
            grounding_rules=adapter.get_grounding_rules(),
            priority=adapter.get_priority(),
        )

    def get_context_hints(self, task_type: Optional[str] = None) -> Optional[str]:
        """Return contextual hints based on detected task type."""
        if task_type and task_type in DATA_ANALYSIS_TASK_TYPE_HINTS:
            return str(DATA_ANALYSIS_TASK_TYPE_HINTS[task_type]["hint"])
        return None
