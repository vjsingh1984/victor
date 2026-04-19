"""Prompt metadata mixin for VerticalBase."""

from __future__ import annotations

from typing import Any, Dict

from victor_sdk.core.types import (
    PromptMetadata,
    normalize_prompt_templates,
    normalize_task_type_hints,
)


class PromptMetadataMixin:
    """Opt-in mixin providing prompt template and task-type hint hooks.

    Methods:
        get_prompt_templates: Return task-specific prompt templates.
        get_task_type_hints: Return task-type hints.
        get_prompt_metadata: Return assembled PromptMetadata.
    """

    @classmethod
    def get_prompt_templates(cls) -> Dict[str, Any]:
        """Return task-specific prompt templates for this vertical."""
        return {}

    @classmethod
    def get_task_type_hints(cls) -> Dict[str, Any]:
        """Return task-type hints for this vertical."""
        return {}

    @classmethod
    def get_prompt_metadata(cls) -> PromptMetadata:
        """Return serializable prompt metadata for this vertical."""
        return PromptMetadata(
            templates=normalize_prompt_templates(cls.get_prompt_templates()),
            task_type_hints=normalize_task_type_hints(cls.get_task_type_hints()),
        )
