"""Workflow metadata mixin for VerticalBase."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from victor_sdk.core.types import WorkflowMetadata


class WorkflowMetadataMixin:
    """Opt-in mixin providing workflow metadata hooks.

    These methods compute derived metadata from the vertical's stage
    definitions. No external vertical currently overrides them.

    Methods:
        get_initial_stage: Return the first stage name.
        get_workflow_spec: Return stage ordering dict.
        get_provider_hints: Return provider selection hints.
        get_evaluation_criteria: Return evaluation criteria.
        get_workflow_metadata: Return assembled WorkflowMetadata.
    """

    @classmethod
    def get_initial_stage(cls) -> Optional[str]:
        """Return the initial stage name for this vertical workflow."""
        stages = cls.get_stages()
        return next(iter(stages.keys()), None)

    @classmethod
    def get_workflow_spec(cls) -> Dict[str, Any]:
        """Return serializable workflow metadata for this vertical."""
        return {"stage_order": list(cls.get_stages().keys())}

    @classmethod
    def get_provider_hints(cls) -> Dict[str, Any]:
        """Return provider selection hints for this vertical."""
        return {}

    @classmethod
    def get_evaluation_criteria(cls) -> List[str]:
        """Return evaluation criteria for this vertical."""
        return []

    @classmethod
    def get_workflow_metadata(cls) -> WorkflowMetadata:
        """Return serializable workflow metadata for this vertical."""
        return WorkflowMetadata(
            initial_stage=cls.get_initial_stage(),
            workflow_spec=cls.get_workflow_spec(),
            provider_hints=cls.get_provider_hints(),
            evaluation_criteria=cls.get_evaluation_criteria(),
        )
