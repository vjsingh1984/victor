"""Pydantic state models for workflow execution.

This module provides Pydantic v2 models for workflow state, replacing
TypedDict with better type checking, validation, and serialization.

Migration Guide:
    OLD: from victor.workflows.context import ExecutionContext
    NEW: from victor.workflows.models import WorkflowExecutionContextModel

Benefits:
    - Type checking with mypy
    - Runtime validation
    - Better error messages
    - Automatic serialization
    - IDE autocomplete
"""

from victor.workflows.models.execution_context import (
    WorkflowExecutionContextModel,
    WorkflowStateModel,
)

__all__ = [
    "WorkflowExecutionContextModel",
    "WorkflowStateModel",
]
