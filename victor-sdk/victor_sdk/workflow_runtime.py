"""SDK host adapters for workflow runtime helpers.

These names are intentionally hosted in the SDK so extracted verticals can
depend on ``victor_sdk`` instead of importing ``victor.framework`` directly.
They remain runtime-backed adapters rather than pure definition-layer
contracts, so they should be treated as a compatibility seam on the migration
path toward descriptor-first workflows.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from victor.framework.workflows import BaseYAMLWorkflowProvider
    from victor.workflows.definition import WorkflowBuilder, WorkflowDefinition, workflow

__all__ = [
    "BaseYAMLWorkflowProvider",
    "WorkflowBuilder",
    "WorkflowDefinition",
    "workflow",
]


_LAZY_IMPORTS = {
    "BaseYAMLWorkflowProvider": "victor.framework.workflows",
    "WorkflowBuilder": "victor.workflows.definition",
    "WorkflowDefinition": "victor.workflows.definition",
    "workflow": "victor.workflows.definition",
}


def __getattr__(name: str):
    """Resolve workflow runtime helpers lazily from the Victor host runtime."""
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'victor_sdk.workflow_runtime' has no attribute {name!r}")

    module = importlib.import_module(module_name)
    return getattr(module, name)
