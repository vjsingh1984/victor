"""Framework protocol definitions."""

from victor_sdk.framework.protocols.orchestrator import Orchestrator
from victor_sdk.framework.protocols.teams import TeamRegistry
from victor_sdk.framework.protocols.workflows import WorkflowCompiler

__all__ = [
    "Orchestrator",
    "TeamRegistry",
    "WorkflowCompiler",
]
