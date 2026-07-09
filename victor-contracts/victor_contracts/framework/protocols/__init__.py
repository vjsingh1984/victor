"""Framework protocol definitions."""

from victor_contracts.framework.protocols.orchestrator import Orchestrator
from victor_contracts.framework.protocols.teams import (
    SupervisorAgent,
    TeamAgent,
    TeamRegistry,
)
from victor_contracts.framework.protocols.workflows import WorkflowCompiler

__all__ = [
    "Orchestrator",
    "SupervisorAgent",
    "TeamAgent",
    "TeamRegistry",
    "WorkflowCompiler",
]
