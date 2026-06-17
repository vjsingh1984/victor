"""Framework protocol definitions.

These protocols define how verticals interact with the Victor framework.
"""

from victor_contracts.framework.protocols.orchestrator import Orchestrator
from victor_contracts.framework.protocols.teams import TeamRegistry
from victor_contracts.framework.protocols.workflows import WorkflowCompiler

__all__ = [
    "Orchestrator",
    "TeamRegistry",
    "WorkflowCompiler",
]
