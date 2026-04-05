"""Framework protocol definitions.

These protocols define how verticals interact with the Victor framework.
"""

from victor_sdk.framework.protocols.orchestrator import Orchestrator
from victor_sdk.framework.protocols.teams import TeamRegistry
from victor_sdk.framework.protocols.workflows import WorkflowCompiler

__all__ = [
    "Orchestrator",
    "TeamRegistry",
    "WorkflowCompiler",
]
