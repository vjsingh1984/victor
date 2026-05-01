# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Workflow domain facade for orchestrator decomposition.

Groups workflow registry, execution, optimization, and coordination-advisor
components behind a single interface.

This facade wraps already-initialized components from the orchestrator,
providing a coherent grouping without changing initialization ordering.
The orchestrator delegates property access through this facade.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Optional

logger = logging.getLogger(__name__)
_DEPRECATED_MODE_WORKFLOW_TEAM_COORDINATOR = object()


class WorkflowFacade:
    """Groups workflow registry, runtime, and optimization components.

    Satisfies ``WorkflowFacadeProtocol`` structurally.  The orchestrator creates
    this facade after all workflow-domain components are initialized, passing
    references to the already-created instances.

    Components managed:
        - workflow_registry: Workflow registry for workflow lookup
        - workflow_runtime: Workflow runtime boundary components
        - workflow_optimization: Workflow optimization components
        - coordination_advisor: Framework-facing team/workflow advisor surface
    """

    def __init__(
        self,
        *,
        workflow_registry: Optional[Any] = None,
        workflow_runtime: Optional[Any] = None,
        workflow_optimization: Optional[Any] = None,
        coordination_advisor: Optional[Any] = None,
        mode_workflow_team_coordinator: Any = _DEPRECATED_MODE_WORKFLOW_TEAM_COORDINATOR,
    ) -> None:
        if mode_workflow_team_coordinator is not _DEPRECATED_MODE_WORKFLOW_TEAM_COORDINATOR:
            warnings.warn(
                "WorkflowFacade(mode_workflow_team_coordinator=...) is deprecated. "
                "Use WorkflowFacade(coordination_advisor=...) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if coordination_advisor is None:
                coordination_advisor = mode_workflow_team_coordinator

        self._workflow_registry = workflow_registry
        self._workflow_runtime = workflow_runtime
        self._workflow_optimization = workflow_optimization
        self._coordination_advisor = coordination_advisor

        logger.debug(
            "WorkflowFacade initialized (registry=%s, optimization=%s, advisor=%s)",
            workflow_registry is not None,
            workflow_optimization is not None,
            self._coordination_advisor is not None,
        )

    # ------------------------------------------------------------------
    # Properties (satisfy WorkflowFacadeProtocol)
    # ------------------------------------------------------------------

    @property
    def workflow_registry(self) -> Optional[Any]:
        """Workflow registry for workflow lookup."""
        return self._workflow_registry

    @workflow_registry.setter
    def workflow_registry(self, value: Any) -> None:
        """Update the workflow registry."""
        self._workflow_registry = value

    @property
    def workflow_runtime(self) -> Optional[Any]:
        """Workflow runtime boundary components."""
        return self._workflow_runtime

    @property
    def workflow_optimization(self) -> Optional[Any]:
        """Workflow optimization components."""
        return self._workflow_optimization

    @property
    def coordination_advisor(self) -> Optional[Any]:
        """Framework-facing advisor for intelligent team/workflow suggestions."""
        return self._coordination_advisor

    @coordination_advisor.setter
    def coordination_advisor(self, value: Any) -> None:
        """Update the framework-facing coordination advisor."""
        self._coordination_advisor = value

    @property
    def mode_workflow_team_coordinator(self) -> Optional[Any]:
        """Compatibility alias for the coordination advisor surface."""
        warnings.warn(
            "WorkflowFacade.mode_workflow_team_coordinator is deprecated. "
            "Use WorkflowFacade.coordination_advisor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._coordination_advisor

    @mode_workflow_team_coordinator.setter
    def mode_workflow_team_coordinator(self, value: Any) -> None:
        """Update the compatibility coordinator alias."""
        warnings.warn(
            "WorkflowFacade.mode_workflow_team_coordinator is deprecated. "
            "Set WorkflowFacade.coordination_advisor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._coordination_advisor = value
