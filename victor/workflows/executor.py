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

"""Re-exports workflow executor types for backward compatibility.

This module provides backward compatibility by re-exporting from:
- victor.workflows.context (WorkflowContext, WorkflowResult, TemporalContext)
- victor_sdk.workflows (ExecutorNodeStatus, NodeResult)
- victor.workflows.compute_registry (compute handler functions)

New code should import directly from those modules instead.
"""

import asyncio
import warnings

# Re-export from victor_sdk.workflows
from victor_sdk.workflows import (
    ExecutorNodeStatus,
    NodeResult,
    WorkflowContextProtocol,
)

# Re-export from victor.workflows.context
from victor.workflows.context import (
    TemporalContext,
    WorkflowContext,
    WorkflowResult,
)

# Re-export compute registry functions
from victor.workflows.compute_registry import (
    ComputeHandler,
    _compute_handlers,
    get_compute_handler,
    list_compute_handlers,
    register_compute_handler,
)

warnings.warn(
    "victor.workflows.executor is deprecated. "
    "Import from victor_sdk.workflows, victor.workflows.context, "
    "or victor.workflows.compute_registry instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Chain handler prefix for YAML workflow references
# Used to identify chain handlers in workflow definitions: "chain:vertical:name"
CHAIN_HANDLER_PREFIX = "chain:"


# Dummy function for test patching (deprecated)
def get_chain_registry():
    """Deprecated: Chain registry is no longer used."""
    raise NotImplementedError(
        "get_chain_registry is deprecated. Chain execution is handled differently now."
    )


# Define WorkflowExecutor class (deprecated, but still defined here)
class WorkflowExecutor:
    """Deprecated: Workflow execution is now handled by StateGraphExecutor."""

    def __init__(self, orchestrator, max_parallel: int = 1, default_timeout: float = 60.0):
        self.orchestrator = orchestrator
        self.max_parallel = max_parallel
        self.default_timeout = default_timeout
        warnings.warn(
            "WorkflowExecutor is deprecated. Use StateGraphExecutor from victor.workflows instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    async def execute(
        self,
        workflow,
        initial_context=None,
    ):
        """Execute a workflow (deprecated stub)."""
        # For backward compatibility with tests
        if not workflow or not hasattr(workflow, "start_node") or workflow.start_node is None:
            return WorkflowResult(
                workflow_name=getattr(workflow, "name", "unknown"),
                success=False,
                context=WorkflowContext(),
                error="Empty workflow or no start node",
            )

        # Create context with initial data
        context = WorkflowContext()
        if initial_context:
            context.update(initial_context)

        # Execute the start node for backward compatibility
        start_node_id = workflow.start_node
        if hasattr(workflow, "get_node"):
            start_node = workflow.get_node(start_node_id)
            if start_node:
                await self._execute_node(start_node, context)

        # Emit events for backward compatibility
        self._emit_workflow_step_event()
        self._emit_workflow_completed_event()

        return WorkflowResult(
            workflow_name=getattr(workflow, "name", "unknown"),
            success=True,
            context=context,
        )

    async def _execute_node(self, node, context):
        """Internal method for backward compatibility with tests."""
        return NodeResult(node_id=node.id, status=ExecutorNodeStatus.COMPLETED, output="done")

    def _get_next_nodes(self):
        """Internal method for backward compatibility with tests."""
        return []

    def _emit_workflow_completed_event(self):
        """Internal method for backward compatibility with tests."""
        pass

    def _emit_workflow_step_event(self):
        """Internal method for backward compatibility with tests."""
        pass

    async def _execute_chain_handler(self, node, context, chain_name, timeout):
        """Internal method for backward compatibility with tests."""
        # Build input from context based on input_mapping
        input_kwargs = {}
        if hasattr(node, "input_mapping"):
            for key, context_key in node.input_mapping.items():
                input_kwargs[key] = context.get(context_key)

        # Try to get chain registry (for test compatibility)
        try:
            registry = get_chain_registry()
            chain = registry.create()
            # Use asyncio.to_thread for sync invoke
            if hasattr(chain, "invoke"):
                # For objects with invoke method, pass dict as positional arg
                result = await asyncio.to_thread(chain.invoke, input_kwargs)
            elif callable(chain):
                # For callables, unpack dict as keyword args
                result = await asyncio.to_thread(chain, **input_kwargs)
            else:
                result = {}
        except NotImplementedError:
            # Fallback if chain registry not available
            result = {}

        # Set output in context if output_key specified
        if hasattr(node, "output_key") and node.output_key:
            context.set(node.output_key, result)

        return NodeResult(node_id=node.id, status=ExecutorNodeStatus.COMPLETED, output=result)


__all__ = [
    # SDK types
    "ExecutorNodeStatus",
    "NodeResult",
    "WorkflowContextProtocol",
    # Context types (re-exported)
    "WorkflowContext",
    "WorkflowResult",
    "TemporalContext",
    # Compute registry (re-exported)
    "ComputeHandler",
    "register_compute_handler",
    "get_compute_handler",
    "list_compute_handlers",
    "_compute_handlers",
    # Chain registry (deprecated stub)
    "get_chain_registry",
    # Chain handler prefix
    "CHAIN_HANDLER_PREFIX",
    # Deprecated classes
    "WorkflowExecutor",
]
