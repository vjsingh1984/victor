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

This module provides backward compatibility by re-exporting from canonical locations:
- victor.workflows.unified_executor (WorkflowExecutor, StateGraphExecutor)
- victor.workflows.context (WorkflowContext, WorkflowResult, TemporalContext)
- victor_sdk.workflows (ExecutorNodeStatus, NodeResult)
- victor.workflows.compute_registry (compute handler functions)

New code should import directly from those modules instead.
"""

import warnings

# Re-export WorkflowExecutor from canonical location
from victor.workflows.unified_executor import (
    CompiledWorkflowExecutor,
    StateGraphExecutor,
)

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
    "Import WorkflowExecutor from victor.workflows.unified_executor instead. "
    "Import types from victor_sdk.workflows and victor.workflows.context.",
    DeprecationWarning,
    stacklevel=2,
)

# Backward compatibility alias
WorkflowExecutor = CompiledWorkflowExecutor

# Chain handler prefix for YAML workflow references
# Used to identify chain handlers in workflow definitions: "chain:vertical:name"
CHAIN_HANDLER_PREFIX = "chain:"


# Dummy function for test patching (deprecated)
def get_chain_registry():
    """Deprecated: Chain registry is no longer used."""
    raise NotImplementedError(
        "get_chain_registry is deprecated. Chain execution is handled differently now."
    )


__all__ = [
    # Executor (canonical, from unified_executor)
    "WorkflowExecutor",
    "CompiledWorkflowExecutor",
    "StateGraphExecutor",
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
]
