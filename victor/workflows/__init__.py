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

"""Workflow definition and execution system.

This package provides a LangGraph-like workflow DSL for defining
reusable multi-agent workflows as Python code.

Example:
    from victor.workflows import WorkflowBuilder, workflow, WorkflowRegistry

    @workflow("code_review", "Review code quality")
    def code_review_workflow():
        return (
            WorkflowBuilder("code_review")
            .add_agent("analyze", "researcher", "Find code patterns")
            .add_agent("review", "reviewer", "Review quality")
            .add_agent("report", "planner", "Summarize findings")
            .build()
        )

    # Execute
    executor = WorkflowExecutor(orchestrator)
    result = await executor.execute("code_review", {"files": ["main.py"]})
"""

from victor.workflows.base import BaseWorkflow
from victor.workflows.definition import (
    NodeType,
    WorkflowNode,
    AgentNode,
    ConditionNode,
    ParallelNode,
    TransformNode,
    WorkflowDefinition,
    WorkflowBuilder,
    workflow,
    get_registered_workflows,
)
from victor.workflows.registry import (
    WorkflowMetadata,
    WorkflowRegistry,
    get_global_registry,
)
from victor.workflows.executor import (
    NodeStatus,
    NodeResult,
    WorkflowContext,
    WorkflowResult,
    WorkflowExecutor,
)
from victor.workflows.yaml_loader import (
    YAMLWorkflowError,
    YAMLWorkflowConfig,
    YAMLWorkflowProvider,
    load_workflow_from_dict,
    load_workflow_from_yaml,
    load_workflow_from_file,
    load_workflows_from_directory,
)
from victor.workflows.cache import (
    WorkflowCacheConfig,
    CacheEntry,
    WorkflowCache,
    WorkflowCacheManager,
    get_workflow_cache_manager,
    configure_workflow_cache,
)

__all__ = [
    # Base
    "BaseWorkflow",
    # Node types
    "NodeType",
    "WorkflowNode",
    "AgentNode",
    "ConditionNode",
    "ParallelNode",
    "TransformNode",
    # Definition
    "WorkflowDefinition",
    "WorkflowBuilder",
    "workflow",
    "get_registered_workflows",
    # Registry
    "WorkflowMetadata",
    "WorkflowRegistry",
    "get_global_registry",
    # Executor
    "NodeStatus",
    "NodeResult",
    "WorkflowContext",
    "WorkflowResult",
    "WorkflowExecutor",
    # YAML Loader
    "YAMLWorkflowError",
    "YAMLWorkflowConfig",
    "YAMLWorkflowProvider",
    "load_workflow_from_dict",
    "load_workflow_from_yaml",
    "load_workflow_from_file",
    "load_workflows_from_directory",
    # Cache
    "WorkflowCacheConfig",
    "CacheEntry",
    "WorkflowCache",
    "WorkflowCacheManager",
    "get_workflow_cache_manager",
    "configure_workflow_cache",
]
