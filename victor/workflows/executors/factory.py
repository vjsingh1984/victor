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

"""Unified node executor factory for workflow nodes.

This factory consolidates the duplicate NodeExecutorFactory implementations
from:
- victor/workflows/unified_compiler.py (lines 171-600)
- victor/workflows/yaml_to_graph_compiler.py (lines 168-600)

Design Pattern: Factory
- Maps node types to executor functions
- Supports registration of custom node types (OCP compliance)
- Protocol-based dependency injection (DIP compliance)

SOLID Compliance:
- SRP: ONLY creates executor functions (doesn't execute them)
- OCP: Open for extension via register_executor_type()
- DIP: Depends on ExecutionContextProtocol, not concrete classes
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:
    from victor.core.container import ServiceContainer
    from victor.workflows.definition import WorkflowNode, WorkflowState
    from victor.workflows.compiler_protocols import NodeExecutorFactoryProtocol

logger = logging.getLogger(__name__)


class NodeExecutorFactory:
    """Factory for creating StateGraph node executor functions.

    This factory converts YAML workflow node definitions into async functions
    that can be used as StateGraph nodes.

    Design Principles:
    - SRP: ONLY creates executor functions (doesn't execute them)
    - OCP: Open for extension via register_executor_type()
    - DIP: Depends on ExecutionContextProtocol, not concrete classes

    Attributes:
        _container: DI container for resolving dependencies
        _executor_types: Registered node executor classes

    Example:
        factory = NodeExecutorFactory(container=container)
        executor_fn = factory.create_executor(agent_node)
        result_state = await executor_fn(initial_state)
    """

    def __init__(self, container: Optional["ServiceContainer"] = None):
        """Initialize the factory.

        Args:
            container: DI container for resolving dependencies
        """
        from victor.core.container import ServiceContainer

        self._container: ServiceContainer = container or ServiceContainer()
        self._executor_types: Dict[str, Any] = {}

    def register_executor_type(
        self,
        node_type: str,
        executor_class: Any,
        *,
        replace: bool = False,
    ) -> None:
        """Register a custom node executor type.

        Args:
            node_type: Node type identifier (e.g., "custom_compute")
            executor_class: Executor class (not instance) that implements
                          NodeExecutorProtocol
            replace: Whether to replace existing registration

        Raises:
            ValueError: If node_type already exists and replace=False

        Example:
            from victor.workflows.executors.agent import AgentNodeExecutor

            factory.register_executor_type("agent", AgentNodeExecutor)
        """
        if node_type in self._executor_types and not replace:
            raise ValueError(
                f"Node type '{node_type}' already registered. " f"Use replace=True to override."
            )

        self._executor_types[node_type] = executor_class
        logger.debug(f"Registered executor type: {node_type} -> {executor_class.__name__}")

    def create_executor(self, node: "WorkflowNode") -> Callable[["WorkflowState"], "WorkflowState"]:
        """Create an executor function for a workflow node.

        This is the main factory method. It dispatches to the appropriate
        executor class based on the node type.

        Args:
            node: Workflow node definition

        Returns:
            Callable: Async executor function that takes WorkflowState and
                     returns WorkflowState

        Raises:
            ValueError: If node type is not supported

        Example:
            factory = NodeExecutorFactory()
            executor_fn = factory.create_executor(agent_node)
            result_state = await executor_fn(initial_state)
        """
        from victor.workflows.definition import WorkflowNodeType

        # Get executor class for this node type
        executor_class = self._get_executor_class(node.node_type)

        # Create executor instance with execution context
        # Note: Executors will resolve dependencies from container as needed
        executor = executor_class(context=self._container)

        # Return wrapper function that matches StateGraph signature
        async def executor_wrapper(state: "WorkflowState") -> "WorkflowState":
            return await executor.execute(node, state)

        return executor_wrapper

    def _get_executor_class(self, node_type: "WorkflowNodeType") -> Any:
        """Get the executor class for a given node type.

        Args:
            node_type: Node type enum

        Returns:
            Executor class

        Raises:
            ValueError: If node type is not supported
        """
        from victor.workflows.definition import WorkflowNodeType

        # Import executor classes
        from victor.workflows.executors.agent import AgentNodeExecutor
        from victor.workflows.executors.compute import ComputeNodeExecutor
        from victor.workflows.executors.condition import ConditionNodeExecutor
        from victor.workflows.executors.parallel import ParallelNodeExecutor
        from victor.workflows.executors.transform import TransformNodeExecutor

        # Map node types to executor classes
        executor_map = {
            WorkflowNodeType.AGENT: AgentNodeExecutor,
            WorkflowNodeType.COMPUTE: ComputeNodeExecutor,
            WorkflowNodeType.TRANSFORM: TransformNodeExecutor,
            WorkflowNodeType.PARALLEL: ParallelNodeExecutor,
            WorkflowNodeType.CONDITION: ConditionNodeExecutor,
        }

        # Check custom registered types first
        if node_type.value in self._executor_types:
            return self._executor_types[node_type.value]

        # Use standard mapping
        if node_type in executor_map:
            return executor_map[node_type]

        # Fallback to legacy for unknown types during migration
        logger.warning(f"Unknown node type '{node_type.value}', using legacy implementation")
        return self._get_legacy_executor_class(node_type)

    def _get_legacy_executor_class(self, node_type: "WorkflowNodeType") -> Any:
        """Get legacy executor class for backward compatibility.

        During migration, falls back to legacy NodeExecutorFactory
        for unknown node types.

        Args:
            node_type: Node type enum

        Returns:
            Legacy executor class

        Deprecated:
            This method is temporary and will be removed once all node types
            have dedicated executor implementations.
        """
        # Import legacy factory
        from victor.workflows.yaml_to_graph_compiler import NodeExecutorFactory as LegacyFactory

        # For now, return the legacy factory itself
        # The legacy factory's create_executor method will be used
        return LegacyFactory

    def _create_legacy_executor(
        self, node: "WorkflowNode"
    ) -> Callable[["WorkflowState"], "WorkflowState"]:
        """Create executor using legacy implementation (temporary compatibility).

        This delegates to the existing NodeExecutorFactory in
        yaml_to_graph_compiler.py during the migration phase.

        Args:
            node: Workflow node definition

        Returns:
            Legacy executor function

        Deprecated:
            This method is temporary and will be removed once migration is complete.
        """
        # Import legacy factory
        from victor.workflows.yaml_to_graph_compiler import NodeExecutorFactory as LegacyFactory

        # Create legacy factory instance
        # Note: This will need orchestrator and tool_registry from container
        legacy_factory = LegacyFactory(
            orchestrator=None,  # Will be set by execution context
            orchestrators=None,  # Will be set by execution context
            tool_registry=None,  # Will be set by execution context
        )

        # Delegate to legacy implementation
        return legacy_factory.create_executor(node)

    def supports_node_type(self, node_type: str) -> bool:
        """Check if a node type is supported.

        Args:
            node_type: Node type identifier

        Returns:
            bool: True if node type is supported

        Example:
            if factory.supports_node_type("agent"):
                executor = factory.create_executor(agent_node)
        """
        from victor.workflows.definition import WorkflowNodeType

        # Standard node types that are supported
        standard_types = {
            WorkflowNodeType.AGENT.value,
            WorkflowNodeType.COMPUTE.value,
            WorkflowNodeType.TRANSFORM.value,
            WorkflowNodeType.PARALLEL.value,
            WorkflowNodeType.CONDITION.value,
        }

        # Check standard types
        if node_type in standard_types:
            return True

        # Check custom registered types
        if node_type in self._executor_types:
            return True

        # During migration, legacy types are also supported
        legacy_types = {"hitl"}  # Human-in-the-loop, etc.
        return node_type in legacy_types


__all__ = [
    "NodeExecutorFactory",
]
