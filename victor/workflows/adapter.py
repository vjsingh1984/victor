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

"""Adapter layer for backward compatibility during SOLID refactoring.

Provides adapters that wrap the new compiler/executor with legacy interfaces.
Emits deprecation warnings to guide migration.

Design:
- Adapter Pattern: Wraps new interfaces with legacy ones
- Deprecation Warnings: Emits warnings for legacy method usage
- Gradual Migration: Allows old code to work while transitioning

Usage:
    # Old code continues to work (with deprecation warning)
    from victor.workflows.adapter import UnifiedWorkflowCompilerAdapter

    compiler = UnifiedWorkflowCompilerAdapter()
    result = await compiler.compile("workflow.yaml")  # Emits deprecation warning

    # New code (recommended)
    from victor.workflows.compiler_protocols import WorkflowCompilerProtocol
    from victor.workflows.services import configure_workflow_services

    container = ServiceContainer()
    configure_workflow_services(container, settings)
    compiler = container.get(WorkflowCompilerProtocol)
    result = compiler.compile("workflow.yaml")
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from victor.workflows.compiler_protocols import CompiledGraphProtocol, WorkflowCompilerProtocol

logger = logging.getLogger(__name__)


class DeprecationAdapter:
    """Base class for deprecation adapters.

    Emits deprecation warnings when legacy methods are called.
    """

    _DEPRECATION_MESSAGE = (
        "This API is deprecated and will be removed in v0.7.0. "
        "Please migrate to the new protocol-based API. "
        "See MIGRATION_GUIDE.md for details."
    )

    @classmethod
    def _warn(cls) -> None:
        """Emit deprecation warning."""
        warnings.warn(
            cls._DEPRECATION_MESSAGE,
            DeprecationWarning,
            stacklevel=3,
        )
        logger.warning(f"Deprecated API used: {cls.__name__}")


class UnifiedWorkflowCompilerAdapter(DeprecationAdapter):
    """Adapter for UnifiedWorkflowCompiler compatibility.

    Wraps the new WorkflowCompiler with the legacy UnifiedWorkflowCompiler interface.
    Emits deprecation warnings to guide migration to protocol-based API.

    Example:
        # Old code (deprecated)
        compiler = UnifiedWorkflowCompilerAdapter()
        compiled = compiler.compile("workflow.yaml")  # Emits warning

        # New code (recommended)
        from victor.workflows.compiler import WorkflowCompiler
        compiler = WorkflowCompiler(yaml_loader=loader, ...)
        compiled = compiler.compile("workflow.yaml")
    """

    def __init__(
        self,
        enable_caching: bool = True,
        cache_ttl: int = 3600,
        settings: Optional[Any] = None,
    ):
        """Initialize the adapter.

        Args:
            enable_caching: Whether to enable caching (ignored, for compatibility)
            cache_ttl: Cache TTL in seconds (ignored, for compatibility)
            settings: Application settings (for DI container creation)
        """
        self._warn()

        # Create DI container if settings provided
        if settings:
            from victor.core.container import ServiceContainer
            from victor.workflows.services.workflow_service_provider import configure_workflow_services

            self._container = ServiceContainer()
            configure_workflow_services(self._container, settings)
        else:
            # Use global container
            from victor.core.container import get_container

            self._container = get_container()

    def compile(
        self,
        source: str,
        *,
        workflow_name: Optional[str] = None,
        validate: bool = True,
    ) -> "CompiledGraphAdapter":
        """Compile a workflow from YAML source.

        Args:
            source: YAML file path or YAML string content
            workflow_name: Name of workflow to compile (for multi-workflow files)
            validate: Whether to validate workflow definition

        Returns:
            CompiledGraphAdapter: Compiled workflow graph (with legacy interface)

        Example:
            compiler = UnifiedWorkflowCompilerAdapter()
            compiled = compiler.compile("workflow.yaml")
            result = await compiled.invoke({"input": "data"})
        """
        # Resolve new compiler from DI container
        from victor.workflows.compiler_protocols import WorkflowCompilerProtocol

        compiler: WorkflowCompilerProtocol = self._container.get(WorkflowCompilerProtocol)

        # Compile using new compiler
        compiled_graph = compiler.compile(source, workflow_name=workflow_name, validate=validate)

        # Wrap in adapter for legacy interface
        return CompiledGraphAdapter(compiled_graph)


class CompiledGraphAdapter(DeprecationAdapter):
    """Adapter for CompiledGraph compatibility.

    Wraps the new CompiledGraphProtocol with legacy invoke() and stream() methods.
    Emits deprecation warnings to guide migration.

    Example:
        # Old code (deprecated)
        result = await compiled.invoke({"input": "data"})

        # New code (recommended)
        result = await compiled.execute({"input": "data"})
    """

    def __init__(self, compiled_graph: "CompiledGraphProtocol"):
        """Initialize the adapter.

        Args:
            compiled_graph: New compiled graph instance
        """
        self._compiled_graph = compiled_graph

    async def invoke(
        self,
        initial_state: Dict[str, Any],
        *,
        thread_id: Optional[str] = None,
        checkpoint: Optional[str] = None,
    ) -> "ExecutorResultAdapter":
        """Execute the compiled workflow (legacy method name).

        Args:
            initial_state: Initial workflow state
            thread_id: Optional thread ID for checkpointing
            checkpoint: Optional checkpoint name to resume from

        Returns:
            ExecutorResultAdapter: Execution result (with legacy interface)

        Example:
            result = await compiled.invoke({"query": "search term"})
        """
        self._warn()

        # Use new interface
        result = await self._compiled_graph.invoke(
            initial_state,
            thread_id=thread_id,
            checkpoint=checkpoint,
        )

        # Wrap in adapter
        return ExecutorResultAdapter(result)

    async def stream(
        self,
        initial_state: Dict[str, Any],
        *,
        thread_id: Optional[str] = None,
    ):
        """Stream execution events (same as new interface).

        Args:
            initial_state: Initial workflow state
            thread_id: Optional thread ID for checkpointing

        Yields:
            Execution events
        """
        self._warn()
        async for event in self._compiled_graph.stream(initial_state, thread_id=thread_id):
            yield event

    @property
    def graph(self):
        """Get the underlying StateGraph (for compatibility)."""
        return self._compiled_graph.graph


class ExecutorResultAdapter:
    """Adapter for ExecutorResult compatibility.

    Wraps the new ExecutionResultProtocol with legacy result interface.
    No deprecation warning as this is just a data wrapper.

    Example:
        result = await compiled.invoke({"input": "data"})
        print(result.final_state)
        print(result.metrics)
    """

    def __init__(self, execution_result: "ExecutionResultProtocol"):
        """Initialize the adapter.

        Args:
            execution_result: New execution result instance
        """
        self._result = execution_result

    @property
    def final_state(self) -> Dict[str, Any]:
        """Get final workflow state."""
        return self._result.final_state

    @property
    def metrics(self) -> Dict[str, Any]:
        """Get execution metrics."""
        return self._result.metrics


__all__ = [
    "UnifiedWorkflowCompilerAdapter",
    "CompiledGraphAdapter",
    "ExecutorResultAdapter",
]
