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

"""Pure executor for compiled StateGraph instances.

This module provides StateGraphExecutor which executes compiled workflows.
This is the execution counterpart to WorkflowCompiler, following SRP by
separating compilation from execution.

Design Principles:
- SRP: ONLY executes compiled graphs, no compilation logic
- DIP: Depends on CompiledGraphProtocol and ExecutionContextProtocol
- ISP: Minimal interface - just invoke() and stream()
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Dict,
    Generic,
    Optional,
    TYPE_CHECKING,
    cast,
)

if TYPE_CHECKING:
    from victor.framework.config import GraphConfig
    from victor.framework.graph import CompiledGraph, GraphExecutionResult
    from victor.workflows.compiler_protocols import (
        ExecutionContextProtocol,
        ExecutionEventProtocol,
        ExecutionResultProtocol,
    )

# StateType is used in Generic at runtime, so import it unconditionally
from victor.framework.graph import StateType

logger = logging.getLogger(__name__)


class StateGraphExecutor(Generic[StateType]):
    """Pure executor for compiled StateGraph instances.

    Responsibility (SRP):
    - Execute compiled StateGraph instances
    - Manage execution context (thread_id, checkpoints)
    - Stream execution events
    - Handle execution errors

    Non-responsibility:
    - Compilation (handled by WorkflowCompiler)
    - Node execution logic (handled by node executors)
    - Caching (handled by decorator/wrapper)
    - Observability (handled by decorator/wrapper)

    Design:
    - SRP compliance: ONLY executes, doesn't compile
    - DIP compliance: Depends on CompiledGraphProtocol
    - Protocol-based: Uses ExecutionContextProtocol for execution parameters

    Attributes:
        _graph: The compiled StateGraph to execute
        _default_config: Default execution configuration

    Example:
        compiler = WorkflowCompiler(...)
        compiled = compiler.compile("workflow.yaml")

        executor = StateGraphExecutor(compiled)
        result = await executor.invoke({"input": "data"})
    """

    def __init__(
        self,
        graph: "CompiledGraph[StateType]",
        default_config: Optional["GraphConfig"] = None,
    ):
        """Initialize the executor with a compiled graph.

        Args:
            graph: Compiled StateGraph to execute
            default_config: Default execution configuration
        """
        from victor.framework.config import GraphConfig

        self._graph = graph
        self._default_config = default_config or GraphConfig()

    async def invoke(
        self,
        initial_state: Dict[str, Any],
        *,
        config: Optional["GraphConfig"] = None,
        thread_id: Optional[str] = None,
        checkpoint: Optional[str] = None,
    ) -> "ExecutionResultProtocol":
        """Execute the compiled workflow graph.

        This is the main execution entry point. It delegates to the
        compiled graph's invoke() method with appropriate configuration.

        Args:
            initial_state: Initial workflow state
            config: Override execution config (uses default if None)
            thread_id: Thread ID for checkpointing (auto-generated if None)
            checkpoint: Checkpoint name to resume from (None for fresh execution)

        Returns:
            ExecutionResultProtocol with execution outcome

        Raises:
            ValueError: If initial_state is invalid
            RuntimeError: If execution fails

        Example:
            executor = StateGraphExecutor(compiled_graph)
            result = await executor.invoke({"query": "search term"})
            print(result.final_state)
        """
        exec_config = config or self._default_config
        thread_id = thread_id or uuid.uuid4().hex

        logger.info(
            f"Executing workflow {getattr(self._graph, 'graph_id', '(unknown)') or '(unknown)'} "
            f"(thread_id={thread_id[:8]}...)"
        )

        # Execute the compiled graph
        # The compiled graph already has all the execution logic,
        # including iteration control, timeout management, etc.
        result = await self._graph.invoke(
            cast(Any, initial_state),
            config=exec_config,
            thread_id=thread_id,
        )

        logger.info(
            f"Execution complete: success={result.success}, "
            f"iterations={result.iterations}, "
            f"duration={result.duration:.2f}s"
        )

        return cast("ExecutionResultProtocol", result)

    async def stream(
        self,
        initial_state: Dict[str, Any],
        *,
        config: Optional["GraphConfig"] = None,
        thread_id: Optional[str] = None,
    ) -> AsyncIterator["ExecutionEventProtocol"]:
        """Stream execution events from the compiled workflow.

        Yields events as the workflow executes, allowing real-time
        monitoring and observability.

        Args:
            initial_state: Initial workflow state
            config: Override execution config (uses default if None)
            thread_id: Thread ID for checkpointing (auto-generated if None)

        Yields:
            ExecutionEventProtocol for each execution step

        Events yielded:
            - graph_started: Graph execution started
            - node_start: Node execution started
            - node_complete: Node execution completed
            - node_error: Node execution failed
            - graph_completed: Graph execution completed
            - graph_failed: Graph execution failed

        Example:
            executor = StateGraphExecutor(compiled_graph)

            async for event in executor.stream({"input": "data"}):
                print(f"Event: {event.event_type}")
                if event.data:
                    print(f"Data: {event.data}")
        """
        from victor.workflows.compiler_protocols import ExecutionEventProtocol

        exec_config = config or self._default_config
        thread_id = thread_id or uuid.uuid4().hex

        logger.info(
            f"Streaming workflow {getattr(self._graph, 'graph_id', '(unknown)') or '(unknown)'} "
            f"(thread_id={thread_id[:8]}...)"
        )

        # Stream execution events from the compiled graph
        async for event in self._graph.stream(
            cast(Any, initial_state),
            config=exec_config,
            thread_id=thread_id,
        ):
            yield cast("ExecutionEventProtocol", event)

    def get_execution_context(
        self,
        thread_id: Optional[str] = None,
    ) -> "ExecutionContextProtocol":
        """Get the execution context for a thread.

        Provides access to execution metadata like thread ID,
        execution statistics, and checkpoint information.

        Args:
            thread_id: Thread ID (uses current if None)

        Returns:
            ExecutionContextProtocol with execution metadata

        Example:
            executor = StateGraphExecutor(compiled_graph)
            context = executor.get_execution_context()
            print(f"Thread ID: {context.thread_id}")
        """
        from dataclasses import dataclass
        from typing import Any

        thread_id = thread_id or uuid.uuid4().hex

        # Create a concrete implementation of ExecutionContextProtocol
        @dataclass
        class _ExecutionContextImpl:
            """Concrete implementation of ExecutionContextProtocol."""
            _thread_id: str
            _graph_id: str
            _start_time: float

            @property
            def orchestrator(self) -> typing_Any:
                return None

            @property
            def settings(self) -> typing_Any:
                return None

            @property
            def services(self) -> typing_Any:
                return None

            @property
            def thread_id(self) -> str:
                return self._thread_id

            @property
            def graph_id(self) -> str:
                return self._graph_id

            @property
            def start_time(self) -> float:
                return self._start_time

        return _ExecutionContextImpl(
            thread_id=thread_id,
            graph_id=getattr(self._graph, 'graph_id', None) or "unknown",
            start_time=time.time(),
        )  # type: ignore[return-value]


__all__ = [
    "StateGraphExecutor",
]
