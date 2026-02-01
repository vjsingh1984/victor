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

"""BaseHandler for workflow compute handlers.

Provides the Template Method pattern to eliminate ~400 lines of boilerplate
across vertical handler implementations. Handles timing, error handling,
metadata tracking, and NodeResult construction.

Subclasses implement the execute() method with handler-specific logic.
"""

from __future__ import annotations

import logging
import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from victor.tools.registry import ToolRegistry
    from victor.workflows.definition import ComputeNode
    from victor.workflows.executor import (
        NodeResult,
        WorkflowContext,
    )

logger = logging.getLogger(__name__)


class HandlerError(Exception):
    """Exception raised by handlers that includes output data.

    Handlers can raise this exception to signal failure while preserving
    the output structure (e.g., status field, partial results).

    Attributes:
        message: Error message
        output: Optional output data to preserve in NodeResult
    """

    def __init__(self, message: str, output: Optional[Any] = None):
        super().__init__(message)
        self.output = output


@dataclass
class BaseHandler:
    """Base class for workflow compute handlers.

    Implements the Template Method pattern to eliminate boilerplate:
    - Automatic timing (start_time, duration_seconds)
    - Error handling and exception catching
    - NodeResult construction with proper status
    - Metadata tracking (duration, tool_calls)
    - Context output storage

    Subclasses only need to implement the execute() method with
    handler-specific business logic.

    Example:
        @dataclass
        class MyHandler(BaseHandler):
            async def execute(
                self,
                node: ComputeNode,
                context: WorkflowContext,
                tool_registry: ToolRegistry,
            ) -> Tuple[Any, int]:
                # Handler-specific logic
                result = await some_operation()
                return result, 0  # (output, tool_calls_count)

        # Handler is automatically callable with full boilerplate:
        handler = MyHandler()
        result = await handler(node, context, tool_registry)
        # Result includes timing, error handling, metadata
    """

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        """Execute handler with automatic timing, error handling, and metadata.

        This method implements the Template Method pattern:
        1. Start timer
        2. Call execute() (subclass-provided logic)
        3. Handle exceptions
        4. Build NodeResult with timing and metadata
        5. Store output in context

        Args:
            node: ComputeNode being executed
            context: Workflow execution context
            tool_registry: Tool registry for tool execution

        Returns:
            NodeResult with status, output, error, timing, and metadata
        """
        from victor.workflows.executor import NodeResult, ExecutorNodeStatus

        start_time = time.time()

        try:
            # Call subclass-specific logic
            output, tool_calls = await self.execute(node, context, tool_registry)

            # Store output in context
            output_key = node.output_key or node.id
            context.set(output_key, output)

            # Build successful result
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.COMPLETED,
                output=output,
                duration_seconds=time.time() - start_time,
                tool_calls_used=tool_calls,
            )

        except HandlerError as e:
            # HandlerError preserves output structure
            logger.exception(f"Handler {self.__class__.__name__} failed for node {node.id}: {e}")
            error_msg = str(e)

            # Use handler-provided output or create error output
            output = e.output if e.output is not None else {"error": error_msg}

            # Store in context for consistency
            output_key = node.output_key or node.id
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error=error_msg,
                output=output,
                duration_seconds=time.time() - start_time,
            )

        except Exception as e:
            logger.exception(f"Handler {self.__class__.__name__} failed for node {node.id}: {e}")
            error_msg = str(e)

            # For backward compatibility, include error in output dict
            # Some handlers/tests expect output["error"] to be set
            output = {"error": error_msg}

            # Still store in context for consistency
            output_key = node.output_key or node.id
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error=error_msg,
                output=output,  # Include output with error for backward compatibility
                duration_seconds=time.time() - start_time,
            )

    @abstractmethod
    async def execute(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> tuple[Any, int]:
        """Execute handler-specific logic.

        Subclasses implement this method with their business logic.
        The base class handles timing, error handling, and NodeResult construction.

        Args:
            node: ComputeNode being executed
            context: Workflow execution context
            tool_registry: Tool registry for tool execution

        Returns:
            Tuple of (output, tool_calls_count)
            - output: Any - The handler's output data
            - tool_calls_count: int - Number of tool calls made

        Example:
            async def execute(
                self,
                node: ComputeNode,
                context: WorkflowContext,
                tool_registry: ToolRegistry,
            ) -> Tuple[Any, int]:
                # Extract inputs
                url = node.input_mapping.get("url")
                data = context.get(url)

                # Perform operation
                result = await tool_registry.execute("process", data=data)

                # Return (output, tool_calls_count)
                return result, 1
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement execute()")


__all__ = ["BaseHandler", "HandlerError"]
