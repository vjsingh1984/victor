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

"""Compute node executor.

Executes compute nodes by calling registered handler functions.
This is a stub that delegates to legacy implementation during migration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from victor.workflows.definition import ComputeNode, WorkflowState

logger = logging.getLogger(__name__)


class ComputeNodeExecutor:
    """Executor for compute nodes.

    Responsibility (SRP):
    - Execute registered handler functions
    - Map handler names to handler instances
    - Pass input context to handlers
    - Handle handler errors

    Non-responsibility:
    - Workflow compilation (handled by WorkflowCompiler)
    - Handler registration (handled by ComputeHandlerRegistry)
    """

    def __init__(self, context: Any = None):
        """Initialize the executor.

        Args:
            context: ExecutionContext with services, settings
        """
        self._context = context

    async def execute(self, node: "ComputeNode", state: "WorkflowState") -> "WorkflowState":
        """Execute a compute node.

        Args:
            node: Compute node definition
            state: Current workflow state

        Returns:
            Updated workflow state

        Raises:
            Exception: If handler execution fails
        """
        import asyncio
        from victor.framework.graph import GraphNodeResult

        logger.info(f"Executing compute node: {node.id}")

        # Step 1: Build params from node.inputs with $ctx. and $state. prefixes
        params = {}
        if node.inputs:
            for param_name, source in node.inputs.items():
                # Handle $ctx. prefix (from state)
                if source.startswith("$ctx."):
                    context_key = source[5:]
                    if context_key in state:
                        params[param_name] = state[context_key]
                    else:
                        params[param_name] = context_key
                # Handle $state. prefix (also from state)
                elif source.startswith("$state."):
                    context_key = source[7:]
                    if context_key in state:
                        params[param_name] = state[context_key]
                    else:
                        params[param_name] = context_key
                # Direct value
                else:
                    params[param_name] = source

        # Step 2: Get tool registry from context
        tool_registry = None
        if self._context and hasattr(self._context, "tool_registry"):
            tool_registry = self._context.tool_registry

        tool_calls_used = 0
        output = None

        # Step 3: Check for custom handler
        if node.handler:
            handler = self._get_compute_handler(node.handler)
            if handler:
                # Create minimal WorkflowContext wrapper
                from victor.workflows.executor import WorkflowContext

                context = WorkflowContext(dict(state))
                result = await handler(node, context, tool_registry)

                # Transfer context changes back to state
                for key, value in context.data.items():
                    if not key.startswith("_"):
                        state[key] = value

                output = result.output if result else None
                tool_calls_used = result.tool_calls_used if result else 0
            else:
                logger.warning(f"Handler '{node.handler}' not found for node '{node.id}'")
                output = {"error": f"Handler '{node.handler}' not found"}
        else:
            # Step 4: Execute tools directly
            outputs = {}
            if tool_registry and node.tools:
                for tool_name in node.tools:
                    # Check constraints
                    if hasattr(node, "constraints") and node.constraints:
                        if not node.constraints.allows_tool(tool_name):
                            logger.debug(f"Tool '{tool_name}' blocked by constraints")
                            continue

                    try:
                        timeout = getattr(node, "timeout", 300)
                        if hasattr(node, "constraints") and node.constraints:
                            timeout = getattr(node.constraints, "timeout", 300)

                        result = await asyncio.wait_for(
                            tool_registry.execute(
                                tool_name,
                                _exec_ctx={
                                    "workflow_context": state,
                                    "constraints": (
                                        node.constraints.to_dict()
                                        if hasattr(node, "constraints") and node.constraints
                                        else {}
                                    ),
                                },
                                **params,
                            ),
                            timeout=timeout,
                        )
                        tool_calls_used += 1

                        if result.success:
                            outputs[tool_name] = result.output
                        else:
                            outputs[tool_name] = {"error": result.error}

                        if hasattr(node, "fail_fast") and node.fail_fast and not result.success:
                            break

                    except asyncio.TimeoutError:
                        outputs[tool_name] = {"error": "Timeout"}
                        if hasattr(node, "fail_fast") and node.fail_fast:
                            break
                    except Exception as e:
                        logger.error(f"Tool '{tool_name}' execution failed: {e}")
                        outputs[tool_name] = {"error": str(e)}
                        if hasattr(node, "fail_fast") and node.fail_fast:
                            break
            else:
                outputs = {"status": "no_tools_executed", "params": params}

            output = outputs

        # Step 5: Store output in state
        output_key = node.output or node.id
        state[output_key] = output

        # Step 6: Update node results for observability
        if "_node_results" not in state:
            state["_node_results"] = {}

        state["_node_results"][node.id] = GraphNodeResult(
            node_id=node.id,
            status="completed",
            result=output,
            metadata={
                "handler": node.handler,
                "tools": node.tools if hasattr(node, "tools") else [],
                "tool_calls_used": tool_calls_used,
            },
        )

        logger.info(f"Compute node {node.id} completed successfully")
        return state

    def _get_compute_handler(self, handler_name: str) -> Any:
        """Get compute handler by name.

        Args:
            handler_name: Name of the registered handler

        Returns:
            Handler instance or None if not found
        """
        from victor.workflows.executor import get_compute_handler

        return get_compute_handler(handler_name)

    def supports_node_type(self, node_type: str) -> bool:
        """Check if this executor supports the given node type."""
        return node_type == "compute"


__all__ = ["ComputeNodeExecutor"]
