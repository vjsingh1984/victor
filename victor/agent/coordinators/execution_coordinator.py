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

"""Execution coordinator for agentic loop execution.

This module contains the ExecutionCoordinator class that extracts the
agentic loop logic from ChatCoordinator into a dedicated, focused coordinator.

The ExecutionCoordinator handles:
- Multi-turn agentic loop (model → tools → model → tools → ...)
- Iteration limit enforcement
- Tool call execution coordination
- Response completion handling
- Error recovery and retry logic

Architecture:
------------
The ExecutionCoordinator depends on protocol-based abstractions rather than
concrete classes, enabling the Dependency Inversion Principle (DIP):

- ChatContextProtocol: For message/conversation access
- ToolContextProtocol: For tool execution
- ProviderContextProtocol: For LLM calls
- ExecutionProvider: For executing model turns

Phase 1: Extract ExecutionCoordinator
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.agent.response_completer import ToolFailureContext
from victor.providers.base import CompletionResponse

if TYPE_CHECKING:
    from victor.agent.coordinators.chat_protocols import (
        ChatContextProtocol,
        ToolContextProtocol,
        ProviderContextProtocol,
    )
    from victor.agent.token_tracker import TokenTracker

logger = logging.getLogger(__name__)


class ExecutionCoordinator:
    """Coordinator for agentic execution loop.

    This coordinator extracts the core agentic loop logic from ChatCoordinator,
    providing a clean separation of concerns and enabling independent testing.

    The agentic loop implements the following flow:
    1. Initialize tracking (tool budget, iteration budget)
    2. Get model response
    3. Execute any tool calls
    4. Continue until model provides final response (no tool calls)
    5. Ensure non-empty response on tool failures

    Args:
        chat_context: Protocol providing conversation/message access
        tool_context: Protocol providing tool selection/execution
        provider_context: Protocol providing LLM provider access
        execution_provider: Protocol for executing model turns
    """

    def __init__(
        self,
        chat_context: "ChatContextProtocol",
        tool_context: "ToolContextProtocol",
        provider_context: "ProviderContextProtocol",
        execution_provider: Any,  # ExecutionProvider protocol
        token_tracker: Optional["TokenTracker"] = None,
    ) -> None:
        """Initialize the ExecutionCoordinator.

        Args:
            chat_context: Chat context protocol implementation
            tool_context: Tool context protocol implementation
            provider_context: Provider context protocol implementation
            execution_provider: Execution provider protocol implementation
            token_tracker: Optional centralized token tracker. When provided,
                token usage is accumulated through the tracker instead of
                direct dict mutation on chat_context.
        """
        self._chat_context = chat_context
        self._tool_context = tool_context
        self._provider_context = provider_context
        self._execution_provider = execution_provider
        self._token_tracker = token_tracker

    # =====================================================================
    # Public API
    # =====================================================================

    async def execute_agentic_loop(
        self,
        user_message: str,
        max_iterations: int = 25,
    ) -> CompletionResponse:
        """Execute the full agentic loop.

        Runs the model, executes tools, and continues until the model
        provides a final response (no tool calls) or budget is exhausted.

        Args:
            user_message: Initial user message
            max_iterations: Maximum model turns

        Returns:
            CompletionResponse with complete response
        """
        # Ensure system prompt is included once at start of conversation
        self._chat_context.conversation.ensure_system_prompt()
        self._chat_context._system_added = True

        # Add user message to history
        self._chat_context.add_message("user", user_message)

        # Initialize tracking for this conversation turn
        self._tool_context.tool_calls_used = 0
        failure_context = ToolFailureContext()
        max_iterations_setting = getattr(self._chat_context.settings, "chat_max_iterations", 10)
        iteration = 0

        # Classify task complexity for appropriate budgeting
        task_classification = self._provider_context.task_classifier.classify(user_message)
        # Ensure at least 1 iteration is always allowed
        task_iteration_budget = max(task_classification.tool_budget * 2, 1)
        iteration_budget = min(
            task_iteration_budget,  # Allow 2x budget for iterations; always run at least one model turn
            max_iterations_setting,
            max_iterations,
        )

        # Agentic loop: continue until no tool calls or budget exhausted
        final_response: Optional[CompletionResponse] = None

        while iteration < iteration_budget:
            iteration += 1

            # Get tool definitions if provider supports them
            tools = None
            if (
                self._provider_context.provider.supports_tools()
                and self._tool_context.tool_calls_used < self._tool_context.tool_budget
            ):
                tools = await self._select_tools_for_turn(user_message)

            # Prepare optional thinking parameter
            provider_kwargs = {}
            if self._provider_context.thinking:
                provider_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": 10000,
                }

            # Check context and compact before API call to prevent overflow
            await self._check_context_compaction(user_message, task_classification)

            # Get response from provider
            response = await self._execute_model_turn(tools=tools, **provider_kwargs)

            # Accumulate token usage
            self._accumulate_token_usage(response)

            # Add assistant response to history if has content
            if response.content:
                self._chat_context.add_message("assistant", response.content)

                # Check compaction after adding assistant response
                await self._check_context_compaction(user_message, task_classification)

            # Check if model wants to use tools
            if response.tool_calls:
                # Handle tool calls and track results
                tool_results = await self._tool_context._handle_tool_calls(response.tool_calls)

                # Update failure context
                for result in tool_results:
                    if result.get("success"):
                        failure_context.successful_tools.append(result)
                    else:
                        failure_context.failed_tools.append(result)
                        failure_context.last_error = result.get("error")

                # Continue loop to get follow-up response
                continue

            # No tool calls - this is the final response
            final_response = response
            break

        # Ensure we have a complete response
        final_response = await self._ensure_complete_response(final_response, failure_context)

        return final_response

    async def execute_single_turn(
        self,
        messages: List[Any],
        tools: Optional[List[Any]] = None,
    ) -> CompletionResponse:
        """Execute a single model turn.

        This is a simpler execution path that makes a single model call
        without the agentic loop. Useful for simple queries or when
        the caller wants full control over iteration.

        Args:
            messages: Conversation history
            tools: Optional tool definitions

        Returns:
            CompletionResponse from model
        """
        return await self._execution_provider.execute_turn(
            messages=messages,
            model=self._provider_context.model,
            temperature=self._provider_context.temperature,
            max_tokens=self._provider_context.max_tokens,
            tools=tools,
        )

    # =====================================================================
    # Private Methods
    # =====================================================================

    async def _select_tools_for_turn(
        self,
        user_message: str,
    ) -> Optional[List[Any]]:
        """Select tools for the current iteration.

        Args:
            user_message: Original user message

        Returns:
            List of tool definitions or None
        """
        conversation_depth = self._chat_context.conversation.message_count()
        conversation_history = (
            [msg.model_dump() for msg in self._chat_context.messages]
            if self._chat_context.messages
            else None
        )

        tools = await self._tool_context.tool_selector.select_tools(
            user_message,
            use_semantic=self._tool_context.use_semantic_selection,
            conversation_history=conversation_history,
            conversation_depth=conversation_depth,
        )

        # Prioritize by stage
        tools = self._tool_context.tool_selector.prioritize_by_stage(user_message, tools)

        return tools

    async def _execute_model_turn(
        self,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Execute a single model turn.

        Args:
            tools: Optional tool definitions
            **kwargs: Additional provider parameters

        Returns:
            CompletionResponse from model
        """
        return await self._execution_provider.execute_turn(
            messages=self._chat_context.messages,
            model=self._provider_context.model,
            temperature=self._provider_context.temperature,
            max_tokens=self._provider_context.max_tokens,
            tools=tools,
            **kwargs,
        )

    def _accumulate_token_usage(self, response: CompletionResponse) -> None:
        """Accumulate token usage for evaluation tracking.

        When a TokenTracker is configured, usage is accumulated through
        the tracker (single source of truth). Otherwise falls back to
        direct dict mutation on chat_context for backward compatibility.

        Args:
            response: Response from model
        """
        if response.usage:
            if self._token_tracker is not None:
                self._token_tracker.accumulate(response.usage)
            else:
                self._chat_context._cumulative_token_usage[
                    "prompt_tokens"
                ] += response.usage.get("prompt_tokens", 0)
                self._chat_context._cumulative_token_usage[
                    "completion_tokens"
                ] += response.usage.get("completion_tokens", 0)
                self._chat_context._cumulative_token_usage[
                    "total_tokens"
                ] += response.usage.get("total_tokens", 0)

    async def _check_context_compaction(
        self,
        user_message: str,
        task_classification: Any,
    ) -> None:
        """Check and perform context compaction if needed.

        Args:
            user_message: Current query
            task_classification: Task complexity classification
        """
        if self._chat_context._context_compactor:
            compaction_action = self._chat_context._context_compactor.check_and_compact(
                current_query=user_message,
                force=False,
                tool_call_count=self._tool_context.tool_calls_used,
                task_complexity=task_classification.complexity.value,
            )
            if compaction_action.action_taken:
                logger.info(
                    f"Compacted context: {compaction_action.messages_removed} messages removed, "
                    f"{compaction_action.tokens_freed} tokens freed"
                )

    async def _ensure_complete_response(
        self,
        final_response: Optional[CompletionResponse],
        failure_context: ToolFailureContext,
    ) -> CompletionResponse:
        """Ensure we have a complete response.

        If the response is empty or None, use the response completer
        to generate a response or use a fallback.

        Args:
            final_response: Response from agentic loop (may be None)
            failure_context: Tool failure context for fallback messages

        Returns:
            CompletionResponse with content
        """
        if final_response is not None and final_response.content:
            return final_response

        # Use response completer to generate a response
        completion_result = await self._provider_context.response_completer.ensure_response(
            messages=self._chat_context.messages,
            model=self._provider_context.model,
            temperature=self._provider_context.temperature,
            max_tokens=self._provider_context.max_tokens,
            failure_context=failure_context if failure_context.failed_tools else None,
        )

        if completion_result.content:
            self._chat_context.add_message("assistant", completion_result.content)
            return CompletionResponse(
                content=completion_result.content,
                role="assistant",
                tool_calls=None,
            )

        # Last resort fallback
        fallback_content = (
            "I was unable to generate a complete response. " "Please try rephrasing your request."
        )
        if failure_context.failed_tools:
            fallback_content = (
                self._provider_context.response_completer.format_tool_failure_message(
                    failure_context
                )
            )

        self._chat_context.add_message("assistant", fallback_content)
        return CompletionResponse(
            content=fallback_content,
            role="assistant",
            tool_calls=None,
        )


__all__ = [
    "ExecutionCoordinator",
]
