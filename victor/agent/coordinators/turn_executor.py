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

This module contains the TurnExecutor class that extracts the
agentic loop logic from ChatCoordinator into a dedicated, focused coordinator.

The TurnExecutor handles:
- Multi-turn agentic loop (model → tools → model → tools → ...)
- Iteration limit enforcement
- Tool call execution coordination
- Response completion handling
- Error recovery and retry logic

Architecture:
------------
The TurnExecutor depends on protocol-based abstractions rather than
concrete classes, enabling the Dependency Inversion Principle (DIP):

- ChatContextProtocol: For message/conversation access
- ToolContextProtocol: For tool execution
- ProviderContextProtocol: For LLM calls
- ExecutionProvider: For executing model turns

Phase 1: Extract TurnExecutor
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from dataclasses import dataclass, field

from victor.agent.response_completer import ToolFailureContext
from victor.providers.base import CompletionResponse

if TYPE_CHECKING:
    from victor.agent.coordinators.chat_protocols import (
        ChatContextProtocol,
        ToolContextProtocol,
        ProviderContextProtocol,
    )
    from victor.agent.token_tracker import TokenTracker


@dataclass
class TurnResult:
    """Result of a single execution turn (one LLM call + tool execution).

    This is the primitive unit returned by execute_turn().
    AgenticLoop uses this to make per-turn evaluation decisions.

    Attributes:
        response: Raw model response
        tool_results: Results from tool execution (empty if no tools called)
        has_tool_calls: Whether the model requested tools
        tool_calls_count: Number of tool calls in this turn
        all_tools_blocked: Whether all tool calls were blocked by dedup
        is_qa_response: Whether this was a Q&A shortcut
        content: Response text content (convenience)
    """

    response: CompletionResponse
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    has_tool_calls: bool = False
    tool_calls_count: int = 0
    all_tools_blocked: bool = False
    is_qa_response: bool = False

    @property
    def content(self) -> str:
        """Response text content."""
        return self.response.content or ""

    @property
    def has_content(self) -> bool:
        """Whether the response has text content."""
        return bool(self.response.content)

    @property
    def successful_tool_count(self) -> int:
        """Number of tools that succeeded."""
        return sum(1 for r in self.tool_results if r.get("success"))

    @property
    def failed_tool_count(self) -> int:
        """Number of tools that failed."""
        return sum(1 for r in self.tool_results if not r.get("success"))


logger = logging.getLogger(__name__)

# Module-level guard to prevent recursive subagent exploration
_EXPLORATION_IN_PROGRESS = False


class TurnExecutor:
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
        """Initialize the TurnExecutor.

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
        self._exploration_done = False  # Instance-level: fires once per conversation

    # =====================================================================
    # Public API
    # =====================================================================

    async def execute_agentic_loop(
        self,
        user_message: str,
        max_iterations: int = 25,
    ) -> CompletionResponse:
        """Execute the full agentic loop.

        When USE_AGENTIC_LOOP flag is enabled (default), delegates to
        AgenticLoop for enhanced execution with perception, evaluation,
        progress tracking, and adaptive termination. Falls back to the
        legacy while-loop when disabled.

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

        # Phase 10: Delegate to AgenticLoop when enabled (Strangler Fig)
        try:
            from victor.core.feature_flags import FeatureFlag, get_feature_flag_manager

            if get_feature_flag_manager().is_enabled(FeatureFlag.USE_AGENTIC_LOOP):
                return await self._execute_via_agentic_loop(user_message, max_iterations)
        except Exception as e:
            logger.warning(f"AgenticLoop delegation failed, falling back to legacy loop: {e}")

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

        # Parallel exploration for complex tasks (before agentic loop)
        await self._run_parallel_exploration(user_message, task_classification)

        # Detect Q&A-style messages that don't need tools.
        # If the model answers without tool calls on the first turn, accept it
        # immediately instead of nudging for tool usage.
        is_qa_task = self._is_question_only(user_message)

        # Agentic loop: continue until no tool calls or budget exhausted
        # Uses shared turn_policy constants for consistency with AgenticLoop
        from victor.agent.turn_policy import (
            SpinDetector as _SpinDetector,
            NudgePolicy as _NudgePolicy,
            SpinState as _SpinState,
            MAX_NO_TOOL_TURNS as _MAX_NO_TOOL_TURNS,
            MAX_ALL_BLOCKED as _MAX_ALL_BLOCKED,
            READ_ONLY_TOOLS as _READ_ONLY_TOOLS,
        )

        final_response: Optional[CompletionResponse] = None
        spin = _SpinDetector()
        nudge_policy = _NudgePolicy()

        while iteration < iteration_budget:
            iteration += 1

            # Get tool definitions if provider supports them.
            # Skip tools entirely for Q&A tasks — sending 48 tool schemas
            # to a local model adds massive latency (68s vs 2s on gemma4:31b).
            tools = None
            if (
                not is_qa_task
                and self._provider_context.provider.supports_tools()
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
                turn_tools = {tc.get("name", "") for tc in response.tool_calls}

                # Handle tool calls and track results
                tool_results = await self._tool_context._handle_tool_calls(response.tool_calls)

                # Update failure context and record for real-time failure hints
                for result in tool_results:
                    if result.get("success"):
                        failure_context.successful_tools.append(result)
                    else:
                        failure_context.failed_tools.append(result)
                        failure_context.last_error = result.get("error")
                        _orch = getattr(self, "_orchestrator", None) or getattr(
                            self._chat_context, "_orchestrator", None
                        )
                        _injector = (
                            getattr(_orch, "_optimization_injector", None) if _orch else None
                        )
                        if _injector and result.get("error"):
                            _injector.record_failure(
                                result.get("tool_name", "unknown"),
                                result["error"],
                            )

                # Spin detection via shared SpinDetector
                _pipeline = getattr(self._tool_context, "_tool_pipeline", None)
                if _pipeline is None:
                    _orch = getattr(self, "_orchestrator", None) or getattr(
                        self._chat_context, "_orchestrator", None
                    )
                    if _orch:
                        _pipeline = getattr(_orch, "_tool_pipeline", None)
                _all_blocked = bool(
                    _pipeline and getattr(_pipeline, "last_batch_all_skipped", False)
                )

                spin.record_turn(
                    has_tool_calls=True,
                    all_blocked=_all_blocked,
                    tool_names=turn_tools,
                    tool_count=len(response.tool_calls),
                )

                # Inject nudges via shared NudgePolicy
                nudge_decision = nudge_policy.evaluate(spin)
                if nudge_decision.should_inject:
                    self._chat_context.add_message(nudge_decision.role, nudge_decision.message)
                    logger.info(f"[nudge] {nudge_decision.nudge_type.value}")

                # Check for spin termination
                if spin.state == _SpinState.TERMINATED:
                    logger.warning(
                        "[spin-detect] Terminated: blocked=%d, no_tool=%d",
                        spin.consecutive_all_blocked,
                        spin.consecutive_no_tool_turns,
                    )
                    final_response = response
                    break

                # Continue loop to get follow-up response
                continue

            # No tool calls — record in spin detector
            spin.record_turn(has_tool_calls=False)

            if spin.state == _SpinState.TERMINATED:
                logger.warning(
                    "Agent stuck: %d turns without tool calls, breaking loop",
                    spin.consecutive_no_tool_turns,
                )
                final_response = response
                break

            if spin.state == _SpinState.WARNING and tools:
                # Inject nudge and budget warning via shared policy
                nudge_decision = nudge_policy.evaluate(spin)
                if nudge_decision.should_inject:
                    self._chat_context.add_message(nudge_decision.role, nudge_decision.message)
                budget_warning = nudge_policy.budget_warning(iteration, iteration_budget)
                if budget_warning.should_inject:
                    self._chat_context.add_message(budget_warning.role, budget_warning.message)
                continue

            # First turn with no tool calls — allow it (model may be
            # providing a final answer after successfully using tools)
            if self._tool_context.tool_calls_used > 0:
                final_response = response
                break

            # Q&A shortcut: if the message is a question/display-only task
            # and the model answered with content, accept it immediately.
            # This avoids wasting 2+ extra LLM calls nudging for tool usage
            # on prompts like "What is 2+2?" or "Explain X".
            if is_qa_task and response.content:
                logger.info("Q&A shortcut: accepting first response for question-only task")
                final_response = response
                break

            # No tools ever called — continue to give the model another chance
            continue

        # Ensure we have a complete response
        final_response = await self._ensure_complete_response(final_response, failure_context)

        return final_response

    async def execute_single_turn(
        self,
        messages: List[Any],
        tools: Optional[List[Any]] = None,
    ) -> CompletionResponse:
        """Execute a single model turn (raw LLM call, no tool execution).

        This is the lowest-level execution path — a single LLM API call.
        For a full turn with tool execution, use execute_turn().

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

    async def execute_turn(
        self,
        user_message: str,
        task_classification: Optional[Any] = None,
        is_qa_task: bool = False,
        enable_thinking: bool = False,
    ) -> TurnResult:
        """Execute a single complete turn: LLM call + tool execution.

        This is the primitive unit for AgenticLoop. It performs exactly
        one model call, executes any requested tools, and returns
        structured results without looping.

        The caller (AgenticLoop) owns the iteration loop and decides
        whether to continue, retry, or complete based on TurnResult.

        Args:
            user_message: User message (for tool selection context)
            task_classification: Task classification for tool selection
            is_qa_task: Whether to skip tool selection for Q&A
            enable_thinking: Whether to enable model thinking

        Returns:
            TurnResult with response, tool results, and metadata
        """
        # Select tools (unless Q&A task)
        tools = None
        if (
            not is_qa_task
            and self._provider_context.provider.supports_tools()
            and self._tool_context.tool_calls_used < self._tool_context.tool_budget
        ):
            tools = await self._select_tools_for_turn(user_message)

        # Prepare thinking parameter
        provider_kwargs: Dict[str, Any] = {}
        if enable_thinking or self._provider_context.thinking:
            provider_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": 10000,
            }

        # Check context compaction before API call
        if task_classification:
            await self._check_context_compaction(user_message, task_classification)

        # Execute model turn
        response = await self._execute_model_turn(tools=tools, **provider_kwargs)

        # Track tokens
        self._accumulate_token_usage(response)

        # Add assistant response to conversation history
        if response.content:
            self._chat_context.add_message("assistant", response.content)
            if task_classification:
                await self._check_context_compaction(user_message, task_classification)

        # Execute tool calls if present
        tool_results: List[Dict[str, Any]] = []
        all_blocked = False
        if response.tool_calls:
            tool_results = await self._tool_context._handle_tool_calls(response.tool_calls)

            # Check for dedup-blocked spin
            _pipeline = getattr(self._tool_context, "_tool_pipeline", None)
            if _pipeline is None:
                _orch = getattr(self, "_orchestrator", None) or getattr(
                    self._chat_context, "_orchestrator", None
                )
                if _orch:
                    _pipeline = getattr(_orch, "_tool_pipeline", None)
            all_blocked = bool(_pipeline and getattr(_pipeline, "last_batch_all_skipped", False))

            # Record failures for optimization hints
            for result in tool_results:
                if not result.get("success"):
                    _orch = getattr(self, "_orchestrator", None) or getattr(
                        self._chat_context, "_orchestrator", None
                    )
                    _injector = getattr(_orch, "_optimization_injector", None) if _orch else None
                    if _injector and result.get("error"):
                        _injector.record_failure(
                            result.get("tool_name", "unknown"),
                            result["error"],
                        )

        return TurnResult(
            response=response,
            tool_results=tool_results,
            has_tool_calls=bool(response.tool_calls),
            tool_calls_count=len(response.tool_calls) if response.tool_calls else 0,
            all_tools_blocked=all_blocked,
            is_qa_response=is_qa_task and bool(response.content),
        )

    # =====================================================================
    # AgenticLoop Integration (Phase 10)
    # =====================================================================

    async def _execute_via_agentic_loop(
        self,
        user_message: str,
        max_iterations: int,
    ) -> CompletionResponse:
        """Delegate execution to AgenticLoop for enhanced evaluation.

        AgenticLoop runs the PERCEIVE -> PLAN -> ACT -> EVALUATE -> DECIDE
        cycle, calling self.execute_turn() for each turn.

        Args:
            user_message: User message (already added to conversation)
            max_iterations: Maximum iterations

        Returns:
            CompletionResponse from the final turn
        """
        from victor.framework.agentic_loop import AgenticLoop

        # Classify task for tool selection
        task_classification = self._provider_context.task_classifier.classify(user_message)
        is_qa = self._is_question_only(user_message)

        # Run parallel exploration for complex tasks (before loop)
        await self._run_parallel_exploration(user_message, task_classification)

        # Calculate iteration budget same as legacy path
        max_iterations_setting = getattr(self._chat_context.settings, "chat_max_iterations", 10)
        task_budget = max(task_classification.tool_budget * 2, 1)
        iteration_budget = min(task_budget, max_iterations_setting, max_iterations)

        # Create AgenticLoop with self as turn_executor
        # Use a lightweight mock orchestrator since AgenticLoop only needs
        # turn_executor for single-turn execution
        loop = AgenticLoop(
            orchestrator=None,
            turn_executor=self,
            max_iterations=iteration_budget,
            enable_fulfillment_check=True,  # Auto-derived criteria via FulfillmentCriteriaBuilder
            enable_adaptive_iterations=True,
        )

        # Inject task classification into state for execute_turn()
        context = {
            "_task_classification": task_classification,
            "_is_qa_task": is_qa,
        }

        loop_result = await loop.run(user_message, context=context)

        # Extract the last TurnResult's response
        if loop_result.iterations:
            last = loop_result.iterations[-1]
            if last.action_result is not None:
                turn_result = last.action_result
                if hasattr(turn_result, "response"):
                    return turn_result.response

        # Fallback: ensure we have a response
        failure_context = ToolFailureContext()
        return await self._ensure_complete_response(None, failure_context)

    # =====================================================================
    # Q&A Detection
    # =====================================================================

    @staticmethod
    def _is_question_only(message: str) -> bool:
        """Detect messages that are pure Q&A and unlikely to need tools.

        Returns True for conversational/knowledge questions like:
        - "What is 2+2?"
        - "Explain Python decorators"
        - "How does X work?"

        Returns False for messages that imply code/file work:
        - "Fix the bug in main.py"
        - "Create a new file"
        - "Refactor the database module"
        """
        msg = message.strip().lower()

        # Short messages ending with ? are almost always Q&A
        if len(msg) < 120 and msg.endswith("?"):
            # Unless they contain action words implying code work
            action_words = (
                "fix",
                "create",
                "write",
                "edit",
                "refactor",
                "add",
                "implement",
                "update",
                "delete",
                "remove",
                "change",
                "modify",
                "build",
                "deploy",
                "run",
                "test",
                "debug",
                "install",
                "configure",
                "setup",
                "migrate",
            )
            words = set(msg.split())
            if not words.intersection(action_words):
                return True

        # Explicit "explain", "what is", "how does" patterns
        qa_prefixes = (
            "what is",
            "what are",
            "what's",
            "who is",
            "who are",
            "how does",
            "how do",
            "how is",
            "how can",
            "why does",
            "why do",
            "why is",
            "when does",
            "when is",
            "explain",
            "describe",
            "define",
            "tell me about",
            "can you explain",
            "what does",
            "reply with",
            "answer",
            "say ",
        )
        if any(msg.startswith(p) for p in qa_prefixes):
            return True

        return False

    # =====================================================================
    # Parallel Exploration
    # =====================================================================

    async def _run_parallel_exploration(
        self,
        user_message: str,
        task_classification: Any,
    ) -> None:
        """Run parallel exploration subagents for complex tasks.

        Spawns concurrent RESEARCHER subagents to explore the codebase
        before the main agentic loop starts. Findings are injected into
        the conversation context as a user message.

        Only fires for COMPLEX/ACTION tasks when parallel_exploration is enabled.
        Falls back gracefully on any error.
        """
        # Only explore once per conversation, not on continuations
        if self._exploration_done:
            return
        if user_message.startswith("You have not edited") or user_message == "Continue.":
            return

        global _EXPLORATION_IN_PROGRESS
        if _EXPLORATION_IN_PROGRESS:
            return  # Prevent recursive subagent exploration

        # Check if exploration is enabled and task warrants it
        try:
            from victor.config.settings import load_settings

            settings = load_settings()
            pipeline = getattr(settings, "pipeline", None)
            if pipeline and not getattr(pipeline, "parallel_exploration", True):
                return

            from victor.framework.task.protocols import TaskComplexity

            if task_classification.complexity not in {
                TaskComplexity.COMPLEX,
                TaskComplexity.ACTION,
                TaskComplexity.ANALYSIS,
            }:
                return
        except Exception:
            return

        try:
            from pathlib import Path

            from victor.agent.coordinators.exploration_coordinator import (
                ExplorationCoordinator,
            )
            from victor.config.settings import get_project_paths

            project_root = Path(get_project_paths().project_root)
            explorer = ExplorationCoordinator()

            # Calculate resource-aware exploration budget
            from victor.agent.budget.resource_calculator import (
                calculate_exploration_budget,
            )

            provider_name = getattr(self._provider_context, "provider_name", "ollama")
            if callable(provider_name):
                provider_name = "ollama"  # Mock fallback
            model_name = getattr(self._provider_context, "model", None)
            complexity_str = getattr(task_classification, "complexity", "action")
            if hasattr(complexity_str, "value"):
                complexity_str = complexity_str.value

            budget = calculate_exploration_budget(
                complexity=complexity_str,
                provider=provider_name,
                model=model_name,
            )

            if budget.max_parallel_agents == 0:
                self._exploration_done = True
                return

            _EXPLORATION_IN_PROGRESS = True
            try:
                findings = await asyncio.wait_for(
                    explorer.explore_parallel(
                        task_description=user_message,
                        project_root=project_root,
                        max_results=budget.tool_budget_per_agent,
                    ),
                    timeout=budget.exploration_timeout,
                )
            finally:
                _EXPLORATION_IN_PROGRESS = False
                self._exploration_done = True  # Never explore again this conversation

            if findings.summary:
                self._chat_context.add_message(
                    "user",
                    f"[Parallel exploration results]\n{findings.summary}",
                )
                logger.info(
                    "Parallel exploration: %d files, %d tool calls, %.1fs",
                    len(findings.file_paths),
                    findings.tool_calls,
                    findings.duration_seconds,
                )

        except asyncio.TimeoutError:
            logger.debug("Parallel exploration timed out (90s), skipping")
        except Exception as e:
            logger.debug("Parallel exploration skipped: %s", e)

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
                cum = self._chat_context._cumulative_token_usage
                cum["prompt_tokens"] += response.usage.get("prompt_tokens", 0)
                cum["completion_tokens"] += response.usage.get("completion_tokens", 0)
                cum["total_tokens"] += response.usage.get("total_tokens", 0)

            # Extract extended token fields from raw_response
            raw = getattr(response, "raw_response", None)
            if raw and isinstance(raw, dict):
                raw_usage = raw.get("usage", {}) or {}
                cum = self._chat_context._cumulative_token_usage
                prompt_details = raw_usage.get("prompt_tokens_details", {}) or {}
                comp_details = raw_usage.get("completion_tokens_details", {}) or {}
                cum["cached_tokens"] = cum.get("cached_tokens", 0) + (
                    prompt_details.get("cached_tokens", 0)
                    or raw_usage.get("prompt_cache_hit_tokens", 0)
                    or raw_usage.get("cache_read_input_tokens", 0)
                )
                cum["cache_miss_tokens"] = cum.get("cache_miss_tokens", 0) + (
                    raw_usage.get("prompt_cache_miss_tokens", 0)
                )
                cum["reasoning_tokens"] = cum.get("reasoning_tokens", 0) + (
                    comp_details.get("reasoning_tokens", 0)
                )
                cum["cost_usd_micros"] = cum.get("cost_usd_micros", 0) + (
                    raw_usage.get("cost_in_usd_ticks", 0)
                )

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
            failure_context=(failure_context if failure_context.failed_tools else None),
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

    # =====================================================================
    # Static Heuristics
    # =====================================================================

    @staticmethod
    def _is_question_only_scored(message: str) -> tuple:
        """Detect Q&A messages with confidence score (HDPO-inspired).

        Returns (is_qa: bool, confidence: float) where confidence indicates
        how certain the heuristic is. Low confidence (<0.7) suggests the
        edge LLM should be consulted for refinement.
        """
        msg = message.strip().lower()

        # Short messages ending with ? — high confidence Q&A
        if len(msg) < 120 and msg.endswith("?"):
            action_words = (
                "fix",
                "create",
                "write",
                "edit",
                "refactor",
                "add",
                "implement",
                "update",
                "delete",
                "remove",
                "change",
                "modify",
                "build",
                "deploy",
                "run",
                "test",
                "debug",
                "install",
                "configure",
                "setup",
                "migrate",
            )
            words = set(msg.split())
            action_matches = words.intersection(action_words)
            if not action_matches:
                return (True, 0.95)  # Strong Q&A signal
            else:
                # Question with action words — ambiguous
                return (False, 0.5)  # Low confidence

        # Explicit QA prefixes — high confidence
        qa_prefixes = (
            "what is",
            "what are",
            "what's",
            "who is",
            "who are",
            "how does",
            "how do",
            "how is",
            "how can",
            "why does",
            "why do",
            "why is",
            "when does",
            "when is",
            "explain",
            "describe",
            "define",
            "tell me about",
            "can you explain",
            "what does",
            "show me",
            "reply with",
            "answer",
            "say ",
        )
        for prefix in qa_prefixes:
            if msg.startswith(prefix):
                # Check if action words follow the QA prefix
                rest = msg[len(prefix) :].strip()
                action_words = (
                    "fix",
                    "create",
                    "write",
                    "edit",
                    "refactor",
                    "implement",
                    "update",
                    "delete",
                    "build",
                    "deploy",
                    "run",
                    "test",
                    "debug",
                )
                rest_words = set(rest.split())
                if rest_words.intersection(action_words):
                    return (True, 0.5)  # QA prefix but action words — ambiguous
                return (True, 0.9)

        # Action-heavy messages — high confidence NOT Q&A
        action_indicators = (
            "fix",
            "create",
            "write",
            "edit",
            "refactor",
            "implement",
            "update",
            "delete",
            "build",
            "deploy",
            "run",
            "test",
            "debug",
            "install",
            "configure",
            "setup",
            "migrate",
        )
        words = set(msg.split())
        action_count = len(words.intersection(action_indicators))
        if action_count >= 2:
            return (False, 0.95)
        if action_count == 1:
            return (False, 0.8)

        # No strong signals — default to needs-tools with low confidence
        return (False, 0.6)

    @staticmethod
    def _is_question_only(message: str) -> bool:
        """Detect messages that are pure Q&A and unlikely to need tools.

        Backward-compatible wrapper around _is_question_only_scored().
        """
        is_qa, _ = TurnExecutor._is_question_only_scored(message)
        return is_qa


__all__ = [
    "TurnExecutor",
]
