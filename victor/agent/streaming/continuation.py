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

"""Continuation decision handler for streaming chat.

This module provides the ContinuationHandler class which encapsulates
the continuation decision logic extracted from _stream_chat_impl (P0 SRP refactor).

The handler processes ContinuationStrategy action results and applies them
to the streaming context, yielding appropriate chunks and returning control
signals for the main loop.

Design Pattern: Command + Strategy
==================================
The handler implements a command-style pattern where each continuation action
(prompt_tool_call, request_summary, finish, etc.) maps to a dedicated handler
method. The ContinuationStrategy determines WHAT action to take, and this
handler determines HOW to execute it.

Usage:
    handler = ContinuationHandler(
        orchestrator=self,
        chunk_generator=self._chunk_generator,
        sanitizer=self.sanitizer,
    )

    result = await handler.handle_continuation_action(
        action_result=action_result,
        stream_ctx=stream_ctx,
        full_content=full_content,
    )

    async for chunk in result.chunks:
        yield chunk

    if result.should_return:
        return
    if result.should_skip_rest:
        continue
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    TYPE_CHECKING,
)

from victor.agent.streaming.context import StreamingChatContext
from victor.providers.base import StreamChunk

if TYPE_CHECKING:
    # Use protocol for type hint to avoid circular dependency (DIP compliance)
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.protocols.agent import IAgentOrchestrator
    from victor.agent.streaming.handler import StreamingChatHandler
    from victor.config.settings import Settings

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols for Dependencies
# =============================================================================


class ChunkGeneratorProtocol(Protocol):
    """Protocol for chunk generation."""

    def generate_content_chunk(
        self, content: str, is_final: bool = False, suffix: str = ""
    ) -> StreamChunk: ...

    def generate_final_marker_chunk(self) -> StreamChunk: ...

    def generate_metrics_chunk(
        self, metrics_line: str, is_final: bool = False, prefix: str = "\n\n"
    ) -> StreamChunk: ...

    def format_completion_metrics(
        self,
        ctx: StreamingChatContext,
        elapsed_time: float,
        cost_str: Optional[str] = None,
    ) -> str: ...


class SanitizerProtocol(Protocol):
    """Protocol for content sanitization."""

    def sanitize(self, content: str) -> str: ...


class MessageAdderProtocol(Protocol):
    """Protocol for adding messages to conversation."""

    def add_message(self, role: str, content: str) -> None: ...


# ProviderProtocol imported from core.protocols for consistency
# Uses **kwargs which accepts model, temperature, max_tokens
from victor.core.protocols import ProviderProtocol


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class ContinuationResult:
    """Result of handling a continuation action.

    Attributes:
        chunks: List of chunks to yield to the stream.
        should_return: Whether the main loop should return (exit).
        should_skip_rest: Whether to skip remaining loop body and continue.
        should_continue_loop: Whether to continue the main while loop.
        state_updates: Dict of state updates to apply to the orchestrator.
    """

    chunks: List[StreamChunk] = field(default_factory=list)
    should_return: bool = False
    should_skip_rest: bool = False
    should_continue_loop: bool = True
    state_updates: Dict[str, Any] = field(default_factory=dict)

    def add_chunk(self, chunk: StreamChunk) -> None:
        """Add a chunk to yield."""
        self.chunks.append(chunk)


@dataclass
class TokenBudget:
    """Token-based continuation limits based on model's context window.

    Uses soft and hard limits as percentages of the actual context window
    to prevent premature synthesis while avoiding context overflow.

    Attributes:
        context_window: Model's context window in tokens
        exploration_budget_pct: % of context for exploration phase
        synthesis_budget_pct: % of context reserved for final output
        soft_limit_pct: % at which to nudge synthesis
        hard_limit_pct: % at which to force synthesis
    """

    context_window: int = 128000
    exploration_budget_pct: float = 0.40  # 40% for exploration
    synthesis_budget_pct: float = 0.15  # 15% for output
    soft_limit_pct: float = 0.30  # 30% - nudge at this point
    hard_limit_pct: float = 0.70  # 70% - force at this point (leaves 30% for output)

    @property
    def exploration_limit(self) -> int:
        """Get soft limit for exploration phase in tokens."""
        return int(self.context_window * self.soft_limit_pct)

    @property
    def hard_limit(self) -> int:
        """Get hard limit for total usage in tokens."""
        return int(self.context_window * self.hard_limit_pct)

    @property
    def synthesis_reserve(self) -> int:
        """Get tokens reserved for final output."""
        return int(self.context_window * self.synthesis_budget_pct)

    def should_nudge_synthesis(self, tokens_used: int) -> bool:
        """Check if synthesis nudge is needed based on token usage.

        Args:
            tokens_used: Total tokens used so far

        Returns:
            True if at or above soft limit
        """
        return tokens_used >= self.exploration_limit

    def should_force_synthesis(self, tokens_used: int) -> bool:
        """Check if synthesis should be forced based on token usage.

        Args:
            tokens_used: Total tokens used so far

        Returns:
            True if at or above hard limit
        """
        return tokens_used >= self.hard_limit

    def get_token_status(self, tokens_used: int) -> Dict[str, Any]:
        """Get token usage status for logging/decisions.

        Args:
            tokens_used: Total tokens used so far

        Returns:
            Dict with token usage metrics and recommendations
        """
        usage_pct = (tokens_used / self.context_window * 100) if self.context_window > 0 else 0

        return {
            "tokens_used": tokens_used,
            "context_window": self.context_window,
            "usage_pct": round(usage_pct, 1),
            "exploration_limit": self.exploration_limit,
            "hard_limit": self.hard_limit,
            "should_nudge": self.should_nudge_synthesis(tokens_used),
            "should_force": self.should_force_synthesis(tokens_used),
            "remaining_for_output": max(0, self.context_window - tokens_used),
        }


@dataclass
class ContinuationSignals:
    """Container for all continuation decision signals.

    Combines multiple signal sources for intelligent continuation decisions:
    - Progress metrics (file discovery, tool usage, stuck loops)
    - Token budget (context window usage)
    - Task complexity (simple/medium/complex)
    - Intervention count (cumulative prompts)

    Attributes:
        progress_metrics: ProgressMetrics instance
        task_complexity: Task complexity level
        cumulative_interventions: Total continuation prompts sent
        max_interventions: Max interventions for this complexity
        max_iterations: Max iterations for this complexity
        estimated_tokens: Current token usage estimate
        content_length: Current response content length
    """

    progress_metrics: Optional[ProgressMetrics] = None
    task_complexity: Optional[str] = None
    cumulative_interventions: int = 0
    max_interventions: int = 10
    max_iterations: int = 25
    estimated_tokens: int = 0
    content_length: int = 0

    def calculate_continuation_score(self) -> Dict[str, Any]:
        """Calculate weighted continuation score combining all signals.

        Returns:
            Dict with:
            - score: Overall continuation score (0.0-1.0, higher = continue)
            - signal_scores: Individual signal scores for debugging
            - recommendation: "continue", "nudge_synthesis", or "force_synthesis"
            - confidence: How confident we are in this decision (0.0-1.0)
        """
        signal_scores = {}
        weights = {
            "progress_velocity": 0.30,  # 30% - Most important, measures actual progress
            "stuck_loop_penalty": 0.25,  # 25% - Strong penalty for being stuck
            "token_budget": 0.20,  # 20% - Context window pressure
            "intervention_ratio": 0.15,  # 15% - Intervention count relative to max
            "complexity_adjustment": 0.10,  # 10% - Task complexity factor
        }

        # 1. Progress Velocity Score (0.0-1.0)
        # Higher velocity = better progress = higher score
        if self.progress_metrics:
            velocity = self.progress_metrics.progress_velocity
            # Cap at 0.5 files per iteration as excellent progress
            progress_score = min(velocity / 0.5, 1.0)
            signal_scores["progress_velocity"] = progress_score
        else:
            signal_scores["progress_velocity"] = 0.5  # Neutral if no metrics

        # 2. Stuck Loop Penalty (0.0-1.0, lower is better)
        # 0 = not stuck, 1 = severely stuck
        if self.progress_metrics and self.progress_metrics.is_stuck_loop:
            stuck_penalty = 1.0  # Maximum penalty
        else:
            stuck_penalty = 0.0
        signal_scores["stuck_loop_penalty"] = stuck_penalty

        # 3. Token Budget Score (0.0-1.0)
        # Higher = more tokens remaining = better
        if self.progress_metrics and self.progress_metrics.token_budget:
            token_status = self.progress_metrics.check_token_limits(self.estimated_tokens)
            if token_status:
                usage_pct = token_status.get("usage_pct", 0) / 100
                # Invert: 100% usage = 0 score, 0% usage = 1 score
                token_score = max(0, 1 - usage_pct)
                signal_scores["token_budget"] = token_score
            else:
                signal_scores["token_budget"] = 0.5  # Neutral
        else:
            signal_scores["token_budget"] = 0.5  # Neutral

        # 4. Intervention Ratio Score (0.0-1.0)
        # Higher ratio of used/max = lower score
        intervention_ratio = self.cumulative_interventions / max(self.max_interventions, 1)
        intervention_score = max(0, 1 - intervention_ratio)
        signal_scores["intervention_ratio"] = intervention_score

        # 5. Complexity Adjustment (0.0-1.0)
        # Complex tasks get more allowance to continue
        complexity_scores = {"simple": 0.3, "medium": 0.5, "complex": 0.8, "generation": 0.6}
        complexity_score = complexity_scores.get(self.task_complexity or "medium", 0.5)
        signal_scores["complexity_adjustment"] = complexity_score

        # Calculate weighted score
        total_score = (
            signal_scores["progress_velocity"] * weights["progress_velocity"]
            - signal_scores["stuck_loop_penalty"] * weights["stuck_loop_penalty"]
            + signal_scores["token_budget"] * weights["token_budget"]
            + signal_scores["intervention_ratio"] * weights["intervention_ratio"]
            + signal_scores["complexity_adjustment"] * weights["complexity_adjustment"]
        )

        # Normalize to 0-1 range
        total_score = max(0, min(1, total_score))

        # Determine recommendation based on score
        if total_score >= 0.6:
            recommendation = "continue"
            confidence = min(1, (total_score - 0.6) / 0.4)  # Higher score = higher confidence
        elif total_score >= 0.3:
            recommendation = "nudge_synthesis"
            confidence = min(1, (0.6 - total_score) / 0.3)
        else:
            recommendation = "force_synthesis"
            confidence = min(1, (0.3 - total_score) / 0.3)

        return {
            "score": round(total_score, 3),
            "signal_scores": {k: round(v, 3) for k, v in signal_scores.items()},
            "recommendation": recommendation,
            "confidence": round(confidence, 3),
            "weights": weights,
        }

    def get_detailed_breakdown(self) -> Dict[str, Any]:
        """Get detailed breakdown of all signals for debugging/observability.

        Returns:
            Dict with comprehensive signal information
        """
        score_result = self.calculate_continuation_score()

        breakdown = {
            "continuation_score": score_result["score"],
            "recommendation": score_result["recommendation"],
            "confidence": score_result["confidence"],
            "signal_scores": score_result["signal_scores"],
            "weights_used": score_result["weights"],
            "raw_signals": {
                "task_complexity": self.task_complexity,
                "cumulative_interventions": self.cumulative_interventions,
                "max_interventions": self.max_interventions,
                "max_iterations": self.max_iterations,
                "intervention_pct": round(
                    (self.cumulative_interventions / max(self.max_interventions, 1)) * 100, 1
                ),
                "estimated_tokens": self.estimated_tokens,
                "content_length": self.content_length,
            },
        }

        # Add progress metrics details if available
        if self.progress_metrics:
            breakdown["progress_details"] = self.progress_metrics.get_progress_summary()

            # Add token budget details if available
            if self.progress_metrics.token_budget:
                breakdown["token_details"] = self.progress_metrics.check_token_limits(
                    self.estimated_tokens
                )

        return breakdown


@dataclass
class ProgressMetrics:
    """Track exploration progress for intelligent continuation decisions.

    Distinguishes between productive exploration (discovering new files, using tools)
    and unproductive cycling (re-reading same files, stuck loops).

    Attributes:
        files_read: Set of file paths that have been read (tracked for uniqueness)
        files_revisited: Set of file paths that were read more than once
        tools_used: Set of tool names that were successfully executed
        iterations_without_tools: Count of consecutive iterations without tool calls
        last_tool_call_iteration: The iteration number when tools were last used
        total_iterations: Total number of iterations tracked
        stuck_patterns: List of detected stuck loop patterns (e.g., "re-reading same 3 files")
        token_budget: Optional TokenBudget for token-aware continuation

    Example:
        metrics = ProgressMetrics()
        metrics.record_file_read("config.py")
        metrics.record_tool_used("read")
        metrics.record_iteration(tools_called=True)

        if metrics.is_making_progress:
            continue_exploring()
    """

    files_read: Set[str] = field(default_factory=set)
    files_revisited: Set[str] = field(default_factory=set)
    tools_used: Set[str] = field(default_factory=set)
    iterations_without_tools: int = 0
    last_tool_call_iteration: int = 0
    total_iterations: int = 0
    stuck_patterns: List[str] = field(default_factory=list)
    token_budget: Optional[TokenBudget] = None

    def record_file_read(self, file_path: str) -> None:
        """Record a file read operation.

        Args:
            file_path: Path to the file that was read
        """
        if file_path in self.files_read:
            self.files_revisited.add(file_path)
        else:
            self.files_read.add(file_path)

    def record_tool_used(self, tool_name: str) -> None:
        """Record a successful tool execution.

        Args:
            tool_name: Name of the tool that was executed
        """
        self.tools_used.add(tool_name)
        self.iterations_without_tools = 0  # Reset on successful tool use
        self.last_tool_call_iteration = self.total_iterations

    def record_iteration(self, tools_called: bool = False) -> None:
        """Record a new iteration.

        Args:
            tools_called: Whether tools were called in this iteration
        """
        self.total_iterations += 1
        if not tools_called:
            self.iterations_without_tools += 1

    @property
    def unique_files_read(self) -> int:
        """Get count of unique files read."""
        return len(self.files_read)

    @property
    def total_file_reads(self) -> int:
        """Get total count of file reads including revisits."""
        return len(self.files_read) + len(self.files_revisited)

    @property
    def revisit_ratio(self) -> float:
        """Calculate ratio of revisited to unique files.

        Returns:
            Float from 0.0 (no revisits) to 1.0+ (more revisits than unique files)
        """
        unique = self.unique_files_read
        if unique == 0:
            return 0.0
        return len(self.files_revisited) / unique

    @property
    def progress_velocity(self) -> float:
        """Calculate unique files read per iteration.

        Returns:
            Float representing new files discovered per iteration.
            Higher values indicate more productive exploration.
        """
        if self.total_iterations == 0:
            return 0.0
        return self.unique_files_read / self.total_iterations

    @property
    def is_making_progress(self) -> bool:
        """Determine if exploration is productive.

        Returns:
            True if the agent is making forward progress (using tools, reading new files).
            False if stuck in a loop without progress.
        """
        # Progress if: used tools recently OR discovering new files
        recent_tools = self.iterations_without_tools < 3
        new_discovery = self.unique_files_read > len(self.files_revisited)

        return recent_tools or new_discovery

    @property
    def is_stuck_loop(self) -> bool:
        """Detect unproductive cycling pattern.

        Returns:
            True if agent is cycling without making progress.
        """
        # Stuck if: many iterations without tools AND more revisits than new discoveries
        no_tool_activity = self.iterations_without_tools > 5
        cycling = self.unique_files_read > 0 and self.revisit_ratio > 1.0

        return no_tool_activity and cycling

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of progress metrics for logging/observability.

        Returns:
            Dict with key metrics for debugging and monitoring
        """
        return {
            "unique_files": self.unique_files_read,
            "total_reads": self.total_file_reads,
            "revisited": len(self.files_revisited),
            "tools_used": len(self.tools_used),
            "iterations_without_tools": self.iterations_without_tools,
            "progress_velocity": round(self.progress_velocity, 3),
            "revisit_ratio": round(self.revisit_ratio, 2),
            "is_making_progress": self.is_making_progress,
            "is_stuck_loop": self.is_stuck_loop,
        }

    def initialize_token_budget(self, context_window: int) -> None:
        """Initialize token budget with the given context window.

        Args:
            context_window: Model's context window in tokens
        """
        self.token_budget = TokenBudget(context_window=context_window)

    def check_token_limits(self, tokens_used: int) -> Dict[str, Any]:
        """Check token budget limits and get recommendations.

        Args:
            tokens_used: Total tokens used so far

        Returns:
            Dict with token status and recommendations (empty if no token budget)
        """
        if self.token_budget is None:
            return {}
        return self.token_budget.get_token_status(tokens_used)


# =============================================================================
# Continuation Handler
# =============================================================================


class ContinuationHandler:
    """Handler for continuation decision execution.

    This class encapsulates the logic for executing continuation actions
    determined by ContinuationStrategy. It extracts ~300 lines of if/elif
    chain from _stream_chat_impl into focused, testable methods.

    Each action type has a dedicated handler method:
    - _handle_continue_asking_input
    - _handle_return_to_user
    - _handle_prompt_tool_call
    - _handle_continue_with_synthesis_hint
    - _handle_request_summary
    - _handle_request_completion
    - _handle_execute_extracted_tool
    - _handle_force_tool_execution
    - _handle_finish

    Example:
        handler = ContinuationHandler(message_adder=orchestrator, ...)
        result = await handler.handle_action(action_result, stream_ctx, content)

        for chunk in result.chunks:
            yield chunk
        if result.should_return:
            return
    """

    def __init__(
        self,
        message_adder: MessageAdderProtocol,
        chunk_generator: ChunkGeneratorProtocol,
        sanitizer: SanitizerProtocol,
        settings: "Settings",
        provider: Optional[ProviderProtocol] = None,
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        messages_getter: Optional[Callable[[], List[Any]]] = None,
        unified_tracker: Optional[Any] = None,
        finalize_metrics: Optional[Callable[[Dict[str, Any]], Any]] = None,
        record_outcome: Optional[Callable[..., None]] = None,
        execute_extracted_tool: Optional[Callable[..., AsyncIterator[StreamChunk]]] = None,
        progress_metrics: Optional[ProgressMetrics] = None,
    ):
        """Initialize the continuation handler.

        Args:
            message_adder: Object that can add messages to conversation.
            chunk_generator: Generator for stream chunks.
            sanitizer: Content sanitizer.
            settings: Application settings.
            provider: Optional LLM provider for forced responses.
            model: Model name for provider calls.
            temperature: Temperature for provider calls.
            max_tokens: Max tokens for provider calls.
            messages_getter: Callable to get current messages list.
            unified_tracker: Task tracker for turn increment.
            finalize_metrics: Callable to finalize stream metrics.
            record_outcome: Callable to record Q-learning outcome.
            execute_extracted_tool: Callable for extracted tool execution.
            progress_metrics: Optional ProgressMetrics for tracking exploration.
        """
        self._message_adder = message_adder
        self._chunk_generator = chunk_generator
        self._sanitizer = sanitizer
        self._settings = settings
        self._provider = provider
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._messages_getter = messages_getter
        self._unified_tracker = unified_tracker
        self._finalize_metrics = finalize_metrics
        self._record_outcome = record_outcome
        self._execute_extracted_tool = execute_extracted_tool
        self._progress_metrics = progress_metrics

        # State tracking (mirrors orchestrator state)
        self._cumulative_prompt_interventions = 0
        self._summary_request_count = 0

    async def handle_action(
        self,
        action_result: Dict[str, Any],
        stream_ctx: StreamingChatContext,
        full_content: str,
    ) -> ContinuationResult:
        """Handle a continuation action from ContinuationStrategy.

        This is the main entry point that dispatches to specific handlers
        based on the action type.

        Args:
            action_result: The result from ContinuationStrategy.determine_continuation_action()
            stream_ctx: The streaming context.
            full_content: The full response content from the model.

        Returns:
            ContinuationResult with chunks and control flags.
        """
        # Apply state updates from action result
        self._apply_state_updates(action_result)

        action = action_result.get("action", "finish")

        # Dispatch to specific handler
        handlers = {
            "continue_asking_input": self._handle_continue_asking_input,
            "return_to_user": self._handle_return_to_user,
            "prompt_tool_call": self._handle_prompt_tool_call,
            "continue_with_synthesis_hint": self._handle_continue_with_synthesis_hint,
            "request_summary": self._handle_request_summary,
            "request_completion": self._handle_request_completion,
            "execute_extracted_tool": self._handle_execute_extracted_tool,
            "force_tool_execution": self._handle_force_tool_execution,
            "finish": self._handle_finish,
        }

        handler = handlers.get(action, self._handle_finish)
        return await handler(action_result, stream_ctx, full_content)

    def _apply_state_updates(self, action_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply state updates from action result.

        Args:
            action_result: The action result with updates.

        Returns:
            Dict of updates that were applied.
        """
        updates = action_result.get("updates", {})
        applied = {}

        if "cumulative_prompt_interventions" in updates:
            # Note: This is tracked internally but also returned for orchestrator sync
            applied["cumulative_prompt_interventions"] = updates["cumulative_prompt_interventions"]

        if action_result.get("set_final_summary_requested"):
            applied["final_summary_requested"] = True

        if action_result.get("set_max_prompts_summary_requested"):
            applied["max_prompts_summary_requested"] = True

        return applied

    async def _handle_continue_asking_input(
        self,
        action_result: Dict[str, Any],
        stream_ctx: StreamingChatContext,
        full_content: str,
    ) -> ContinuationResult:
        """Handle continue_asking_input action.

        Adds a follow-up message and continues the loop.
        """
        result = ContinuationResult(should_skip_rest=True)
        message = action_result.get("message", "")
        if message:
            self._message_adder.add_message("user", message)
        return result

    async def _handle_return_to_user(
        self,
        action_result: Dict[str, Any],
        stream_ctx: StreamingChatContext,
        full_content: str,
    ) -> ContinuationResult:
        """Handle return_to_user action.

        Yields accumulated content and exits the loop.
        """
        result = ContinuationResult(should_return=True)

        # Yield accumulated content
        if full_content:
            sanitized = self._sanitizer.sanitize(full_content)
            if sanitized:
                result.add_chunk(self._chunk_generator.generate_content_chunk(sanitized))

        # Add final marker
        result.add_chunk(self._chunk_generator.generate_final_marker_chunk())
        return result

    async def _handle_prompt_tool_call(
        self,
        action_result: Dict[str, Any],
        stream_ctx: StreamingChatContext,
        full_content: str,
    ) -> ContinuationResult:
        """Handle prompt_tool_call action.

        Prompts the model to make a tool call.
        """
        result = ContinuationResult(should_skip_rest=True)
        message = action_result.get("message", "")
        if message:
            self._message_adder.add_message("user", message)

        # Increment turn in tracker
        if self._unified_tracker:
            self._unified_tracker.increment_turn()

        # Track intervention
        self._cumulative_prompt_interventions += 1
        result.state_updates["cumulative_prompt_interventions"] = (
            self._cumulative_prompt_interventions
        )

        # Track iteration in progress metrics (no tools called in this iteration)
        if self._progress_metrics:
            self._progress_metrics.record_iteration(tools_called=False)

        return result

    async def _handle_continue_with_synthesis_hint(
        self,
        action_result: Dict[str, Any],
        stream_ctx: StreamingChatContext,
        full_content: str,
    ) -> ContinuationResult:
        """Handle continue_with_synthesis_hint action.

        Gentle nudge to synthesize when all required files are read.
        """
        result = ContinuationResult(should_skip_rest=True)
        message = action_result.get("message", "")
        if message:
            self._message_adder.add_message("user", message)

        # Update synthesis nudge count in tracker
        updates = action_result.get("updates", {})
        if "synthesis_nudge_count" in updates and self._unified_tracker:
            if hasattr(self._unified_tracker, "synthesis_nudge_count"):
                self._unified_tracker.synthesis_nudge_count = updates["synthesis_nudge_count"]

        return result

    async def _handle_request_summary(
        self,
        action_result: Dict[str, Any],
        stream_ctx: StreamingChatContext,
        full_content: str,
    ) -> ContinuationResult:
        """Handle request_summary action.

        Requests a summary from the model, with forced response on second request.
        """
        # Check if this is a repeated request
        if self._summary_request_count >= 1:
            return await self._force_summary_response(stream_ctx)

        # First summary request
        self._summary_request_count += 1
        result = ContinuationResult(should_skip_rest=True)
        message = action_result.get("message", "")
        if message:
            self._message_adder.add_message("user", message)

        return result

    async def _force_summary_response(
        self,
        stream_ctx: StreamingChatContext,
    ) -> ContinuationResult:
        """Force a summary response when model ignores summary request.

        Disables tools and makes a direct provider call.
        """
        result = ContinuationResult(should_return=True)
        logger.warning(
            "Model ignored previous summary request - forcing final response with tools disabled"
        )

        if not self._provider or not self._messages_getter:
            # Can't force without provider
            result.add_chunk(self._chunk_generator.generate_final_marker_chunk())
            return result

        try:
            # Import Message here to avoid circular dependency
            from victor.providers.base import Message

            messages = self._messages_getter() + [
                Message(
                    role="user",
                    content="CRITICAL: Provide your FINAL ANALYSIS NOW. "
                    "Do NOT mention any more tools or files. "
                    "Summarize what you found from the tool calls you already executed.",
                )
            ]

            response = await self._provider.chat(
                messages=messages,
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                tools=None,  # Disable tools to force text response
            )

            if response and response.content:
                sanitized = self._sanitizer.sanitize(response.content)
                if sanitized:
                    self._message_adder.add_message("assistant", sanitized)
                    result.add_chunk(self._chunk_generator.generate_content_chunk(sanitized))

            # Finalize metrics
            if self._finalize_metrics:
                final_metrics = self._finalize_metrics(stream_ctx.cumulative_usage)
                elapsed_time = (
                    final_metrics.total_duration
                    if final_metrics
                    else time.time() - stream_ctx.start_time
                )
                cost_str = None
                if self._settings.show_cost_metrics and final_metrics:
                    cost_str = final_metrics.format_cost()
                metrics_line = self._chunk_generator.format_completion_metrics(
                    stream_ctx, elapsed_time, cost_str
                )
                result.add_chunk(self._chunk_generator.generate_metrics_chunk(metrics_line))

            result.add_chunk(self._chunk_generator.generate_final_marker_chunk())

        except Exception as e:
            logger.warning(f"Error forcing final response: {e}")
            result.add_chunk(self._chunk_generator.generate_final_marker_chunk())

        return result

    async def _handle_request_completion(
        self,
        action_result: Dict[str, Any],
        stream_ctx: StreamingChatContext,
        full_content: str,
    ) -> ContinuationResult:
        """Handle request_completion action.

        Requests the model to complete its response.
        """
        result = ContinuationResult(should_skip_rest=True)
        message = action_result.get("message", "")
        if message:
            self._message_adder.add_message("user", message)
        return result

    async def _handle_execute_extracted_tool(
        self,
        action_result: Dict[str, Any],
        stream_ctx: StreamingChatContext,
        full_content: str,
    ) -> ContinuationResult:
        """Handle execute_extracted_tool action.

        Executes a tool call extracted from the model's text response.
        """
        result = ContinuationResult(should_skip_rest=True)
        extracted_call = action_result.get("extracted_call")

        if extracted_call and self._execute_extracted_tool:
            logger.info(
                f"Executing extracted tool call: {extracted_call.tool_name} "
                f"(confidence: {extracted_call.confidence:.2f})"
            )
            # Execute and collect chunks
            async for chunk in self._execute_extracted_tool(stream_ctx, extracted_call):
                result.add_chunk(chunk)

        return result

    async def _handle_force_tool_execution(
        self,
        action_result: Dict[str, Any],
        stream_ctx: StreamingChatContext,
        full_content: str,
    ) -> ContinuationResult:
        """Handle force_tool_execution action.

        Forces tool execution when model mentions tools but doesn't call them.
        """
        result = ContinuationResult(should_skip_rest=True)
        mentioned_tools = action_result.get("mentioned_tools", [])
        force_message = action_result.get("message", "")

        # Track force attempt
        attempt_count = stream_ctx.record_force_tool_attempt()

        if attempt_count >= 3:
            # Give up after 3 attempts
            logger.warning(
                "Giving up on forced tool execution after 3 attempts - requesting summary"
            )
            self._message_adder.add_message(
                "user",
                "You are unable to make tool calls. Please provide your response "
                "NOW based on what you know. Do not mention any tools.",
            )
            stream_ctx.reset_force_tool_attempts()
        elif force_message:
            self._message_adder.add_message("user", force_message)
        else:
            # Default message
            tools_str = ", ".join(mentioned_tools)
            default_message = (
                f"You mentioned using {tools_str} but did not actually call the tool(s). "
                "Please make the actual tool call now, or provide your final answer without "
                "mentioning tools you cannot use."
            )
            self._message_adder.add_message("user", default_message)

        # Increment turn in tracker (mirrors _handle_force_tool_execution_with_handler)
        if self._unified_tracker:
            self._unified_tracker.increment_turn()

        return result

    async def _handle_finish(
        self,
        action_result: Dict[str, Any],
        stream_ctx: StreamingChatContext,
        full_content: str,
    ) -> ContinuationResult:
        """Handle finish action.

        Finalizes the response with metrics and exits.
        """
        result = ContinuationResult(should_return=True)

        # Yield accumulated content
        if full_content:
            sanitized = self._sanitizer.sanitize(full_content)
            if sanitized:
                result.add_chunk(self._chunk_generator.generate_content_chunk(sanitized))

        # Finalize and display metrics
        if self._finalize_metrics:
            final_metrics = self._finalize_metrics(stream_ctx.cumulative_usage)
            elapsed_time = (
                final_metrics.total_duration
                if final_metrics
                else time.time() - stream_ctx.start_time
            )
            cost_str = None
            if self._settings.show_cost_metrics and final_metrics:
                cost_str = final_metrics.format_cost()
            metrics_line = self._chunk_generator.format_completion_metrics(
                stream_ctx, elapsed_time, cost_str
            )
            result.add_chunk(self._chunk_generator.generate_metrics_chunk(metrics_line))

        # Record outcome for Q-learning
        if self._record_outcome:
            outcome = self._record_outcome(
                success=True,
                quality_score=stream_ctx.last_quality_score,
                user_satisfied=True,
                completed=True,
            )
            if asyncio.iscoroutine(outcome):
                await outcome  # type: ignore[unreachable]

        result.add_chunk(self._chunk_generator.generate_final_marker_chunk())
        return result


# =============================================================================
# Factory Function
# =============================================================================


def create_continuation_handler(
    orchestrator: "AgentOrchestrator",
) -> ContinuationHandler:
    """Factory function to create a ContinuationHandler from an orchestrator.

    Args:
        orchestrator: The AgentOrchestrator instance.

    Returns:
        Configured ContinuationHandler.
    """
    chunk_gen = orchestrator._chunk_generator or getattr(orchestrator, "chunk_generator", None)
    if chunk_gen is None:
        raise ValueError("orchestrator must have a chunk_generator")
    return ContinuationHandler(
        message_adder=orchestrator,
        chunk_generator=chunk_gen,
        sanitizer=orchestrator.sanitizer,
        settings=orchestrator.settings,
        provider=orchestrator.provider,
        model=orchestrator.model,
        temperature=orchestrator.temperature,
        max_tokens=orchestrator.max_tokens,
        messages_getter=lambda: orchestrator.messages,
        unified_tracker=orchestrator.unified_tracker,
        finalize_metrics=orchestrator._finalize_stream_metrics,
        record_outcome=orchestrator._record_intelligent_outcome if not asyncio.iscoroutinefunction(orchestrator._record_intelligent_outcome) else None,  # type: ignore[arg-type]
        execute_extracted_tool=orchestrator._execute_extracted_tool_call,
        progress_metrics=getattr(orchestrator, "_progress_metrics", None),
    )
