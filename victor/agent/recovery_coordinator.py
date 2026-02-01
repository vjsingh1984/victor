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
# See the License for the specific language governing permissions
# limitations under the License.

"""Streaming recovery coordinator for streaming chat sessions.

This module consolidates all recovery and error handling logic previously
scattered across AgentOrchestrator._handle_stream_chunk and related methods.

Extracted from CRITICAL-001 Phase 2A: Extract RecoveryCoordinator

NOTE: This is StreamingRecoveryCoordinator, distinct from:
- victor.agent.recovery.coordinator.RecoveryCoordinator (SOLID recovery system)
The name was changed to avoid confusion between the two different classes.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.recovery import RecoveryHandler
    from victor.agent.orchestrator_recovery import (
        OrchestratorRecoveryIntegration,
        OrchestratorRecoveryAction,
    )
    from victor.agent.streaming import StreamingChatHandler, StreamingChatContext
    from victor.agent.context_compactor import ContextCompactor
    from victor.agent.unified_task_tracker import UnifiedTaskTracker
    from victor.config.settings import Settings
    from victor.agent.presentation import PresentationProtocol

from victor.providers.base import StreamChunk
from victor.core.events import ObservabilityBus

logger = logging.getLogger(__name__)


@dataclass
class StreamingRecoveryContext:
    """Context for streaming recovery decisions.

    This dataclass encapsulates all state needed for streaming recovery coordination,
    eliminating the need for StreamingRecoveryCoordinator to hold mutable state.

    NOTE: This is distinct from victor.agent.recovery.protocols.StreamingRecoveryContext
    which is used by the SOLID recovery system for failure detection/strategy.

    Attributes:
        iteration: Current iteration number
        elapsed_time: Elapsed time since session start (seconds)
        tool_calls_used: Number of tool calls made so far
        tool_budget: Maximum allowed tool calls
        max_iterations: Maximum allowed iterations
        session_start_time: Unix timestamp when session started
        last_quality_score: Quality score from last iteration (0-1)
        streaming_context: StreamingChatContext for current session
        provider_name: Name of current LLM provider
        model: Name of current model
        temperature: Current temperature setting
        unified_task_type: Task type from UnifiedTaskTracker
        is_analysis_task: Whether this is an analysis task
        is_action_task: Whether this is an action task
    """

    iteration: int
    elapsed_time: float
    tool_calls_used: int
    tool_budget: int
    max_iterations: int
    session_start_time: float
    last_quality_score: float
    streaming_context: "StreamingChatContext"
    provider_name: str
    model: str
    temperature: float
    unified_task_type: Any  # TaskType from UnifiedTaskTracker
    is_analysis_task: bool
    is_action_task: bool


class StreamingRecoveryCoordinator:
    """Coordinates recovery actions during streaming chat sessions.

    Responsibilities:
    - Check recovery conditions (time, iterations, budget, progress, blocked tools)
    - Apply recovery actions (truncation, completion, fallback, retry)
    - Format recovery prompts and metrics
    - Integrate with RecoveryHandler for decision-making
    - Delegate to StreamingChatHandler for stream-specific logic

    This component centralizes all recovery logic previously scattered across
    AgentOrchestrator, providing a clean interface for recovery coordination.

    NOTE: This is distinct from victor.agent.recovery.coordinator.RecoveryCoordinator
    which handles failure detection and strategy selection for the SOLID recovery system.

    Architecture:
    - StreamingRecoveryCoordinator: High-level streaming session coordination
    - StreamingChatHandler: Low-level stream manipulation and chunk generation
    - RecoveryHandler: Recovery strategy selection and decision-making
    - OrchestratorRecoveryIntegration: Integration between orchestrator and handler

    Design Pattern:
    - Facade: Simplifies complex recovery subsystem
    - Strategy: RecoveryHandler provides pluggable strategies
    - Delegation: Most logic delegated to specialized handlers
    """

    def __init__(
        self,
        recovery_handler: Optional["RecoveryHandler"],
        recovery_integration: Optional["OrchestratorRecoveryIntegration"],
        streaming_handler: "StreamingChatHandler",
        context_compactor: Optional["ContextCompactor"],
        unified_tracker: "UnifiedTaskTracker",
        settings: "Settings",
        event_bus: Optional[ObservabilityBus] = None,
        presentation: Optional["PresentationProtocol"] = None,
    ):
        """Initialize StreamingRecoveryCoordinator.

        Args:
            recovery_handler: Recovery strategy handler (optional)
            recovery_integration: Integration with recovery system (optional)
            streaming_handler: Handler for streaming logic
            context_compactor: Context management (optional)
            unified_tracker: Task progress tracker
            settings: Application settings
            event_bus: Optional ObservabilityBus instance. If None, uses DI container.
            presentation: Optional presentation adapter for icons (creates default if None)
        """
        self.recovery_handler = recovery_handler
        self.recovery_integration = recovery_integration
        self.streaming_handler = streaming_handler
        self.context_compactor = context_compactor
        self.unified_tracker = unified_tracker
        self.settings = settings
        self._event_bus = event_bus or self._get_default_bus()
        # Lazy init for backward compatibility
        if presentation is None:
            from victor.agent.presentation import create_presentation_adapter

            self._presentation = create_presentation_adapter()
        else:
            self._presentation = presentation

    def _get_default_bus(self) -> Optional[ObservabilityBus]:
        """Get default ObservabilityBus from DI container.

        Returns:
            ObservabilityBus instance or None if unavailable
        """
        try:
            from victor.core.events import get_observability_bus

            return get_observability_bus()
        except Exception:
            return None

    # =====================================================================
    # Condition Checking Methods
    # =====================================================================

    def check_time_limit(
        self,
        ctx: StreamingRecoveryContext,
    ) -> Optional[StreamChunk]:
        """Check if session has exceeded time limit.

        Delegates to StreamingChatHandler for time limit checking,
        then records intelligent outcome for RL learning.

        Args:
            ctx: Recovery context

        Returns:
            StreamChunk if time limit reached, None otherwise
        """
        result = self.streaming_handler.check_time_limit(ctx.streaming_context)
        if result:
            # Emit STATE event for time limit reached (fire-and-forget)
            if self._event_bus:
                try:
                    import asyncio

                    # Schedule the coroutine without blocking - avoids RuntimeWarning
                    asyncio.create_task(
                        self._event_bus.emit(
                            topic="state.recovery.time_limit_reached",
                            data={
                                "elapsed_time": ctx.elapsed_time,
                                "iteration": ctx.iteration,
                                "tool_calls_used": ctx.tool_calls_used,
                                "category": "state",  # Preserve for observability
                            },
                            source="RecoveryCoordinator",
                        )
                    )
                except Exception as e:
                    logger.debug(f"Failed to emit time limit event: {e}")

            # Return the chunk from the result
            if result.chunks:
                return result.chunks[0]
            pending_icon = self._presentation.icon("pending", with_color=False)
            return StreamChunk(
                content=f"\n\n{pending_icon} Session time limit reached. "
                "Providing summary of progress so far.\n",
                is_final=False,
            )
        return None

    def check_iteration_limit(
        self,
        ctx: StreamingRecoveryContext,
    ) -> Optional[StreamChunk]:
        """Check if session has exceeded iteration limit.

        Args:
            ctx: Recovery context

        Returns:
            StreamChunk if iteration limit reached, None otherwise
        """
        result = self.streaming_handler.check_iteration_limit(ctx.streaming_context)
        if result and result.chunks:
            # Emit STATE event for iteration limit reached
            if self._event_bus:
                try:
                    import asyncio

                    asyncio.create_task(
                        self._event_bus.emit(
                            topic="state.recovery.iteration_limit_reached",
                            data={
                                "iteration": ctx.iteration,
                                "max_iterations": ctx.max_iterations,
                                "tool_calls_used": ctx.tool_calls_used,
                                "category": "state",  # Preserve for observability
                            },
                            source="RecoveryCoordinator",
                        )
                    )
                except Exception as e:
                    logger.debug(f"Failed to emit iteration limit event: {e}")
            return result.chunks[0]
        return None

    def check_natural_completion(
        self,
        ctx: StreamingRecoveryContext,
        has_tool_calls: bool,
        content_length: int,
    ) -> Optional[StreamChunk]:
        """Check for natural completion (no tool calls, sufficient content).

        Args:
            ctx: Recovery context
            has_tool_calls: Whether there are tool calls
            content_length: Length of current content

        Returns:
            StreamChunk if natural completion detected, None otherwise
        """
        result = self.streaming_handler.check_natural_completion(
            ctx.streaming_context, has_tool_calls, content_length
        )
        if result:
            return StreamChunk(content="", is_final=True)
        return None

    def check_tool_budget(
        self,
        ctx: StreamingRecoveryContext,
        warning_threshold: int = 250,
    ) -> Optional[StreamChunk]:
        """Check tool budget and generate warning if approaching limit.

        Warning is triggered when:
        - Used at least warning_threshold tool calls (e.g., 250)
        - Budget is not yet exhausted

        Args:
            ctx: Recovery context
            warning_threshold: Number of tool calls before warning (default 250)

        Returns:
            StreamChunk with warning if approaching limit, None otherwise
        """
        remaining = ctx.tool_budget - ctx.tool_calls_used

        # Warn when we've used at least warning_threshold calls and still have remaining
        if ctx.tool_calls_used >= warning_threshold and remaining > 0:
            warning_icon = self._presentation.icon("warning", with_color=False)
            return StreamChunk(
                content=f"[tool] {warning_icon} Approaching tool budget limit: {ctx.tool_calls_used}/{ctx.tool_budget} calls used\n"
            )
        return None

    def is_budget_exhausted(self, ctx: StreamingRecoveryContext) -> bool:
        """Check if tool budget has been exhausted.

        Args:
            ctx: Recovery context

        Returns:
            True if budget exhausted, False otherwise
        """
        return ctx.tool_calls_used >= ctx.tool_budget

    def check_progress(
        self,
        ctx: StreamingRecoveryContext,
    ) -> bool:
        """Check if session is making progress (not looping).

        Uses UnifiedTaskTracker to detect repetitive tool calls.

        Args:
            ctx: Recovery context

        Returns:
            True if making progress, False if stuck/looping
        """
        decision = self.unified_tracker.should_stop()
        return not decision.should_stop

    def check_blocked_threshold(
        self,
        ctx: StreamingRecoveryContext,
        all_blocked: bool,
    ) -> Optional[tuple[StreamChunk, bool]]:
        """Check if too many tools have been blocked.

        Args:
            ctx: Recovery context
            all_blocked: Whether all tool calls were blocked

        Returns:
            Tuple of (chunk, should_clear_tools) if threshold exceeded, None otherwise
        """
        consecutive_limit = getattr(self.settings, "recovery_blocked_consecutive_threshold", 4)
        total_limit = getattr(self.settings, "recovery_blocked_total_threshold", 6)

        result = self.streaming_handler.check_blocked_threshold(
            ctx.streaming_context, all_blocked, consecutive_limit, total_limit
        )
        if result:
            warning_icon = self._presentation.icon("warning", with_color=False)
            chunk = (
                result.chunks[0]
                if result.chunks
                else StreamChunk(
                    content=f"\n[loop] {warning_icon} Multiple blocked attempts - forcing completion\n"
                )
            )
            return (chunk, result.clear_tool_calls)
        return None

    def check_force_action(
        self,
        ctx: StreamingRecoveryContext,
    ) -> tuple[bool, Optional[str]]:
        """Check if recovery handler recommends force action.

        Args:
            ctx: Recovery context

        Returns:
            Tuple of (should_force, action_type)
        """
        if not self.recovery_handler:
            return False, None

        # Check if recovery handler recommends force action
        # This would integrate with RecoveryHandler.should_force_action()
        # For now, return False (no force action)
        return False, None

    # =====================================================================
    # Action Handling Methods
    # =====================================================================

    def handle_empty_response(
        self,
        ctx: StreamingRecoveryContext,
    ) -> tuple[Optional[StreamChunk], bool]:
        """Handle empty model response.

        Args:
            ctx: Recovery context

        Returns:
            Tuple of (StreamChunk if threshold exceeded, should_force_completion flag)
        """
        result = self.streaming_handler.handle_empty_response(ctx.streaming_context)
        if result and result.chunks:
            # Emit ERROR event for empty response
            if self._event_bus:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(
                        self._event_bus.emit(
                            topic="error.raised",
                            data={
                                "error_type": "RuntimeError",
                                "error_message": "Empty model response",
                                "category": "error",
                                "recoverable": True,
                                "context": {
                                    "iteration": ctx.iteration,
                                    "provider": ctx.provider_name,
                                    "model": ctx.model,
                                    "force_completion": ctx.streaming_context.force_completion,
                                },
                            },
                        )
                    )
                except RuntimeError:
                    # No event loop running
                    logger.debug("No event loop, skipping error event emission")
                except Exception as e:
                    logger.debug(f"Failed to emit empty response error event: {e}")
            # Handler sets ctx.force_completion = True when threshold exceeded
            return result.chunks[0], ctx.streaming_context.force_completion
        return None, False

    def handle_blocked_tool(
        self,
        ctx: StreamingRecoveryContext,
        tool_name: str,
        tool_args: dict[str, Any],
        block_reason: str,
    ) -> StreamChunk:
        """Handle blocked tool call.

        Args:
            ctx: Recovery context
            tool_name: Name of blocked tool
            tool_args: Arguments that were passed
            block_reason: Reason for blocking

        Returns:
            StreamChunk with block notification
        """
        # Emit ERROR event for blocked tool
        if self._event_bus:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self._event_bus.emit(
                        topic="error.raised",
                        data={
                            "error_type": "RuntimeError",
                            "error_message": f"Tool blocked: {tool_name}",
                            "category": "error",
                            "recoverable": True,
                            "context": {
                                "tool_name": tool_name,
                                "block_reason": block_reason,
                                "iteration": ctx.iteration,
                            },
                        },
                    )
                )
            except RuntimeError:
                # No event loop running
                logger.debug("No event loop, skipping error event emission")
            except Exception as e:
                logger.debug(f"Failed to emit blocked tool error event: {e}")
        return self.streaming_handler.handle_blocked_tool_call(
            ctx.streaming_context, tool_name, tool_args, block_reason
        )

    def handle_force_tool_execution(
        self,
        ctx: StreamingRecoveryContext,
    ) -> tuple[bool, Optional[list[StreamChunk]]]:
        """Handle forced tool execution.

        Args:
            ctx: Recovery context

        Returns:
            Tuple of (should_execute, chunks)
        """
        # Check if recovery handler recommends force execution
        # For now, return False (no force execution)
        return False, None

    def handle_force_completion(
        self,
        ctx: StreamingRecoveryContext,
    ) -> Optional[list[StreamChunk]]:
        """Handle forced completion.

        Args:
            ctx: Recovery context

        Returns:
            List of StreamChunks if forced completion, None otherwise
        """
        if not ctx.streaming_context.force_completion:
            return None

        # Emit STATE event for forced completion
        if self._event_bus:
            try:
                import asyncio

                asyncio.create_task(
                    self._event_bus.emit(
                        topic="state.recovery.force_completion",
                        data={
                            "iteration": ctx.iteration,
                            "tool_calls_used": ctx.tool_calls_used,
                            "category": "state",  # Preserve for observability
                        },
                        source="RecoveryCoordinator",
                    )
                )
            except Exception as e:
                logger.debug(f"Failed to emit force completion event: {e}")

        # Generate forced completion chunks
        clipboard_icon = self._presentation.icon("clipboard", with_color=False)
        chunks = []
        chunks.append(
            StreamChunk(
                content=f"\n\n{clipboard_icon} Providing summary of progress...\n",
                is_final=False,
            )
        )
        return chunks

    def handle_loop_warning(
        self,
        ctx: StreamingRecoveryContext,
    ) -> Optional[list[StreamChunk]]:
        """Handle loop detection warning.

        Args:
            ctx: Recovery context

        Returns:
            List of warning chunks, None if no loop detected
        """
        decision = self.unified_tracker.should_stop()
        if not decision.should_stop:
            return None

        # Generate loop warning chunks
        warning_icon = self._presentation.icon("warning", with_color=False)
        chunks = []
        chunks.append(
            StreamChunk(
                content=f"\n[loop] {warning_icon} {decision.reason}\n",
                is_final=False,
            )
        )
        return chunks

    async def handle_recovery_with_integration(
        self,
        ctx: StreamingRecoveryContext,
        full_content: str,
        tool_calls: Optional[list[dict[str, Any]]],
        mentioned_tools: Optional[list[str]] = None,
        message_adder: Any = None,
    ) -> "OrchestratorRecoveryAction":
        """Handle response using the recovery integration.

        This method delegates to OrchestratorRecoveryIntegration to:
        - Detect failures from model responses
        - Apply recovery strategies via RecoveryHandler
        - Provide recovery prompts and temperature adjustments

        Args:
            ctx: Recovery context
            full_content: Full response content
            tool_calls: Tool calls made (if any)
            mentioned_tools: Tools mentioned but not called
            message_adder: Callback to add messages (for retry action)

        Returns:
            RecoveryAction with action to take (continue, retry, abort, force_summary)
        """
        # Import here to avoid circular import
        from victor.agent.orchestrator_recovery import OrchestratorRecoveryAction

        if self.recovery_integration is None or not self.recovery_integration.enabled:
            # Return a continue action if recovery not enabled
            return OrchestratorRecoveryAction(action="continue", reason="Recovery disabled")

        # Get context utilization for recovery decisions
        context_utilization = None
        if self.context_compactor:
            stats = self.context_compactor.get_statistics()
            context_utilization = stats.get("current_utilization")

        # Call recovery integration
        recovery_action = await self.recovery_integration.handle_response(
            content=full_content,
            tool_calls=tool_calls,
            mentioned_tools=mentioned_tools,
            provider_name=ctx.provider_name,
            model_name=ctx.model,
            tool_calls_made=ctx.tool_calls_used,
            tool_budget=ctx.tool_budget,
            iteration_count=ctx.streaming_context.total_iterations,
            max_iterations=ctx.streaming_context.max_total_iterations,
            current_temperature=ctx.temperature,
            quality_score=ctx.last_quality_score,
            task_type=ctx.unified_task_type.value,
            is_analysis_task=ctx.is_analysis_task,
            is_action_task=ctx.is_action_task,
            context_utilization=context_utilization,
        )

        # Log recovery decision
        if recovery_action.action != "continue":
            logger.info(
                f"Recovery integration: action={recovery_action.action}, "
                f"reason={recovery_action.reason}, "
                f"failure_type={recovery_action.failure_type}, "
                f"strategy={recovery_action.strategy_name}"
            )

        return recovery_action

    def apply_recovery_action(
        self,
        recovery_action: "OrchestratorRecoveryAction",
        ctx: StreamingRecoveryContext,
        message_adder: Any = None,
    ) -> Optional[StreamChunk]:
        """Apply a recovery action from the recovery integration.

        This method handles the various recovery actions:
        - continue: No action needed
        - retry: Add recovery message and optionally adjust temperature
        - force_summary: Force completion with summary request
        - abort: Return abort chunk

        Args:
            recovery_action: The recovery action to apply
            ctx: Recovery context
            message_adder: Callback to add messages (callable with (role, content))

        Returns:
            StreamChunk if action requires immediate yield, None otherwise
        """
        if recovery_action.action == "continue":
            return None

        # Emit STATE event for recovery action
        if self._event_bus:
            try:
                import asyncio

                asyncio.create_task(
                    self._event_bus.emit(
                        topic=f"state.recovery.action_{recovery_action.action}",
                        data={
                            "action": recovery_action.action,
                            "reason": recovery_action.reason,
                            "category": "state",  # Preserve for observability
                        },
                        source="RecoveryCoordinator",
                    )
                )
            except Exception as e:
                logger.debug(f"Failed to emit recovery action event: {e}")

        if recovery_action.action == "retry":
            # Add recovery message if provided
            if recovery_action.message and message_adder:
                message_adder("user", recovery_action.message)

            # Adjust temperature if provided (for next iteration)
            if recovery_action.new_temperature is not None:
                logger.debug(
                    f"Recovery: adjusting temperature from {ctx.temperature} to "
                    f"{recovery_action.new_temperature}"
                )
                # Note: Temperature adjustment is per-call, not persistent
                # The caller should use this in provider_kwargs

            return None

        if recovery_action.action == "force_summary":
            ctx.streaming_context.force_completion = True
            if recovery_action.message and message_adder:
                message_adder("user", recovery_action.message)
            elif message_adder:
                message_adder(
                    "user",
                    "Please provide a brief summary of what you've accomplished and any findings.",
                )
            return None

        if recovery_action.action == "abort":
            # Emit ERROR event for abort
            if self._event_bus:
                try:
                    loop = asyncio.get_running_loop()

                    loop.create_task(
                        self._event_bus.emit(
                            topic="error.raised",
                            data={
                                "error_type": "RuntimeError",
                                "error_message": f"Session aborted: {recovery_action.reason}",
                                "category": "error",
                                "recoverable": False,
                                "context": {
                                    "failure_type": recovery_action.failure_type,
                                    "iteration": ctx.iteration,
                                },
                            },
                        )
                    )
                except RuntimeError:
                    # No event loop running
                    logger.debug("No event loop, skipping error event emission")
                except Exception as e:
                    logger.debug(f"Failed to emit abort error event: {e}")
            return StreamChunk(
                content=f"\n[recovery] Session aborted: {recovery_action.reason}\n",
                is_final=True,
            )

        return None

    # =====================================================================
    # Filtering and Truncation Methods
    # =====================================================================

    def filter_blocked_tool_calls(
        self,
        ctx: StreamingRecoveryContext,
        tool_calls: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[StreamChunk], int]:
        """Filter out blocked tool calls.

        Uses unified_tracker.is_blocked_after_warning as the block checker.

        Args:
            ctx: Recovery context
            tool_calls: List of tool calls to filter

        Returns:
            Tuple of (filtered_tool_calls, blocked_chunks, blocked_count)
        """
        return self.streaming_handler.filter_blocked_tool_calls(
            ctx.streaming_context,
            tool_calls,
            self.unified_tracker.is_blocked_after_warning,
        )

    def truncate_tool_calls(
        self,
        ctx: StreamingRecoveryContext,
        tool_calls: list[dict[str, Any]],
        max_calls: int,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Truncate tool calls to budget limit.

        Args:
            ctx: Recovery context
            tool_calls: List of tool calls to truncate
            max_calls: Maximum allowed calls

        Returns:
            Tuple of (truncated_tool_calls, was_truncated)
        """
        if len(tool_calls) <= max_calls:
            return tool_calls, False

        logger.warning(f"Truncating {len(tool_calls)} tool calls to budget limit of {max_calls}")
        return tool_calls[:max_calls], True

    # =====================================================================
    # Prompt and Message Generation
    # =====================================================================

    def get_recovery_prompts(
        self,
        ctx: StreamingRecoveryContext,
    ) -> list[str]:
        """Get recovery prompts for current context.

        Args:
            ctx: Recovery context

        Returns:
            List of recovery prompts
        """
        if not self.recovery_handler:
            return []

        # Get recovery prompts from handler
        # This would integrate with RecoveryHandler.get_recovery_prompts()
        # For now, return empty list
        return []

    def get_recovery_fallback_message(
        self,
        ctx: StreamingRecoveryContext,
    ) -> str:
        """Get fallback message when recovery fails.

        Args:
            ctx: Recovery context

        Returns:
            Fallback message
        """
        return (
            "I apologize, but I'm having difficulty completing this task. "
            "Here's a summary of what I've accomplished so far..."
        )

    def should_use_tools_for_recovery(
        self,
        ctx: StreamingRecoveryContext,
    ) -> bool:
        """Determine if tools should be used during recovery.

        Args:
            ctx: Recovery context

        Returns:
            True if tools should be used, False otherwise
        """
        # Don't use tools if budget exhausted
        if ctx.tool_calls_used >= ctx.tool_budget:
            return False

        # Don't use tools if looping
        if not self.check_progress(ctx):
            return False

        return True

    # =====================================================================
    # Metrics and Formatting
    # =====================================================================

    def format_completion_metrics(
        self,
        ctx: StreamingRecoveryContext,
    ) -> dict[str, Any]:
        """Format completion metrics for display.

        Args:
            ctx: Recovery context

        Returns:
            Dictionary of formatted metrics
        """
        return {
            "iterations": ctx.iteration,
            "tool_calls": ctx.tool_calls_used,
            "elapsed_time": ctx.elapsed_time,
            "quality_score": ctx.last_quality_score,
        }

    def format_budget_exhausted_metrics(
        self,
        ctx: StreamingRecoveryContext,
    ) -> dict[str, Any]:
        """Format budget exhausted metrics.

        Args:
            ctx: Recovery context

        Returns:
            Dictionary of formatted metrics
        """
        return {
            "tool_calls_used": ctx.tool_calls_used,
            "tool_budget": ctx.tool_budget,
            "iterations": ctx.iteration,
            "max_iterations": ctx.max_iterations,
        }

    def generate_tool_result_chunks(
        self,
        results: list[Any],  # List[ToolCallResult]
        ctx: StreamingRecoveryContext,
    ) -> list[StreamChunk]:
        """Generate stream chunks from tool results.

        Args:
            results: List of ToolCallResult objects
            ctx: Recovery context

        Returns:
            List of StreamChunk objects
        """
        # Delegate to streaming handler for chunk generation
        # This would call streaming_handler.generate_tool_result_chunks()
        # For now, return empty list
        return []
