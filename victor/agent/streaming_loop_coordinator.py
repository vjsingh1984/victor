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

"""StreamingLoopCoordinator - Extracted from AgentOrchestrator.

This module coordinates the main streaming iteration loop, extracted from
the AgentOrchestrator to reduce class size while maintaining the facade pattern.

The coordinator handles:
- Iteration lifecycle (start, termination conditions)
- Tool call parsing and validation
- Recovery integration
- Continuation strategy decisions
- Tool execution coordination

Example:
    coordinator = StreamingLoopCoordinator(
        tool_pipeline=tool_pipeline,
        streaming_controller=streaming_controller,
        recovery_handler=recovery_handler,
        context_compactor=context_compactor,
    )

    async for chunk in coordinator.run_loop(stream_ctx, provider, messages, tools):
        yield chunk
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.agent.context_compactor import ContextCompactor
    from victor.agent.conversation_controller import ConversationController
    from victor.agent.orchestrator import StreamingChatContext
    from victor.agent.recovery_handler import RecoveryHandler
    from victor.agent.streaming_controller import StreamingController
    from victor.agent.tool_pipeline import ToolPipeline
    from victor.completion import CompletionResponse, StreamChunk
    from victor.providers.base import BaseProvider

logger = logging.getLogger(__name__)


@runtime_checkable
class LoopTerminationHandler(Protocol):
    """Protocol for handling loop termination conditions."""

    def check_cancellation(self, quality_score: float) -> Optional[StreamChunk]:
        """Check if loop should be cancelled."""
        ...

    def check_time_limit(self, stream_ctx: Any) -> Optional[StreamChunk]:
        """Check if time limit exceeded."""
        ...

    def check_iteration_limits(
        self,
        user_message: str,
        max_iterations: int,
        max_context: int,
        current_iteration: int,
        quality_score: float,
    ) -> Tuple[bool, Optional[StreamChunk]]:
        """Check context and iteration limits."""
        ...


@runtime_checkable
class ToolCallHandler(Protocol):
    """Protocol for handling tool calls."""

    def parse_and_validate(
        self, tool_calls: List[Dict[str, Any]], content: str
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Parse and validate tool calls."""
        ...

    def truncate_to_budget(
        self, tool_calls: List[Dict[str, Any]], stream_ctx: Any
    ) -> List[Dict[str, Any]]:
        """Truncate tool calls to remaining budget."""
        ...

    def filter_blocked(
        self, stream_ctx: Any, tool_calls: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[StreamChunk], int]:
        """Filter blocked tool calls after loop warning."""
        ...


@runtime_checkable
class RecoveryIntegrationHandler(Protocol):
    """Protocol for recovery integration."""

    async def handle_recovery(
        self,
        stream_ctx: Any,
        content: str,
        tool_calls: List[Dict[str, Any]],
        mentioned_tools: Optional[List[str]],
    ) -> Any:
        """Handle recovery for failed responses."""
        ...

    def apply_recovery_action(self, action: Any, stream_ctx: Any) -> Optional[StreamChunk]:
        """Apply recovery action and return chunk if needed."""
        ...


@dataclass
class LoopIterationResult:
    """Result of a single loop iteration."""

    should_continue: bool = True
    should_return: bool = False
    chunks: List[StreamChunk] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    content: str = ""
    quality_score: float = 0.0
    mentioned_tools: List[str] = field(default_factory=list)


@dataclass
class ContinuationDecision:
    """Decision from continuation strategy."""

    action: str  # 'continue', 'finish', 'prompt_tool_call', etc.
    reason: str
    message: Optional[str] = None
    updates: Dict[str, Any] = field(default_factory=dict)
    mentioned_tools: List[str] = field(default_factory=list)


class StreamingLoopCoordinator:
    """Coordinates the main streaming iteration loop.

    Extracted from AgentOrchestrator to reduce class size while
    maintaining the facade pattern. The orchestrator delegates to
    this coordinator for the main while-loop body.

    This class coordinates existing handler methods rather than
    duplicating their logic. It serves as a higher-level orchestrator
    for the streaming iteration lifecycle.

    Attributes:
        termination_handler: Handler for loop termination conditions
        tool_call_handler: Handler for tool call processing
        recovery_handler: Handler for recovery integration
        chunk_generator: Generator for stream chunks
    """

    def __init__(
        self,
        termination_handler: LoopTerminationHandler,
        tool_call_handler: ToolCallHandler,
        recovery_handler: RecoveryIntegrationHandler,
        chunk_generator: Any,  # ChunkGenerator from orchestrator
        intent_classifier: Any,  # IntentClassifier
        continuation_strategy: Any,  # ContinuationStrategy
        settings: Any,  # Settings
    ):
        """Initialize the streaming loop coordinator.

        Args:
            termination_handler: Handler for termination conditions
            tool_call_handler: Handler for tool call processing
            recovery_handler: Handler for recovery integration
            chunk_generator: Generator for stream chunks
            intent_classifier: Classifier for response intent
            continuation_strategy: Strategy for continuation decisions
            settings: Application settings
        """
        self._termination_handler = termination_handler
        self._tool_call_handler = tool_call_handler
        self._recovery_handler = recovery_handler
        self._chunk_generator = chunk_generator
        self._intent_classifier = intent_classifier
        self._continuation_strategy = continuation_strategy
        self._settings = settings

        # State tracking
        self._continuation_prompts = 0
        self._asking_input_prompts = 0
        self._consecutive_blocked_attempts = 0
        self._summary_request_count = 0
        self._final_summary_requested = False
        self._max_prompts_summary_requested = False

    def reset_state(self) -> None:
        """Reset coordinator state for new conversation."""
        self._continuation_prompts = 0
        self._asking_input_prompts = 0
        self._consecutive_blocked_attempts = 0
        self._summary_request_count = 0
        self._final_summary_requested = False
        self._max_prompts_summary_requested = False

    async def check_pre_iteration_conditions(
        self,
        stream_ctx: Any,
        user_message: str,
        max_iterations: int,
        max_context: int,
    ) -> LoopIterationResult:
        """Check conditions before starting an iteration.

        This handles:
        1. Cancellation check
        2. Compaction handling
        3. Time limit check
        4. Context and iteration limits

        Args:
            stream_ctx: Streaming context
            user_message: Original user message
            max_iterations: Maximum iterations allowed
            max_context: Maximum context size

        Returns:
            LoopIterationResult with chunks and continuation decision
        """
        result = LoopIterationResult()

        # 1. Cancellation check
        cancellation_chunk = self._termination_handler.check_cancellation(
            stream_ctx.last_quality_score
        )
        if cancellation_chunk:
            result.chunks.append(cancellation_chunk)
            result.should_return = True
            result.should_continue = False
            return result

        # 2. Compaction handling (handled externally, just check for chunk)
        # The actual compaction is done by context_compactor

        # 3. Time limit check
        time_limit_chunk = self._termination_handler.check_time_limit(stream_ctx)
        if time_limit_chunk:
            result.chunks.append(time_limit_chunk)
            # Handler sets stream_ctx.force_completion = True

        # 4. Context and iteration limits
        handled, iter_chunk = await self._termination_handler.check_iteration_limits(
            user_message,
            max_iterations,
            max_context,
            stream_ctx.total_iterations,
            stream_ctx.last_quality_score,
        )
        if iter_chunk:
            result.chunks.append(iter_chunk)
        if handled:
            result.should_continue = False

        return result

    async def handle_provider_response(
        self,
        stream_ctx: Any,
        content: str,
        tool_calls: List[Dict[str, Any]],
        garbage_detected: bool,
    ) -> LoopIterationResult:
        """Handle the provider response.

        This processes:
        1. Garbage detection
        2. Tool call parsing and validation
        3. Recovery integration

        Args:
            stream_ctx: Streaming context
            content: Response content
            tool_calls: Raw tool calls from provider
            garbage_detected: Whether garbage was detected

        Returns:
            LoopIterationResult with processed tool calls
        """
        result = LoopIterationResult()
        result.content = content

        # 1. Handle garbage detection
        if garbage_detected and not tool_calls:
            stream_ctx.force_completion = True
            logger.info("Setting force_completion due to garbage detection")

        # 2. Parse and validate tool calls
        parsed_calls, updated_content = self._tool_call_handler.parse_and_validate(
            tool_calls, content
        )
        result.tool_calls = parsed_calls
        result.content = updated_content

        # 3. Detect mentioned tools for recovery
        if result.content and not parsed_calls:
            from victor.agent.continuation_strategy import ContinuationStrategy
            from victor.tools import TOOL_ALIASES, _ALL_TOOL_NAMES

            result.mentioned_tools = ContinuationStrategy.detect_mentioned_tools(
                result.content, list(_ALL_TOOL_NAMES), TOOL_ALIASES
            )

        # 4. Recovery integration
        recovery_action = await self._recovery_handler.handle_recovery(
            stream_ctx=stream_ctx,
            content=result.content,
            tool_calls=result.tool_calls,
            mentioned_tools=result.mentioned_tools or None,
        )

        if recovery_action.action != "continue":
            recovery_chunk = self._recovery_handler.apply_recovery_action(
                recovery_action, stream_ctx
            )
            if recovery_chunk:
                result.chunks.append(recovery_chunk)
                if recovery_chunk.is_final:
                    result.should_return = True
                    result.should_continue = False
                    return result
            if recovery_action.action in ("retry", "force_summary"):
                # Continue loop with updated state
                result.should_continue = True

        return result

    def determine_continuation(
        self,
        stream_ctx: Any,
        content: str,
        content_length: int,
        mentioned_tools: List[str],
        provider_name: str,
        model: str,
        tool_budget: int,
        unified_tracker_config: Any,
        rl_coordinator: Any,
    ) -> ContinuationDecision:
        """Determine what action to take when no tool calls were made.

        Uses the ContinuationStrategy to decide whether to:
        - Continue with tool prompting
        - Request summary
        - Return to user
        - Force tool execution
        - Finish

        Args:
            stream_ctx: Streaming context
            content: Response content
            content_length: Length of content
            mentioned_tools: Tools mentioned in response
            provider_name: Name of current provider
            model: Current model
            tool_budget: Tool call budget
            unified_tracker_config: UnifiedTaskTracker config
            rl_coordinator: RL coordinator for learning

        Returns:
            ContinuationDecision with action and updates
        """
        # Use the last 500 chars for intent classification
        intent_text = content
        if len(intent_text) > 500:
            intent_text = intent_text[-500:]

        intent_result = self._intent_classifier.classify_intent_sync(intent_text)

        logger.debug(
            f"Intent classification: {intent_result.intent.name} "
            f"(confidence={intent_result.confidence:.3f})"
        )

        # Use ContinuationStrategy to determine action
        one_shot_mode = getattr(self._settings, "one_shot_mode", False)
        action_result = self._continuation_strategy.determine_continuation_action(
            intent_result=intent_result,
            is_analysis_task=stream_ctx.is_analysis_task,
            is_action_task=stream_ctx.is_action_task,
            content_length=content_length,
            full_content=content,
            continuation_prompts=self._continuation_prompts,
            asking_input_prompts=self._asking_input_prompts,
            one_shot_mode=one_shot_mode,
            mentioned_tools=mentioned_tools,
            max_prompts_summary_requested=self._max_prompts_summary_requested,
            settings=self._settings,
            rl_coordinator=rl_coordinator,
            provider_name=provider_name,
            model=model,
            tool_budget=tool_budget,
            unified_tracker_config=unified_tracker_config,
        )

        # Apply state updates
        if "continuation_prompts" in action_result.get("updates", {}):
            self._continuation_prompts = action_result["updates"]["continuation_prompts"]
        if "asking_input_prompts" in action_result.get("updates", {}):
            self._asking_input_prompts = action_result["updates"]["asking_input_prompts"]
        if action_result.get("set_final_summary_requested"):
            self._final_summary_requested = True
        if action_result.get("set_max_prompts_summary_requested"):
            self._max_prompts_summary_requested = True

        return ContinuationDecision(
            action=action_result["action"],
            reason=action_result.get("reason", ""),
            message=action_result.get("message"),
            updates=action_result.get("updates", {}),
            mentioned_tools=action_result.get("mentioned_tools", []),
        )

    async def process_tool_calls(
        self,
        stream_ctx: Any,
        tool_calls: List[Dict[str, Any]],
        tool_executor: Callable,
    ) -> Tuple[List[Dict[str, Any]], List[StreamChunk]]:
        """Process tool calls through validation and execution.

        This handles:
        1. Budget warning check
        2. Budget exhaustion check
        3. Progress and force completion checks
        4. Tool call truncation and filtering
        5. Tool execution

        Args:
            stream_ctx: Streaming context
            tool_calls: Tool calls to process
            tool_executor: Callable to execute tool calls

        Returns:
            Tuple of (tool_results, chunks)
        """
        chunks = []

        # Sync tool tracking
        stream_ctx.tool_calls_used = stream_ctx.tool_calls_used or 0

        remaining = stream_ctx.get_remaining_budget()

        if remaining <= 0:
            # Budget exhausted - handled by orchestrator
            return [], chunks

        # Truncate to budget
        tool_calls = self._tool_call_handler.truncate_to_budget(tool_calls or [], stream_ctx)

        # Filter blocked calls
        filtered_calls, blocked_chunks, blocked_count = self._tool_call_handler.filter_blocked(
            stream_ctx, tool_calls
        )
        chunks.extend(blocked_chunks)

        # Execute tools
        for tool_call in filtered_calls:
            tool_name = tool_call.get("name", "tool")
            tool_args = tool_call.get("arguments", {})
            # Generate tool start chunk
            status_msg = f"Executing {tool_name}..."
            chunks.append(
                self._chunk_generator.generate_tool_start_chunk(tool_name, tool_args, status_msg)
            )

        # Execute all tool calls
        if filtered_calls:
            tool_results = await tool_executor(filtered_calls)

            # Generate result chunks
            for result in tool_results:
                for chunk in self._chunk_generator.generate_tool_result_chunks(result):
                    chunks.append(chunk)

            return tool_results, chunks

        return [], chunks

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get current coordinator state for debugging.

        Returns:
            Dict with current state values
        """
        return {
            "continuation_prompts": self._continuation_prompts,
            "asking_input_prompts": self._asking_input_prompts,
            "consecutive_blocked_attempts": self._consecutive_blocked_attempts,
            "summary_request_count": self._summary_request_count,
            "final_summary_requested": self._final_summary_requested,
            "max_prompts_summary_requested": self._max_prompts_summary_requested,
        }


__all__ = [
    "StreamingLoopCoordinator",
    "LoopTerminationHandler",
    "ToolCallHandler",
    "RecoveryIntegrationHandler",
    "LoopIterationResult",
    "ContinuationDecision",
]
