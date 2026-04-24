# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Shared streaming helper implementations for the canonical chat service path."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Optional

from victor.agent.prompt_requirement_extractor import extract_prompt_requirements
from victor.agent.unified_task_tracker import TrackerTaskType
from victor.core.errors import (
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from victor.framework.task import TaskComplexity
from victor.providers.base import Message, StreamChunk

if TYPE_CHECKING:
    from victor.agent.streaming.context import StreamingChatContext

logger = logging.getLogger(__name__)


class ChatStreamHelperMixin:
    """Shared streaming helper methods reused by service and compatibility shims."""

    async def _handle_context_and_iteration_limits(
        self,
        user_message: str,
        max_total_iterations: int,
        max_context: int,
        total_iterations: int,
        last_quality_score: float,
    ) -> tuple[bool, Optional[StreamChunk]]:
        """Compatibility delegate for context and iteration limit handling."""
        orch = self._orchestrator

        runtime_helper = None
        helper_resolver = getattr(self, "_get_orchestrator_runtime_helper", None)
        if callable(helper_resolver):
            runtime_helper = helper_resolver("_handle_context_and_iteration_limits_runtime")
        else:
            runtime_helper = getattr(orch, "_handle_context_and_iteration_limits_runtime", None)

        if callable(runtime_helper):
            return await runtime_helper(
                user_message,
                max_total_iterations,
                max_context,
                total_iterations,
                last_quality_score,
            )

        return False, None

    async def _prepare_stream(self, user_message: str, **kwargs: Any) -> tuple[
        Any,
        float,
        float,
        Dict[str, int],
        int,
        int,
        int,
        bool,
        TrackerTaskType,
        Any,
        int,
    ]:
        """Prepare streaming state and return commonly used values."""
        orch = self._orchestrator

        orch._cancel_event = asyncio.Event()
        orch._is_streaming = True

        stream_metrics = orch._metrics_collector.init_stream_metrics()
        start_time = stream_metrics.start_time
        total_tokens: float = 0

        cumulative_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }

        orch.conversation.ensure_system_prompt()
        orch._system_added = True
        orch._session_state.reset_for_new_turn()
        orch.unified_tracker.reset()
        orch.reminder_manager.reset()

        if orch.has_capability("usage_analytics") and orch.get_capability_value("usage_analytics"):
            orch.get_capability_value("usage_analytics").start_session()

        if orch.has_capability("tool_sequence_tracker") and orch.get_capability_value(
            "tool_sequence_tracker"
        ):
            orch.get_capability_value("tool_sequence_tracker").clear_history()

        if orch._context_manager and hasattr(orch._context_manager, "start_background_compaction"):
            await orch._context_manager.start_background_compaction(interval_seconds=15.0)

        max_total_iterations = orch.unified_tracker.config.get("max_total_iterations", 50)

        fallback_iteration = kwargs.get("_fallback_iteration", 0)
        if fallback_iteration > 0:
            total_iterations = fallback_iteration
            logger.info(
                f"[Fallback] Preserving iteration count: continuing from iteration {total_iterations} "
                f"(preserved {total_iterations} iterations of progress)"
            )
        else:
            total_iterations = 0

        force_completion = False

        orch.add_message("user", user_message)

        if orch.has_capability("usage_analytics") and orch.get_capability_value("usage_analytics"):
            orch.get_capability_value("usage_analytics").record_turn()

        unified_task_type = orch.unified_tracker.detect_task_type(user_message)
        logger.info(f"Task type detected: {unified_task_type.value}")

        prompt_requirements = extract_prompt_requirements(user_message)
        if prompt_requirements.has_explicit_requirements():
            orch.unified_tracker._progress.has_prompt_requirements = True

            if (
                prompt_requirements.tool_budget
                and prompt_requirements.tool_budget > orch.unified_tracker._progress.tool_budget
            ):
                orch.unified_tracker.set_tool_budget(prompt_requirements.tool_budget)
                logger.info(
                    f"Dynamic budget from prompt: {prompt_requirements.tool_budget} "
                    f"(files={prompt_requirements.file_count}, fixes={prompt_requirements.fix_count})"
                )

            if (
                prompt_requirements.iteration_budget
                and prompt_requirements.iteration_budget
                > orch.unified_tracker._task_config.max_exploration_iterations
            ):
                orch.unified_tracker.set_max_iterations(prompt_requirements.iteration_budget)
                logger.info(
                    f"Dynamic iterations from prompt: {prompt_requirements.iteration_budget}"
                )

        intelligent_task = asyncio.create_task(
            orch._prepare_intelligent_request(
                task=user_message,
                task_type=unified_task_type.value,
            )
        )

        max_exploration_iterations = orch.unified_tracker.max_exploration_iterations

        task_classification, complexity_tool_budget = self._prepare_task(
            user_message, unified_task_type
        )

        intelligent_context = await intelligent_task
        if intelligent_context:
            if intelligent_context.get("system_prompt_addition"):
                orch.add_message(
                    "user",
                    f"[SYSTEM-REMINDER: {intelligent_context['system_prompt_addition']}]",
                )
                logger.debug("Injected intelligent pipeline optimized prompt")

        return (
            stream_metrics,
            start_time,
            total_tokens,
            cumulative_usage,
            max_total_iterations,
            max_exploration_iterations,
            total_iterations,
            force_completion,
            unified_task_type,
            task_classification,
            complexity_tool_budget,
        )

    async def _create_stream_context(
        self, user_message: str, **kwargs: Any
    ) -> "StreamingChatContext":
        """Create a StreamingChatContext with all prepared data."""
        from victor.agent.streaming import create_stream_context

        orch = self._orchestrator
        (
            stream_metrics,
            start_time,
            total_tokens,
            cumulative_usage,
            max_total_iterations,
            max_exploration_iterations,
            total_iterations,
            force_completion,
            unified_task_type,
            task_classification,
            complexity_tool_budget,
        ) = await self._prepare_stream(user_message, **kwargs)

        task_keywords = orch._classify_task_keywords(user_message)

        ctx = create_stream_context(
            user_message=user_message,
            max_iterations=max_total_iterations,
            max_exploration=max_exploration_iterations,
            tool_budget=complexity_tool_budget,
        )

        ctx.stream_metrics = stream_metrics
        ctx.start_time = start_time
        ctx.total_tokens = total_tokens
        ctx.cumulative_usage = cumulative_usage
        ctx.total_iterations = total_iterations
        ctx.force_completion = force_completion
        ctx.unified_task_type = unified_task_type
        ctx.task_classification = task_classification
        ctx.complexity_tool_budget = complexity_tool_budget

        task_type_val = task_keywords.get("task_type", "default")
        ctx.is_analysis_task = task_keywords.get(
            "is_analysis_task",
            task_type_val in ("analysis", "analyze"),
        ) or unified_task_type.value in ("analyze", "analysis")
        ctx.is_action_task = task_keywords.get(
            "is_action_task",
            task_type_val in ("action", "create_simple", "create_complex"),
        )
        ctx.needs_execution = task_keywords.get(
            "needs_execution",
            task_type_val in ("execution", "action"),
        )
        ctx.coarse_task_type = task_keywords.get("coarse_task_type", task_type_val)

        if task_classification and hasattr(task_classification, "complexity"):
            ctx.is_complex_task = task_classification.complexity in (
                TaskComplexity.COMPLEX,
                TaskComplexity.ANALYSIS,
            )

        from victor.agent.services.turn_execution_runtime import TurnExecutor

        ctx.is_qa_task = TurnExecutor._is_question_only(user_message)
        ctx.goals = orch._tool_planner.infer_goals_from_message(user_message)
        ctx.tool_budget = orch.tool_budget
        ctx.tool_calls_used = orch.tool_calls_used
        ctx.task_completion_detector = orch._task_completion_detector

        return ctx

    def _prepare_task(
        self, user_message: str, unified_task_type: TrackerTaskType
    ) -> tuple[Any, int]:
        """Prepare task-specific guidance and budget adjustments."""
        orch = self._orchestrator

        if orch.task_coordinator._reminder_manager is None:
            orch.task_coordinator.set_reminder_manager(orch.reminder_manager)

        return orch.task_coordinator.prepare_task(
            user_message, unified_task_type, orch.conversation_controller
        )

    async def _run_iteration_pre_checks(
        self,
        stream_ctx: "StreamingChatContext",
        user_message: str,
    ) -> AsyncIterator[StreamChunk]:
        """Run pre-iteration checks: cancellation, compaction, time limit."""
        orch = self._orchestrator

        if orch._check_cancellation():
            logger.info("Stream cancelled by user request")
            orch._is_streaming = False
            orch._record_intelligent_outcome(
                success=False,
                quality_score=stream_ctx.last_quality_score,
                user_satisfied=False,
                completed=False,
            )
            yield StreamChunk(
                content="\n\n[Cancelled by user]\n",
                is_final=True,
            )
            return

        if orch._context_compactor:
            compaction_action = orch._context_compactor.check_and_compact(
                current_query=user_message,
                force=False,
                tool_call_count=orch.tool_calls_used,
                task_complexity=TaskComplexity.COMPLEX.value,
            )
            if compaction_action.action_taken:
                logger.info(
                    f"Compacted context: {compaction_action.messages_removed} messages removed, "
                    f"{compaction_action.tokens_freed} tokens freed"
                )

        time_limit = getattr(orch.settings, "stream_idle_timeout_seconds", 300)
        if stream_ctx.is_over_time_limit(time_limit):
            logger.warning(f"Stream time limit exceeded: {stream_ctx.elapsed_time():.1f}s")
            yield StreamChunk(
                content=f"\n\n[Session exceeded {time_limit}s idle timeout - providing summary]\n",
                is_final=False,
            )
            stream_ctx.force_completion = True

        stream_ctx.increment_iteration()

        if stream_ctx.pending_grounding_feedback:
            logger.info("Injecting pending grounding feedback as system message")
            orch.add_message(
                "user", f"[GROUNDING-FEEDBACK: {stream_ctx.pending_grounding_feedback}]"
            )
            stream_ctx.pending_grounding_feedback = ""

    async def _stream_provider_response(
        self,
        tools: Any,
        provider_kwargs: Dict[str, Any],
        stream_ctx: "StreamingChatContext",
    ) -> tuple[str, Any, float, bool]:
        """Stream response from provider with rate limit retry."""
        return await self._stream_with_rate_limit_retry(tools, provider_kwargs, stream_ctx)

    def _get_rate_limit_wait_time(self, exc: Exception, attempt: int) -> float:
        """Get wait time for rate limit retry."""
        orch = self._orchestrator
        base_wait = orch._provider_service.get_rate_limit_wait_time(exc)
        backoff_multiplier = 2**attempt
        wait_time = base_wait * backoff_multiplier
        return min(wait_time, 300.0)

    async def _stream_with_rate_limit_retry(
        self,
        tools: Any,
        provider_kwargs: Dict[str, Any],
        stream_ctx: "StreamingChatContext",
        max_retries: int = 3,
    ) -> tuple[str, Any, float, bool]:
        """Stream provider response with automatic rate limit retry."""
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return await self._stream_provider_response_inner(
                    tools, provider_kwargs, stream_ctx
                )
            except ProviderRateLimitError as exc:
                last_exception = exc
                if attempt < max_retries:
                    wait_time = self._get_rate_limit_wait_time(exc, attempt)
                    endpoint_info = (
                        f"{self.provider_name}:{self.model}"
                        if hasattr(self, "provider_name")
                        else "API"
                    )
                    logger.warning(
                        f"[yellow]⚠ Rate limit hit for {endpoint_info}[/] "
                        f"(attempt {attempt + 1}/{max_retries + 1}). "
                        f"Waiting {wait_time:.0f}s before retry..."
                    )
                    logger.debug(f"Rate limit detail: {str(exc)[:300]}")

                    if attempt == 0:
                        logger.info(
                            "[dim]Tip: To avoid rate limits, consider:[/]\n"
                            "[dim]  • Using an API key instead of free tier[/]\n"
                            "[dim]  • Adding a small delay between requests[/]\n"
                            "[dim]  • Reducing request frequency[/]"
                        )

                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Rate limit persisted after {max_retries + 1} attempts")
            except Exception as exc:
                exc_str = str(exc).lower()
                if "rate_limit" in exc_str or "429" in exc_str or "rate limit" in exc_str:
                    last_exception = exc
                    if attempt < max_retries:
                        wait_time = self._get_rate_limit_wait_time(exc, attempt)
                        logger.warning(
                            f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). "
                            f"Waiting {wait_time:.1f}s before retry..."
                        )
                        logger.debug(f"Rate limit detail: {type(exc).__name__}: {str(exc)[:200]}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Rate limit persisted after {max_retries + 1} attempts")
                else:
                    raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Rate limit retry exhausted without exception")

    async def _stream_provider_response_inner(
        self,
        tools: Any,
        provider_kwargs: Dict[str, Any],
        stream_ctx: "StreamingChatContext",
    ) -> tuple[str, Any, float, bool]:
        """Inner implementation of stream_provider_response without retry logic."""
        orch = self._orchestrator

        full_content = ""
        tool_calls = None
        garbage_detected = False
        consecutive_garbage_chunks = 0
        max_garbage_chunks = 3
        total_tokens: float = 0

        assembled = orch.get_assembled_messages(
            current_query=stream_ctx.user_message if stream_ctx else None
        )
        async for chunk in orch.provider.stream(
            messages=assembled,
            model=orch.model,
            temperature=orch.temperature,
            max_tokens=orch.max_tokens,
            tools=tools,
            **provider_kwargs,
        ):
            chunk, consecutive_garbage_chunks, garbage_detected = self._handle_stream_chunk(
                chunk,
                consecutive_garbage_chunks,
                max_garbage_chunks,
                garbage_detected,
            )
            if chunk is None:
                continue

            full_content += chunk.content
            stream_ctx.stream_metrics.total_chunks += 1
            if chunk.content:
                orch._metrics_collector.record_first_token()
                total_tokens += len(chunk.content) / 4
                stream_ctx.stream_metrics.total_content_length += len(chunk.content)

            if chunk.tool_calls:
                logger.debug(f"Received tool_calls in chunk: {chunk.tool_calls}")
                tool_calls = chunk.tool_calls
                stream_ctx.stream_metrics.tool_calls_count += len(chunk.tool_calls)

            if chunk.usage:
                for key in stream_ctx.cumulative_usage:
                    stream_ctx.cumulative_usage[key] += chunk.usage.get(key, 0)
                logger.debug(
                    f"Chunk usage: in={chunk.usage.get('prompt_tokens', 0)} "
                    f"out={chunk.usage.get('completion_tokens', 0)} "
                    f"cache_read={chunk.usage.get('cache_read_input_tokens', 0)}"
                )

            if tool_calls:
                # Tool calls signal the end of this turn — the SSE stream is done.
                # Break immediately; do NOT start a new stream call with the same
                # messages (that would duplicate full_content).
                logger.debug("Tool calls received, breaking stream loop")
                break

        if garbage_detected and not tool_calls:
            logger.info("Setting force_completion due to garbage detection")

        if not tool_calls and not stream_ctx.force_completion:
            content_length = len(full_content.strip()) if full_content else 0

            if content_length == 0:
                logger.warning("Stream completed without content or tool calls")
                full_content = "[No content received from provider - stream may have failed]"
            elif content_length < 50:
                logger.warning(f"Stream completed with very short content ({content_length} chars)")
                logger.debug(f"Short content: {full_content}")

        logger.debug(
            "Stream completion summary - content: %s chars, tool_calls: %s",
            len(full_content) if full_content else 0,
            len(tool_calls) if tool_calls else 0,
        )

        stream_ctx.total_tokens = total_tokens
        return full_content, tool_calls, total_tokens, garbage_detected

    def _handle_stream_chunk(
        self,
        chunk: Any,
        consecutive_garbage_chunks: int,
        max_garbage_chunks: int,
        garbage_detected: bool,
    ) -> tuple[Any, int, bool]:
        """Handle garbage detection for a streaming chunk."""
        orch = self._orchestrator
        if chunk.content and orch.sanitizer.is_garbage_content(chunk.content):
            consecutive_garbage_chunks += 1
            if consecutive_garbage_chunks >= max_garbage_chunks:
                if not garbage_detected:
                    garbage_detected = True
                    logger.warning(
                        f"Garbage content detected after {len(chunk.content)} chars - stopping stream early"
                    )
                return None, consecutive_garbage_chunks, garbage_detected
        else:
            consecutive_garbage_chunks = 0
        return chunk, consecutive_garbage_chunks, garbage_detected

    async def _handle_empty_response_recovery(
        self,
        stream_ctx: "StreamingChatContext",
        tools: Any,
    ) -> tuple[bool, Any, Optional[StreamChunk]]:
        """Handle empty response recovery with multi-strategy retry."""
        orch = self._orchestrator

        recovery_temps = [0.7, 0.9]
        for temp in recovery_temps:
            try:
                orch.add_message(
                    "system",
                    "Please provide a response to the user's question. "
                    "If you need to use tools, go ahead. Otherwise, provide a text answer.",
                )

                provider_kwargs: Dict[str, Any] = {}
                if orch.thinking:
                    provider_kwargs["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": 10000,
                    }

                full_content = ""
                recovered_tool_calls = None

                retry_assembled = orch.get_assembled_messages(
                    current_query=stream_ctx.user_message if stream_ctx else None
                )
                async for chunk in orch.provider.stream(
                    messages=retry_assembled,
                    model=orch.model,
                    temperature=temp,
                    max_tokens=orch.max_tokens,
                    tools=tools,
                    **provider_kwargs,
                ):
                    if chunk.content:
                        full_content += chunk.content
                    if chunk.tool_calls:
                        recovered_tool_calls = chunk.tool_calls
                        break

                if recovered_tool_calls:
                    logger.info(f"Recovery at temperature {temp} produced tool calls")
                    return True, recovered_tool_calls, None

                if full_content.strip():
                    logger.info(
                        f"Recovery at temperature {temp} produced content "
                        f"({len(full_content)} chars)"
                    )
                    sanitized = orch.sanitizer.sanitize(full_content)
                    if sanitized:
                        orch.add_message("assistant", sanitized)
                    final_chunk = orch._chunk_generator.generate_content_chunk(
                        sanitized or full_content, is_final=True
                    )
                    return True, None, final_chunk

            except Exception as exc:
                logger.warning(f"Recovery attempt at temperature {temp} failed: {exc}")
                continue

        return False, None, None
