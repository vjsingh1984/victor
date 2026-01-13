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

"""Evaluation coordinator for agent orchestration.

This module provides the EvaluationCoordinator which handles evaluation
and analytics operations for the orchestrator including:

- RL feedback signal collection
- Usage analytics tracking
- Intelligent outcome recording
- Analytics flushing and persistence

Extracted from AgentOrchestrator as part of SOLID refactoring
to improve modularity and testability.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.usage_analytics import UsageAnalytics
    from victor.agent.tool_sequence_tracker import ToolSequenceTracker

logger = logging.getLogger(__name__)


class EvaluationCoordinator:
    """Coordinates evaluation and analytics operations for the orchestrator.

    This coordinator handles:
    - RL feedback signal collection for model selection
    - Usage analytics tracking for data-driven optimization
    - Intelligent outcome recording for Q-learning
    - Analytics flushing for graceful shutdown

    The coordinator uses callback functions for accessing orchestrator
    state to maintain loose coupling.

    Example:
        coordinator = EvaluationCoordinator(
            usage_analytics=analytics,
            sequence_tracker=tracker,
            get_rl_coordinator_fn=lambda: orchestrator._rl_coordinator,
            get_vertical_context_fn=lambda: orchestrator._vertical_context,
            get_stream_context_fn=lambda: getattr(orchestrator, '_current_stream_context', None),
        )

        # Record outcome
        coordinator.record_intelligent_outcome(
            success=True,
            quality_score=0.9,
        )

        # Flush analytics
        results = await coordinator.flush_analytics()
    """

    def __init__(
        self,
        usage_analytics: Optional["UsageAnalytics"],
        sequence_tracker: Optional["ToolSequenceTracker"],
        get_rl_coordinator_fn: Callable[[], Optional[Any]],
        get_vertical_context_fn: Callable[[], Optional[Any]],
        get_stream_context_fn: Callable[[], Optional[Any]],
        get_provider_fn: Callable[[], Optional[Any]],
        get_model_fn: Callable[[], str],
        get_tool_calls_used_fn: Callable[[], int],
        get_intelligent_integration_fn: Callable[[], Optional[Any]],
    ) -> None:
        """Initialize the evaluation coordinator.

        Args:
            usage_analytics: The usage analytics instance
            sequence_tracker: The tool sequence tracker instance
            get_rl_coordinator_fn: Function to get RL coordinator
            get_vertical_context_fn: Function to get vertical context
            get_stream_context_fn: Function to get current stream context
            get_provider_fn: Function to get current provider
            get_model_fn: Function to get current model name
            get_tool_calls_used_fn: Function to get tool calls used count
            get_intelligent_integration_fn: Function to get intelligent integration
        """
        self._usage_analytics = usage_analytics
        self._sequence_tracker = sequence_tracker
        self._get_rl_coordinator_fn = get_rl_coordinator_fn
        self._get_vertical_context_fn = get_vertical_context_fn
        self._get_stream_context_fn = get_stream_context_fn
        self._get_provider_fn = get_provider_fn
        self._get_model_fn = get_model_fn
        self._get_tool_calls_used_fn = get_tool_calls_used_fn
        self._get_intelligent_integration_fn = get_intelligent_integration_fn

    @property
    def usage_analytics(self) -> Optional["UsageAnalytics"]:
        """Get the usage analytics instance.

        Returns:
            UsageAnalytics instance or None if disabled
        """
        return self._usage_analytics

    def record_intelligent_outcome(
        self,
        success: bool,
        quality_score: float = 0.5,
        user_satisfied: bool = True,
        completed: bool = True,
    ) -> None:
        """Record outcome for Q-learning feedback.

        Delegates to OrchestratorIntegration.record_intelligent_outcome().

        Args:
            success: Whether the task was completed successfully
            quality_score: Final quality score (0.0-1.0)
            user_satisfied: Whether user seemed satisfied
            completed: Whether task reached completion
        """
        integration = self._get_intelligent_integration_fn()
        if not integration:
            return

        # Get orchestrator state to pass to integration
        stream_context = self._get_stream_context_fn()
        vertical_context = self._get_vertical_context_fn()
        provider = self._get_provider_fn()
        model = self._get_model_fn()
        tool_calls_used = self._get_tool_calls_used_fn()
        rl_coordinator = self._get_rl_coordinator_fn()

        # Get additional state attributes (if available)
        continuation_prompts = getattr(stream_context, "_continuation_prompts", 0) if stream_context else 0
        max_continuation_prompts_used = getattr(stream_context, "_max_continuation_prompts_used", 6) if stream_context else 6
        stuck_loop_detected = getattr(stream_context, "_stuck_loop_detected", False) if stream_context else False

        try:
            integration.record_intelligent_outcome(
                success=success,
                quality_score=quality_score,
                user_satisfied=user_satisfied,
                completed=completed,
                rl_coordinator=rl_coordinator,
                stream_context=stream_context,
                vertical_context=vertical_context,
                provider_name=provider.name if provider else "unknown",
                model=model,
                tool_calls_used=tool_calls_used,
                continuation_prompts=continuation_prompts,
                max_continuation_prompts_used=max_continuation_prompts_used,
                stuck_loop_detected=stuck_loop_detected,
            )
        except Exception as e:
            logger.debug(f"IntelligentPipeline record_outcome failed: {e}")

    def send_rl_reward_signal(self, session: Any) -> None:
        """Send reward signal to RL model selector for Q-value updates.

        Converts StreamingSession data into RLOutcome and updates Q-values
        based on session outcome (success, latency, throughput, tool usage).

        Args:
            session: StreamingSession instance
        """
        try:
            from victor.framework.rl.base import RLOutcome

            rl_coordinator = self._get_rl_coordinator_fn()
            if not rl_coordinator:
                return

            # Extract metrics from session
            token_count = 0
            if hasattr(session, "metrics") and session.metrics:
                # Estimate tokens from chunks (streaming metrics)
                token_count = getattr(session.metrics, "total_chunks", 0) or 0

            # Get tool execution count from metrics collector
            tool_calls_made = 0
            # Note: MetricsCollector access handled via orchestrator state if needed

            # Determine success: no error and not cancelled
            success = not hasattr(session, "error") or session.error is None
            if hasattr(session, "cancelled"):
                success = success and not session.cancelled

            # Compute quality score (0-1) based on success and metrics
            quality_score = 0.5
            if success:
                quality_score = 0.8
                # Bonus for fast responses
                if hasattr(session, "duration") and session.duration < 10:
                    quality_score += 0.1
                # Bonus for tool usage
                if tool_calls_made > 0:
                    quality_score += 0.1
            quality_score = min(1.0, quality_score)

            # Create outcome
            provider = getattr(session, "provider", "unknown")
            model = getattr(session, "model", "unknown")
            vertical_context = self._get_vertical_context_fn()
            vertical_name = getattr(vertical_context, "vertical_name", None) if vertical_context else None

            outcome = RLOutcome(
                provider=provider,
                model=model,
                task_type=getattr(session, "task_type", "unknown"),
                success=success,
                quality_score=quality_score,
                metadata={
                    "latency_seconds": getattr(session, "duration", 0),
                    "token_count": token_count,
                    "tool_calls_made": tool_calls_made,
                    "session_id": getattr(session, "session_id", "unknown"),
                },
                vertical=vertical_name or "default",
            )

            # Record outcome for model selector
            rl_coordinator.record_outcome("model_selector", outcome, vertical_name or "default")

            logger.debug(
                f"RL feedback: provider={provider} success={success} "
                f"quality={quality_score:.2f} duration={getattr(session, 'duration', 0):.1f}s"
            )

        except ImportError:
            # RL module not available - skip silently
            pass
        except (KeyError, AttributeError) as e:
            # RL coordinator not properly initialized
            logger.debug(f"RL reward signal skipped (not configured): {e}")
        except (ValueError, TypeError) as e:
            # Invalid reward data
            logger.warning(f"Failed to send RL reward signal (invalid data): {e}")

    def flush_analytics(self) -> Dict[str, bool]:
        """Flush all analytics and cached data to persistent storage.

        Call this method before shutdown or when you need to ensure
        all analytics data is persisted to disk. Useful for graceful
        shutdown scenarios.

        Returns:
            Dictionary indicating success/failure for each component:
            - usage_analytics: Whether analytics were flushed
            - sequence_tracker: Whether patterns were saved
            - tool_cache: Whether cache was flushed
        """
        results: Dict[str, bool] = {}

        # Flush usage analytics
        if self._usage_analytics:
            try:
                self._usage_analytics.flush()
                results["usage_analytics"] = True
                logger.debug("UsageAnalytics flushed to disk")
            except Exception as e:
                logger.warning(f"Failed to flush usage analytics: {e}")
                results["usage_analytics"] = False
        else:
            results["usage_analytics"] = False

        # Flush sequence tracker patterns
        if self._sequence_tracker:
            try:
                # SequenceTracker learns patterns in memory; no explicit flush needed
                # but we capture statistics for reporting
                stats = self._sequence_tracker.get_statistics()
                results["sequence_tracker"] = True
                logger.debug(
                    f"SequenceTracker has {stats.get('unique_transitions', 0)} learned patterns"
                )
            except Exception as e:
                logger.warning(f"Failed to get sequence tracker stats: {e}")
                results["sequence_tracker"] = False
        else:
            results["sequence_tracker"] = False

        # Note: Tool cache flush would be handled by the caller if needed
        # as it's not directly managed by this coordinator
        results["tool_cache"] = False

        logger.info(f"Analytics flush complete: {results}")
        return results
