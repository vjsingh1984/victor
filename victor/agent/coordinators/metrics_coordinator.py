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

"""Metrics coordinator for agent orchestration.

This module provides the MetricsCoordinator which acts as a facade for
metrics collection, cost tracking, and usage analytics in the orchestrator.

The coordinator wraps:
- MetricsCollector: Stream metrics, tool selection, execution stats
- SessionCostTracker: Per-request and session-cumulative cost tracking
- Token usage tracking for evaluation/benchmarking

Extracted from AgentOrchestrator to improve modularity and testability
as part of the SOLID refactoring initiative.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.metrics_collector import MetricsCollector
    from victor.agent.session_cost_tracker import SessionCostTracker
    from victor.agent.stream_handler import StreamMetrics
    from victor.evaluation.protocol import TokenUsage

logger = logging.getLogger(__name__)


class MetricsCoordinator:
    """Coordinates all metrics collection and tracking for the orchestrator.

    This class serves as a facade for:
    - Stream metrics (time-to-first-token, duration, tokens/second)
    - Tool selection and execution statistics
    - Cost tracking (per-request and session-cumulative)
    - Token usage for evaluation tracking

    The coordinator delegates to specialized components:
    - MetricsCollector: Real-time metrics collection
    - SessionCostTracker: Cost tracking and export

    Example:
        coordinator = MetricsCoordinator(
            metrics_collector=collector,
            session_cost_tracker=cost_tracker,
            cumulative_token_usage={"prompt_tokens": 0, ...}
        )

        # After streaming
        metrics = coordinator.finalize_stream_metrics(usage_data)

        # Get cost summary
        summary = coordinator.get_session_cost_summary()
    """

    def __init__(
        self,
        metrics_collector: "MetricsCollector",
        session_cost_tracker: "SessionCostTracker",
        cumulative_token_usage: Dict[str, int],
    ) -> None:
        """Initialize the metrics coordinator.

        Args:
            metrics_collector: The metrics collector instance
            session_cost_tracker: The session cost tracker instance
            cumulative_token_usage: Mutable dict for cumulative token tracking
        """
        self._metrics_collector = metrics_collector
        self._session_cost_tracker = session_cost_tracker
        self._cumulative_token_usage = cumulative_token_usage

    # ========================================================================
    # Stream Metrics
    # ========================================================================

    def finalize_stream_metrics(
        self, usage_data: Optional[Dict[str, int]] = None
    ) -> Optional["StreamMetrics"]:
        """Finalize stream metrics at end of streaming session.

        This method:
        1. Finalizes stream metrics via MetricsCollector
        2. Records to SessionCostTracker for cumulative tracking
        3. Returns the complete metrics

        Args:
            usage_data: Optional cumulative token usage from provider API.
                       When provided, enables accurate token counts.

        Returns:
            StreamMetrics with complete session data or None
        """
        metrics = self._metrics_collector.finalize_stream_metrics(usage_data)

        # Record to session cost tracker for cumulative tracking
        if metrics:
            self._session_cost_tracker.record_request(
                prompt_tokens=metrics.prompt_tokens,
                completion_tokens=metrics.completion_tokens,
                cache_read_tokens=metrics.cache_read_tokens,
                cache_write_tokens=metrics.cache_write_tokens,
                duration_seconds=metrics.total_duration,
                tool_calls=metrics.tool_calls_count,
            )

        return metrics

    def get_last_stream_metrics(self) -> Optional["StreamMetrics"]:
        """Get metrics from the last streaming session.

        Returns:
            StreamMetrics from the last session or None if no metrics available
        """
        return self._metrics_collector.get_last_stream_metrics()

    def get_streaming_metrics_summary(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive streaming metrics summary.

        Returns:
            Dictionary with aggregated metrics or None if metrics disabled.
        """
        return self._metrics_collector.get_streaming_metrics_summary()

    def get_streaming_metrics_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent streaming metrics history.

        Args:
            limit: Maximum number of recent metrics to return

        Returns:
            List of recent metrics dictionaries
        """
        return self._metrics_collector.get_streaming_metrics_history(limit)

    # ========================================================================
    # Cost Tracking
    # ========================================================================

    def get_session_cost_summary(self) -> Dict[str, Any]:
        """Get session cost summary.

        Returns:
            Dictionary with session cost statistics including:
            - Total tokens and costs by type
            - Per-request breakdown
            - Session totals
        """
        return self._session_cost_tracker.get_summary()

    def get_session_cost_formatted(self) -> str:
        """Get formatted session cost string.

        Returns:
            Cost string like "$0.0123" or "cost n/a"
        """
        return self._session_cost_tracker.format_inline_cost()

    def export_session_costs(self, path: str, format: str = "json") -> None:
        """Export session costs to file.

        Args:
            path: Output file path
            format: Export format ("json" or "csv")
        """
        output_path = Path(path)
        if format == "csv":
            self._session_cost_tracker.export_csv(output_path)
        else:
            self._session_cost_tracker.export_json(output_path)

    # ========================================================================
    # Token Usage (for Evaluation/Benchmarking)
    # ========================================================================

    def get_token_usage(self) -> "TokenUsage":
        """Get cumulative token usage for evaluation tracking.

        Returns cumulative tokens used across all stream_chat calls.
        Used by VictorAgentAdapter for benchmark token tracking.

        Returns:
            TokenUsage dataclass with input/output/total token counts
        """
        from victor.evaluation.protocol import TokenUsage

        return TokenUsage(
            input_tokens=self._cumulative_token_usage.get("prompt_tokens", 0),
            output_tokens=self._cumulative_token_usage.get("completion_tokens", 0),
            total_tokens=self._cumulative_token_usage.get("total_tokens", 0),
        )

    def reset_token_usage(self) -> None:
        """Reset cumulative token usage tracking.

        Call this at the start of a new evaluation task to get fresh counts.
        """
        for key in self._cumulative_token_usage:
            self._cumulative_token_usage[key] = 0

    def update_cumulative_token_usage(self, usage_data: Dict[str, int]) -> None:
        """Update cumulative token usage from provider response.

        Args:
            usage_data: Token usage dict with prompt_tokens, completion_tokens, etc.
        """
        for key, value in usage_data.items():
            if key in self._cumulative_token_usage:
                self._cumulative_token_usage[key] += value

    # ========================================================================
    # Tool Usage Statistics
    # ========================================================================

    def get_tool_usage_stats(
        self, conversation_state_summary: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get comprehensive tool usage statistics.

        Args:
            conversation_state_summary: Optional conversation state to include

        Returns:
            Dictionary with usage analytics including:
            - Selection stats (semantic/keyword/fallback counts)
            - Per-tool execution stats (calls, success rate, timing)
            - Cost tracking (by tier and total)
            - Overall metrics
        """
        return self._metrics_collector.get_tool_usage_stats(
            conversation_state_summary=conversation_state_summary
        )

    # ========================================================================
    # Metrics Collector Delegation
    # ========================================================================

    @property
    def metrics_collector(self) -> "MetricsCollector":
        """Get the underlying metrics collector.

        Returns:
            The MetricsCollector instance
        """
        return self._metrics_collector

    @property
    def session_cost_tracker(self) -> "SessionCostTracker":
        """Get the underlying session cost tracker.

        Returns:
            The SessionCostTracker instance
        """
        return self._session_cost_tracker

    # Convenience methods for common metrics operations

    def record_tool_selection(self, method: str, num_tools: int) -> None:
        """Record tool selection statistics.

        Args:
            method: Selection method used ('semantic', 'keyword', 'fallback')
            num_tools: Number of tools selected
        """
        self._metrics_collector.record_tool_selection(method, num_tools)

    def record_tool_execution(self, tool_name: str, success: bool, elapsed_ms: float) -> None:
        """Record tool execution statistics.

        Args:
            tool_name: Name of the tool executed
            success: Whether execution succeeded
            elapsed_ms: Execution time in milliseconds
        """
        self._metrics_collector.record_tool_execution(tool_name, success, elapsed_ms)

    def on_tool_start(self, tool_name: str, arguments: Dict[str, Any], iteration: int = 0) -> None:
        """Callback when tool execution starts (from ToolPipeline).

        Args:
            tool_name: Name of the tool being executed
            arguments: Tool arguments
            iteration: Current iteration count
        """
        self._metrics_collector.on_tool_start(tool_name, arguments, iteration)

    def on_tool_complete(self, result: Any) -> None:
        """Callback when tool execution completes (from ToolPipeline).

        Args:
            result: The tool call result
        """
        self._metrics_collector.on_tool_complete(result)

    def on_streaming_session_complete(self, session: Any) -> None:
        """Callback when streaming session completes (from StreamingController).

        Args:
            session: The completed streaming session
        """
        self._metrics_collector.on_streaming_session_complete(session)

    def record_first_token(self) -> None:
        """Record the time of first token received."""
        self._metrics_collector.record_first_token()

    def init_stream_metrics(self) -> "StreamMetrics":
        """Initialize fresh stream metrics for a new streaming session.

        Returns:
            New StreamMetrics instance
        """
        return self._metrics_collector.init_stream_metrics()

    def update_model_info(self, model: str, provider: str) -> None:
        """Update model and provider info (e.g., after hot-swap).

        Args:
            model: New model name
            provider: New provider name
        """
        self._metrics_collector.update_model_info(model, provider)

    def get_optimization_status(
        self,
        context_compactor: Optional[Any] = None,
        usage_analytics: Optional[Any] = None,
        sequence_tracker: Optional[Any] = None,
        code_correction_middleware: Optional[Any] = None,
        safety_checker: Optional[Any] = None,
        auto_committer: Optional[Any] = None,
        search_router: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Get comprehensive status of all integrated optimization components.

        Provides visibility into the health and statistics of all optimization
        components for debugging, monitoring, and observability.

        Args:
            context_compactor: Optional context compactor instance
            usage_analytics: Optional usage analytics instance
            sequence_tracker: Optional sequence tracker instance
            code_correction_middleware: Optional code correction middleware
            safety_checker: Optional safety checker instance
            auto_committer: Optional auto committer instance
            search_router: Optional search router instance

        Returns:
            Dictionary with component status and statistics
        """
        status: Dict[str, Any] = {
            "timestamp": time.time(),
            "components": {},
        }

        # Context Compactor
        if context_compactor:
            status["components"]["context_compactor"] = context_compactor.get_statistics()

        # Usage Analytics
        if usage_analytics:
            try:
                status["components"]["usage_analytics"] = {
                    "session_active": usage_analytics._current_session is not None,
                    "tool_records_count": len(usage_analytics._tool_records),
                    "provider_records_count": len(usage_analytics._provider_records),
                }
            except Exception:
                status["components"]["usage_analytics"] = {"status": "error"}

        # Sequence Tracker
        if sequence_tracker:
            try:
                status["components"]["sequence_tracker"] = sequence_tracker.get_statistics()
            except Exception:
                status["components"]["sequence_tracker"] = {"status": "error"}

        # Code Correction Middleware
        status["components"]["code_correction"] = {
            "enabled": code_correction_middleware is not None,
        }
        if code_correction_middleware:
            if hasattr(code_correction_middleware, "config"):
                status["components"]["code_correction"]["config"] = {
                    "auto_fix": code_correction_middleware.config.auto_fix,
                    "max_iterations": code_correction_middleware.config.max_iterations,
                }
            else:
                status["components"]["code_correction"]["config"] = {
                    "auto_fix": getattr(code_correction_middleware, "auto_fix", True),
                    "max_iterations": getattr(code_correction_middleware, "max_iterations", 1),
                }

        # Safety Checker
        status["components"]["safety_checker"] = {
            "enabled": safety_checker is not None,
            "has_confirmation_callback": (
                safety_checker.confirmation_callback is not None if safety_checker else False
            ),
        }

        # Auto Committer
        status["components"]["auto_committer"] = {
            "enabled": auto_committer is not None,
        }
        if auto_committer:
            status["components"]["auto_committer"]["auto_commit"] = auto_committer.auto_commit

        # Search Router
        status["components"]["search_router"] = {
            "enabled": search_router is not None,
        }

        # Overall health
        enabled_count = sum(
            1
            for c in status["components"].values()
            if c.get("enabled", True) and c.get("status") != "error"
        )
        status["health"] = {
            "enabled_components": enabled_count,
            "total_components": len(status["components"]),
            "status": "healthy" if enabled_count >= 4 else "degraded",
        }

        return status

    def reset_stats(self) -> None:
        """Reset all statistics (e.g., after conversation reset)."""
        self._metrics_collector.reset_stats()


def create_metrics_coordinator(
    metrics_collector: "MetricsCollector",
    session_cost_tracker: "SessionCostTracker",
    cumulative_token_usage: Optional[Dict[str, int]] = None,
) -> MetricsCoordinator:
    """Factory function to create a MetricsCoordinator.

    Args:
        metrics_collector: The metrics collector instance
        session_cost_tracker: The session cost tracker instance
        cumulative_token_usage: Optional token usage dict (creates new if None)

    Returns:
        Configured MetricsCoordinator instance
    """
    if cumulative_token_usage is None:
        cumulative_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }

    return MetricsCoordinator(
        metrics_collector=metrics_collector,
        session_cost_tracker=session_cost_tracker,
        cumulative_token_usage=cumulative_token_usage,
    )
