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

"""Protocol for analytics coordination.

This protocol defines the interface for collecting, aggregating, and reporting
session-level analytics and metrics.
"""

from typing import Any, Dict, List, Optional, Protocol
from dataclasses import dataclass, field


@dataclass
class ToolExecutionStats:
    """Statistics for tool execution.

    Attributes:
        tool_name: Name of the tool
        total_calls: Total number of calls
        successful_calls: Number of successful calls
        failed_calls: Number of failed calls
        total_duration_ms: Total execution duration in milliseconds
        avg_duration_ms: Average execution duration in milliseconds
        last_error: Last error message if any
    """

    tool_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    last_error: Optional[str] = None


@dataclass
class SessionStats:
    """Session-level statistics.

    Attributes:
        session_id: Session identifier
        start_time: Session start timestamp
        duration_seconds: Session duration in seconds
        total_iterations: Total iterations
        tool_calls_count: Total tool calls
        total_tokens_used: Total tokens consumed
        total_cost_usd: Total cost in USD
        tools_used: List of tools used
        tool_stats: Per-tool statistics
        provider_switches: Number of provider switches
        model_switches: Number of model switches
        errors_encountered: Total errors encountered
        cancellation_requested: Whether cancellation was requested
        completion_status: Session completion status
    """

    session_id: str
    start_time: float
    duration_seconds: float = 0.0
    total_iterations: int = 0
    tool_calls_count: int = 0
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    tools_used: List[str] = field(default_factory=list)
    tool_stats: Dict[str, ToolExecutionStats] = field(default_factory=dict)
    provider_switches: int = 0
    model_switches: int = 0
    errors_encountered: int = 0
    cancellation_requested: bool = False
    completion_status: str = "in_progress"


@dataclass
class OptimizationStatus:
    """Optimization metrics status.

    Attributes:
        context_window_usage: Context window usage percentage
        token_usage_efficiency: Token usage efficiency score
        tool_selection_accuracy: Tool selection accuracy (0-1)
        average_iteration_time: Average iteration time in seconds
        cache_hit_rate: Cache hit rate (0-1)
        recommendations: List of optimization recommendations
    """

    context_window_usage: float = 0.0
    token_usage_efficiency: float = 0.0
    tool_selection_accuracy: float = 0.0
    average_iteration_time: float = 0.0
    cache_hit_rate: float = 0.0
    recommendations: List[str] = field(default_factory=list)


class IAnalyticsCoordinator(Protocol):
    """Protocol for analytics coordination.

    This protocol defines the interface for collecting, aggregating, and reporting
    session-level analytics and metrics.

    The coordinator is responsible for:
    - Tracking tool execution statistics
    - Collecting session-level metrics
    - Computing optimization status
    - Recording reward signals for RL
    - Flushing analytics to backends

    Example:
        coordinator = container.get(IAnalyticsCoordinator)
        await coordinator.record_tool_execution(
            tool_name="read_file",
            arguments={"path": "test.py"},
            result={"content": "..."}
        )
        stats = await coordinator.get_session_stats()
    """

    async def get_session_stats(self) -> SessionStats:
        """Get comprehensive session statistics.

        Returns:
            Session statistics including tool usage, tokens, cost, etc.
        """
        ...

    def get_optimization_status(self) -> OptimizationStatus:
        """Get optimization metrics and recommendations.

        Returns:
            Optimization status with metrics and recommendations
        """
        ...

    async def flush_analytics(self) -> Dict[str, bool]:
        """Flush analytics to backends.

        Flushes accumulated analytics data to metrics backends and
        analytics exporters.

        Returns:
            Dictionary mapping backend name to flush success status
        """
        ...

    async def record_tool_execution(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        duration_ms: float = 0.0,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Record tool execution for analytics.

        Args:
            tool_name: Name of the executed tool
            arguments: Tool arguments
            result: Tool execution result
            duration_ms: Execution duration in milliseconds
            success: Whether execution succeeded
            error: Error message if failed
        """
        ...

    def get_tool_execution_stats(
        self,
        tool_name: Optional[str] = None,
    ) -> Optional[ToolExecutionStats]:
        """Get tool execution statistics.

        Args:
            tool_name: Specific tool name, or None for aggregate stats

        Returns:
            Tool execution statistics, or None if tool not found
        """
        ...

    async def record_reward_signal(
        self,
        reward: float,
        task_type: str,
        success: bool,
        quality_score: float = 0.5,
    ) -> None:
        """Record reward signal for reinforcement learning.

        Args:
            reward: Reward value (typically -1.0 to 1.0)
            task_type: Type of task (e.g., "code_edit", "debug")
            success: Whether task was completed successfully
            quality_score: Quality score (0.0 to 1.0)
        """
        ...

    def reset_session_stats(self) -> None:
        """Reset session statistics.

        Clears all accumulated statistics for a fresh session.
        """
        ...

    def increment_iteration_count(self) -> int:
        """Increment iteration counter.

        Returns:
            New iteration count
        """
        ...

    def get_iteration_count(self) -> int:
        """Get current iteration count.

        Returns:
            Current iteration count
        """
        ...


class AnalyticsCoordinatorConfig:
    """Configuration for AnalyticsCoordinator.

    Attributes:
        track_tool_stats: Whether to track per-tool statistics
        track_token_usage: Whether to track token usage
        track_costs: Whether to track costs
        auto_flush_interval: Auto-flush interval in seconds (0 = disabled)
        max_cached_events: Maximum events to cache before auto-flush
        enable_rl_tracking: Whether to enable RL reward tracking
    """

    def __init__(
        self,
        track_tool_stats: bool = True,
        track_token_usage: bool = True,
        track_costs: bool = True,
        auto_flush_interval: float = 0.0,
        max_cached_events: int = 1000,
        enable_rl_tracking: bool = False,
    ) -> None:
        self.track_tool_stats = track_tool_stats
        self.track_token_usage = track_token_usage
        self.track_costs = track_costs
        self.auto_flush_interval = auto_flush_interval
        self.max_cached_events = max_cached_events
        self.enable_rl_tracking = enable_rl_tracking


__all__ = [
    "IAnalyticsCoordinator",
    "SessionStats",
    "ToolExecutionStats",
    "OptimizationStatus",
    "AnalyticsCoordinatorConfig",
]
