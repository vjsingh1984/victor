"""Callback coordinator for orchestrator tool/streaming lifecycle events.

Extracts callback logic from AgentOrchestrator into a focused component
that delegates to existing coordinators (MetricsCoordinator, ToolCoordinator,
UsageAnalytics, RL coordinator).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.tool_pipeline import ToolCallResult
    from victor.agent.streaming_controller import StreamingSession
    from victor.agent.coordinators.metrics_coordinator import MetricsCoordinator

logger = logging.getLogger(__name__)


class CallbackCoordinator:
    """Coordinates callbacks for tool and streaming lifecycle events.

    Replaces inline callback methods on the orchestrator with clean delegation
    to MetricsCoordinator, ToolCoordinator, and UsageAnalytics.
    """

    def __init__(
        self,
        *,
        metrics_coordinator: "MetricsCoordinator",
        get_tool_coordinator: Callable[[], Any],
        get_observability: Callable[[], Optional[Any]],
        get_pipeline_calls_used: Callable[[], int],
        get_usage_analytics: Callable[[], Optional[Any]],
        get_rl_coordinator: Callable[[], Any],
        get_vertical_context: Callable[[], Any],
    ) -> None:
        self._metrics = metrics_coordinator
        self._get_tool_coordinator = get_tool_coordinator
        self._get_observability = get_observability
        self._get_pipeline_calls_used = get_pipeline_calls_used
        self._get_usage_analytics = get_usage_analytics
        self._get_rl_coordinator = get_rl_coordinator
        self._get_vertical_context = get_vertical_context

    def on_tool_start(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Called when tool execution starts (from ToolPipeline)."""
        iteration = self._get_pipeline_calls_used()
        self._metrics.on_tool_start(tool_name, arguments, iteration)

        obs = self._get_observability()
        if obs:
            tool_id = f"tool-{iteration}"
            obs.on_tool_start(tool_name, arguments, tool_id)

    def on_tool_complete(
        self,
        result: "ToolCallResult",
        *,
        read_files_session: Set[str],
        required_files: List[str],
        required_outputs: List[str],
        nudge_sent_flag: List[bool],
        add_message: Callable,
    ) -> None:
        """Called when tool execution completes (from ToolPipeline)."""
        self._get_tool_coordinator().on_tool_complete(
            result=result,
            metrics_collector=self._metrics.metrics_collector,
            read_files_session=read_files_session,
            required_files=required_files,
            required_outputs=required_outputs,
            nudge_sent_flag=nudge_sent_flag,
            add_message=add_message,
            observability=self._get_observability(),
            pipeline_calls_used=self._get_pipeline_calls_used(),
        )

    def on_streaming_session_complete(self, session: "StreamingSession") -> None:
        """Called when streaming session completes (from StreamingController)."""
        self._metrics.on_streaming_session_complete(session)

        analytics = self._get_usage_analytics()
        if analytics:
            analytics.end_session()

        self._send_rl_reward_signal(session)

    def _send_rl_reward_signal(self, session: "StreamingSession") -> None:
        """Send reward signal to RL model selector."""
        self._metrics.send_rl_reward_signal(
            session=session,
            rl_coordinator=self._get_rl_coordinator(),
            vertical_context=self._get_vertical_context(),
        )
