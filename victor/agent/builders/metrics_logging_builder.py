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

"""Metrics and logging builder for orchestrator initialization.

Part of HIGH-005: Initialization Complexity reduction.
"""

import asyncio
import logging
from typing import Any, Optional, TYPE_CHECKING

from victor.agent.builders.base import FactoryAwareBuilder

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.orchestrator_factory import OrchestratorFactory

logger = logging.getLogger(__name__)


class MetricsLoggingBuilder(FactoryAwareBuilder):
    """Build usage logging, metrics, and background task tracking."""

    def __init__(self, settings: Any, factory: Optional["OrchestratorFactory"] = None):
        """Initialize the builder.

        Args:
            settings: Application settings
            factory: Optional OrchestratorFactory instance
        """
        super().__init__(settings, factory)

    def build(self, orchestrator: "AgentOrchestrator", **_kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        """Build metrics and logging components and attach them to orchestrator."""
        factory = self._ensure_factory()
        components: dict[str, Any] = {}

        # Analytics - usage logger with DI support (via factory)
        orchestrator.usage_logger = factory.create_usage_logger()
        orchestrator.usage_logger.log_event(
            "session_start",
            {"model": orchestrator.model, "provider": orchestrator.provider.__class__.__name__},
        )
        components["usage_logger"] = orchestrator.usage_logger

        # Streaming metrics collector for performance monitoring (via factory)
        orchestrator.streaming_metrics_collector = factory.create_streaming_metrics_collector()
        components["streaming_metrics_collector"] = orchestrator.streaming_metrics_collector
        if orchestrator.streaming_metrics_collector:
            logger.info("StreamingMetricsCollector initialized via factory")

        # Debug logger for incremental output and conversation tracking (via factory)
        orchestrator.debug_logger = factory.create_debug_logger_configured()
        components["debug_logger"] = orchestrator.debug_logger

        # Background task tracking for graceful shutdown
        if not hasattr(orchestrator, "_background_tasks"):
            orchestrator._background_tasks: set[asyncio.Task[Any]] = set()  # type: ignore[misc]
        if not hasattr(orchestrator, "_embedding_preload_task"):
            orchestrator._embedding_preload_task: Optional[asyncio.Task[Any]] = None  # type: ignore[misc]
        components["background_tasks"] = orchestrator._background_tasks
        components["embedding_preload_task"] = orchestrator._embedding_preload_task

        # Metrics collection (via factory)
        from victor.tools.base import CostTier

        orchestrator._metrics_collector = factory.create_metrics_collector(
            streaming_metrics_collector=orchestrator.streaming_metrics_collector,
            usage_logger=orchestrator.usage_logger,
            debug_logger=orchestrator.debug_logger,
            tool_cost_lookup=lambda name: (
                orchestrator.tools.get_tool_cost(name)
                if hasattr(orchestrator, "tools") and orchestrator.tools is not None
                else CostTier.FREE
            ),
        )
        components["metrics_collector"] = orchestrator._metrics_collector

        # Session cost tracking (for LLM API cost monitoring)
        from victor.agent.session_cost_tracker import SessionCostTracker

        orchestrator._session_cost_tracker = SessionCostTracker(
            provider=orchestrator.provider.name,
            model=orchestrator.model,
        )
        components["session_cost_tracker"] = orchestrator._session_cost_tracker

        # Metrics coordinator (Phase 2 refactoring - aggregates metrics/cost/token tracking)
        from victor.agent.coordinators.metrics_coordinator import MetricsCoordinator

        orchestrator._metrics_coordinator = MetricsCoordinator(
            metrics_collector=orchestrator._metrics_collector,
            session_cost_tracker=orchestrator._session_cost_tracker,
            cumulative_token_usage=orchestrator._cumulative_token_usage,
        )
        components["metrics_coordinator"] = orchestrator._metrics_coordinator

        self._register_components(components)
        return components
