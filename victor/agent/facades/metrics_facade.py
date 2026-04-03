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

"""Metrics domain facade for orchestrator decomposition.

Groups usage tracking, analytics, streaming metrics, cost accounting,
and debug logging components behind a single interface.

This facade wraps already-initialized components from the orchestrator,
providing a coherent grouping without changing initialization ordering.
The orchestrator delegates property access through this facade.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MetricsFacade:
    """Groups metrics, analytics, and observability components.

    Satisfies ``MetricsFacadeProtocol`` structurally.  The orchestrator creates
    this facade after all metrics-domain components are initialized, passing
    references to the already-created instances.

    Components managed:
        - metrics_runtime: Metrics runtime boundary components
        - metrics_collector: Optional metrics collector
        - usage_analytics: Optional usage analytics collector
        - usage_logger: Usage event logger
        - streaming_metrics_collector: Streaming metrics collector
        - session_cost_tracker: Session cost tracker
        - metrics_coordinator: Metrics coordinator service
        - debug_logger: Debug logger for incremental output
        - callback_coordinator: CallbackCoordinator for tool/streaming events
    """

    def __init__(
        self,
        *,
        metrics_runtime: Optional[Any] = None,
        metrics_collector: Optional[Any] = None,
        usage_analytics: Optional[Any] = None,
        usage_logger: Optional[Any] = None,
        streaming_metrics_collector: Optional[Any] = None,
        session_cost_tracker: Optional[Any] = None,
        metrics_coordinator: Optional[Any] = None,
        debug_logger: Optional[Any] = None,
        callback_coordinator: Optional[Any] = None,
    ) -> None:
        self._metrics_runtime = metrics_runtime
        self._metrics_collector = metrics_collector
        self._usage_analytics = usage_analytics
        self._usage_logger = usage_logger
        self._streaming_metrics_collector = streaming_metrics_collector
        self._session_cost_tracker = session_cost_tracker
        self._metrics_coordinator = metrics_coordinator
        self._debug_logger = debug_logger
        self._callback_coordinator = callback_coordinator

        logger.debug(
            "MetricsFacade initialized (analytics=%s, streaming=%s, cost=%s)",
            usage_analytics is not None,
            streaming_metrics_collector is not None,
            session_cost_tracker is not None,
        )

    # ------------------------------------------------------------------
    # Properties (satisfy MetricsFacadeProtocol)
    # ------------------------------------------------------------------

    @property
    def metrics_runtime(self) -> Optional[Any]:
        """Metrics runtime boundary components."""
        return self._metrics_runtime

    @property
    def metrics_collector(self) -> Optional[Any]:
        """Optional metrics collector."""
        return self._metrics_collector

    @property
    def usage_analytics(self) -> Optional[Any]:
        """Optional usage analytics collector."""
        return self._usage_analytics

    @property
    def usage_logger(self) -> Optional[Any]:
        """Usage event logger."""
        return self._usage_logger

    @property
    def streaming_metrics_collector(self) -> Optional[Any]:
        """Streaming metrics collector."""
        return self._streaming_metrics_collector

    @property
    def session_cost_tracker(self) -> Optional[Any]:
        """Session cost tracker."""
        return self._session_cost_tracker

    @property
    def metrics_coordinator(self) -> Optional[Any]:
        """Metrics coordinator service."""
        return self._metrics_coordinator

    @property
    def debug_logger(self) -> Optional[Any]:
        """Debug logger for incremental output and conversation tracking."""
        return self._debug_logger

    @property
    def callback_coordinator(self) -> Optional[Any]:
        """CallbackCoordinator for tool/streaming event delegation."""
        return self._callback_coordinator
