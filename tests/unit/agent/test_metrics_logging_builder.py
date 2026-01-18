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

"""Tests for MetricsLoggingBuilder."""

from unittest.mock import MagicMock, patch

from victor.agent.builders.metrics_logging_builder import MetricsLoggingBuilder


def test_metrics_logging_builder_wires_components():
    """MetricsLoggingBuilder assigns logging and metrics components."""
    settings = MagicMock()
    factory = MagicMock()
    usage_logger = MagicMock()
    factory.create_usage_logger.return_value = usage_logger
    factory.create_streaming_metrics_collector.return_value = "streaming-collector"
    factory.create_debug_logger_configured.return_value = "debug-logger"
    factory.create_metrics_collector.return_value = "metrics-collector"

    orchestrator = MagicMock()
    orchestrator.model = "model"
    provider = MagicMock()
    provider.name = "provider"
    orchestrator.provider = provider
    orchestrator._cumulative_token_usage = {"total_tokens": 0}
    orchestrator.tools = MagicMock()

    with (
        patch("victor.agent.session_cost_tracker.SessionCostTracker") as tracker_cls,
        patch(
            "victor.agent.coordinators.metrics_coordinator.MetricsCoordinator"
        ) as metrics_coord_cls,
    ):
        tracker_instance = MagicMock()
        tracker_cls.return_value = tracker_instance
        metrics_coord_instance = MagicMock()
        metrics_coord_cls.return_value = metrics_coord_instance

        builder = MetricsLoggingBuilder(settings=settings, factory=factory)
        components = builder.build(orchestrator)

    usage_logger.log_event.assert_called_once()
    factory.create_metrics_collector.assert_called_once()
    metrics_args = factory.create_metrics_collector.call_args.kwargs
    assert metrics_args["streaming_metrics_collector"] == "streaming-collector"
    assert metrics_args["usage_logger"] == usage_logger
    assert metrics_args["debug_logger"] == "debug-logger"

    tracker_cls.assert_called_once_with(provider="provider", model="model")
    metrics_coord_cls.assert_called_once_with(
        metrics_collector="metrics-collector",
        session_cost_tracker=tracker_instance,
        cumulative_token_usage=orchestrator._cumulative_token_usage,
    )

    assert orchestrator.usage_logger == usage_logger
    assert orchestrator.streaming_metrics_collector == "streaming-collector"
    assert orchestrator.debug_logger == "debug-logger"
    assert orchestrator._metrics_collector == "metrics-collector"
    assert orchestrator._metrics_coordinator == metrics_coord_instance
    assert isinstance(orchestrator._background_tasks, set)
    assert orchestrator._embedding_preload_task is None
    assert components["metrics_collector"] == "metrics-collector"
