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

"""Tests for MetricsFacade domain facade."""

import pytest
from unittest.mock import MagicMock

from victor.agent.facades.metrics_facade import MetricsFacade
from victor.agent.facades.protocols import MetricsFacadeProtocol


class TestMetricsFacadeInit:
    """Tests for MetricsFacade initialization."""

    def test_init_with_all_components(self):
        """MetricsFacade initializes with all components provided."""
        analytics = MagicMock()
        logger_mock = MagicMock()
        collector = MagicMock()

        facade = MetricsFacade(
            metrics_runtime=MagicMock(),
            metrics_collector=collector,
            usage_analytics=analytics,
            usage_logger=logger_mock,
            streaming_metrics_collector=MagicMock(),
            session_cost_tracker=MagicMock(),
            metrics_coordinator=MagicMock(),
            debug_logger=MagicMock(),
            callback_coordinator=MagicMock(),
        )

        assert facade.usage_analytics is analytics
        assert facade.usage_logger is logger_mock
        assert facade.metrics_collector is collector

    def test_init_with_minimal_components(self):
        """MetricsFacade initializes with no required components (all optional)."""
        facade = MetricsFacade()

        assert facade.metrics_runtime is None
        assert facade.metrics_collector is None
        assert facade.usage_analytics is None
        assert facade.usage_logger is None
        assert facade.streaming_metrics_collector is None
        assert facade.session_cost_tracker is None
        assert facade.metrics_coordinator is None
        assert facade.debug_logger is None
        assert facade.callback_coordinator is None


class TestMetricsFacadeProperties:
    """Tests for MetricsFacade property access."""

    @pytest.fixture
    def facade(self):
        """Create a MetricsFacade with mock components."""
        return MetricsFacade(
            metrics_runtime=MagicMock(name="runtime"),
            metrics_collector=MagicMock(name="collector"),
            usage_analytics=MagicMock(name="analytics"),
            usage_logger=MagicMock(name="logger"),
            streaming_metrics_collector=MagicMock(name="streaming"),
            session_cost_tracker=MagicMock(name="cost"),
            metrics_coordinator=MagicMock(name="coordinator"),
            debug_logger=MagicMock(name="debug"),
            callback_coordinator=MagicMock(name="callback"),
        )

    def test_metrics_runtime_property(self, facade):
        """MetricsRuntime property returns the runtime."""
        assert facade.metrics_runtime._mock_name == "runtime"

    def test_metrics_collector_property(self, facade):
        """MetricsCollector property returns the collector."""
        assert facade.metrics_collector._mock_name == "collector"

    def test_usage_analytics_property(self, facade):
        """UsageAnalytics property returns the analytics."""
        assert facade.usage_analytics._mock_name == "analytics"

    def test_usage_logger_property(self, facade):
        """UsageLogger property returns the logger."""
        assert facade.usage_logger._mock_name == "logger"

    def test_streaming_metrics_property(self, facade):
        """StreamingMetricsCollector property returns the collector."""
        assert facade.streaming_metrics_collector._mock_name == "streaming"

    def test_session_cost_tracker_property(self, facade):
        """SessionCostTracker property returns the tracker."""
        assert facade.session_cost_tracker._mock_name == "cost"

    def test_debug_logger_property(self, facade):
        """DebugLogger property returns the logger."""
        assert facade.debug_logger._mock_name == "debug"

    def test_callback_coordinator_property(self, facade):
        """CallbackCoordinator property returns the coordinator."""
        assert facade.callback_coordinator._mock_name == "callback"


class TestMetricsFacadeProtocolConformance:
    """Tests that MetricsFacade satisfies MetricsFacadeProtocol."""

    def test_satisfies_protocol(self):
        """MetricsFacade structurally conforms to MetricsFacadeProtocol."""
        facade = MetricsFacade()
        assert isinstance(facade, MetricsFacadeProtocol)

    def test_protocol_properties_present(self):
        """All protocol-required properties are present on MetricsFacade."""
        required = [
            "usage_analytics",
            "usage_logger",
            "metrics_collector",
        ]
        facade = MetricsFacade()
        for prop in required:
            assert hasattr(facade, prop), f"Missing protocol property: {prop}"
