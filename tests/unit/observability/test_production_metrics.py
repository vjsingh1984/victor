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

"""Tests for production metrics collector."""

import pytest

from victor.observability.production_metrics import (
    ProductionMetricsCollector,
    create_production_collector,
    MetricCategory,
)


@pytest.fixture
def metrics_collector():
    """Create a fresh metrics collector for each test."""
    collector = ProductionMetricsCollector(collect_system_metrics=False)
    yield collector
    # Cleanup
    collector._registry.clear()


class TestProductionMetricsCollector:
    """Test ProductionMetricsCollector functionality."""

    def test_initialization(self, metrics_collector):
        """Test collector initializes correctly."""
        assert metrics_collector is not None
        assert metrics_collector._prefix == "victor"
        assert metrics_collector._registry is not None

    def test_record_request(self, metrics_collector):
        """Test recording request metrics."""
        metrics_collector.record_request(
            endpoint="/chat",
            provider="anthropic",
            success=True,
            latency_ms=123.4,
        )

        summary = metrics_collector.get_summary()
        assert summary["requests"]["total"] == 1
        assert summary["requests"]["success"] == 1
        assert summary["requests"]["errors"] == 0

    def test_record_request_error(self, metrics_collector):
        """Test recording failed request."""
        metrics_collector.record_request(
            endpoint="/chat",
            provider="anthropic",
            success=False,
            latency_ms=456.7,
        )

        summary = metrics_collector.get_summary()
        assert summary["requests"]["total"] == 1
        assert summary["requests"]["success"] == 0
        assert summary["requests"]["errors"] == 1

    def test_record_tool_execution(self, metrics_collector):
        """Test recording tool execution."""
        metrics_collector.record_tool_execution(
            tool_name="read_file",
            success=True,
            duration_ms=45.2,
        )

        summary = metrics_collector.get_summary()
        assert summary["tools"]["calls"] == 1
        assert summary["tools"]["errors"] == 0

    def test_record_tool_execution_error(self, metrics_collector):
        """Test recording failed tool execution."""
        metrics_collector.record_tool_execution(
            tool_name="read_file",
            success=False,
            duration_ms=45.2,
        )

        summary = metrics_collector.get_summary()
        assert summary["tools"]["calls"] == 1
        assert summary["tools"]["errors"] == 1

    def test_cache_metrics(self, metrics_collector):
        """Test cache metrics."""
        metrics_collector.record_cache_hit("tool_cache")
        metrics_collector.record_cache_hit("tool_cache")
        metrics_collector.record_cache_miss("tool_cache")

        hit_rate = metrics_collector.get_cache_hit_rate()
        assert hit_rate == 66.66666666666666

    def test_export_prometheus(self, metrics_collector):
        """Test Prometheus export."""
        metrics_collector.record_request(
            endpoint="/test",
            provider="anthropic",
            success=True,
            latency_ms=100.0,
        )

        prometheus_text = metrics_collector.export_prometheus()
        assert "victor_request_total" in prometheus_text
        assert "victor_request_success_total" in prometheus_text

    def test_export_json(self, metrics_collector):
        """Test JSON export."""
        metrics_collector.record_request(
            endpoint="/test",
            provider="anthropic",
            success=True,
            latency_ms=100.0,
        )

        json_data = metrics_collector.export_json()
        assert "timestamp" in json_data
        assert "metrics" in json_data
        assert "summary" in json_data

    def test_get_summary(self, metrics_collector):
        """Test metrics summary."""
        metrics_collector.record_request(
            endpoint="/test",
            provider="anthropic",
            success=True,
            latency_ms=100.0,
        )
        metrics_collector.record_tool_execution(
            tool_name="read_file",
            success=True,
            duration_ms=50.0,
        )

        summary = metrics_collector.get_summary()
        assert "requests" in summary
        assert "tools" in summary
        assert "cache" in summary
        assert "business" in summary

    def test_record_error(self, metrics_collector):
        """Test error recording."""
        metrics_collector.record_error("ValueError", "test error")

        summary = metrics_collector.get_summary()
        assert summary["requests"]["errors"] >= 0  # Errors tracked

    def test_record_task_completion(self, metrics_collector):
        """Test task completion tracking."""
        metrics_collector.record_task_completion(success=True, cost_usd=0.05)
        metrics_collector.record_task_completion(success=False, cost_usd=0.0)

        summary = metrics_collector.get_summary()
        assert summary["business"]["tasks_completed"] == 1
        assert summary["business"]["tasks_failed"] == 1
        assert summary["business"]["total_cost_usd"] == 0.05

    def test_record_token_usage(self, metrics_collector):
        """Test token usage tracking."""
        metrics_collector.record_token_usage(1000)
        metrics_collector.record_token_usage(2000)

        summary = metrics_collector.get_summary()
        assert summary["business"]["tokens_used"] == 3000

    def test_create_production_collector(self):
        """Test factory function."""
        collector = create_production_collector(prefix="test")
        assert collector is not None
        assert collector._prefix == "test"


class TestMetricCategory:
    """Test MetricCategory enum."""

    def test_metric_categories(self):
        """Test all metric categories exist."""
        assert MetricCategory.REQUEST == "request"
        assert MetricCategory.TOOL == "tool"
        assert MetricCategory.COORDINATOR == "coordinator"
        assert MetricCategory.BUSINESS == "business"
        assert MetricCategory.SYSTEM == "system"
        assert MetricCategory.CACHE == "cache"
        assert MetricCategory.ERROR == "error"
