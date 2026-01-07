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

# """Tests for EventBus sampling and batching (Phase 3 - Scalability)."""

import time
from unittest.mock import MagicMock

import pytest

# Old event_bus imports removed - migration complete


@pytest.fixture(autouse=True)
def reset_event_bus():
    # """Reset EventBus singleton before and after each test."""
    # EventBus.reset_instance()
    yield
    # EventBus.reset_instance()


# =============================================================================
# SamplingConfig Tests
# =============================================================================


class TestSamplingConfig:
    """Tests for SamplingConfig."""

    def test_default_config(self):
        """Default config should have 100% sampling."""
        config = SamplingConfig()
        assert config.default_rate == 1.0
        assert config.preserve_errors is True
        assert config.preserve_critical is True

    def test_rate_validation(self):
        """Invalid rates should raise ValueError."""
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            SamplingConfig(rates={EventCategory.TOOL: 1.5})

        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            SamplingConfig(rates={EventCategory.TOOL: -0.1})

        with pytest.raises(ValueError, match="Default sampling rate"):
            SamplingConfig(default_rate=2.0)

    def test_get_rate_uses_category_specific(self):
        """Should return category-specific rate if configured."""
        config = SamplingConfig(
            rates={EventCategory.METRIC: 0.1},
            default_rate=0.5,
        )
        assert config.get_rate(EventCategory.METRIC) == 0.1
        assert config.get_rate(EventCategory.TOOL) == 0.5  # default

    def test_preserve_errors_overrides_rate(self):
        """ERROR category should always return 1.0 when preserve_errors=True."""
        config = SamplingConfig(
            rates={EventCategory.ERROR: 0.1},
            preserve_errors=True,
        )
        assert config.get_rate(EventCategory.ERROR) == 1.0

    def test_preserve_errors_disabled(self):
        """ERROR category should use configured rate when preserve_errors=False."""
        config = SamplingConfig(
            rates={EventCategory.ERROR: 0.1},
            preserve_errors=False,
        )
        assert config.get_rate(EventCategory.ERROR) == 0.1

    def test_should_sample_critical_always_passes(self):
        """CRITICAL priority events should always be sampled."""
        config = SamplingConfig(
            rates={EventCategory.TOOL: 0.0},
            preserve_critical=True,
        )
        event = VictorEvent(
            category=EventCategory.TOOL,
            name="test",
            priority=EventPriority.CRITICAL,
        )
        assert config.should_sample(event) is True

    def test_should_sample_zero_rate_blocks(self):
        """0% rate should block all events."""
        config = SamplingConfig(
            rates={EventCategory.TOOL: 0.0},
            preserve_critical=False,
        )
        event = VictorEvent(
            category=EventCategory.TOOL,
            name="test",
            priority=EventPriority.NORMAL,
        )
        assert config.should_sample(event) is False

    def test_should_sample_full_rate_passes(self):
        """100% rate should pass all events."""
        config = SamplingConfig(rates={EventCategory.TOOL: 1.0})
        for _ in range(100):
            event = VictorEvent(category=EventCategory.TOOL, name="test")
            assert config.should_sample(event) is True

    def test_should_sample_deterministic(self):
        """Same event should always produce same result."""
        config = SamplingConfig(rates={EventCategory.TOOL: 0.5})
        event = VictorEvent(category=EventCategory.TOOL, name="test", id="fixed-id")

        # Same event should always give same result
        results = [config.should_sample(event) for _ in range(10)]
        assert all(r == results[0] for r in results)


# =============================================================================
# SamplingMetrics Tests
# =============================================================================


class TestSamplingMetrics:
    """Tests for SamplingMetrics."""

    def test_record_sampled(self):
        """Recording sampled events should increment correct counters."""
        metrics = SamplingMetrics()
        metrics.record(EventCategory.TOOL, sampled=True)

        assert metrics.events_sampled == 1
        assert metrics.events_dropped == 0
        assert metrics.by_category["tool"]["sampled"] == 1
        assert metrics.by_category["tool"]["dropped"] == 0

    def test_record_dropped(self):
        """Recording dropped events should increment correct counters."""
        metrics = SamplingMetrics()
        metrics.record(EventCategory.TOOL, sampled=False)

        assert metrics.events_sampled == 0
        assert metrics.events_dropped == 1
        assert metrics.by_category["tool"]["dropped"] == 1

    def test_to_dict(self):
        """Should serialize to dictionary."""
        metrics = SamplingMetrics()
        metrics.record(EventCategory.TOOL, sampled=True)
        metrics.record(EventCategory.METRIC, sampled=False)

        data = metrics.to_dict()
        assert data["events_sampled"] == 1
        assert data["events_dropped"] == 1
        assert "tool" in data["by_category"]
        assert "metric" in data["by_category"]


# =============================================================================
# BatchConfig Tests
# =============================================================================


class TestBatchConfig:
    """Tests for BatchConfig."""

    def test_default_disabled(self):
        """Batching should be disabled by default."""
        config = BatchConfig()
        assert config.enabled is False

    def test_should_batch_when_disabled(self):
        """should_batch should return False when disabled."""
        config = BatchConfig(enabled=False)
        event = VictorEvent(category=EventCategory.TOOL, name="test")
        assert config.should_batch(event) is False

    def test_should_batch_all_categories_when_empty(self):
        """Empty categories means batch all."""
        config = BatchConfig(enabled=True, categories=set())
        event = VictorEvent(category=EventCategory.TOOL, name="test")
        assert config.should_batch(event) is True

    def test_should_batch_only_specified_categories(self):
        """Should only batch specified categories."""
        config = BatchConfig(
            enabled=True,
            categories={EventCategory.METRIC},
        )
        metric_event = VictorEvent(category=EventCategory.METRIC, name="test")
        tool_event = VictorEvent(category=EventCategory.TOOL, name="test")

        assert config.should_batch(metric_event) is True
        assert config.should_batch(tool_event) is False


# =============================================================================
# EventBatcher Tests
# =============================================================================


class TestEventBatcher:
    """Tests for EventBatcher."""

    def test_add_event_returns_none_below_threshold(self):
        """Events below threshold should not trigger flush."""
        config = BatchConfig(enabled=True, batch_size=10)
        batcher = EventBatcher(config)

        event = VictorEvent(category=EventCategory.TOOL, name="test")
        result = batcher.add_event(event)

        assert result is None
        assert batcher.get_pending_count() == 1

    def test_add_event_flushes_at_batch_size(self):
        """Should flush when batch size is reached."""
        config = BatchConfig(enabled=True, batch_size=3)
        batcher = EventBatcher(config)

        events = [VictorEvent(category=EventCategory.TOOL, name=f"event{i}") for i in range(3)]

        # First two events don't flush
        assert batcher.add_event(events[0]) is None
        assert batcher.add_event(events[1]) is None

        # Third event triggers flush
        result = batcher.add_event(events[2])
        assert result is not None
        assert len(result) == 3
        assert batcher.get_pending_count() == 0

    def test_flush_all(self):
        """flush_all should return all pending events."""
        config = BatchConfig(enabled=True, batch_size=100)
        batcher = EventBatcher(config)

        tool_event = VictorEvent(category=EventCategory.TOOL, name="tool")
        metric_event = VictorEvent(category=EventCategory.METRIC, name="metric")

        batcher.add_event(tool_event)
        batcher.add_event(metric_event)

        result = batcher.flush_all()

        assert EventCategory.TOOL in result
        assert EventCategory.METRIC in result
        assert len(result[EventCategory.TOOL]) == 1
        assert len(result[EventCategory.METRIC]) == 1
        assert batcher.get_pending_count() == 0


# =============================================================================
# ExporterConfig Tests
# =============================================================================


class TestExporterConfig:
    """Tests for ExporterConfig."""

    def test_default_allows_all(self):
        """Default config should allow all events."""
        config = ExporterConfig()
        event = VictorEvent(
            category=EventCategory.TOOL,
            name="test",
            priority=EventPriority.LOW,
        )
        assert config.should_export(event) is True

    def test_min_priority_filtering(self):
        """Should filter events below min priority."""
        config = ExporterConfig(min_priority=EventPriority.HIGH)

        low_event = VictorEvent(
            category=EventCategory.TOOL,
            name="test",
            priority=EventPriority.LOW,
        )
        high_event = VictorEvent(
            category=EventCategory.TOOL,
            name="test",
            priority=EventPriority.HIGH,
        )

        assert config.should_export(low_event) is False
        assert config.should_export(high_event) is True

    def test_category_inclusion(self):
        """Should only export specified categories."""
        config = ExporterConfig(categories={EventCategory.ERROR})

        error_event = VictorEvent(category=EventCategory.ERROR, name="test")
        tool_event = VictorEvent(category=EventCategory.TOOL, name="test")

        assert config.should_export(error_event) is True
        assert config.should_export(tool_event) is False

    def test_category_exclusion(self):
        """Should exclude specified categories."""
        config = ExporterConfig(exclude_categories={EventCategory.METRIC})

        metric_event = VictorEvent(category=EventCategory.METRIC, name="test")
        tool_event = VictorEvent(category=EventCategory.TOOL, name="test")

        assert config.should_export(metric_event) is False
        assert config.should_export(tool_event) is True

    def test_sampling_override(self):
        """Should apply per-exporter sampling."""
        sampling = SamplingConfig(
            rates={EventCategory.TOOL: 0.0},
            preserve_critical=False,
        )
        config = ExporterConfig(sampling_override=sampling)

        event = VictorEvent(
            category=EventCategory.TOOL,
            name="test",
            priority=EventPriority.NORMAL,
        )
        assert config.should_export(event) is False

    # =============================================================================
    # EventBus Integration Tests
    # =============================================================================

    # class TestEventBusSamplingIntegration:
    # """Tests for EventBus sampling integration."""

    def test_configure_sampling(self):
        """Should configure and apply sampling."""
        # bus = EventBus.get_instance()
        handler = MagicMock()
        bus.subscribe(EventCategory.TOOL, handler)

        # Configure 0% sampling for TOOL events
        bus.configure_sampling(
            SamplingConfig(
                rates={EventCategory.TOOL: 0.0},
                preserve_critical=False,
            )
        )

        # Event should be dropped
        event = VictorEvent(
            category=EventCategory.TOOL,
            name="test",
            priority=EventPriority.NORMAL,
        )
        bus.publish(event)

        handler.assert_not_called()

    def test_disable_sampling(self):
        """Should disable sampling and allow all events."""
        # bus = EventBus.get_instance()
        handler = MagicMock()
        bus.subscribe(EventCategory.TOOL, handler)

        # Configure then disable sampling
        bus.configure_sampling(
            SamplingConfig(
                rates={EventCategory.TOOL: 0.0},
                preserve_critical=False,
            )
        )
        bus.disable_sampling()

        event = VictorEvent(category=EventCategory.TOOL, name="test")
        bus.publish(event)

        handler.assert_called_once()

    def test_sampling_metrics(self):
        """Should track sampling metrics."""
        # bus = EventBus.get_instance()
        bus.configure_sampling(
            SamplingConfig(
                rates={EventCategory.TOOL: 0.0},
                preserve_critical=False,
            )
        )

        event = VictorEvent(
            category=EventCategory.TOOL,
            name="test",
            priority=EventPriority.NORMAL,
        )
        bus.publish(event)

        metrics = bus.get_sampling_metrics()
        assert metrics.events_dropped == 1

    # class TestEventBusExporterConfig:
    # """Tests for EventBus exporter config integration."""

    def test_add_exporter_with_config(self):
        """Should store exporter config."""
        # bus = EventBus.get_instance()
        exporter = MagicMock()
        config = ExporterConfig(min_priority=EventPriority.HIGH)

        bus.add_exporter(exporter, config)

        retrieved_config = bus.get_exporter_config(exporter)
        assert retrieved_config is config

    def test_exporter_config_filtering(self):
        """Should filter events per exporter config."""
        # bus = EventBus.get_instance()
        exporter = MagicMock()
        config = ExporterConfig(categories={EventCategory.ERROR})

        bus.add_exporter(exporter, config)

        # TOOL event should be filtered
        tool_event = VictorEvent(category=EventCategory.TOOL, name="test")
        bus.publish(tool_event)
        exporter.export.assert_not_called()

        # ERROR event should pass
        error_event = VictorEvent(category=EventCategory.ERROR, name="test")
        bus.publish(error_event)
        exporter.export.assert_called_once()

    def test_remove_exporter_cleans_config(self):
        """Removing exporter should clean up config."""
        # bus = EventBus.get_instance()
        exporter = MagicMock()
        config = ExporterConfig()

        bus.add_exporter(exporter, config)
        bus.remove_exporter(exporter)

        assert bus.get_exporter_config(exporter) is None

    def test_set_exporter_config_requires_registered(self):
        """set_exporter_config should require registered exporter."""
        # bus = EventBus.get_instance()
        exporter = MagicMock()

        with pytest.raises(ValueError, match="not registered"):
            bus.set_exporter_config(exporter, ExporterConfig())

    # class TestEventBusBatchingIntegration:
    # """Tests for EventBus batching integration."""

    def test_configure_batching(self):
        """Should configure batching."""
        # bus = EventBus.get_instance()
        bus.configure_batching(BatchConfig(enabled=True, batch_size=100))

        assert bus.get_batch_pending_count() == 0

    def test_disable_batching(self):
        """Should disable batching."""
        # bus = EventBus.get_instance()
        bus.configure_batching(BatchConfig(enabled=True))
        bus.disable_batching()

        assert bus._batcher is None

    def test_flush_batches(self):
        """Should flush batches."""
        # bus = EventBus.get_instance()

        # With no batcher, flush returns empty
        result = bus.flush_batches()
        assert result == {}
