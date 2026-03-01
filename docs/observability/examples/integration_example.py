"""Example: Integrating observability into a custom component.

This example shows how to add observability to your own components
by implementing the MetricSource protocol.
"""

import time
import threading
from typing import Optional
from victor.framework.observability import ObservabilityManager
from victor.framework.observability.metrics import (
    MetricSource,
    MetricsSnapshot,
    CounterMetric,
    GaugeMetric,
    MetricLabel,
)


class CustomProcessor:
    """Example component that we want to monitor."""

    def __init__(self, name: str):
        self.name = name
        self._items_processed = 0
        self._items_failed = 0
        self._processing_time_ms = 0.0
        self._queue_size = 0
        self._lock = threading.Lock()

    def process_item(self, item: str) -> bool:
        """Process an item and track metrics."""
        start_time = time.time()

        try:
            # Simulate processing
            time.sleep(0.01)  # 10ms processing time

            with self._lock:
                self._items_processed += 1

            processing_time = (time.time() - start_time) * 1000
            with self._lock:
                self._processing_time_ms += processing_time

            return True

        except Exception:
            with self._lock:
                self._items_failed += 1
            return False

    def enqueue(self, count: int):
        """Add items to the queue."""
        with self._lock:
            self._queue_size += count

    def dequeue(self, count: int):
        """Remove items from the queue."""
        with self._lock:
            self._queue_size = max(0, self._queue_size - count)


class ObservableProcessor(MetricSource):
    """Observable wrapper for CustomProcessor.

    This class implements MetricSource to provide metrics for the processor.
    """

    def __init__(self, processor: CustomProcessor):
        self._processor = processor

    @property
    def source_id(self) -> str:
        return f"processor:{self._processor.name}"

    @property
    def source_type(self) -> str:
        return "processor"

    def get_metrics(self) -> MetricsSnapshot:
        """Get current metrics from the processor."""

        with self._processor._lock:
            items_processed = self._processor._items_processed
            items_failed = self._processor._items_failed
            processing_time = self._processor._processing_time_ms
            queue_size = self._processor._queue_size

        # Calculate average processing time
        avg_processing_time = (
            processing_time / items_processed if items_processed > 0 else 0.0
        )

        # Calculate success rate
        total_items = items_processed + items_failed
        success_rate = (
            items_processed / total_items if total_items > 0 else 1.0
        )

        metrics = [
            CounterMetric(
                name="items_processed_total",
                description="Total items processed",
                value=items_processed,
            ),
            CounterMetric(
                name="items_failed_total",
                description="Total items that failed processing",
                value=items_failed,
            ),
            GaugeMetric(
                name="queue_size",
                description="Number of items in the queue",
                value=queue_size,
                labels=(MetricLabel(key="processor", value=self._processor.name),),
            ),
            GaugeMetric(
                name="average_processing_time_ms",
                description="Average processing time per item",
                value=avg_processing_time,
            ),
            GaugeMetric(
                name="success_rate",
                description="Processing success rate",
                value=success_rate,
            ),
        ]

        return MetricsSnapshot(
            source_id=self.source_id,
            source_type=self.source_type,
            metrics=tuple(metrics),
        )


# Usage example
if __name__ == "__main__":
    # Create processor
    processor = CustomProcessor("image-resizer")

    # Create observable wrapper
    observable_processor = ObservableProcessor(processor)

    # Register with ObservabilityManager
    manager = ObservabilityManager.get_instance()
    manager.register_source(observable_processor)

    print("Starting processor with observability...")
    print(f"Registered source: {observable_processor.source_id}")
    print()

    # Simulate processing
    processor.enqueue(100)

    for i in range(100):
        processor.process_item(f"item-{i}")
        if i % 20 == 0:
            # Get dashboard data to see metrics
            data = manager.get_dashboard_data()
            print(f"Processed {i + 1}/100 items")
            print(f"  Queue size: {processor._queue_size}")
            print(f"  Total sources: {data.total_sources}")
            print()

    # Get final metrics
    snapshot = observable_processor.get_metrics()

    print("\nFinal Metrics:")
    print("=" * 50)
    for metric in snapshot.metrics:
        if isinstance(metric, GaugeMetric) and metric.labels:
            label_str = ", ".join(f"{l.key}={l.value}" for l in metric.labels)
            print(f"{metric.name} [{label_str}]: {metric.value}")
        else:
            print(f"{metric.name}: {metric.value}")
