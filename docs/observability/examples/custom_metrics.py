"""Example: Creating custom metrics sources.

This example demonstrates how to create a custom metrics source
that implements the MetricSource protocol.
"""

from victor.framework.observability import ObservabilityManager
from victor.framework.observability.metrics import (
    MetricSource,
    MetricsSnapshot,
    CounterMetric,
    GaugeMetric,
    HistogramMetric,
    MetricLabel,
)


class DatabaseMetrics(MetricSource):
    """Custom metrics source for database operations."""

    def __init__(self):
        self._query_count = 0
        self._connection_count = 0
        self._query_latencies = []
        self._error_count = 0

    @property
    def source_id(self) -> str:
        return "database"

    @property
    def source_type(self) -> str:
        return "custom"

    def record_query(self, latency_ms: float):
        """Record a database query."""
        self._query_count += 1
        self._query_latencies.append(latency_ms)

    def record_connection(self, count: int):
        """Record active connections."""
        self._connection_count = count

    def record_error(self):
        """Record a database error."""
        self._error_count += 1

    def get_metrics(self) -> MetricsSnapshot:
        """Get current database metrics."""

        # Calculate histogram from latencies
        if self._query_latencies:
            sorted_latencies = sorted(self._query_latencies)
            count = len(sorted_latencies)
            total = sum(sorted_latencies)

            # Create buckets: <10ms, <50ms, <100ms, <500ms, >=500ms
            bucket_counts = [
                sum(1 for l in sorted_latencies if l < 10),
                sum(1 for l in sorted_latencies if 10 <= l < 50),
                sum(1 for l in sorted_latencies if 50 <= l < 100),
                sum(1 for l in sorted_latencies if 100 <= l < 500),
                sum(1 for l in sorted_latencies if l >= 500),
            ]
            bucket_bounds = (10.0, 50.0, 100.0, 500.0)

            histogram = HistogramMetric(
                name="query_latency_ms",
                description="Database query latency in milliseconds",
                count=count,
                sum=float(total),
                bucket_counts=tuple(bucket_counts),
                bucket_bounds=bucket_bounds,
            )
        else:
            histogram = HistogramMetric(
                name="query_latency_ms",
                description="Database query latency in milliseconds",
                count=0,
                sum=0.0,
                bucket_counts=(0, 0, 0, 0, 0),
                bucket_bounds=(10.0, 50.0, 100.0, 500.0),
            )

        metrics = [
            CounterMetric(
                name="queries_total",
                description="Total number of database queries",
                value=self._query_count,
            ),
            GaugeMetric(
                name="active_connections",
                description="Number of active database connections",
                value=self._connection_count,
            ),
            CounterMetric(
                name="errors_total",
                description="Total number of database errors",
                value=self._error_count,
            ),
            histogram,
        ]

        return MetricsSnapshot(
            source_id=self.source_id,
            source_type=self.source_type,
            metrics=tuple(metrics),
        )


# Usage example
if __name__ == "__main__":
    # Create metrics source
    db_metrics = DatabaseMetrics()

    # Register with ObservabilityManager
    manager = ObservabilityManager.get_instance()
    manager.register_source(db_metrics)

    # Simulate database operations
    db_metrics.record_query(25.0)  # 25ms query
    db_metrics.record_query(150.0)  # 150ms query
    db_metrics.record_query(8.0)    # 8ms query
    db_metrics.record_connection(5)
    db_metrics.record_query(45.0)
    db_metrics.record_error()

    # Get dashboard data
    dashboard_data = manager.get_dashboard_data()

    # Print custom metrics
    print("Database Metrics:")
    print(f"  Queries: {db_metrics._query_count}")
    print(f"  Connections: {db_metrics._connection_count}")
    print(f"  Errors: {db_metrics._error_count}")

    # Get dashboard data with our custom metrics
    dashboard_data = manager.get_dashboard_data()
    print(f"\nTotal Sources: {dashboard_data.total_sources}")
    print(f"Sources by Type: {dashboard_data.sources_by_type}")
