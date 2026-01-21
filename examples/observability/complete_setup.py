#!/usr/bin/env python3
"""Complete observability setup example for Victor AI.

This example demonstrates:
1. OpenTelemetry tracing setup
2. Structured JSON logging
3. Prometheus metrics endpoint
4. Health check endpoints
5. Integration with all observability features

Usage:
    python complete_setup.py
"""

import asyncio
import os
import logging
from pathlib import Path

# Ensure we can import victor
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


async def main():
    """Run complete observability setup."""

    print("=" * 60)
    print("Victor AI 0.5.1 - Observability Setup Example")
    print("=" * 60)

    # ============================================================================
    # 1. Configuration
    # ============================================================================
    print("\n1. Configuring environment...")

    # Enable observability features
    os.environ["VICTOR_TELEMETRY_ENABLED"] = "true"
    os.environ["VICTOR_TELEMETRY_ENDPOINT"] = "http://localhost:4317"
    os.environ["VICTOR_TELEMETRY_SAMPLING"] = "1.0"
    os.environ["VICTOR_TELEMETRY_EXPORTER"] = "console"  # Use console for demo

    os.environ["VICTOR_LOG_FORMAT"] = "json"
    os.environ["VICTOR_LOG_LEVEL"] = "info"

    os.environ["VICTOR_METRICS_ENABLED"] = "true"
    os.environ["VICTOR_METRICS_PORT"] = "9090"

    print("   ✓ Telemetry: Enabled (console exporter)")
    print("   ✓ Logging: JSON format")
    print("   ✓ Metrics: Enabled on port 9090")

    # ============================================================================
    # 2. OpenTelemetry Setup
    # ============================================================================
    print("\n2. Setting up OpenTelemetry...")

    from victor.config.telemetry_config import setup_telemetry, get_telemetry_config

    config = get_telemetry_config()
    tracer, meter = setup_telemetry(config)

    if tracer:
        print("   ✓ Tracing initialized")
    else:
        print("   ✗ Tracing not available (install: pip install victor-ai[observability])")

    # ============================================================================
    # 3. Structured Logging Setup
    # ============================================================================
    print("\n3. Setting up structured logging...")

    from victor.observability.structured_logging import (
        setup_structured_logging,
        set_correlation_id,
    )

    logger, perf_logger, req_logger = setup_structured_logging(
        log_format="json",
        log_level="info",
        service_name="victor-example",
        environment="development",
    )

    print("   ✓ Structured logging initialized")

    # ============================================================================
    # 4. Metrics Setup
    # ============================================================================
    print("\n4. Setting up metrics collection...")

    from victor.observability.prometheus_metrics import get_prometheus_registry

    registry = get_prometheus_registry()

    # Create custom metrics
    request_counter = registry.counter(
        name="example_requests_total",
        help="Total example requests",
    )

    latency_histogram = registry.histogram(
        name="example_latency_ms",
        help="Example request latency",
        buckets=[10, 50, 100, 500, 1000],
    )

    print("   ✓ Metrics initialized")

    # ============================================================================
    # 5. Health Checks
    # ============================================================================
    print("\n5. Setting up health checks...")

    from victor.core.health import (
        HealthChecker,
        MemoryHealthCheck,
        create_default_health_checker,
    )

    health_checker = create_default_health_checker(
        version="0.5.1",
        include_memory=True,
    )

    print("   ✓ Health checker initialized")

    # ============================================================================
    # 6. Demonstration
    # ============================================================================
    print("\n6. Demonstrating observability features...")

    # Set correlation ID for request
    set_correlation_id("demo-request-123")
    print("\n   [Correlation ID: demo-request-123]")

    # Log structured message
    logger.info("Starting example operation", extra={"operation": "demo"})

    # Trace an operation
    if tracer:
        from victor.observability.telemetry import trace_function

        @trace_function(name="example.operation")
        async def example_operation():
            """Example operation with tracing."""
            logger.info("Inside traced operation")

            # Simulate work
            import time
            time.sleep(0.1)

            # Record metrics
            request_counter.inc()
            latency_histogram.observe(100)

            logger.info("Operation completed successfully")
            return "success"

        result = await example_operation()
        print(f"\n   Operation result: {result}")

    # Performance logging
    with perf_logger.track_operation("database_query", metadata={"table": "users"}):
        # Simulate query
        import time
        time.sleep(0.05)
        logger.info("Query executed")

    # Request logging
    req_logger.log_request(
        method="POST",
        path="/api/example",
        headers={"user-agent": "Mozilla/5.0"},
        correlation_id="demo-request-123",
    )

    # Simulate processing
    await asyncio.sleep(0.1)

    req_logger.log_response(
        status_code=200,
        duration_ms=100,
    )

    # Health check
    print("\n   Running health check...")
    health_report = await health_checker.check_health()
    print(f"   Health status: {health_report.status.value}")
    print(f"   Components checked: {len(health_report.components)}")

    # ============================================================================
    # 7. Metrics Export
    # ============================================================================
    print("\n7. Exporting metrics...")

    from victor.observability.prometheus_metrics import get_prometheus_exporter

    exporter = get_prometheus_exporter()
    metrics_text = exporter.export_metrics()

    print("   Prometheus metrics:")
    print("   " + "-" * 56)
    for line in metrics_text.split("\n")[:10]:  # Show first 10 lines
        print(f"   {line}")
    if len(metrics_text.split("\n")) > 10:
        print(f"   ... ({len(metrics_text.split('\n')) - 10} more lines)")

    # ============================================================================
    # 8. HTTP Endpoints (Optional)
    # ============================================================================
    print("\n8. HTTP endpoints available:")
    print("   - Metrics:    http://localhost:9090/metrics")
    print("   - Health:     http://localhost:9090/health")
    print("   - Readiness:  http://localhost:9090/ready")
    print("   - Detailed:   http://localhost:9090/health/detailed")
    print("\n   To start HTTP server:")
    print("   python -m victor.observability.metrics_endpoint")

    # ============================================================================
    # 9. Summary
    # ============================================================================
    print("\n" + "=" * 60)
    print("Observability Setup Complete!")
    print("=" * 60)
    print("\nFeatures enabled:")
    print("  ✓ OpenTelemetry tracing (console exporter)")
    print("  ✓ Structured JSON logging")
    print("  ✓ Prometheus metrics")
    print("  ✓ Health checks")
    print("\nNext steps:")
    print("  1. Configure OTLP exporter for production:")
    print("     export VICTOR_TELEMETRY_ENDPOINT=http://jaeger:4317")
    print("  2. Start metrics server:")
    print("     python -m victor.observability.metrics_endpoint")
    print("  3. View traces in Jaeger:")
    print("     http://localhost:16686")
    print("  4. Query metrics in Prometheus:")
    print("     http://localhost:9091")
    print("\nFor more information, see:")
    print("  docs/observability/README.md")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
