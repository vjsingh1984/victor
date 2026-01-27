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

"""OpenTelemetry setup and configuration."""

try:
    from opentelemetry import trace, metrics  # type: ignore[import-untyped]
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-untyped]
    from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore[import-untyped]
    from opentelemetry.sdk.metrics import MeterProvider  # type: ignore[import-untyped]
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader  # type: ignore[import-untyped]
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter  # type: ignore[import-untyped]
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter  # type: ignore[import-untyped]
    from opentelemetry.sdk.resources import Resource  # type: ignore[import-untyped]

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    # Optional dependency, skip if not available
    OPENTELEMETRY_AVAILABLE = False

    # Create dummy objects to avoid import errors
    class TracerProvider:
        def __init__(self, *args, **kwargs):
            pass

    class BatchSpanProcessor:
        def __init__(self, *args, **kwargs):
            pass

    class MeterProvider:
        def __init__(self, *args, **kwargs):
            pass

    class PeriodicExportingMetricReader:
        def __init__(self, *args, **kwargs):
            pass

    class OTLPSpanExporter:
        def __init__(self, *args, **kwargs):
            pass

    class OTLPMetricExporter:
        def __init__(self, *args, **kwargs):
            pass

    class Resource:
        @staticmethod
        def create(*args, **kwargs):
            return {}


def setup_opentelemetry(service_name: str, service_version: str):
    """Initializes OpenTelemetry tracing and metrics."""
    if not OPENTELEMETRY_AVAILABLE:
        return None, None

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": service_version,
        }
    )

    # Tracing
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

    # Metrics
    metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter())
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    return trace.get_tracer(__name__), metrics.get_meter(__name__)
