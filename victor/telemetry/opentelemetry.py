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
    from opentelemetry import trace, metrics  # type: ignore[import]
    from opentelemetry.sdk.trace import TracerProvider as _TracerProvider  # type: ignore[import]
    from opentelemetry.sdk.trace.export import BatchSpanProcessor as _BatchSpanProcessor  # type: ignore[import]
    from opentelemetry.sdk.metrics import MeterProvider as _MeterProvider  # type: ignore[import]
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader as _PeriodicExportingMetricReader  # type: ignore[import]
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as _OTLPSpanExporter  # type: ignore[import]
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter as _OTLPMetricExporter  # type: ignore[import]
    from opentelemetry.sdk.resources import Resource as _Resource  # type: ignore[import]

    TracerProvider = _TracerProvider
    BatchSpanProcessor = _BatchSpanProcessor
    MeterProvider = _MeterProvider
    PeriodicExportingMetricReader = _PeriodicExportingMetricReader
    OTLPSpanExporter = _OTLPSpanExporter
    OTLPMetricExporter = _OTLPMetricExporter
    Resource = _Resource

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    # Optional dependency, skip if not available
    OPENTELEMETRY_AVAILABLE = False

    # Create dummy objects to avoid import errors
    class TracerProvider:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

    class BatchSpanProcessor:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

    class MeterProvider:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

    class PeriodicExportingMetricReader:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

    class OTLPSpanExporter:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

    class OTLPMetricExporter:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

    class Resource:  # type: ignore[no-redef]
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
