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

"""Victor Observability - Unified event bus and telemetry.

This module provides a centralized event bus for all Victor observations,
implementing the Pub/Sub pattern for decoupled event handling.

Architecture:
    EventBus (Singleton)
        │
        ├── Subscribers (Observer Pattern)
        │   ├── LoggingSubscriber
        │   ├── MetricsSubscriber
        │   └── Custom handlers
        │
        └── Exporters (Strategy Pattern)
            ├── JsonLineExporter
            ├── CallbackExporter
            └── OpenTelemetryExporter

Example:
    from victor.observability import EventBus, VictorEvent, EventCategory

    # Get singleton bus
    bus = EventBus.get_instance()

    # Subscribe to events
    def on_tool_event(event: VictorEvent):
        print(f"Tool {event.name}: {event.data}")

    bus.subscribe(EventCategory.TOOL, on_tool_event)

    # Publish events
    bus.publish(VictorEvent(
        category=EventCategory.TOOL,
        name="read",
        data={"path": "/tmp/test.txt"}
    ))

    # Add exporters for persistence
    bus.add_exporter(JsonLineExporter("events.jsonl"))
"""

from victor.observability.event_bus import (
    EventBus,
    EventCategory,
    EventPriority,
    VictorEvent,
)
from victor.observability.exporters import (
    BaseExporter,
    BufferedExporter,
    CallbackExporter,
    CompositeExporter,
    FilteringExporter,
    JsonLineExporter,
)
from victor.observability.hooks import (
    LoggingHook,
    MetricsHook,
    StateHookManager,
    StateTransitionHook,
    TransitionHistory,
    TransitionRecord,
)
from victor.observability.integration import (
    ObservabilityIntegration,
    ToolEventMiddleware,
    setup_observability,
)
from victor.observability.metrics import (
    Counter,
    Gauge,
    Histogram,
    Metric,
    MetricsCollector,
    MetricsRegistry,
    Timer,
)
from victor.observability.resilience import (
    BackoffStrategy,
    Bulkhead,
    BulkheadFullError,
    CircuitBreaker,
    CircuitBreakerError,
    ConstantBackoff,
    DecorrelatedJitterBackoff,
    ExponentialBackoff,
    LinearBackoff,
    RateLimiter,
    ResiliencePolicy,
    ObservabilityRetryConfig,
    retry_with_backoff,
    with_timeout,
)
from victor.observability.otel_exporter import (
    AsyncBatchingExporter,
    create_otel_exporter,
)
from victor.observability.cqrs_adapter import (
    AdapterConfig,
    CQRSEventAdapter,
    EventDirection,
    EventMappingRule,
    UnifiedEventBridge,
    create_unified_bridge,
)

# Conditional OTEL exports
try:
    from victor.observability.otel_exporter import (
        OpenTelemetryExporter,
        OTELSpanManager,
    )

    _OTEL_EXPORTS = ["OpenTelemetryExporter", "OTELSpanManager"]
except ImportError:
    _OTEL_EXPORTS = []

__all__ = [
    # Core
    "EventBus",
    "VictorEvent",
    "EventCategory",
    "EventPriority",
    # Exporters
    "BaseExporter",
    "JsonLineExporter",
    "CallbackExporter",
    "CompositeExporter",
    "FilteringExporter",
    "BufferedExporter",
    "AsyncBatchingExporter",
    "create_otel_exporter",
    # Hooks
    "StateHookManager",
    "StateTransitionHook",
    "TransitionRecord",
    "TransitionHistory",
    "LoggingHook",
    "MetricsHook",
    # Integration
    "ObservabilityIntegration",
    "ToolEventMiddleware",
    "setup_observability",
    # CQRS Adapter
    "AdapterConfig",
    "CQRSEventAdapter",
    "EventDirection",
    "EventMappingRule",
    "UnifiedEventBridge",
    "create_unified_bridge",
    # Metrics
    "Metric",
    "Counter",
    "Gauge",
    "Histogram",
    "Timer",
    "MetricsRegistry",
    "MetricsCollector",
    # Resilience
    "BackoffStrategy",
    "ExponentialBackoff",
    "LinearBackoff",
    "ConstantBackoff",
    "DecorrelatedJitterBackoff",
    "ObservabilityRetryConfig",
    "retry_with_backoff",
    "CircuitBreaker",
    "CircuitBreakerError",
    "Bulkhead",
    "BulkheadFullError",
    "RateLimiter",
    "ResiliencePolicy",
    "with_timeout",
] + _OTEL_EXPORTS

# Submodules - access via victor.observability.analytics, etc.
from victor.observability import analytics
from victor.observability import debug
from victor.observability import profiler
from victor.observability import telemetry
from victor.observability import pipeline

__version__ = "0.2.0"
