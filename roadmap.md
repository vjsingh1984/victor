# Roadmap

This roadmap is directional and may change based on community feedback.

## Horizons

| Horizon | Focus | Examples |
|---------|-------|----------|
| **0-3 months** | Stability + UX | Faster startup, clearer errors, smoother TUI |
| **3-6 months** | Workflow scale + Reliability | Better scheduling, richer observability, parallel execution, AT_LEAST_ONCE event delivery |
| **6-12 months** | Platform maturity | Distributed execution, plugin marketplace, enterprise controls |

## Recently delivered (v0.5.x)

### Framework Refactoring

- **Workflow Coordinators**: Split monolithic WorkflowEngine into focused coordinators (YAML, Graph, HITL, Cache) following SRP
- **Resilience Framework**: Unified retry strategies, circuit breakers, specialized handlers (NetworkRetryHandler, RateLimitRetryHandler, DatabaseRetryHandler)
- **Service Lifecycle Management**: ServiceLifecycleProtocol, ServiceManager, ServiceRegistry for unified service management
- **Validation Pipeline**: Generic ValidationPipeline with built-in validators (Threshold, Range, Pattern, Type, Length, Composite)
- **Health Checking**: HealthChecker, ComponentHealth, ProviderHealthCheck for system monitoring
- **Metrics Collection**: Counter, Gauge, Histogram, Timer, MetricsRegistry for observability
- **Configuration Capability**: ConfigurationCapabilityProvider replacing vertical-specific @capability decorators
- **Agent Components**: AgentBuilder, AgentSession, AgentBridge for flexible agent composition
- **Tool Configuration**: ToolConfigurator with filters (AirgappedFilter, CostTierFilter, SecurityFilter)
- **Tool Naming**: Canonical tool name enforcement to eliminate alias confusion

### Documentation

- **Migration Guide**: Comprehensive guide for migrating to new framework capabilities
- **Framework README**: Detailed API documentation for all framework modules
- **Updated CLAUDE.md**: Added new framework modules and canonical import locations

### Earlier Releases

- Provider switching with context independence
- Workflow DSL with graph execution
- Multi-agent team formations
- Provider/tool/vertical registries and validation
- Universal Registry System for cross-vertical entity management

## Upcoming Features

### Event System Reliability (3-6 months)

**AT_LEAST_ONCE Event Delivery with Ack/Nack**

Current implementation provides AT_MOST_ONCE delivery (fire-and-forget). Planned enhancements:

- **Ack/Nack Protocol**: Add `ack()` and `nack(requeue=True)` methods to `MessagingEvent`
- **Pending Event Tracking**: Backend keeps events in `_pending_events` until ACK'd
- **Automatic Retry**: Re-deliver events on NACK or timeout (with exponential backoff)
- **Dead Letter Queue (DLQ)**: Move events to DLQ after max retry attempts
- **Delivery Count Tracking**: Prevent infinite retry loops with `_max_delivery_count`

**Per-Subscriber Queues (Fan-out Model)**

Current dispatcher uses shared queue with pattern matching. Planned enhancement:

- **Independent Queues**: Each subscriber gets their own queue (isolated failures)
- **Exchange Pattern**: Backend copies events to all matching subscriber queues
- **Backpressure per Subscriber**: Slow subscribers don't block fast ones
- **Offline Subscriber Support**: Messages queue up for offline subscribers

**Delivery Guarantees**

```python
from victor.core.events import DeliveryGuarantee

# Current (AT_MOST_ONCE)
backend = InMemoryEventBackend()
await backend.publish(event)  # Fire and forget

# Planned (AT_LEAST_ONCE)
backend = InMemoryEventBackend(delivery_guarantee=DeliveryGuarantee.AT_LEAST_ONCE)

async def handler(event):
    try:
        await process(event)
        await event.ack()  # Remove from queue
    except Exception:
        await event.nack(requeue=True)  # Retry
```

**Architecture Split**

| Component | Responsibility |
|-----------|---------------|
| **Backend** | Queue storage, persistence, transport protocol |
| **EventBus** | Ack/Nack tracking, retry logic, DLQ, circuit breaker |

**Best Practices from AMQP/Kafka**

- RabbitMQ-style basic ACK/NACK
- Kafka-style consumer group offsets
- Dead letter exchange for failed events
- Circuit breaker for failing handlers
- Metrics: delivery rate, ack rate, nack rate, DLQ size

## How to influence the roadmap

- Open an issue or discussion with a concrete use case.
- Submit a [FEP](feps/fep-0000-template.md) for framework-level changes.
