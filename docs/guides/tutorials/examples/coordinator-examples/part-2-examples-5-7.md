# Coordinator Examples - Part 2

**Part 2 of 3:** Examples 5-7 (Analytics, Combined Coordinators, Context)

---

## Navigation

- [Part 1: Examples 1-4](part-1-examples-1-4.md)
- **[Part 2: Examples 5-7](#)** (Current)
- [Part 3: Examples 8-10](part-3-examples-8-10.md)
- [**Complete Guide**](../coordinator_examples.md)

---

### Scenario

Export analytics to custom destinations (database, API, monitoring system).

### Code

```python
import asyncio
from typing import List
from victor.protocols import IAnalyticsExporter, ExportResult, AnalyticsEvent
from victor.agent.coordinators.analytics_coordinator import BaseAnalyticsExporter
from victor.agent.orchestrator import AgentOrchestrator

class DatabaseAnalyticsExporter(BaseAnalyticsExporter):
    """Export analytics to a database."""

    def __init__(self, db_connection):
        self.db = db_connection

    async def export(self, events: List[AnalyticsEvent]) -> ExportResult:
        """Export events to database."""
        try:
            # Batch insert events
            query = """
                INSERT INTO analytics (session_id, event_type, event_data, timestamp)
                VALUES ($1, $2, $3, $4)
            """

            records = [
                (e.session_id, e.type, e.data, e.timestamp)
                for e in events
            ]

            await self.db.executemany(query, records)

            return ExportResult(
                success=True,
                exported_count=len(events)
            )
        except Exception as e:
            return ExportResult(
                success=False,
                error=str(e)
            )

class WebhookAnalyticsExporter(BaseAnalyticsExporter):
    """Export analytics to a webhook endpoint."""

    def __init__(self, webhook_url: str, headers: dict = None):
        self.webhook_url = webhook_url
        self.headers = headers or {}

    async def export(self, events: List[AnalyticsEvent]) -> ExportResult:
        """Send events to webhook endpoint."""
        import httpx

        try:
            payload = [e.model_dump() for e in events]

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()

            return ExportResult(
                success=True,
                exported_count=len(events)
            )
        except Exception as e:
            return ExportResult(
                success=False,
                error=str(e)
            )

class PrometheusMetricsExporter(BaseAnalyticsExporter):
    """Export analytics as Prometheus metrics."""

    def __init__(self, pushgateway_url: str, job_name: str):
        self.pushgateway_url = pushgateway_url
        self.job_name = job_name

    async def export(self, events: List[AnalyticsEvent]) -> ExportResult:
        """Export metrics to Prometheus Pushgateway."""
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

        try:
            registry = CollectorRegistry()

            # Create metrics from events
            event_types = {}
            for event in events:
                event_types[event.type] = event_types.get(event.type, 0) + 1

            # Create gauge for each event type
            for event_type, count in event_types.items():
                gauge = Gauge(
                    f'victor_analytics_{event_type}',
                    f'Number of {event_type} events',
                    registry=registry
                )
                gauge.set(count)

            # Push to Prometheus
            push_to_gateway(
                self.pushgateway_url,
                job=self.job_name,
                registry=registry
            )

            return ExportResult(
                success=True,
                exported_count=len(events)
            )
        except Exception as e:
            return ExportResult(
                success=False,
                error=str(e)
            )

async def custom_analytics_example():
    """Use custom analytics exporters."""

    # Step 1: Create custom exporters
    db_exporter = DatabaseAnalyticsExporter(db_connection=your_db)
    webhook_exporter = WebhookAnalyticsExporter(
        webhook_url="https://hooks.example.com/analytics",
        headers={"Authorization": "Bearer your-token"}
    )
    prometheus_exporter = PrometheusMetricsExporter(
        pushgateway_url="http://localhost:9091",
        job_name="victor-analytics"
    )

    # Step 2: Create analytics coordinator with custom exporters
    from victor.agent.coordinators import AnalyticsCoordinator

    analytics_coordinator = AnalyticsCoordinator(exporters=[
        db_exporter,       # Export to database
        webhook_exporter,  # Send to webhook
        prometheus_exporter,  # Export to Prometheus
    ])

    # Step 3: Use with orchestrator
    orchestrator = AgentOrchestrator(
        settings=settings,
        provider=provider,
        model=model,
        _analytics_coordinator=analytics_coordinator
    )

    # Step 4: Use the orchestrator
    # Analytics will be automatically tracked and exported
    response = await orchestrator.chat("Hello!")

    # Step 5: Manually export analytics (optional)
    result = await analytics_coordinator.export_analytics(
        session_id=orchestrator.session_id
    )

    print(f"Exported {result.exported_count} events")

if __name__ == "__main__":
    asyncio.run(custom_analytics_example())
```text

### Key Takeaways

- Create custom analytics exporters by extending `BaseAnalyticsExporter`
- Exporters run in parallel for efficiency
- Can export to multiple destinations simultaneously
- Useful for monitoring, observability, data pipelines

---

## Example 6: Combining Multiple Coordinators

### Scenario

Combine multiple custom coordinators for advanced use cases.

### Code

```python
import asyncio
from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.coordinators import (
    ConfigCoordinator,
    PromptCoordinator,
    ContextCoordinator,
    AnalyticsCoordinator,
)
from victor.config.settings import Settings

async def combined_coordinators_example():
    """Combine multiple custom coordinators."""

    # Step 1: Create custom config coordinator
    config_coordinator = ConfigCoordinator(providers=[
        DatabaseConfigProvider(your_db),
        EnvironmentConfigProvider(),
    ])

    # Step 2: Create custom prompt coordinator
    prompt_coordinator = PromptCoordinator(contributors=[
        CompliancePromptContributor(),
        CodeStylePromptContributor(style_guide),
        MultilingualPromptContributor(),
    ])

    # Step 3: Create custom context coordinator
    context_coordinator = ContextCoordinator(
        compaction_strategy=RecentMessagesCompactionStrategy(keep_last_n=15)
    )

    # Step 4: Create custom analytics coordinator
    analytics_coordinator = AnalyticsCoordinator(exporters=[
        DatabaseAnalyticsExporter(your_db),
        PrometheusMetricsExporter(pushgateway_url, job_name),
    ])

    # Step 5: Create orchestrator with all custom coordinators
    orchestrator = AgentOrchestrator(
        settings=Settings(
            enable_analytics=True,
            context_compaction_threshold=0.8,
        ),
        provider=provider,
        model=model,
        _config_coordinator=config_coordinator,
        _prompt_coordinator=prompt_coordinator,
        _context_coordinator=context_coordinator,
        _analytics_coordinator=analytics_coordinator,
    )

    # Step 6: Use the orchestrator
    # All custom coordinators work together
    response = await orchestrator.chat("Generate Python code...")

    print(response.content)

if __name__ == "__main__":
    asyncio.run(combined_coordinators_example())
```

### Key Takeaways

- You can combine multiple custom coordinators
- Each coordinator operates independently
- Coordinators work together through the orchestrator facade
- Useful for complex, multi-dimensional customization


**Reading Time:** 3 min
**Last Updated:** February 08, 2026**

---

## See Also

- [Documentation Home](../../README.md)


## Example 7: Context Management
