# Coordinator-Based Architecture: Usage Examples

**Version**: 1.0
**Date**: 2025-01-13
**Audience**: Developers, Advanced Users

---

## Table of Contents

1. [Example 1: Basic Chat with Coordinators](#example-1-basic-chat-with-coordinators)
2. [Example 2: Custom Config Provider](#example-2-custom-config-provider)
3. [Example 3: Custom Prompt Contributor](#example-3-custom-prompt-contributor)
4. [Example 4: Custom Compaction Strategy](#example-4-custom-compaction-strategy)
5. [Example 5: Custom Analytics Exporter](#example-5-custom-analytics-exporter)
6. [Example 6: Combining Multiple Coordinators](#example-6-combining-multiple-coordinators)
7. [Example 7: Context Management](#example-7-context-management)
8. [Example 8: Tool Selection Strategies](#example-8-tool-selection-strategies)
9. [Example 9: Session Management](#example-9-session-management)
10. [Example 10: Error Handling](#example-10-error-handling)

---

## Example 1: Basic Chat with Coordinators

### Scenario

Use the coordinator-based orchestrator for basic chat operations.

### Code

```python
import asyncio
from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings
from victor.providers.anthropic import AnthropicProvider

async def basic_chat_example():
    """Basic chat using coordinator-based orchestrator."""

    # Step 1: Initialize settings and provider
    settings = Settings()
    provider = AnthropicProvider(api_key="sk-ant-...")

    # Step 2: Create orchestrator
    # Coordinators are automatically initialized:
    # - ConfigCoordinator: loads configuration
    # - PromptCoordinator: builds prompts
    # - ContextCoordinator: manages context
    # - ChatCoordinator: handles chat
    # - ToolCoordinator: executes tools
    # - AnalyticsCoordinator: tracks analytics
    # ... (15 coordinators total)
    orchestrator = AgentOrchestrator(
        settings=settings,
        provider=provider,
        model="claude-sonnet-4-5"
    )

    # Step 3: Use the orchestrator
    response = await orchestrator.chat(
        "Explain what coordinators are in Victor."
    )

    # Step 4: Access response
    print(f"Response: {response.content}")
    print(f"Tokens used: {response.usage.total_tokens}")

    # Coordinators worked together:
    # 1. ConfigCoordinator loaded configuration
    # 2. PromptCoordinator built the system prompt
    # 3. ContextCoordinator managed conversation context
    # 4. ChatCoordinator executed the chat
    # 5. AnalyticsCoordinator tracked token usage

if __name__ == "__main__":
    asyncio.run(basic_chat_example())
```

### Output

```
Response: Coordinators in Victor are specialized components that each handle
one specific aspect of the orchestration process...
Tokens used: 150
```

### Key Takeaways

- Coordinators are automatically initialized when you create `AgentOrchestrator`
- You don't need to interact with coordinators directly for basic usage
- All 15 coordinators work together behind the scenes

---

## Example 2: Custom Config Provider

### Scenario

Load configuration from a custom source (e.g., database, API, file).

### Code

```python
import asyncio
from typing import Dict
from victor.protocols import IConfigProvider
from victor.agent.coordinators import ConfigCoordinator
from victor.agent.orchestrator import AgentOrchestrator

class DatabaseConfigProvider(IConfigProvider):
    """Load configuration from a database."""

    def __init__(self, db_connection):
        self.db = db_connection

    def priority(self) -> int:
        """Higher priority = checked first."""
        return 100

    async def get_config(self, session_id: str) -> Dict:
        """Fetch configuration from database."""
        query = "SELECT config FROM sessions WHERE session_id = $1"
        result = await self.db.fetchrow(query, session_id)

        if result:
            return result["config"]
        return {}  # Empty config = let next provider try

class APIConfigProvider(IConfigProvider):
    """Load configuration from a remote API."""

    def __init__(self, api_base_url: str, api_key: str):
        self.api_base_url = api_base_url
        self.api_key = api_key

    def priority(self) -> int:
        """Lower priority = fallback."""
        return 50

    async def get_config(self, session_id: str) -> Dict:
        """Fetch configuration from API."""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.api_base_url}/config/{session_id}",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            return response.json() if response.status_code == 200 else {}

async def custom_config_example():
    """Use custom configuration providers."""

    # Step 1: Create custom providers
    db_provider = DatabaseConfigProvider(db_connection=your_db)
    api_provider = APIConfigProvider(
        api_base_url="https://api.example.com",
        api_key="your-api-key"
    )

    # Step 2: Create coordinator with custom providers
    config_coordinator = ConfigCoordinator(providers=[
        db_provider,    # Try database first (priority 100)
        api_provider,   # Fall back to API (priority 50)
    ])

    # Step 3: Use with orchestrator
    orchestrator = AgentOrchestrator(
        settings=settings,
        provider=provider,
        model=model,
        _config_coordinator=config_coordinator  # Inject custom coordinator
    )

    # Step 4: Load configuration
    config = await config_coordinator.load_config(
        session_id="user-session-123"
    )

    print(f"Loaded config: {config}")
    # Config from database if available, otherwise from API

if __name__ == "__main__":
    asyncio.run(custom_config_example())
```

### Key Takeaways

- Create custom config providers by implementing `IConfigProvider`
- Use `priority()` to control provider order (higher = checked first)
- Providers are tried in order until one returns a valid config
- Useful for multi-tenant applications, A/B testing, feature flags

---

## Example 3: Custom Prompt Contributor

### Scenario

Add custom prompt building logic for domain-specific requirements.

### Code

```python
import asyncio
from victor.protocols import IPromptContributor, PromptContext
from victor.agent.coordinators.prompt_coordinator import BasePromptContributor
from victor.agent.orchestrator import AgentOrchestrator

class CompliancePromptContributor(BasePromptContributor):
    """Add compliance requirements to prompts."""

    def priority(self) -> int:
        return 80  # High priority

    async def get_contribution(self, context: PromptContext) -> str:
        """Add compliance instructions if needed."""
        if context.get("requires_compliance"):
            return """

## Compliance Requirements
- Follow all regulatory requirements (GDPR, HIPAA, etc.)
- Do not include personal identifiable information (PII)
- Ensure data handling meets security standards
"""
        return ""

class CodeStylePromptContributor(BasePromptContributor):
    """Add code style guidelines to prompts."""

    def __init__(self, style_guide: dict):
        self.style_guide = style_guide

    def priority(self) -> int:
        return 60  # Medium priority

    async def get_contribution(self, context: PromptContext) -> str:
        """Add code style instructions for coding tasks."""
        if context.get("task") == "code_generation":
            return f"""

## Code Style Guidelines
- Max line length: {self.style_guide['max_line_length']}
- Use {self.style_guide['indentation']} for indentation
- Follow {self.style_guide['naming_convention']} naming convention
- Type hints required for all functions
"""
        return ""

class MultilingualPromptContributor(BasePromptContributor):
    """Add language-specific instructions."""

    def priority(self) -> int:
        return 70  # Medium-high priority

    async def get_contribution(self, context: PromptContext) -> str:
        """Add language instructions based on user preference."""
        language = context.get("language", "en")

        instructions = {
            "en": "Respond in English.",
            "es": "Responde en español.",
            "fr": "Répondre en français.",
            "de": "Auf Deutsch antworten.",
            "ja": "日本語で回答してください。",
        }

        return f"\nLanguage: {instructions.get(language, instructions['en'])}"

async def custom_prompt_example():
    """Use custom prompt contributors."""

    # Step 1: Create custom contributors
    compliance_contributor = CompliancePromptContributor()
    code_style_contributor = CodeStylePromptContributor(style_guide={
        "max_line_length": 100,
        "indentation": "4 spaces",
        "naming_convention": "PEP8",
    })
    multilingual_contributor = MultilingualPromptContributor()

    # Step 2: Create prompt coordinator with custom contributors
    from victor.agent.coordinators import PromptCoordinator

    prompt_coordinator = PromptCoordinator(contributors=[
        compliance_contributor,
        code_style_contributor,
        multilingual_contributor,
    ])

    # Step 3: Use with orchestrator
    orchestrator = AgentOrchestrator(
        settings=settings,
        provider=provider,
        model=model,
        _prompt_coordinator=prompt_coordinator
    )

    # Step 4: Build custom prompt
    from victor.protocols import PromptContext

    context = PromptContext({
        "task": "code_generation",
        "requires_compliance": True,
        "language": "en",
    })

    prompt = await prompt_coordinator.build_system_prompt(context)

    print("Generated prompt:")
    print(prompt)
    # Contains:
    # - Base system prompt
    # - Compliance requirements
    # - Code style guidelines
    # - Language instruction

if __name__ == "__main__":
    asyncio.run(custom_prompt_example())
```

### Key Takeaways

- Create custom prompt contributors by extending `BasePromptContributor`
- Use `priority()` to control contribution order
- Contributors are called in priority order (highest first)
- Later contributors can override earlier ones
- Useful for domain-specific prompts, multilingual support, style guides

---

## Example 4: Custom Compaction Strategy

### Scenario

Implement a custom context compaction strategy for specific use cases.

### Code

```python
import asyncio
from victor.protocols import Context
from victor.agent.coordinators.context_coordinator import BaseCompactionStrategy
from victor.agent.orchestrator import AgentOrchestrator

class RecentMessagesCompactionStrategy(BaseCompactionStrategy):
    """Keep only the most recent N messages."""

    def __init__(self, keep_last_n: int = 10):
        self.keep_last_n = keep_last_n

    async def compact(self, context: Context) -> Context:
        """Compaction strategy: Keep only recent messages."""
        if len(context.messages) <= self.keep_last_n:
            return context  # No compaction needed

        # Keep system messages + recent messages
        system_messages = [m for m in context.messages if m.role == "system"]
        recent_messages = context.messages[-self.keep_last_n:]

        return Context(messages=system_messages + recent_messages)

class SemanticCompactionStrategy(BaseCompactionStrategy):
    """Use semantic similarity to keep important messages."""

    def __init__(self, embedding_service):
        self.embedding_service = embedding_service

    async def compact(self, context: Context) -> Context:
        """Compaction strategy: Keep semantically diverse messages."""
        if len(context.messages) <= 10:
            return context

        # Compute embeddings for all messages
        embeddings = await self._compute_embeddings(context.messages)

        # Select diverse messages using clustering
        selected_indices = await self._select_diverse_messages(
            context.messages,
            embeddings,
            target_count=10
        )

        selected_messages = [context.messages[i] for i in selected_indices]
        return Context(messages=selected_messages)

    async def _compute_embeddings(self, messages):
        # Simplified: use actual embedding service in practice
        return [self.embedding_service.embed(m.content) for m in messages]

    async def _select_diverse_messages(self, messages, embeddings, target_count):
        # Simplified: use k-means or similar in practice
        # For now, just return the last N messages
        return list(range(-target_count, 0))

class TaskAwareCompactionStrategy(BaseCompactionStrategy):
    """Keep messages relevant to the current task."""

    async def compact(self, context: Context) -> Context:
        """Compaction strategy: Keep task-relevant messages."""
        current_task = context.metadata.get("current_task")

        if not current_task:
            # No task identified, use default strategy
            return await self._default_compaction(context)

        # Keep task-relevant messages
        relevant_messages = [
            m for m in context.messages
            if self._is_relevant_to_task(m, current_task)
        ]

        # Always keep system messages
        system_messages = [m for m in context.messages if m.role == "system"]

        return Context(messages=system_messages + relevant_messages)

    def _is_relevant_to_task(self, message, task):
        """Check if message is relevant to the current task."""
        # Simplified: use keyword matching
        # In practice, use semantic similarity
        task_keywords = task.lower().split()
        message_content = message.content.lower()
        return any(keyword in message_content for keyword in task_keywords)

    async def _default_compaction(self, context):
        """Fallback to recent messages strategy."""
        if len(context.messages) <= 10:
            return context
        return Context(messages=context.messages[-10:])

async def custom_compaction_example():
    """Use custom compaction strategies."""

    # Step 1: Create custom compaction strategy
    recent_strategy = RecentMessagesCompactionStrategy(keep_last_n=15)

    # Step 2: Create context coordinator with custom strategy
    from victor.agent.coordinators import ContextCoordinator

    context_coordinator = ContextCoordinator(
        compaction_strategy=recent_strategy
    )

    # Step 3: Use with orchestrator
    orchestrator = AgentOrchestrator(
        settings=settings,
        provider=provider,
        model=model,
        _context_coordinator=context_coordinator
    )

    # Step 4: Use the orchestrator
    # Context will be automatically compacted when it exceeds thresholds
    response = await orchestrator.chat("Have a long conversation...")

if __name__ == "__main__":
    asyncio.run(custom_compaction_example())
```

### Key Takeaways

- Create custom compaction strategies by extending `BaseCompactionStrategy`
- Implement `compact()` to define your strategy
- Strategies can be: recent messages, semantic, task-aware, hybrid
- Useful for managing long conversations, reducing token usage

---

## Example 5: Custom Analytics Exporter

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
```

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

---

## Example 7: Context Management

### Scenario

Monitor and manage conversation context manually.

### Code

```python
import asyncio
from victor.agent.orchestrator import AgentOrchestrator

async def context_management_example():
    """Monitor and manage conversation context."""

    orchestrator = AgentOrchestrator(settings=settings, provider=provider, model=model)

    # Step 1: Chat and monitor context size
    print("Initial context size:",
          len(orchestrator._context_coordinator.get_context().messages))

    response1 = await orchestrator.chat("Tell me about Python")
    print("After chat 1:",
          len(orchestrator._context_coordinator.get_context().messages))

    response2 = await orchestrator.chat("What about JavaScript?")
    print("After chat 2:",
          len(orchestrator._context_coordinator.get_context().messages))

    # Step 2: Manually trigger compaction
    context = orchestrator._context_coordinator.get_context()
    print(f"Context size before compaction: {len(context.messages)}")

    compacted_context = await orchestrator._context_coordinator.compact_context(context)
    print(f"Context size after compaction: {len(compacted_context.messages)}")

    # Step 3: Reset context
    orchestrator._context_coordinator.reset()
    print("After reset:",
          len(orchestrator._context_coordinator.get_context().messages))

if __name__ == "__main__":
    asyncio.run(context_management_example())
```

### Key Takeaways

- Monitor context size with `get_context()`
- Manually trigger compaction with `compact_context()`
- Reset context with `reset()`
- Useful for managing long conversations, reducing costs

---

## Example 8: Tool Selection Strategies

### Scenario

Use different tool selection strategies.

### Code

```python
import asyncio
from victor.config.settings import Settings
from victor.agent.orchestrator import AgentOrchestrator

async def tool_selection_example():
    """Use different tool selection strategies."""

    # Keyword-based selection
    settings_keyword = Settings(
        tool_selection_strategy="keyword",
    )
    orchestrator_keyword = AgentOrchestrator(
        settings=settings_keyword,
        provider=provider,
        model=model
    )

    # Semantic-based selection
    settings_semantic = Settings(
        tool_selection_strategy="semantic",
    )
    orchestrator_semantic = AgentOrchestrator(
        settings=settings_semantic,
        provider=provider,
        model=model
    )

    # Hybrid selection (default)
    settings_hybrid = Settings(
        tool_selection_strategy="hybrid",
        tool_selection_hybrid_alpha=0.7,  # 70% semantic, 30% keyword
    )
    orchestrator_hybrid = AgentOrchestrator(
        settings=settings_hybrid,
        provider=provider,
        model=model
    )

    # Use the orchestrator
    response = await orchestrator_hybrid.chat("Read the file config.yaml")

if __name__ == "__main__":
    asyncio.run(tool_selection_example())
```

### Key Takeaways

- Three strategies: keyword, semantic, hybrid
- Configure via `tool_selection_strategy` setting
- Hybrid strategy combines both approaches
- Useful for optimizing tool selection accuracy

---

## Example 9: Session Management

### Scenario

Manage multiple sessions with different configurations.

### Code

```python
import asyncio
from victor.agent.orchestrator import AgentOrchestrator

async def session_management_example():
    """Manage multiple sessions."""

    # Create orchestrator
    orchestrator = AgentOrchestrator(settings=settings, provider=provider, model=model)

    # Session 1: Code review
    session1_id = "code-review-123"
    response1 = await orchestrator.chat(
        "Review this Python code...",
        session_id=session1_id
    )

    # Session 2: Documentation
    session2_id = "docs-456"
    response2 = await orchestrator.chat(
        "Write documentation for...",
        session_id=session2_id
    )

    # Get analytics for each session
    analytics1 = await orchestrator._analytics_coordinator.get_session_analytics(session1_id)
    analytics2 = await orchestrator._analytics_coordinator.get_session_analytics(session2_id)

    print(f"Session 1 events: {len(analytics1.events)}")
    print(f"Session 2 events: {len(analytics2.events)}")

    # Reset specific session
    orchestrator._session_coordinator.reset_session(session1_id)

if __name__ == "__main__":
    asyncio.run(session_management_example())
```

### Key Takeaways

- Use `session_id` parameter for multi-tenancy
- Each session has isolated context and analytics
- Reset individual sessions with `reset_session()`
- Useful for serving multiple users/contexts

---

## Example 10: Error Handling

### Scenario

Handle errors gracefully across coordinators.

### Code

```python
import asyncio
import logging
from victor.agent.orchestrator import AgentOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def error_handling_example():
    """Handle errors gracefully."""

    orchestrator = AgentOrchestrator(settings=settings, provider=provider, model=model)

    try:
        response = await orchestrator.chat("Hello!")
    except Exception as e:
        logger.error(f"Chat failed: {e}")

        # Check coordinator health
        try:
            config = orchestrator._config_coordinator.get_config()
            logger.info(f"ConfigCoordinator OK: {config}")
        except Exception as config_error:
            logger.error(f"ConfigCoordinator failed: {config_error}")

        try:
            context = orchestrator._context_coordinator.get_context()
            logger.info(f"ContextCoordinator OK: {len(context.messages)} messages")
        except Exception as context_error:
            logger.error(f"ContextCoordinator failed: {context_error}")

        # Fallback behavior
        logger.info("Using fallback configuration...")
        # Implement fallback logic

if __name__ == "__main__":
    asyncio.run(error_handling_example())
```

### Key Takeaways

- Wrap coordinator calls in try/except
- Check coordinator health individually
- Implement fallback behavior
- Log errors for debugging

---

## Summary

This guide provided 10 comprehensive examples covering:

1. **Basic chat** - Standard usage without direct coordinator access
2. **Custom config providers** - Load config from databases, APIs
3. **Custom prompt contributors** - Add domain-specific prompts
4. **Custom compaction strategies** - Control context management
5. **Custom analytics exporters** - Export to databases, webhooks, Prometheus
6. **Combining coordinators** - Use multiple custom coordinators together
7. **Context management** - Monitor and control conversation context
8. **Tool selection** - Use different selection strategies
9. **Session management** - Handle multiple isolated sessions
10. **Error handling** - Graceful error handling and recovery

### Next Steps

- [Migration Guide](../migration/orchestrator_refactoring_guide.md) - Migrate from legacy code
- [Recipes](../tutorials/coordinator_recipes.md) - Step-by-step solutions
- [Architecture](../architecture/coordinator_based_architecture.md) - Deep dive into design

---

**Document Version**: 1.0
**Last Updated**: 2025-01-13
**Next Review**: 2025-04-13

---

**End of Usage Examples**
