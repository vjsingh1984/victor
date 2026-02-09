# Coordinator Examples - Part 1

**Part 1 of 3:** Examples 1-4 (Basic Chat through Custom Compaction)

---

## Navigation

- **[Part 1: Examples 1-4](#)** (Current)
- [Part 2: Examples 5-7](part-2-examples-5-7.md)
- [Part 3: Examples 8-10](part-3-examples-8-10.md)
- [**Complete Guide**](../coordinator_examples.md)

---
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
```text

### Output

```
Response: Coordinators in Victor are specialized components that each handle
one specific aspect of the orchestration process...
Tokens used: 150
```text

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
```text

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


**Reading Time:** 7 min
**Last Updated:** February 08, 2026**

---

## See Also

- [Documentation Home](../../README.md)


## Example 5: Custom Analytics Exporter
