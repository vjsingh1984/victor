# Coordinator Examples - Part 3

**Part 3 of 3:** Examples 8-10 (Tool Selection, Session, Error Handling)

---

## Navigation

- [Part 1: Examples 1-4](part-1-examples-1-4.md)
- [Part 2: Examples 5-7](part-2-examples-5-7.md)
- **[Part 3: Examples 8-10](#)** (Current)
- [**Complete Guide**](../coordinator_examples.md)

---

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


**Reading Time:** 2 min
**Last Updated:** February 08, 2026**

---

## See Also

- [Documentation Home](../../README.md)


## Summary
