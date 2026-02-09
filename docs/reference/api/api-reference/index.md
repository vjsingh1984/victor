# Victor AI 0.5.0 API Reference

Complete API reference documentation for Victor AI's public interfaces.

---

## Quick Summary

Victor AI's API consists of:
- **AgentOrchestrator** - Main facade for all agent operations
- **Coordinators** - 20 specialized coordinators for different concerns
- **Intelligent Pipeline** - Tool selection, execution, and response processing
- **Provider Management** - Provider lifecycle and switching
- **Tool System** - Tool execution and budgeting
- **Conversation Management** - State and history tracking
- **Streaming** - Real-time response streaming
- **Vertical Base** - Base class for verticals

---

## Reference Parts

### [Part 1: Orchestrator, Coordinators, Pipeline](part-1-orchestrator-coordinators-pipeline.md)
- AgentOrchestrator (Facade)
- Coordinators (20 specialized coordinators)
- Intelligent Pipeline

### [Part 2: Provider, Tools, Conversation, Streaming](part-2-provider-tools-conversation-streaming.md)
- Provider Management
- Tool System
- Conversation Management
- Streaming
- Vertical Base

---

## Quick Start

```python
from victor.agent import AgentOrchestrator
from victor.config.settings import Settings

# Create orchestrator
settings = Settings()
orchestrator = AgentOrchestrator(settings=settings)

# Chat
response = await orchestrator.chat(
    messages=[{"role": "user", "content": "Hello!"}]
)

# Stream
async for chunk in orchestrator.stream_chat(
    messages=[{"role": "user", "content": "Hello!"}]
):
    print(chunk.content, end="")
```text

---

## Related Documentation

- [Architecture Overview](../../architecture/README.md)
- [Provider Reference](./PROVIDER_REFERENCE.md)
- [Tools API](./internals/tools-api.md)

---

**Last Updated:** February 01, 2026
**Reading Time:** 12 min (all parts)
