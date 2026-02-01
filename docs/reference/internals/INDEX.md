# Internal API Reference

Developer documentation for Victor's internal protocols, components, and APIs. These documents are internal snapshots and may lag behind code changes.

## Quick Links

| Document | Description |
|----------|-------------|
| [Protocols API](protocols-api.md) | Protocol interfaces |
| [Providers API](providers-api.md) | Provider implementations |
| [Tools API](tools-api.md) | Tool creation and registration |
| [Workflows API](workflows-api.md) | Workflow system internals |

## Overview

Victor's internal architecture emphasizes protocol-based design, dependency inversion, and interface segregation. Verify specifics against the current codebase when making changes.

## Protocol Categories

### Core Protocols
- `ProviderProtocol` - LLM provider interface
- `ToolProtocol` - Tool interface
- `ToolExecutorProtocol` - Tool execution
- `ChatCoordinatorProtocol` - Chat coordination

### Agent Protocols
- `AgentProtocol` - Agent interface
- `TeamMemberProtocol` - Team member interface
- `TeamCoordinatorProtocol` - Team coordination

### Infrastructure Protocols
- `CacheProtocol` - Caching operations
- `MetricsProtocol` - Metrics collection
- `EventBusProtocol` - Event handling

## Usage

```python
from victor.agent.protocols import ToolExecutorProtocol
from victor.core.container import ServiceContainer

container = ServiceContainer()
executor = container.get(ToolExecutorProtocol)

result = await executor.execute_tool(tool, arguments)
```

## See Also

- [Architecture Overview](../../architecture/overview.md)
- [Developer Guide](../../development/CONTRIBUTING.md)
- [API Reference](../api.md)

---

<div align="center">

**[<- Back to Reference](../index.md)**

**Internal APIs**

*Developer documentation for Victor internals*

</div>

---

**Last Updated:** February 01, 2026
**Reading Time:** 1 min
