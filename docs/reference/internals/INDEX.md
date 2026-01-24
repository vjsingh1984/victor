# Internal API Reference

Developer documentation for Victor's internal protocols, components, and APIs.

## Quick Links

| Document | Description | Lines |
|----------|-------------|-------|
| [Protocols API](protocols-api.md) | Protocol interfaces (ISP-compliant) | 1,371 |
| [Providers API](providers-api.md) | LLM provider implementations | 1,218 |
| [Tools API](tools-api.md) | Tool creation and registration | 1,010 |
| [Workflows API](workflows-api.md) | Workflow system internals | 1,291 |

## Overview

Victor's internal architecture follows SOLID principles with:
- **98 protocols** for loose coupling and testability
- **Protocol-based design** for dependency inversion
- **Interface segregation** for focused, composable interfaces

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

**[← Back to Reference](../index.md)**

**Internal APIs**

*Developer documentation for Victor internals*

</div>
