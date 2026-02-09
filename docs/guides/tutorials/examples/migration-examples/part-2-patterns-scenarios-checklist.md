# Coordinator-Based Architecture: Migration Examples - Part 2

**Part 2 of 2:** Common Migration Patterns, Real-World Scenarios, Migration Checklist, Summary

---

## Navigation

- [Part 1: Examples 1-10](part-1-examples-1-10.md)
- **[Part 2: Patterns, Scenarios, Checklist](#)** (Current)
- [**Complete Guide](../migration_examples.md)**

---

## Common Migration Patterns

### Pattern 1: Private Attribute Access

**Before**:
```python
# Direct access to private attributes
orchestrator._enabled_tools = tools
orchestrator._middleware = middleware
```text

**After**:
```python
# Use coordinators
await orchestrator.tool_coordinator.set_tools(tools)
await orchestrator.middleware_coordinator.set_middleware(middleware)
```

### Pattern 2: Direct Provider Instantiation

**Before**:
```python
from victor.providers.anthropic import AnthropicProvider

provider = AnthropicProvider(api_key="...")
response = provider.chat(messages)
```text

**After**:
```python
# Use ProviderCoordinator
provider = await orchestrator.provider_coordinator.get_provider()
response = await provider.chat(messages)
```

### Pattern 3: Manual State Management

**Before**:
```python
# Manual state tracking
state = {
    "messages": [],
    "context": {},
    "stage": "initial"
}
```text

**After**:
```python
# Use StateCoordinator
state = await orchestrator.state_coordinator.get_state()
await orchestrator.state_coordinator.update_stage("executing")
```

---

## Real-World Migration Scenarios

### Scenario 1: Migrating a Chat Application

**Challenge**: Application uses direct provider access.

**Solution**:
```python
# Before
from victor.providers.anthropic import AnthropicProvider

class ChatApp:
    def __init__(self):
        self.provider = AnthropicProvider(api_key=settings.key)

    async def send_message(self, message: str):
        response = self.provider.chat([{"role": "user", "content": message}])
        return response.content

# After
from victor.agent import AgentOrchestrator

class ChatApp:
    def __init__(self):
        self.orchestrator = AgentOrchestrator(settings=settings)

    async def send_message(self, message: str):
        response = await self.orchestrator.chat(
            messages=[{"role": "user", "content": message}]
        )
        return response.content
```text

### Scenario 2: Migrating Tool Execution

**Challenge**: Application directly calls tools.

**Solution**:
```python
# Before
from victor.tools import read_file, write_file

content = read_file("file.txt")
modified = process(content)
write_file("file.txt", modified)

# After
from victor.agent import AgentOrchestrator

orchestrator = AgentOrchestrator()

result = await orchestrator.tool_coordinator.execute_tool(
    "read",
    path="file.txt"
)

# Process result
modified = process(result.data)

# Write back
await orchestrator.tool_coordinator.execute_tool(
    "write",
    path="file.txt",
    content=modified
)
```

---

## Migration Checklist

### Phase 1: Assessment
- [ ] Identify all orchestrator usage in codebase
- [ ] List all private attribute accesses
- [ ] Document current behavior
- [ ] Plan migration strategy

### Phase 2: Migration
- [ ] Update imports (if needed)
- [ ] Replace private attribute access with coordinator calls
- [ ] Update tests
- [ ] Verify functionality

### Phase 3: Validation
- [ ] Run integration tests
- [ ] Perform manual testing
- [ ] Monitor for issues
- [ ] Update documentation

---

## Summary

Migrating to coordinator-based architecture provides:
- **Better testability**: Test coordinators in isolation
- **Clearer separation**: Each coordinator has single responsibility
- **Easier extension**: Add new functionality via new coordinators
- **Backward compatibility**: Existing code continues to work

**Key Takeaways**:
1. Most code requires no changes
2. Changes are only needed for direct internal access
3. Migration can be gradual
4. Use coordinators for all new code

**Next Steps**:
- Explore [Coordinator Guide](../../guides/coordinators/)
- Review [Architecture Overview](../../architecture/README.md)
- Check [Best Practices](../../architecture/BEST_PRACTICES.md)

---

**Reading Time:** 2 min
**Last Updated:** January 13, 2025
