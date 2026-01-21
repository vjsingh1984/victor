# Breaking Changes Checklist: Victor 0.5.x to 1.0.0

This document provides a comprehensive checklist of all breaking changes in Victor 1.0.0.

## Severity Levels

- **Critical**: Requires immediate attention, code will not work without changes
- **Moderate**: Important but may not immediately break all code
- **Minor**: Low impact, deprecation warnings or minor changes

---

## Critical Breaking Changes

### 1. Orchestrator Initialization [CRITICAL]

**Severity**: Critical
**Affected Component**: Core orchestrator creation
**Impact**: All code that creates orchestrators directly

**Change**:
```python
# Before (0.5.x)
from victor.agent.orchestrator import AgentOrchestrator
orchestrator = AgentOrchestrator(provider=OpenAIProvider(api_key="..."))

# After (1.0.0)
from victor.core.bootstrap import bootstrap_orchestrator
orchestrator = bootstrap_orchestrator(Settings())
```

**Migration Steps**:
1. Replace all `AgentOrchestrator()` instantiations with `bootstrap_orchestrator()`
2. Move provider configuration to `Settings` or environment variables
3. Update all import statements

**Validation**:
```bash
# Search for direct instantiation
grep -r "AgentOrchestrator(" --include="*.py" .
```

---

### 2. Provider API Key Handling [CRITICAL]

**Severity**: Critical
**Affected Component**: All provider instantiations
**Impact**: All code that passes API keys to providers

**Change**:
```python
# Before (0.5.x)
provider = OpenAIProvider(api_key="sk-...")

# After (1.0.0)
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
provider = OpenAIProvider()
```

**Migration Steps**:
1. Remove all `api_key` parameters from provider constructors
2. Move API keys to environment variables
3. Update configuration files

**Validation**:
```bash
# Search for API key parameters
grep -r "api_key=" --include="*.py" .
```

---

### 3. Tool Registry Access [CRITICAL]

**Severity**: Critical
**Affected Component**: Tool registry usage
**Impact**: All code accessing tool registry

**Change**:
```python
# Before (0.5.x)
from victor.tools.registry import SharedToolRegistry
registry = SharedToolRegistry.get_instance()

# After (1.0.0)
from victor.core.container import ServiceContainer
from victor.protocols import ToolRegistryProtocol
container = ServiceContainer()
registry = container.get(ToolRegistryProtocol)
```

**Migration Steps**:
1. Replace all `SharedToolRegistry.get_instance()` calls
2. Use DI container to resolve `ToolRegistryProtocol`
3. Update all tool registry access patterns

**Validation**:
```bash
# Search for singleton access
grep -r "SharedToolRegistry" --include="*.py" .
```

---

### 4. Event Bus Usage [CRITICAL]

**Severity**: Critical
**Affected Component**: Event publishing and subscription
**Impact**: All code using EventBus

**Change**:
```python
# Before (0.5.x)
from victor.observability.event_bus import EventBus
bus = EventBus()
bus.publish("tool.complete", data={...})

# After (1.0.0)
from victor.core.events import create_event_backend, MessagingEvent
backend = create_event_backend()
await backend.publish(MessagingEvent(topic="tool.complete", data={...}))
```

**Migration Steps**:
1. Replace `EventBus` with `create_event_backend()`
2. Use `MessagingEvent` wrapper for events
3. Make all publish/subscribe calls async
4. Update event handlers to be async

**Validation**:
```bash
# Search for EventBus usage
grep -r "EventBus" --include="*.py" .
```

---

### 5. Async Chat API [CRITICAL]

**Severity**: Critical
**Affected Component**: Orchestrator chat method
**Impact**: All code calling `orchestrator.chat()`

**Change**:
```python
# Before (0.5.x)
result = orchestrator.chat("Hello")

# After (1.0.0)
result = await orchestrator.chat("Hello")
```

**Migration Steps**:
1. Make all chat calls async
2. Add `await` keyword to all `orchestrator.chat()` calls
3. Update function signatures to be async

**Validation**:
```bash
# Search for sync chat calls
grep -r "orchestrator.chat(" --include="*.py" .
```

---

## Moderate Breaking Changes

### 6. Workflow Execution [MODERATE]

**Severity**: Moderate
**Affected Component**: Workflow execution API
**Impact**: All code using workflows

**Change**:
```python
# Before (0.5.x)
from victor.workflows.executor import WorkflowExecutor
executor = WorkflowExecutor(orchestrator)
result = await executor.execute(workflow, context)

# After (1.0.0)
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler
compiler = UnifiedWorkflowCompiler(orchestrator)
compiled = compiler.compile(workflow)
result = await compiled.invoke(context)
```

**Migration Steps**:
1. Replace `WorkflowExecutor` with `UnifiedWorkflowCompiler`
2. Use `compile()` + `invoke()` pattern
3. Update workflow definitions to YAML format

---

### 7. Tool Selection [MODERATE]

**Severity**: Moderate
**Affected Component**: Tool selection API
**Impact**: All code using tool selection

**Change**:
```python
# Before (0.5.x)
from victor.agent.tool_selector import ToolSelector
selector = ToolSelector(strategy="keyword")
tools = selector.select_tools(query, context)

# After (1.0.0)
from victor.protocols import IToolSelector
selector = container.get(IToolSelector)
tools = await selector.select_tools(query, context)
```

**Migration Steps**:
1. Use protocol-based dependency injection
2. Make tool selection calls async
3. Configure strategy via settings

---

### 8. Conversation State Machine [MODERATE]

**Severity**: Moderate
**Affected Component**: State machine API
**Impact**: All code using state machine

**Change**:
```python
# Before (0.5.x)
from victor.agent.state_machine import ConversationStateMachine
sm = ConversationStateMachine()
sm.transition_to("READING")
state = sm.get_state()

# After (1.0.0)
from victor.protocols import IConversationStateMachine
from victor.framework import Stage
sm = container.get(IConversationStateMachine)
await sm.transition_to(Stage.READING)
state = await sm.get_state()
```

**Migration Steps**:
1. Use protocol-based dependency injection
2. Make all state machine calls async
3. Use `Stage` enum instead of strings

---

## Minor Breaking Changes

### 9. Import Paths [MINOR]

**Severity**: Minor
**Affected Component**: Import statements
**Impact**: Code using old import paths

**Changes**:

| Old Path | New Path |
|----------|----------|
| `victor.config` | `victor.config.settings` |
| `victor.state_machine` | `victor.agent.coordinators.state_coordinator` |
| `victor.tool_selector` | `victor.agent.coordinators.tool_coordinator` |
| `victor.event_bus` | `victor.core.events` |
| `victor.protocols.base` | `victor.protocols` |

**Migration Steps**:
1. Update all import statements
2. Run `victor-lint-vertical` to check for issues

---

### 10. Type Hints [MINOR]

**Severity**: Minor
**Affected Component**: Function signatures
**Impact**: Type-checked code

**Change**:
```python
# Before (0.5.x)
def execute_tool(tool: BaseTool, args: Dict) -> Any:
    ...

# After (1.0.0)
async def execute_tool(
    tool: BaseTool,
    arguments: Dict[str, Any],
) -> ToolCallResult:
    ...
```

**Migration Steps**:
1. Update function signatures
2. Make functions async where needed
3. Use proper return types

---

## Complete Checklist

### Pre-Migration Checklist

- [ ] Read main migration guide
- [ ] Review this breaking changes checklist
- [ ] Backup code and configuration
- [ ] Identify all affected code areas
- [ ] Create migration branch

### Code Migration Checklist

- [ ] Update orchestrator initialization
- [ ] Update provider initialization (remove API keys)
- [ ] Update tool registry access
- [ ] Update event bus usage
- [ ] Make chat calls async
- [ ] Update workflow execution
- [ ] Update tool selection
- [ ] Update state machine usage
- [ ] Update import paths
- [ ] Update type hints

### Test Migration Checklist

- [ ] Update test fixtures
- [ ] Update provider mocks
- [ ] Make tests async
- [ ] Update assertions for new return types
- [ ] Add test markers (unit, integration, etc.)
- [ ] Run all tests and verify they pass

### Configuration Migration Checklist

- [ ] Move API keys to environment variables
- [ ] Update settings files
- [ ] Update YAML configurations
- [ ] Update feature flags
- [ ] Validate configuration

### Validation Checklist

- [ ] Run migration validation script
- [ ] Run linter
- [ ] Run type checker
- [ ] Run unit tests
- [ ] Run integration tests
- [ ] Run smoke tests
- [ ] Test basic functionality
- [ ] Monitor logs for errors

### Post-Migration Checklist

- [ ] Update documentation
- [ ] Update team members
- [ ] Monitor production (if applicable)
- [ ] Collect feedback
- [ ] Address any issues
- [ ] Consider rollback if needed

---

## Automated Detection

Use these commands to detect breaking changes:

```bash
# Detect orchestrator instantiation
grep -rn "AgentOrchestrator(" --include="*.py" .

# Detect API key parameters
grep -rn "api_key=" --include="*.py" .

# Detect singleton usage
grep -rn "SharedToolRegistry" --include="*.py" .

# Detect EventBus usage
grep -rn "EventBus" --include="*.py" .

# Detect sync chat calls
grep -rn "\.chat(" --include="*.py" . | grep -v "await"

# Detect old imports
grep -rn "from victor.config import" --include="*.py" .
grep -rn "from victor.state_machine" --include="*.py" .
grep -rn "from victor.tool_selector" --include="*.py" .
grep -rn "from victor.event_bus" --include="*.py" .
```

---

## Migration Script

Run the automated migration script:

```bash
# Detect breaking changes
python scripts/detect_breaking_changes.py

# Generate migration report
python scripts/detect_breaking_changes.py --report migration_report.json

# Auto-fix some issues
python scripts/detect_breaking_changes.py --fix
```

---

## Risk Assessment

### High-Risk Areas

1. **Direct orchestrator instantiation** - Critical for all applications
2. **API key handling** - Security-critical, must update
3. **Tool registry access** - Affects all tool usage
4. **Event bus usage** - Affects all event-driven code
5. **Async chat API** - Affects all chat interactions

### Medium-Risk Areas

1. **Workflow execution** - Affects workflow-based applications
2. **Tool selection** - Affects custom tool selection logic
3. **State machine** - Affects conversation management

### Low-Risk Areas

1. **Import paths** - Easy to update, compiler will catch
2. **Type hints** - Optional for dynamic typing

---

## Timeline Estimate

| Change | Estimated Time | Priority |
|--------|---------------|----------|
| Orchestrator initialization | 2-4 hours | Critical |
| Provider API keys | 1-2 hours | Critical |
| Tool registry access | 2-3 hours | Critical |
| Event bus usage | 2-4 hours | Critical |
| Async chat API | 1-2 hours | Critical |
| Workflow execution | 3-5 hours | Moderate |
| Tool selection | 2-3 hours | Moderate |
| State machine | 1-2 hours | Moderate |
| Import paths | 1-2 hours | Minor |
| Type hints | 2-4 hours | Minor |

**Total Estimated Time**: 17-31 hours

---

## Support Resources

- [Main Migration Guide](./MIGRATION_GUIDE.md)
- [API Migration Guide](./MIGRATION_API.md)
- [Configuration Migration Guide](./MIGRATION_CONFIG.md)
- [Workflow Migration Guide](./MIGRATION_WORKFLOWS.md)
- [Testing Migration Guide](./MIGRATION_TESTING.md)
- [Rollback Guide](./ROLLBACK_GUIDE.md)

---

**Last Updated**: 2025-01-21
**Version**: 1.0.0
