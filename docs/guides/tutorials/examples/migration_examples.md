# Coordinator-Based Architecture: Migration Examples

**Version**: 1.0
**Date**: 2025-01-13
**Audience**: Developers, Technical Leads

---

## Table of Contents

1. [Introduction](#introduction)
2. [Example 1: Basic Chat](#example-1-basic-chat)
3. [Example 2: Custom Configuration](#example-2-custom-configuration)
4. [Example 3: Context Management](#example-3-context-management)
5. [Example 4: Analytics Tracking](#example-4-analytics-tracking)
6. [Example 5: Tool Execution](#example-5-tool-execution)
7. [Example 6: Provider Switching](#example-6-provider-switching)
8. [Example 7: Streaming Responses](#example-7-streaming-responses)
9. [Example 8: Error Handling](#example-8-error-handling)
10. [Example 9: Session Management](#example-9-session-management)
11. [Example 10: Advanced Customization](#example-10-advanced-customization)

---

## Introduction

This document provides side-by-side comparisons of code before and after migration to the coordinator-based architecture. Each example shows:

- **Before**: Code using the legacy monolithic orchestrator
- **After**: Code using the new coordinator-based orchestrator
- **Migration Notes**: What changed and why

### Key Points

- **Most code requires NO changes** - The coordinator-based architecture is 100% backward compatible
- **Changes are only needed** if you directly access internal orchestrator attributes
- **The migration is gradual** - You can migrate incrementally

---

## Example 1: Basic Chat

### Scenario

Simple chat operations without any advanced features.

### Before (Legacy)

```python
from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings
from victor.providers.anthropic import AnthropicProvider

async def basic_chat():
    settings = Settings()
    provider = AnthropicProvider(api_key="sk-ant-...")

    orchestrator = AgentOrchestrator(
        settings=settings,
        provider=provider,
        model="claude-sonnet-4-5"
    )

    response = await orchestrator.chat("Hello, Victor!")
    print(response.content)
```

### After (Coordinator-Based)

```python
from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings
from victor.providers.anthropic import AnthropicProvider

async def basic_chat():
    settings = Settings()
    provider = AnthropicProvider(api_key="sk-ant-...")

    orchestrator = AgentOrchestrator(
        settings=settings,
        provider=provider,
        model="claude-sonnet-4-5"
    )

    response = await orchestrator.chat("Hello, Victor!")
    print(response.content)
```

### Migration Notes

**No changes required!** The basic API remains identical. Coordinators are used internally, but the public API is unchanged.

---

## Example 2: Custom Configuration

### Scenario

Accessing internal configuration state.

### Before (Legacy)

```python
from victor.agent.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator(settings=settings, provider=provider, model=model)

# Direct access to internal config (not recommended)
config = orchestrator._config
print(f"Model: {config.model}")
print(f"Temperature: {config.temperature}")
```

### After (Coordinator-Based)

```python
from victor.agent.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator(settings=settings, provider=provider, model=model)

# Access through ConfigCoordinator
config = orchestrator._config_coordinator.get_config()
print(f"Model: {config['model']}")
print(f"Temperature: {config['temperature']}")
```

### Migration Notes

- `_config` → `_config_coordinator.get_config()`
- Returns a dictionary instead of a config object
- More flexible for multi-source configuration

---

## Example 3: Context Management

### Scenario

Monitoring and managing conversation context.

### Before (Legacy)

```python
from victor.agent.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator(settings=settings, provider=provider, model=model)

# Direct access to context (not recommended)
messages = orchestrator._context_manager.get_messages()
print(f"Context size: {len(messages)}")

# Manual compaction
compacted = orchestrator._context_compactor.compact(messages)
orchestrator._context_manager.set_messages(compacted)
```

### After (Coordinator-Based)

```python
from victor.agent.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator(settings=settings, provider=provider, model=model)

# Access through ContextCoordinator
context = orchestrator._context_coordinator.get_context()
print(f"Context size: {len(context.messages)}")

# Automatic compaction (threshold-based)
# Or manual compaction:
compacted_context = await orchestrator._context_coordinator.compact_context(context)
```

### Migration Notes

- `_context_manager` → `_context_coordinator`
- `_context_compactor` → `_context_coordinator.compact_context()`
- Context is returned as a `Context` object, not raw messages
- Compaction is automatic but can be triggered manually

---

## Example 4: Analytics Tracking

### Scenario

Tracking usage analytics and events.

### Before (Legacy)

```python
from victor.agent.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator(settings=settings, provider=provider, model=model)

# Direct access to analytics (not recommended)
orchestrator._usage_analytics.track_event(
    event_type="tool_call",
    data={"tool": "read", "file": "config.yaml"}
)

# Export analytics
analytics_data = orchestrator._usage_analytics.get_analytics()
```

### After (Coordinator-Based)

```python
from victor.agent.orchestrator import AgentOrchestrator
from victor.protocols import AnalyticsEvent

orchestrator = AgentOrchestrator(settings=settings, provider=provider, model=model)

# Access through AnalyticsCoordinator
await orchestrator._analytics_coordinator.track_event(
    session_id=orchestrator.session_id,
    event=AnalyticsEvent(
        type="tool_call",
        data={"tool": "read", "file": "config.yaml"}
    )
)

# Export analytics
result = await orchestrator._analytics_coordinator.export_analytics(
    session_id=orchestrator.session_id
)
```

### Migration Notes

- `_usage_analytics` → `_analytics_coordinator`
- Events are now structured `AnalyticsEvent` objects
- Export is async and returns `ExportResult`
- Multiple exporters can be registered

---

## Example 5: Tool Execution

### Scenario

Executing tools and handling tool calls.

### Before (Legacy)

```python
from victor.agent.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator(settings=settings, provider=provider, model=model)

# Direct tool execution (not recommended)
result = await orchestrator._tool_executor.execute_tool(
    tool_name="read",
    parameters={"file_path": "config.yaml"}
)

# Manual tool call handling
tool_calls = orchestrator._tool_call_parser.parse(response)
for call in tool_calls:
    result = await orchestrator._tool_executor.execute_tool(
        tool_name=call.name,
        parameters=call.arguments
    )
```

### After (Coordinator-Based)

```python
from victor.agent.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator(settings=settings, provider=provider, model=model)

# Tools are automatically handled by ChatCoordinator and ToolCoordinator
# No manual intervention needed for most cases

# If you need to execute tools directly:
tool_coordinator = orchestrator._tool_coordinator
result = await tool_coordinator.execute_tool_calls(
    tool_calls=[tool_call],
    context=orchestrator._context_coordinator.get_context()
)
```

### Migration Notes

- Tool execution is now fully automatic
- `_tool_executor` → `_tool_coordinator`
- `_tool_call_parser` is handled internally
- Tool selection is delegated to `ToolSelectionCoordinator`

---

## Example 6: Provider Switching

### Scenario

Switching providers mid-conversation.

### Before (Legacy)

```python
from victor.agent.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator(settings=settings, provider=provider, model=model)

# Manual provider switching (not recommended)
orchestrator._provider = new_provider
orchestrator._model = "new-model"

# Reinitialize components that depend on provider
orchestrator._tool_executor.set_provider(new_provider)
```

### After (Coordinator-Based)

```python
from victor.agent.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator(settings=settings, provider=provider, model=model)

# Use ProviderCoordinator for clean switching
await orchestrator.switch_provider(
    provider=new_provider,
    model="new-model"
)

# All coordinators are automatically updated
```

### Migration Notes

- Manual provider switching → `switch_provider()` method
- Coordinators are automatically notified of provider changes
- No need to manually update dependencies
- Cleaner, more reliable switching

---

## Example 7: Streaming Responses

### Scenario

Streaming chat responses for real-time output.

### Before (Legacy)

```python
from victor.agent.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator(settings=settings, provider=provider, model=model)

# Manual stream handling (not recommended)
async for chunk in orchestrator._provider.stream_chat(messages):
    print(chunk.content, end="", flush=True)
    # Manual context and analytics updates
```

### After (Coordinator-Based)

```python
from victor.agent.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator(settings=settings, provider=provider, model=model)

# Streaming is fully managed by ChatCoordinator
async for chunk in orchestrator.stream_chat("Tell me a story"):
    print(chunk.content, end="", flush=True)
    # Context and analytics are automatically updated
```

### Migration Notes

- No changes needed for basic streaming
- Coordinators handle context updates automatically
- Analytics are tracked automatically
- Cleaner separation of concerns

---

## Example 8: Error Handling

### Scenario

Handling errors from various orchestrator components.

### Before (Legacy)

```python
from victor.agent.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator(settings=settings, provider=provider, model=model)

try:
    response = await orchestrator.chat("Hello!")
except Exception as e:
    # Manual error handling
    if "tool" in str(e):
        orchestrator._metrics_collector.track_error("tool_execution", e)
    elif "provider" in str(e):
        orchestrator._metrics_collector.track_error("provider", e)
    else:
        orchestrator._metrics_collector.track_error("unknown", e)
```

### After (Coordinator-Based)

```python
from victor.agent.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator(settings=settings, provider=provider, model=model)

try:
    response = await orchestrator.chat("Hello!")
except Exception as e:
    # Coordinators handle their own error tracking
    # MetricsCoordinator automatically logs errors

    # Optional: Check specific coordinator health
    config = orchestrator._config_coordinator.get_config()
    context = orchestrator._context_coordinator.get_context()

    # Implement fallback logic
```

### Migration Notes

- Error tracking is automatic
- Coordinators log their own errors
- Metrics are collected automatically
- Less boilerplate code needed

---

## Example 9: Session Management

### Scenario

Managing multiple conversation sessions.

### Before (Legacy)

```python
from victor.agent.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator(settings=settings, provider=provider, model=model)

# Manual session management (not recommended)
session1_context = []
session2_context = []

async def chat_in_session(session_id, message):
    if session_id == "session1":
        session1_context.append({"role": "user", "content": message})
        response = await orchestrator._provider.chat(session1_context)
        session1_context.append({"role": "assistant", "content": response.content})
    else:
        session2_context.append({"role": "user", "content": message})
        # ... duplicate logic
```

### After (Coordinator-Based)

```python
from victor.agent.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator(settings=settings, provider=provider, model=model)

# Session management is automatic
response1 = await orchestrator.chat("Hello!", session_id="session1")
response2 = await orchestrator.chat("Hi there!", session_id="session2")

# Sessions are isolated
# Context is automatically managed per session
# Analytics are tracked per session
```

### Migration Notes

- No manual session management needed
- Pass `session_id` parameter to `chat()`
- Context and analytics are automatically isolated
- Much cleaner code

---

## Example 10: Advanced Customization

### Scenario

Deep customization of orchestrator behavior.

### Before (Legacy)

```python
from victor.agent.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator(settings=settings, provider=provider, model=model)

# Subclass monolithic orchestrator for customization
class CustomOrchestrator(AgentOrchestrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override internal components
        self._context_compactor = CustomCompactor()
        self._prompt_builder = CustomPromptBuilder()

    async def chat(self, message):
        # Override chat logic
        custom_prompt = self._build_custom_prompt(message)
        response = await self._provider.chat(custom_prompt)
        return self._process_response(response)
```

### After (Coordinator-Based)

```python
from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.coordinators import (
    ContextCoordinator,
    PromptCoordinator,
    AnalyticsCoordinator,
)

# Create custom coordinators
context_coordinator = ContextCoordinator(
    compaction_strategy=CustomCompactionStrategy()
)

prompt_coordinator = PromptCoordinator(contributors=[
    CustomPromptContributor(),
])

analytics_coordinator = AnalyticsCoordinator(exporters=[
    CustomAnalyticsExporter(),
])

# Inject custom coordinators
orchestrator = AgentOrchestrator(
    settings=settings,
    provider=provider,
    model=model,
    _context_coordinator=context_coordinator,
    _prompt_coordinator=prompt_coordinator,
    _analytics_coordinator=analytics_coordinator,
)

# No need to subclass orchestrator
# Standard chat() method works with custom coordinators
response = await orchestrator.chat("Hello!")
```

### Migration Notes

- Subclassing → coordinator injection
- More modular and maintainable
- Easier to test individual components
- Cleaner separation of concerns
- No need to override core methods

---

## Common Migration Patterns

### Pattern 1: Direct Attribute Access

**Before**:
```python
config = orchestrator._config
messages = orchestrator._context_manager.get_messages()
```

**After**:
```python
config = orchestrator._config_coordinator.get_config()
context = orchestrator._context_coordinator.get_context()
messages = context.messages
```

### Pattern 2: Custom Component Injection

**Before**:
```python
class CustomOrchestrator(AgentOrchestrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_component = CustomComponent()
```

**After**:
```python
custom_coordinator = CustomCoordinator()
orchestrator = AgentOrchestrator(
    ...,
    _custom_coordinator=custom_coordinator,
)
```

### Pattern 3: Async Method Calls

**Before**:
```python
result = orchestrator._some_component.process(data)
```

**After**:
```python
result = await orchestrator._some_coordinator.process(data)
```

### Pattern 4: Event Tracking

**Before**:
```python
orchestrator._usage_analytics.track_event(type="tool_call", data={...})
```

**After**:
```python
await orchestrator._analytics_coordinator.track_event(
    session_id=session_id,
    event=AnalyticsEvent(type="tool_call", data={...}),
)
```

---

## Real-World Migration Scenarios

### Scenario 1: Enterprise Multi-Tenant Application

**Before**:
```python
class TenantOrchestrator(AgentOrchestrator):
    def __init__(self, tenant_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tenant_id = tenant_id
        self._load_tenant_config()

    def _load_tenant_config(self):
        self._config = self.db.get_config(self.tenant_id)
```

**After**:
```python
class TenantConfigProvider(IConfigProvider):
    def __init__(self, db, tenant_id):
        self.db = db
        self.tenant_id = tenant_id

    async def get_config(self, session_id):
        return await self.db.get_config(self.tenant_id)

# Use with orchestrator
config_provider = TenantConfigProvider(db, tenant_id)
config_coordinator = ConfigCoordinator(providers=[config_provider])

orchestrator = AgentOrchestrator(
    ...,
    _config_coordinator=config_coordinator,
)
```

### Scenario 2: Compliance-Heavy Application

**Before**:
```python
class ComplianceOrchestrator(AgentOrchestrator):
    async def chat(self, message):
        # Pre-processing
        if self._contains_pii(message):
            message = self._sanitize_pii(message)

        # Chat
        response = await super().chat(message)

        # Post-processing
        if self._contains_pii(response.content):
            response.content = self._sanitize_pii(response.content)

        return response
```

**After**:
```python
class CompliancePromptContributor(BasePromptContributor):
    async def get_contribution(self, context):
        return "\nDo not include PII in responses."

class PIISanitizationExporter(BaseAnalyticsExporter):
    async def export(self, events):
        sanitized = [self._sanitize_pii(e) for e in events]
        return await self.db.export(sanitized)

orchestrator = AgentOrchestrator(
    ...,
    _prompt_coordinator=PromptCoordinator(contributors=[
        CompliancePromptContributor(),
    ]),
    _analytics_coordinator=AnalyticsCoordinator(exporters=[
        PIISanitizationExporter(),
    ]),
)
```

### Scenario 3: Analytics-Driven Optimization

**Before**:
```python
class OptimizedOrchestrator(AgentOrchestrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cost_tracker = CostTracker()

    async def chat(self, message):
        cost_estimate = self._cost_tracker.estimate_cost(message)
        if cost_estimate > self.budget:
            raise BudgetExceededError()
        return await super().chat(message)
```

**After**:
```python
class BudgetAwareCompactionStrategy(BaseCompactionStrategy):
    def __init__(self, max_tokens_per_request):
        self.max_tokens = max_tokens_per_request

    async def compact(self, context):
        current_tokens = self._count_tokens(context)
        if current_tokens <= self.max_tokens:
            return context

        # Compact to fit budget
        return await self._compact_to_budget(context, self.max_tokens)

orchestrator = AgentOrchestrator(
    ...,
    _context_coordinator=ContextCoordinator(
        compaction_strategy=BudgetAwareCompactionStrategy(max_tokens=4000)
    ),
)
```

---

## Migration Checklist

### Phase 1: Assessment (1-2 hours)

- [ ] Identify all direct orchestrator attribute access
- [ ] List all custom orchestrator subclasses
- [ ] Document all custom components
- [ ] Identify critical integration points

### Phase 2: Planning (2-4 hours)

- [ ] Map old attributes to new coordinators
- [ ] Create migration plan for each component
- [ ] Estimate migration effort per component
- [ ] Identify testing requirements

### Phase 3: Migration (4-8 hours)

- [ ] Migrate config access patterns
- [ ] Migrate context management
- [ ] Migrate analytics tracking
- [ ] Migrate tool execution
- [ ] Migrate custom components

### Phase 4: Testing (2-4 hours)

- [ ] Unit test migrated components
- [ ] Integration test coordinator interactions
- [ ] Performance test coordinator overhead
- [ ] Verify backward compatibility

### Phase 5: Deployment (1-2 hours)

- [ ] Deploy to staging
- [ ] Run smoke tests
- [ ] Monitor metrics
- [ ] Deploy to production

**Total Estimated Time**: 10-20 hours for complex applications, 0 hours for simple applications

---

## Summary

This guide demonstrated 10 migration examples covering:

1. **Basic chat** - No changes needed
2. **Custom configuration** - Use ConfigCoordinator
3. **Context management** - Use ContextCoordinator
4. **Analytics tracking** - Use AnalyticsCoordinator
5. **Tool execution** - Automatic via ToolCoordinator
6. **Provider switching** - Use `switch_provider()` method
7. **Streaming responses** - No changes needed
8. **Error handling** - Automatic tracking
9. **Session management** - Built-in support
10. **Advanced customization** - Coordinator injection

### Key Takeaways

- **Most code requires no changes** - Backward compatible
- **Direct internal access** → Coordinator methods
- **Subclassing** → Coordinator injection
- **Manual management** → Automatic handling
- **Cleaner, more maintainable** architecture

### Next Steps

- [Quick Start Guide](../tutorials/coordinator_quickstart.md) - Get started quickly
- [Usage Examples](../examples/coordinator_examples.md) - More examples
- [Recipes](../tutorials/coordinator_recipes.md) - Step-by-step solutions

---

**Document Version**: 1.0
**Last Updated**: 2025-01-13
**Next Review**: 2025-04-13

---

**End of Migration Examples**

---

**Last Updated:** February 01, 2026
**Reading Time:** 5 minutes
