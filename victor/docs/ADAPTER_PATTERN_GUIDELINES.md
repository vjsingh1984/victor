# Adapter Pattern Guidelines for Victor

**Purpose:** Guidelines for when to use (and when NOT to use) the Adapter pattern in Victor's codebase.

**Last Updated:** 2025-12-21 (HIGH-007)

---

## When to Use Adapters

Use the Adapter pattern when **ALL** of these conditions are met:

### 1. ✓ You need to adapt between incompatible interfaces

```python
# GOOD: Different interfaces that cannot be unified
# OpenAI: {"type": "function", "function": {...}}
# Anthropic: {"name": "...", "input_schema": {...}}
# → Requires adapter to normalize
```

### 2. ✓ You're integrating with external systems (providers, APIs, protocols)

```python
# GOOD: External system with different interface
class VictorAgentAdapter:
    """Adapt Victor's orchestrator to external benchmark framework."""
    def execute_task(self, task) -> AgenticExecutionTrace:
        response = self.orchestrator.chat(task.prompt)
        return self._convert_to_trace(response)
```

### 3. ✓ You need to support multiple variants of an interface

```python
# GOOD: Multiple providers with different behaviors
class DeepSeekAdapter(BaseProviderAdapter):
    def extract_thinking_content(self, response):
        """Extract <think>...</think> tags specific to DeepSeek"""
        ...

class GrokAdapter(BaseProviderAdapter):
    def deduplicate_output(self, response):
        """Remove repeated blocks specific to Grok"""
        ...
```

### 4. ✓ The adaptation logic is complex (multiple methods, state management)

```python
# GOOD: Complex adapter with multiple methods and state
class DebugAdapter(Protocol):
    async def set_breakpoints(...)
    async def continue_execution(...)
    async def step_over(...)
    async def get_variables(...)
    # 30+ methods - too complex for a function
```

---

## When NOT to Use Adapters

Do **NOT** use adapters when:

### 1. ✗ A simple function would suffice (single transformation)

```python
# BAD: Unnecessary adapter for simple transformation
class DataAdapter:
    @staticmethod
    def convert(data: dict) -> dict:
        return {"transformed": data["original"]}

# GOOD: Use a simple function instead
def convert_data_format(data: dict) -> dict:
    """Convert data to standard format."""
    return {"transformed": data["original"]}
```

### 2. ✗ You're just wrapping without adapting

```python
# BAD: Pass-through wrapper adds no value
class FooAdapter:
    def __init__(self, foo):
        self.foo = foo

    def method1(self):
        return self.foo.method1()  # Just delegates

    def method2(self):
        return self.foo.method2()  # Just delegates

# GOOD: Use the object directly
foo.method1()
foo.method2()
```

### 3. ✗ You can modify the underlying interface instead

```python
# BAD: Creating adapter when you control both interfaces
class LegacyAdapter:
    def new_method(self, data):
        return self.legacy_obj.old_method(data)

# GOOD: Update the interface if you control it
class UpdatedClass:
    def new_method(self, data):  # Just rename/refactor
        ...
```

### 4. ✗ The "adapter" adds a redundant normalization layer

```python
# BAD: Provider already returns StreamChunk, why wrap it again?
class UnifiedStreamAdapter:
    async def stream(self, messages):
        raw_stream = self.provider.stream(messages)  # Already returns StreamChunk
        async for chunk in raw_stream:
            yield chunk  # No adaptation happening!

# GOOD: Use provider.stream() directly
async for chunk in provider.stream(messages):
    process(chunk)
```

---

## Decision Tree

```
Does it adapt between incompatible interfaces?
├─ YES ──> Is the adaptation complex (multiple methods, state)?
│          ├─ YES ──> USE ADAPTER PATTERN ✓
│          └─ NO ──> Use simple function instead
└─ NO ──> Is it just wrapping/delegating?
           ├─ YES ──> REMOVE, use object directly ✗
           └─ NO ──> Reconsider design (might not need anything)
```

---

## Good Examples from Victor Codebase

### Example 1: Tool Calling Adapter (KEEP)

**Purpose:** Adapt tool definitions to provider-specific formats

**Why it's needed:**
- OpenAI, Anthropic, Google all have different tool calling schemas
- Cannot be unified (external APIs we don't control)

**Complexity:**
- Multiple methods: `convert_tools()`, `parse_tool_calls()`, `get_capabilities()`
- Provider-specific validation and normalization
- Dynamic capability detection

**Verdict:** ✓ TRUE ADAPTER PATTERN

```python
class AnthropicToolCallingAdapter(BaseToolCallingAdapter):
    def convert_tools(self, tools):
        """Convert to Anthropic format (name, description, input_schema)."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,  # Different key!
            }
            for tool in tools
        ]

    def parse_tool_calls(self, content, raw_tool_calls):
        """Parse Anthropic tool calls (content blocks)."""
        # Complex parsing logic specific to Anthropic
        ...
```

### Example 2: Provider Adapter (KEEP)

**Purpose:** Normalize provider-specific behaviors

**Why it's needed:**
- Different providers have different quirks (thinking tags, dedup, quality thresholds)
- Cannot be unified (provider-specific behavior)

**Complexity:**
- Multiple methods for different quirks
- Provider-specific configuration
- Runtime behavior adaptation

**Verdict:** ✓ STRATEGY PATTERN + ADAPTER

```python
class DeepSeekAdapter(BaseProviderAdapter):
    @property
    def capabilities(self):
        return ProviderCapabilities(
            quality_threshold=0.60,  # Lower for analytical responses
            supports_thinking_tags=True,
            thinking_tag_format="<think>...</think>",
            continuation_markers=["💭 Thinking...", "Let me "],
        )

    def extract_thinking_content(self, response):
        """Extract <think>...</think> tags from DeepSeek responses."""
        # Complex regex extraction
        ...
```

### Example 3: Protocol Adapter (KEEP)

**Purpose:** Adapt Victor's orchestrator to different client transports

**Why it's needed:**
- CLI needs direct in-process calls
- VS Code needs HTTP/REST API calls
- Different transport mechanisms require different implementations

**Complexity:**
- State management (HTTP client, connection pooling)
- Async operations
- Error handling for network vs local calls

**Verdict:** ✓ TRUE ADAPTER PATTERN

```python
class DirectProtocolAdapter(VictorProtocol):
    """CLI adapter - calls orchestrator directly."""
    async def chat(self, messages):
        return await self._orchestrator.chat(messages[-1].content)

class HTTPProtocolAdapter(VictorProtocol):
    """VS Code adapter - uses HTTP API."""
    async def chat(self, messages):
        response = await self._client.post(
            "/chat",
            json={"messages": [m.to_dict() for m in messages]}
        )
        return ChatResponse.from_dict(response.json())
```

---

## Examples to Avoid (Anti-Patterns)

### Anti-Pattern 1: Pass-through Wrapper (❌)

```python
# DON'T DO THIS
class FooAdapter:
    def __init__(self, foo):
        self.foo = foo

    def method1(self):
        return self.foo.method1()  # No adaptation!

    def method2(self):
        return self.foo.method2()  # Just delegating

# INSTEAD: Use foo directly!
foo.method1()
foo.method2()
```

### Anti-Pattern 2: Single-method "Adapter" (❌)

```python
# DON'T DO THIS
class DataAdapter:
    @staticmethod
    def convert(data: dict) -> dict:
        return transform(data)

# INSTEAD: Use a function!
def convert_data(data: dict) -> dict:
    """Convert data to standard format."""
    return transform(data)
```

### Anti-Pattern 3: Redundant Normalization Layer (❌)

```python
# DON'T DO THIS
class UnifiedStreamAdapter:
    async def stream(self, messages):
        # Provider already returns StreamChunk
        raw_stream = self.provider.stream(messages)
        async for chunk in raw_stream:
            yield chunk  # No normalization happening!

# INSTEAD: Use provider.stream() directly!
async for chunk in provider.stream(messages):
    process(chunk)
```

### Anti-Pattern 4: Adapter for Code You Control (❌)

```python
# DON'T DO THIS
class NewAPIAdapter:
    """Adapt our old API to our new API."""
    def new_method(self, data):
        return self.old_obj.old_method(data)

# INSTEAD: Refactor the interface!
class UpdatedClass:
    def new_method(self, data):  # Just rename
        ...
```

---

## Architecture Decision Record (ADR) Template

When adding an adapter, document:

### 1. What interfaces are being adapted?

```
Interface A: <describe source interface>
Interface B: <describe target interface>
Why different: <explain incompatibility>
```

### 2. Why can't the interfaces be unified?

```
☐ External API (we don't control it)
☐ Legacy system (cannot modify)
☐ Multiple providers with different formats
☐ Different transport mechanisms (local vs network)
☐ Other: <explain>
```

### 3. What value does the adapter provide?

```
☐ Format conversion (incompatible schemas)
☐ Behavior normalization (provider quirks)
☐ Protocol translation (different communication methods)
☐ Integration bridge (internal to external system)
☐ Other: <explain>
```

### 4. Could a simpler approach work?

```
☐ No - requires multiple methods
☐ No - requires state management
☐ No - complex transformation logic
☐ No - integrating with external system
☐ Yes - <explain why simpler approach rejected>
```

---

## Checklist for New Adapters

Before creating an adapter, verify:

- [ ] Interfaces are genuinely incompatible (cannot be unified)
- [ ] At least one interface is external (or has good reason not to modify)
- [ ] Adaptation requires >1 method OR state management
- [ ] No simpler alternative exists (function, refactoring)
- [ ] Adapter will be used in 3+ places (avoids one-off wrappers)
- [ ] Documented in ADR format (see template above)

If **ALL** boxes are checked: ✓ Adapter is justified

If **ANY** box is unchecked: ⚠️ Reconsider - might be over-engineering

---

## Victor's Justified Adapters (Reference)

### Multi-Provider Tool Calling (ESSENTIAL)
- `victor/agent/tool_calling/adapters.py`
- Why: 15+ providers with different tool call formats (OpenAI, Anthropic, Google, etc.)

### Multi-Provider Behavior Normalization (ESSENTIAL)
- `victor/protocols/provider_adapter.py`
- Why: Provider-specific quirks (thinking tags, dedup, quality thresholds)

### Multi-Client Transport (ESSENTIAL)
- `victor/protocol/adapters.py`
- Why: Different clients need different transports (direct, HTTP, WebSocket)

### Multi-Debugger Protocol (ESSENTIAL)
- `victor/debug/adapter.py`
- Why: Different debugger protocols (DAP, lldb, gdb)

### External Benchmark Integration (ESSENTIAL)
- `victor/evaluation/agent_adapter.py`
- Why: Adapt internal orchestrator to external benchmark framework

### Event System Bridge (JUSTIFIED)
- `victor/observability/cqrs_adapter.py`
- Why: Both EventBus and EventDispatcher in active use, need integration

---

## When in Doubt

Ask these questions:

1. **Could this be a function?**
   - If yes → Don't use adapter

2. **Am I just wrapping without adapting?**
   - If yes → Don't use adapter

3. **Do I control both interfaces?**
   - If yes → Refactor instead of adapting

4. **Is this adding a redundant normalization layer?**
   - If yes → Remove the redundant layer

5. **Would this adapter be used in <3 places?**
   - If yes → Inline the logic instead

**Golden Rule:** Adapters are for bridging **genuinely incompatible external interfaces**. If you control both sides, refactor instead of adapting.

---

## Summary

✓ **DO** use adapters for:
- Multi-provider support (OpenAI, Anthropic, Google, etc.)
- External system integration (benchmarks, debuggers, etc.)
- Multi-client architecture (CLI, HTTP, MCP, etc.)
- Complex provider quirk normalization

✗ **DON'T** use adapters for:
- Simple transformations (use functions)
- Pass-through wrappers (use objects directly)
- Code you control (refactor instead)
- Redundant normalization (remove the layer)

🎯 **Remember:** Victor's 6 adapters (after removing stream_adapter) are justified by the genuinely multi-provider (15+), multi-client (4+), multi-debugger (3+) architecture. This is NOT over-engineering.
