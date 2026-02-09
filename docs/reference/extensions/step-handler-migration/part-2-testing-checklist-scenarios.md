# Step Handler Migration Guide - Part 2

**Part 2 of 2:** Testing Migrated Extensions, Common Mistakes, Migration Checklist, Advanced Scenarios

---

## Navigation

- [Part 1: Migration Patterns](part-1-migration-patterns.md)
- **[Part 2: Testing, Mistakes, Checklist, Scenarios](#)** (Current)
- [**Complete Guide](../step_handler_migration.md)**

---

## Testing Migrated Extensions

### Unit Testing Handlers

Test handlers in isolation:

```python
import pytest
from victor.framework.step_handlers import BaseStepHandler
from victor.framework.vertical_integration import IntegrationResult

def test_custom_tools_handler():
    """Test custom tools handler."""
    handler = CustomToolsHandler()

    # Create mock objects
    orchestrator = MockOrchestrator()
    vertical = MockVertical()
    context = MockContext()
    result = IntegrationResult("test")

    # Apply handler
    handler.apply(orchestrator, vertical, context, result)

    # Verify
    assert "tools" in context.data
    assert result.success is True
```

### Integration Testing

Test handler interaction:

```python
@pytest.mark.asyncio
async def test_full_integration():
    """Test complete integration with handlers."""
    orchestrator = AgentOrchestrator()
    vertical = MyVertical()

    # Apply vertical
    await orchestrator.apply_vertical(vertical)

    # Verify
    assert len(orchestrator.tools) > 0
    assert orchestrator.middleware is not None
```

---

## Common Migration Mistakes

### Mistake 1: Preserving Private Access

**Don't do this:**
```python
def _do_apply(self, orchestrator, vertical, context, result):
    # Still using private attributes
    orchestrator._enabled_tools = tools  # Bad!
```

**Do this instead:**
```python
def _do_apply(self, orchestrator, vertical, context, result):
    # Use context
    context.apply_enabled_tools(tools)  # Good!
```

### Mistake 2: Monolithic Handlers

**Don't create:**
```python
class EverythingHandler(BaseStepHandler):
    """Does everything - bad!"""
    def _do_apply(self, orchestrator, vertical, context, result):
        # Tools, middleware, safety, configs, all in one
        tools = vertical.get_tools()
        middleware = vertical.get_middleware()
        # ... 200 lines
```

**Create instead:**
```python
class ToolsHandler(BaseStepHandler):
    """Just tools"""
    def _do_apply(self, orchestrator, vertical, context, result):
        tools = vertical.get_tools()
        context.apply_enabled_tools(tools)

class MiddlewareHandler(BaseStepHandler):
    """Just middleware"""
    def _do_apply(self, orchestrator, vertical, context, result):
        middleware = vertical.get_middleware()
        context.apply_middleware(middleware)
```

### Mistake 3: Wrong Order

**Be careful with handler order:**
```python
# Wrong: Safety before tools
class SafetyHandler(BaseStepHandler):
    order = 5  # Too early!

# Correct: Safety after capability config (5)
class SafetyHandler(BaseStepHandler):
    order = 30  # Right time
```

---

## Migration Checklist

- [ ] Identify all `apply_to_orchestrator()` implementations
- [ ] Extract distinct concerns into handlers
- [ ] Determine correct handler order
- [ ] Implement handlers inheriting `BaseStepHandler`
- [ ] Register handlers with `StepHandlerRegistry`
- [ ] Remove old `apply_to_orchestrator()` method
- [ ] Add unit tests for each handler
- [ ] Add integration tests
- [ ] Verify backward compatibility
- [ ] Update documentation

---

## Advanced Migration Scenarios

### Scenario 1: Migrating Complex Verticals

**Challenge**: Vertical with complex conditional logic.

**Solution**: Use conditional handlers:

```python
class ConditionalToolsHandler(BaseStepHandler):
    def _do_apply(self, orchestrator, vertical, context, result):
        if vertical.name == "coding":
            tools = CODING_TOOLS
        elif vertical.name == "devops":
            tools = DEVOPS_TOOLS
        else:
            tools = DEFAULT_TOOLS

        context.apply_enabled_tools(tools)
```

### Scenario 2: Migrating with Dependencies

**Challenge**: Handlers depend on each other.

**Solution**: Use order and context:

```python
class Handler1(BaseStepHandler):
    order = 10
    def _do_apply(self, orchestrator, vertical, context, result):
        data = self._compute_data()
        context.apply_handler1_data(data)

class Handler2(BaseStepHandler):
    order = 20
    def _do_apply(self, orchestrator, vertical, context, result):
        # Use data from Handler1
        data = context.get("handler1_data")
        self._process_data(data)
```

### Scenario 3: Migrating Async Operations

**Challenge**: Handlers need async operations.

**Solution**: Use `apply_async`:

```python
class AsyncResourceHandler(BaseStepHandler):
    async def apply_async(
        self, orchestrator, vertical, context, result, **kwargs
    ):
        resources = await self._load_resources_async()
        context.apply_resources(resources)
```

---

## Summary

Migrating to step handlers provides:
- **Better testability**: Test handlers in isolation
- **Clearer separation**: Each handler has single responsibility
- **Reusability**: Share handlers across verticals
- **Observability**: Track handler execution
- **SOLID compliance**: Follow software design principles

**Key Takeaways:**
1. Extract distinct concerns into focused handlers
2. Use protocols for loose coupling
3. Respect handler ordering
4. Test thoroughly
5. Document handler purpose

---

## See Also

- [Documentation Home](../../README.md)


**Reading Time:** 2 min
**Last Updated:** February 01, 2026
