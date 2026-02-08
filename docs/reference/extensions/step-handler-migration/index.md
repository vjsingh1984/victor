# Step Handler Migration Guide

Guide to migrating from direct vertical extension patterns to the StepHandlerRegistry-based approach.

---

## Quick Summary

This guide helps you migrate from monolithic `apply_to_orchestrator()` methods to focused, reusable step handlers.

**Benefits:**
- Focused, testable handlers
- Protocol-based communication
- Clear separation of concerns
- Reusable across verticals
- Observable execution
- SOLID compliant

---

## Guide Parts

### [Part 1: Migration Patterns](part-1-migration-patterns.md)
- Overview
- Why Migrate?
- Migration Patterns

### [Part 2: Testing, Mistakes, Checklist, Scenarios](part-2-testing-checklist-scenarios.md)
- Testing Migrated Extensions
- Common Migration Mistakes
- Migration Checklist
- Advanced Migration Scenarios

---

## Quick Start

**Before:**
```python
class MyVertical(VerticalBase):
    def apply_to_orchestrator(self, orchestrator):
        orchestrator._enabled_tools = self.get_tools()
        orchestrator._middleware = self.get_middleware()
        # ... everything in one method
```

**After:**
```python
# Handler 1: Tools
class CustomToolsHandler(BaseStepHandler):
    order = 10
    def _do_apply(self, orchestrator, vertical, context, result):
        tools = vertical.get_tools()
        context.apply_enabled_tools(tools)

# Handler 2: Middleware
class CustomMiddlewareHandler(BaseStepHandler):
    order = 50
    def _do_apply(self, orchestrator, vertical, context, result):
        middleware = vertical.get_middleware()
        context.apply_middleware(middleware)

# Register
registry = StepHandlerRegistry.default()
registry.add_handler(CustomToolsHandler())
registry.add_handler(CustomMiddlewareHandler())
```

---

## Related Documentation

- [Step Handler Guide](../step_handler_guide.md)
- [Step Handler Examples](../step_handler_examples.md)
- [Vertical Development Guide](../../guides/tutorials/creating-verticals/)

---

**Last Updated:** February 01, 2026
**Reading Time:** 10 min (all parts)
