# Step Handler Migration Guide - Part 1

**Part 1 of 2:** Overview, Why Migrate, and Migration Patterns

---

## Navigation

- **[Part 1: Migration Patterns](#)** (Current)
- [Part 2: Testing, Mistakes, Checklist, Scenarios](part-2-testing-checklist-scenarios.md)
- [**Complete Guide](../step_handler_migration.md)**

---

## Overview

This guide helps you migrate from direct vertical extension patterns to the StepHandlerRegistry-based approach. This
  migration improves testability,
  maintainability, and reusability of your vertical extensions.

## Why Migrate?

### Before: Direct Extension

**Problems:**
- Monolithic `apply_to_orchestrator()` methods
- Hard to test individual concerns
- Tight coupling between vertical and orchestrator
- Private attribute access violates SOLID principles
- No clear separation of concerns
- Difficult to extend or modify

**Example:**
```python
class MyVertical(VerticalBase):
    def apply_to_orchestrator(self, orchestrator):
        # Everything in one method
        orchestrator._enabled_tools = self.get_tools()
        orchestrator._middleware = self.get_middleware()
        orchestrator._safety_patterns = self.get_safety_patterns()
        orchestrator._mode_configs = self.get_mode_configs()
        orchestrator._workflows = self.get_workflows()
        # ... 100+ lines
        # Violates SRP, DIP, encapsulation
```text

### After: Step Handlers

**Benefits:**
- Focused, testable handlers
- Protocol-based communication
- Clear separation of concerns
- Reusable across verticals
- Observable execution
- SOLID compliant

**Example:**
```python
# Handler 1: Tools
class CustomToolsHandler(BaseStepHandler):
    order = 10
    def _do_apply(self, orchestrator, vertical, context, result):
        tools = vertical.get_tools()
        context.apply_enabled_tools(tools)
```

[Content continues through Migration Patterns...]


**Reading Time:** 1 min
**Last Updated:** February 08, 2026**

---

## See Also

- [Documentation Home](../../README.md)


**Continue to [Part 2: Testing, Mistakes, Checklist, Scenarios](part-2-testing-checklist-scenarios.md)**
