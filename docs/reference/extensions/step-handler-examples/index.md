# Step Handler Examples

Complete guide with practical, working examples of step handlers for common vertical extension scenarios.

---

## Quick Summary

This guide provides 19 step handler examples organized into 3 parts:
- **Part 1**: Basic Examples, Tool Management, Middleware Integration, and Workflow Registration
- **Part 2**: Safety & Security and Configuration Management
- **Part 3**: Advanced Patterns and Testing Examples

---

## Guide Parts

### [Part 1: Basic Examples through Workflow Registration](part-1-basic-middleware-workflow.md)
- Example 1: Minimal Step Handler
- Example 2: Logging Handler
- Example 3: Conditional Handler
- Example 4: Custom Tool Registration with Validation
- Example 5: Tiered Tool Configuration
- Example 6: Tool Dependency Resolution
- Example 7: Custom Middleware Injection
- Example 8: Middleware Chain Validation
- Example 9: Workflow Registration with Validation
- Example 10: Workflow Trigger Registration

### [Part 2: Safety & Security and Configuration Management](part-2-safety-config.md)
- Example 11: Safety Pattern Validation
- Example 12: Strict Mode Enforcement
- Example 13: Mode Configuration Application
- Example 14: Stage Configuration

### [Part 3: Advanced Patterns and Testing Examples](part-3-advanced-testing.md)
- Example 15: Handler Composition
- Example 16: Async Handler
- Example 17: Conditional Handler with Retry
- Example 18: Testable Handler
- Example 19: Handler with Test Hooks

---

## Key Patterns

**Validation**: Validate before applying (Examples 4, 11, 15)
**Filtering**: Filter based on conditions (Examples 5, 6)
**Composition**: Compose multiple operations (Example 15)
**Async**: Load resources asynchronously (Example 16)
**Retry**: Retry on failure (Example 17)
**Testability**: Design for testing (Examples 18, 19)

---

## Best Practices

- Use clear, descriptive names
- Choose appropriate order values
- Handle errors gracefully
- Provide step details for observability
- Design for testability

---

## Related Documentation

- [Step Handler Guide](step_handler_guide.md) - Concepts and architecture
- [Migration Guide](step_handler_migration.md) - Migration patterns
- [Quick Reference](step_handler_quick_reference.md) - API details

---

**Last Updated:** February 01, 2026
**Reading Time:** 15 min (all parts)
