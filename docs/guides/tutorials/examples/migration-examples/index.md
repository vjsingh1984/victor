# Coordinator-Based Architecture: Migration Examples

Examples demonstrating migration to coordinator-based architecture.

---

## Quick Summary

This document provides side-by-side comparisons of code before and after migration to the coordinator-based
  architecture.

**Key Points:**
- **Most code requires NO changes** - 100% backward compatible
- **Changes are only needed** if you directly access internal orchestrator attributes
- **The migration is gradual** - You can migrate incrementally

---

## Guide Parts

### [Part 1: Examples 1-10](part-1-examples-1-10.md)
- Introduction
- Example 1: Basic Chat
- Example 2: Custom Configuration
- Example 3: Context Management
- Example 4: Analytics Tracking
- Example 5: Tool Execution
- Example 6: Provider Switching
- Example 7: Streaming Responses
- Example 8: Error Handling
- Example 9: Session Management
- Example 10: Advanced Customization

### [Part 2: Patterns, Scenarios, Checklist](part-2-patterns-scenarios-checklist.md)
- Common Migration Patterns
- Real-World Migration Scenarios
- Migration Checklist
- Summary

---

## Quick Start

**Before (Legacy):**
```python
from victor.agent.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator()
response = orchestrator.chat("Hello, Victor!")
```text

**After (Coordinator-Based):**
```python
from victor.agent import AgentOrchestrator

orchestrator = AgentOrchestrator()
response = await orchestrator.chat(
    messages=[{"role": "user", "content": "Hello, Victor!"}]
)
```

---

## Related Documentation

- [Coordinator Guide](../../guides/coordinators/)
- [Architecture Overview](../../architecture/README.md)
- [Best Practices](../../architecture/BEST_PRACTICES.md)

---

**Last Updated:** January 13, 2025
**Reading Time:** 15 min (all parts)
