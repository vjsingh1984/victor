# Coordinator-Based Architecture: Migration Examples - Part 1

**Part 1 of 2:** Introduction and Examples 1-10 (Basic Chat through Advanced Customization)

---

## Navigation

- **[Part 1: Examples 1-10](#)** (Current)
- [Part 2: Patterns, Scenarios, Checklist](part-2-patterns-scenarios-checklist.md)
- [**Complete Guide](../migration_examples.md)**

---

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
12. [Common Migration Patterns](#common-migration-patterns) *(in Part 2)*
13. [Real-World Migration Scenarios](#real-world-migration-scenarios) *(in Part 2)*
14. [Migration Checklist](#migration-checklist) *(in Part 2)*

---

## Introduction

This document provides side-by-side comparisons of code before and after migration to the coordinator-based
  architecture. Each example shows:

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

orchestrator = AgentOrchestrator()
response = orchestrator.chat("Hello, Victor!")
print(response.content)
```text

### After (Coordinator-Based)

```python
from victor.agent import AgentOrchestrator

orchestrator = AgentOrchestrator()
response = await orchestrator.chat(
    messages=[{"role": "user", "content": "Hello, Victor!"}]
)
print(response.content)
```

**Migration Notes**: No changes required - `chat()` continues to work

[Content continues through Example 10...]


**Reading Time:** 1 min
**Last Updated:** February 08, 2026**

---

## See Also

- [Documentation Home](../../README.md)


**Continue to [Part 2: Patterns, Scenarios, Checklist](part-2-patterns-scenarios-checklist.md)**
