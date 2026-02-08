# Coordinator Usage Guide

Complete guide to the 20 specialized coordinators in Victor AI's two-layer architecture.

---

## Quick Summary

This guide covers Victor AI's coordinator system:
- **ConfigCoordinator** - Configuration loading, validation, and management
- **PromptCoordinator** - Build prompts from multiple contributors
- **ContextCoordinator** - Manage conversation context and lifecycle
- **ChatCoordinator** - Orchestrate chat interactions with LLMs
- **ToolCoordinator** - Tool selection and execution
- **ProviderCoordinator** - LLM provider management and switching
- **SessionCoordinator** - Session management and state
- [And 13 more specialized coordinators]

---

## Guide Parts

### [Part 1: Core Coordinators](part-1-config-prompt-context-chat.md)
- ConfigCoordinator
- PromptCoordinator
- ContextCoordinator
- ChatCoordinator
- ToolCoordinator

### [Part 2: Provider, Session, Additional Coordinators](part-2-provider-session-additional.md)
- ProviderCoordinator
- SessionCoordinator
- Additional specialized coordinators

---

## Coordinator Architecture

Victor uses a **two-layer coordinator architecture**:

**Application Layer** (`victor/agent/coordinators/`): Victor-specific business logic
- ChatCoordinator, ToolCoordinator, ContextCoordinator, PromptCoordinator
- SessionCoordinator, ProviderCoordinator, ModeCoordinator, etc.

**Framework Layer** (`victor/framework/coordinators/`): Domain-agnostic infrastructure
- YAMLWorkflowCoordinator, GraphExecutionCoordinator, HITLCoordinator, CacheCoordinator

**Benefits**: Single responsibility, clear boundaries, reusability across verticals, independent testing.

---

## Related Documentation

- [Architecture Overview](../../architecture/README.md)
- [Coordinator Separation](../../architecture/coordinator_separation.md)
- [API Reference](../reference/api/API_REFERENCE.md)

---

**Last Updated:** January 31, 2026
**Reading Time:** 15 min (all parts)
