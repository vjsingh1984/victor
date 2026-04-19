# Architecture Decision Records (ADRs)

This directory contains the Architecture Decision Records for the Victor AI Framework.

## What are ADRs?

Architecture Decision Records (ADRs) document significant architectural decisions made during the development of Victor. Each ADR captures:

- **Context**: The problem or situation
- **Decision**: What was decided
- **Rationale**: Why the decision was made
- **Consequences**: Positive, negative, and neutral impacts

## ADR Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-001](001-agent-orchestration.md) | Agent Orchestrator Architecture | Accepted | 2025-02-26 |
| [ADR-002](002-state-management.md) | State Management System | Accepted | 2025-02-26 |
| [ADR-003](003-workflow-engine.md) | Workflow Engine Architecture | Accepted | 2025-02-26 |
| [ADR-004](004-tool-system.md) | Tool System Architecture | Accepted | 2025-02-26 |
| [ADR-005](005-event-system.md) | Event System Architecture | Accepted | 2025-02-26 |
| [ADR-006](006-provider-integration-improvements.md) | Provider Integration Improvements for Non-Interactive Environments | Proposed | 2026-02-28 |
| [ADR-007](007-vertical-distribution-and-sdk-boundary.md) | Vertical Distribution Model and SDK Boundary | Proposed | 2026-03-10 |
| [ADR-008](008-registry-performance-optimization.md) | Tool Registry Performance Optimization | Accepted | 2025-04-19 |

## ADR Summaries

### ADR-001: Agent Orchestrator Architecture

**Decision**: Use Coordinator Pattern with specialized coordinators for conversation, safety, tools, and metrics.

**Key Points**:
- Central AgentOrchestrator with delegated responsibilities
- Single Responsibility Principle compliance
- Easy to extend with new coordinators

### ADR-002: State Management System

**Decision**: Four-scope state management (CONVERSATION, WORKFLOW, TEAM, GLOBAL) with copy-on-write optimization.

**Key Points**:
- Clear state boundaries
- Efficient memory usage
- Checkpointing support

### ADR-003: Workflow Engine Architecture

**Decision**: Two-level architecture with WorkflowEngine facade and specialized coordinators.

**Key Points**:
- YAML and Python workflow support
- Two-level caching
- Human-in-the-loop integration

### ADR-004: Tool System Architecture

**Decision**: Three-phase tool system with metadata-driven discovery and progressive disclosure.

**Key Points**:
- Rich tool metadata
- Cost-based tiers
- 33+ built-in tools

### ADR-005: Event System Architecture

**Decision**: Structured event model with middleware pipeline and event bus.

**Key Points**:
- Type-safe events
- Middleware processing
- Pub/Sub event distribution

### ADR-008: Tool Registry Performance Optimization

**Decision**: Multi-layered caching and batch operations for 6-11× performance improvement.

**Key Points**:
- Batch registration API with single cache invalidation
- Feature flag caching with TTL expiration
- Query result caching with LRU eviction
- Performance regression tests with CI gates
- 3-5× faster for 100+ items, 6-11× overall improvement

## Creating New ADRs

When making a significant architectural decision:

1. Copy the [template](000-template.md)
2. Fill in all sections
3. Use next sequential number (e.g., ADR-006)
4. Update this index
5. Submit for review

## ADR Lifecycle

```
Proposed → Accepted → Superseded/Deprecated
```

- **Proposed**: Initial draft for discussion
- **Accepted**: Decision made and implemented
- **Superseded**: Replaced by newer decision (link to new ADR)
- **Deprecated**: No longer applicable

## Related Documentation

- [Victor Architecture](../architecture-analysis-phase3.md)
- [Implementation Plan](../roadmap/improvement-plan-v1.md)
- [Vertical Platform Convergence Plan](../roadmap/vertical-platform-convergence-plan.md)
- [API Reference](../api/README.md)

## References

- [Architecture Decision Records](https://adr.github.io/)
- [Markdown ADR Template](https://github.com/joelparkerhenderson/architecture_decision_record_template)
