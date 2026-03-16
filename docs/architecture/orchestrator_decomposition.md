# Orchestrator Decomposition

## Overview

The `AgentOrchestrator` has been decomposed from a monolithic god class into a thin facade coordinating extracted components. This document describes the current topology.

## Component Topology

```
AgentOrchestrator (facade)
├── CallbackCoordinator          ← tool/streaming lifecycle callbacks
│   ├── MetricsCoordinator       ← metrics collection
│   ├── ToolCoordinator          ← tool completion handling
│   └── UsageAnalytics           ← session analytics
├── OrchestratorPropertyFacade   ← ~25 property delegations
│   ├── Simple accessors         ← conversation_controller, tool_pipeline, etc.
│   └── Lazy coordinators        ← protocol_adapter, execution_coordinator, etc.
├── InitializationPhaseManager   ← 8 initialization phases
├── SessionStateAccessor         ← session state delegation
├── Runtime Boundaries
│   ├── ProviderRuntime          ← provider coordinator + pool
│   ├── MetricsRuntime           ← metrics collectors
│   ├── WorkflowRuntime          ← lazy workflow registry
│   ├── MemoryRuntime            ← memory manager
│   ├── ResilienceRuntime        ← recovery handler/integration
│   ├── CoordinationRuntime      ← recovery/chunk/planner/task
│   └── InteractionRuntime       ← chat/tool/session coordinators
└── Vertical Integration
    ├── VerticalContext           ← unified vertical state
    └── VerticalIntegrationAdapter
```

## Extraction Summary

| Component | LOC Saved | Phase |
|-----------|-----------|-------|
| CallbackCoordinator | ~40 | 3A |
| OrchestratorPropertyFacade | ~350 | 3B |
| InitializationPhaseManager | ~120 (future) | 3C |
| SessionStateAccessor | ~100 (prior) | — |

## Backward Compatibility

All property access patterns are preserved via `__getattr__` delegation:
```python
# Still works exactly the same
orchestrator.conversation_controller
orchestrator.vertical_context
orchestrator.protocol_adapter  # lazy-inits on first access
```

## Extension Loader Decomposition

| Component | Purpose |
|-----------|---------|
| ExtensionModuleResolver | Module path resolution, availability checking |
| ExtensionCacheManager | Thread-safe extension caching with namespaces |
| ExtensionLoaderPressureMonitor | Metrics and queue pressure monitoring |
