# Victor Architecture Consolidation Migration Guide

## Overview

Victor has undergone a significant architectural consolidation to unify the framework layer and CLI/chat layer. This guide helps you migrate your code and understand the changes.

## What Changed

### Before: Two Parallel Architectures

1. **Framework Layer** (`victor/framework/`): Deterministic primitives
   - StateGraph, WorkflowEngine, Teams
   - Used by: Advanced users, workflow authors

2. **CLI/Chat Layer** (`victor/agent/`): Service-oriented, custom loops
   - AgentOrchestrator, AgenticLoop
   - Used by: CLI chat, API server

### After: Unified Architecture

- **Framework as foundation**: All execution uses StateGraph-based patterns
- **CLI/Chat as thin surface**: Composes framework components
- **Service injection**: Services accessed via ExecutionContext
- **Team formations**: 5 formations (SEQUENTIAL, PARALLEL, HIERARCHICAL, PIPELINE, CONSENSUS)

## Migration Paths

### 1. AgenticLoop Users

**Before** (Custom while loop - now legacy):
```python
from victor.framework.agentic_loop import AgenticLoop

loop = AgenticLoop(orchestrator)
result = await loop.run("Write tests")
```

**After** (StateGraph-based - now default with feature flag):
```python
# Enable StateGraph path (opt-in during migration)
export VICTOR_USE_STATEGRAPH_AGENTIC_LOOP=true

# Same API - no code changes needed
from victor.framework.agentic_loop import AgenticLoop

loop = AgenticLoop(orchestrator)
result = await loop.run("Write tests")
```

### 2. Coordinator Users

**Before** (Direct coordinator instantiation):
```python
from victor.agent.coordinators.exploration_state_passed import ExplorationStatePassedCoordinator

coordinator = ExplorationStatePassedCoordinator()
result = await coordinator.explore(context_snapshot, user_message="Find files")
```

**After** (StateGraph node):
```python
from victor.framework.agentic_graph.coordinator_nodes import exploration_node

result = await exploration_node(state, exploration_coordinator=coordinator)
```

### 3. Service Access

**Before** (Direct service access):
```python
chat_service = orchestrator.chat_service
response = await chat_service.chat(message="Hello")
```

**After** (Via ExecutionContext):
```python
from victor.framework.agentic_graph.service_nodes import chat_service_node

result = await chat_service_node(state, message="Hello")
response = result.context.get("last_response")
```

### 4. Team Formation Users

**Before** (Direct coordinator usage):
```python
from victor.teams import UnifiedTeamCoordinator, TeamFormation

coordinator = UnifiedTeamCoordinator(orchestrator)
coordinator.set_formation(TeamFormation.PARALLEL)
coordinator.add_member(agent1).add_member(agent2)
result = await coordinator.execute_task("Task", {})
```

**After** (Use UnifiedTeamCoordinator directly as a StateGraph node):
```python
from victor.framework import StateGraph
from victor.teams import UnifiedTeamCoordinator, TeamFormation

# Create coordinator
coordinator = UnifiedTeamCoordinator(orchestrator)
coordinator.set_formation(TeamFormation.PARALLEL)
coordinator.add_member(agent1).add_member(agent2)

# Use directly in StateGraph
graph = StateGraph(AgentState)
graph.add_node("parallel_research", coordinator)
graph.add_edge("analyze", "parallel_research")
graph.add_edge("parallel_research", "synthesize")
```

**With formation selection:**
```python
from victor.framework.agentic_graph.team_selector import select_formation

formation = select_formation(state)
coordinator.set_formation(formation)
```

## Feature Flags

Control consolidation behavior with environment variables:

| Flag | Default | Purpose |
|------|---------|---------|
| `VICTOR_USE_STATEGRAPH_AGENTIC_LOOP` | `false` | Enable StateGraph-based agentic loop |
| `VICTOR_USE_FRAMEWORK_COORDINATORS` | `false` | Use framework coordinator nodes |
| `VICTOR_USE_CONTEXT_SERVICE_INJECTION` | `false` | Enable service injection via context |
| `VICTOR_USE_FRAMEWORK_TEAMS` | `false` | Use framework team formations |

## Rollback

If you encounter issues, disable specific phases:

```bash
# Rollback to legacy agentic loop
export VICTOR_USE_STATEGRAPH_AGENTIC_LOOP=false

# Rollback to legacy coordinators
export VICTOR_USE_FRAMEWORK_COORDINATORS=false

# Rollback to legacy team handling
export VICTOR_USE_FRAMEWORK_TEAMS=false
```

## Breaking Changes

### None During Migration

All changes are backward compatible. Legacy code paths remain available via feature flags.

### Future Deprecations

The following will be deprecated in future releases:

1. **Direct while-loop execution** in AgenticLoop
2. **Direct coordinator instantiation** (use nodes instead)
3. **Global service access** (use ExecutionContext)

## Performance

Expected performance impact:
- **<5% overhead** from StateGraph execution
- **No degradation** in streaming performance
- **Improved** parallel team execution

## Testing

Run your tests with the new architecture:

```bash
# Enable all consolidation features
export VICTOR_USE_STATEGRAPH_AGENTIC_LOOP=true

# Run tests
pytest tests/
```

## Getting Help

- GitHub Issues: https://github.com/your-org/victor/issues
- Documentation: `CLAUDE.md` in repository
- Migration support: `migration.md` (this file)
