# Victor Architecture Roadmap

**Last Updated**: December 2025
**Purpose**: Persistent roadmap for architecture improvements across development sessions

---

## Recent Completions (December 2025)

### Phase 1: Foundation Fixes ✅

| Task | Status | Evidence |
|------|--------|----------|
| Remove duplicate vertical files | ✅ | Deleted `coding.py`, `research.py` (v1.0.0 shadowed by packages) |
| Enforce TaskTypeHint return type | ✅ | All prompts.py files return `Dict[str, TaskTypeHint]` |
| Fix CapabilityRegistryProtocol coupling | ✅ | Removed 18 private attribute fallbacks from vertical_integration.py |
| Add vertical config caching | ✅ | 26x speedup via `_config_cache` in VerticalBase |

**Commit**: `refactor(verticals): Phase 1 Foundation Fixes - SOLID compliance and caching`

### Phase 2: Core Extraction ✅

| Task | Status | Notes |
|------|--------|-------|
| Extract TaskTypeHint to victor/core/ | ✅ | Created `victor/core/vertical_types.py` as canonical source |
| Unify ModeConfig schema | ✅ | Schema unified in `victor/core/mode_config.py`; vertical adoption incremental |
| Merge tool dependency models | ✅ | Analysis complete - models serve different purposes, no merge needed |
| Create core/vertical_types.py | ✅ | Contains StageDefinition, TaskTypeHint, MiddlewarePriority, MiddlewareResult, TieredToolConfig |

**Key Changes**:
- `victor/core/vertical_types.py` - New canonical location for cross-vertical types
- All existing imports maintained for backward compatibility
- 209 framework unit tests pass

---

## Current Phase: Architecture Improvements (Phase 3)

### Objective
Improve architecture patterns for better testability and maintainability.

### Tasks

| Task | Priority | Description | Benefit |
|------|----------|-------------|---------|
| Split VerticalIntegrationPipeline | P1 | Separate 10 steps into focused classes | Better testability |
| Add capability versioning | P2 | Version capabilities for backward compat | Safe upgrades |
| Fix UI layer protocol violations | P1 | UI accesses private `_attributes` | ISP compliance |
| Add workflow caching | P2 | Cache workflow execution results | Performance |

---

## Upcoming Phase: Advanced Patterns (Phase 4)

| Task | Description | Reference |
|------|-------------|-----------|
| StateGraph DSL | Declarative workflow definitions | Similar to LangGraph patterns |
| Crew personas via TeamMember | Rich agent personas with backstory | victor/agent/teams/ |
| LCEL-style composition | Pipe-based tool chaining | Functional composition |
| Dynamic capability loading | Hot-reload capabilities at runtime | Plugin architecture |

---

## Technical Debt Tracker

### High Priority

| Issue | Location | Impact | Effort |
|-------|----------|--------|--------|
| UI layer protocol violations | victor/ui/*.py | ISP violation | Medium |
| Workflow caching needed | victor/workflows/ | Performance | Medium |
| MCP client resource leak | victor/mcp/client.py | Resource leak | Low |

### Medium Priority

| Issue | Location | Impact | Effort |
|-------|----------|--------|--------|
| Tool DI still uses global state | victor/tools/*.py | Testing difficulty | High |
| Orchestrator size (3,178 lines) | victor/agent/orchestrator.py | Maintainability | High |
| Async blocking in MCP | victor/mcp/server.py | Performance | Medium |

---

## SOLID Compliance Status

| Principle | Status | Evidence | Remaining Work |
|-----------|--------|----------|----------------|
| SRP | ✅ | Pipeline methods focused | - |
| OCP | ✅ | Capability registry extensible | - |
| LSP | ✅ | TaskTypeHint return types fixed | - |
| ISP | ⚠️ | Protocols are focused | UI layer still accesses private attrs |
| DIP | ✅ | Uses CapabilityRegistryProtocol | - |

---

## Architecture Metrics

### Protocol Coverage

| Module | Protocols | Runtime Checkable | Status |
|--------|-----------|-------------------|--------|
| victor/protocols/ | 15+ | Yes | ✅ |
| victor/framework/ | 10+ | Yes | ✅ |
| victor/agent/ | 8+ | Yes | ✅ |
| victor/verticals/ | 5+ | Yes | ✅ |
| **Total** | **150+** | **Yes** | ✅ |

### Performance Baselines

| Operation | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Config access (cached) | 26ms | 1ms | 26x |
| Extensions access (cached) | ~20ms | ~1ms | ~20x |
| Tool selection (semantic) | 50ms | 50ms | Baseline |
| Tool selection (keyword) | <1ms | <1ms | Baseline |

---

## Session Resume Context

### Key Files for Continuing Work

```
# Phase 2 completed - canonical locations
victor/core/vertical_types.py                  # TaskTypeHint, StageDefinition, etc.
victor/core/mode_config.py                     # Unified ModeDefinition schema
victor/core/tool_types.py                      # Tool dependency types

# For Phase 3
victor/framework/vertical_integration.py       # Split pipeline (10 steps)
victor/ui/tui/app.py                          # Fix protocol violations

# Reference implementations
victor/verticals/base.py:_config_cache        # Caching pattern (26x speedup)
victor/verticals/coding/assistant.py          # Vertical template
```

### Import Patterns

```python
# Current (Phase 2 complete) - canonical imports
from victor.core.vertical_types import TaskTypeHint, StageDefinition, TieredToolConfig
from victor.core.mode_config import ModeDefinition, ModeConfigRegistry
from victor.core import TaskTypeHint  # Also works via __init__.py

# Backward compatible (still works)
from victor.verticals.protocols import TaskTypeHint, PromptContributorProtocol
from victor.verticals.base import StageDefinition
```

---

## Architecture Strengths Summary

| Capability | Evidence | Status |
|------------|----------|--------|
| 150+ Protocols | victor/protocols/, framework/, agent/ | ✅ |
| 26x Config Caching | verticals/base.py:_config_cache | ✅ |
| Provider Agnostic | 25+ providers via BaseProvider | ✅ |
| Tool Ecosystem | 45 tools with cost-aware selection | ✅ |
| Multi-Agent Teams | 4 formations in victor/agent/teams/ | ✅ |
| DAG Workflows | YAML-based DSL with HITL nodes | ✅ |
| Adaptive Learning | 13 RL learners for tool selection | ✅ |
| Vertical Extensibility | 4 domain verticals | ✅ |
| Air-Gapped Support | 100% local operation | ✅ |
| MCP Protocol | Both client and server | ✅ |

---

*This roadmap is maintained for development session continuity. Update after completing phases.*
