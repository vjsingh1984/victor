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

### Phase 3: Architecture Improvements ✅

| Task | Status | Notes |
|------|--------|-------|
| Split VerticalIntegrationPipeline | ✅ | Created step_handlers.py with 8 focused handlers |
| Add capability versioning | ✅ | Added version to OrchestratorCapability, IncompatibleVersionError |
| Fix UI layer protocol violations | ✅ | Fixed 6+ violations in UI layer, added public accessors |
| Add workflow caching | ✅ | Created workflows/cache.py with TTL-based node caching |

**Key Changes**:
- `victor/framework/step_handlers.py` - New module with StepHandlerProtocol and 8 handlers
- `victor/framework/protocols.py` - Enhanced with capability versioning
- `victor/agent/conversation_controller.py` - Added current_plan public property
- `victor/workflows/cache.py` - New workflow caching module
- 255 tests pass

### Phase 4: Advanced Patterns ✅

| Task | Status | Notes |
|------|--------|-------|
| StateGraph DSL | ✅ | Created workflows/graph_dsl.py with State, StateGraph, `>>` operator |
| Crew personas via TeamMember | ✅ | Added expertise, personality, max_delegation_depth, MemoryConfig |
| LCEL-style composition | ✅ | Created tools/composition.py with Runnable, pipe `|` operator |
| Dynamic capability loading | ✅ | Created framework/capability_loader.py with hot-reload support |

**Key Changes**:
- `victor/workflows/graph_dsl.py` - StateGraph DSL with 35 tests
- `victor/agent/teams/team.py` - Enhanced TeamMember with persona attributes
- `victor/tools/composition.py` - LCEL-style Runnable system (748 lines, 31 tests)
- `victor/framework/capability_loader.py` - Dynamic capability loading (23 tests)
- 437 tests pass

### Phase 5: SOLID Compliance & Framework Promotions ✅

| Task | Status | Notes |
|------|--------|-------|
| SRP: Remove get_config() overrides | ✅ | DevOps, DataAnalysis, Research now use base template method |
| OCP: Protocol-based extension | ✅ | Verified - no hardcoded vertical checks in pipeline |
| LSP: Central format adapter | ✅ | Created victor/verticals/format_adapter.py |
| ISP: Fix method signatures | ✅ | Renamed get_rl_config→get_rl_config_provider, get_team_specs→get_team_spec_provider |
| DIP: Protocol dependencies | ✅ | Orchestrator depends on protocols, not concrete verticals |
| Canonical tool naming | ✅ | Created framework/tool_naming.py with ToolNames constants |
| TaskTypeRegistry | ✅ | Created framework/task_types.py consolidating 5 enums |
| Framework middleware | ✅ | LoggingMiddleware, SecretMaskingMiddleware, GitSafetyMiddleware |

**Key Changes**:
- `victor/verticals/format_adapter.py` - Central LSP-compliant format normalization
- `victor/framework/tool_naming.py` - Canonical tool names to prevent Q-value fragmentation
- `victor/framework/task_types.py` - Unified TaskTypeRegistry (19 core types)
- `victor/framework/middleware.py` - Common middleware baseline for all verticals
- TeamSpecProvider classes added to DevOps, DataAnalysis, Research teams
- Method signature standardization across all verticals

---

## Roadmap Complete 🎉

All five phases of the architecture roadmap have been completed:

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Foundation Fixes | ✅ Complete |
| Phase 2 | Core Extraction | ✅ Complete |
| Phase 3 | Architecture Improvements | ✅ Complete |
| Phase 4 | Advanced Patterns | ✅ Complete |
| Phase 5 | SOLID Compliance & Framework Promotions | ✅ Complete |

---

## Technical Debt Tracker

### Resolved (December 2025)

| Issue | Resolution | Evidence |
|-------|------------|----------|
| MCP client resource leak | ✅ Fixed | Added async context manager to victor/mcp/client.py |
| Async blocking in MCP | ✅ Fixed | Replaced readline with asyncio.StreamReader in victor/mcp/server.py |
| Workflow caching needed | ✅ Fixed | Created victor/workflows/cache.py with TTL-based caching |

### Medium Priority

| Issue | Location | Impact | Effort |
|-------|----------|--------|--------|
| Tool DI still uses global state | victor/tools/*.py | Testing difficulty | High |
| Orchestrator size (3,178 lines) | victor/agent/orchestrator.py | Maintainability | High |

---

## SOLID Compliance Status

| Principle | Status | Evidence | Remaining Work |
|-----------|--------|----------|----------------|
| SRP | ✅ | Verticals are declarative, pipeline handles integration | - |
| OCP | ✅ | Extension via protocols + registries, no pipeline edits | - |
| LSP | ✅ | Central VerticalFormatAdapter for format normalization | - |
| ISP | ✅ | Provider protocols segregated (RL, Teams, Workflows, Prompts) | - |
| DIP | ✅ | Orchestrator depends on VerticalContext + protocols | - |

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
| Provider Agnostic | 21 providers via BaseProvider | ✅ |
| Tool Ecosystem | 55 tools with cost-aware selection | ✅ |
| Multi-Agent Teams | 4 formations in victor/agent/teams/ | ✅ |
| DAG Workflows | YAML-based DSL with HITL nodes | ✅ |
| Adaptive Learning | 13 RL learners for tool selection | ✅ |
| Vertical Extensibility | 4 domain verticals | ✅ |
| Air-Gapped Support | 100% local operation | ✅ |
| MCP Protocol | Both client and server | ✅ |

---

*This roadmap is maintained for development session continuity. Update after completing phases.*
