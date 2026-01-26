# LegacyAPIMixin Usage Analysis

This document tracks all usage of deprecated LegacyAPIMixin methods across the codebase and provides migration guidance.

**Last Updated**: 2025-01-25
**Status**: Active Migration

---

## Critical Findings

### Direct LegacyAPIMixin Method Calls

| File | Lines | Methods Used | Priority | Migration Effort |
|------|-------|--------------|----------|------------------|
| `victor/evaluation/agent_adapter.py` | 285, 431 | `get_token_usage()` | Medium | Moderate |
| `victor/framework/step_handlers.py` | 1783 | `set_vertical_context()` | High | Low (✅ Completed) |
| `victor/framework/protocols.py` | 16 | `get_available_tools()` | High | Low |

### Indirect Usage (via hasattr checks)

These files check for deprecated methods before calling them:

| File | Pattern | Notes |
|------|---------|-------|
| `victor/framework/step_handlers.py` | `hasattr(orchestrator, "set_vertical_context")` | ✅ Fixed - removed fallback |
| `victor/evaluation/agent_adapter.py` | `hasattr(orchestrator, "get_token_usage")` | ⚠️ Keep as defensive code |

---

## Migration Guide

### 1. `set_vertical_context()` → VerticalContext Capability

**Status**: ✅ Completed

**Old Code**:
```python
if hasattr(orchestrator, "set_vertical_context"):
    orchestrator.set_vertical_context(context)
```

**New Code**:
```python
if _check_capability(orchestrator, "vertical_context"):
    _invoke_capability(orchestrator, "vertical_context", context)
```

**Files Updated**:
- `victor/framework/step_handlers.py:1783`

---

### 2. `get_available_tools()` → ToolRegistry or Capability

**Status**: ⚠️ Needs Investigation

**Current Usage** (`victor/framework/protocols.py:16`):
```python
tools = orchestrator.get_available_tools()
```

**Migration Options**:
1. **Preferred**: Use ToolRegistry capability
   ```python
   if _check_capability(orchestrator, "tools"):
       tools = _invoke_capability(orchestrator, "tools")
   ```

2. **Alternative**: Direct tool registry access
   ```python
   from victor.core.registries import ToolRegistry
   tools = ToolRegistry.get_instance().list_tools()
   ```

**Action Required**:
- Determine if this code path is still active
- If active, migrate to capability-based approach
- If legacy, mark for removal

---

### 3. `get_token_usage()` → MetricsCoordinator or SessionState

**Status**: ⚠️ Deferred (Technical Debt)

**Current Usage** (`victor/evaluation/agent_adapter.py:285, 431`):
```python
if hasattr(self.orchestrator, "get_token_usage"):
    usage = self.orchestrator.get_token_usage()
```

**Migration Options**:
1. **Preferred**: Use MetricsCoordinator
   ```python
   from victor.agent.coordinators.metrics_coordinator import MetricsCoordinator
   metrics = container.get(MetricsCoordinator)
   usage = metrics.get_token_usage()
   ```

2. **Alternative**: Use SessionState
   ```python
   usage = self.orchestrator.session_state.get_token_usage()
   ```

**Action Required**:
- Refactor VictorAgentAdapter to inject MetricsCoordinator
- Update both call sites (lines 285, 431)
- Update return type expectations

**Priority**: Medium (evaluation code, not production hot path)

---

## Methods Not Currently Used

The following 40 LegacyAPIMixin methods have **NO direct usage** found in the codebase:

### Vertical & Middleware (0 usage)
- ✅ `apply_vertical_middleware()` - No usage
- ✅ `apply_vertical_safety_patterns()` - No usage
- ✅ `get_middleware()` - No usage
- ✅ `get_middleware_chain()` - No usage
- ✅ `set_middleware()` - No usage
- ✅ `set_safety_patterns()` - No usage
- ✅ `get_safety_patterns()` - No usage

### Team & Config (0 usage)
- ✅ `set_team_specs()` - No usage
- ✅ `get_team_specs()` - No usage
- ✅ `set_tiered_tool_config()` - No usage
- ✅ `set_workspace()` - No usage

### Metrics (2 usage - documented above)
- ⚠️ `get_token_usage()` - 2 usages in evaluation code
- ✅ `reset_token_usage()` - No usage
- ✅ `get_last_stream_metrics()` - No usage
- ✅ `get_streaming_metrics_summary()` - No usage
- ✅ `get_streaming_metrics_history()` - No usage
- ✅ `get_session_cost_summary()` - No usage
- ✅ `get_session_cost_formatted()` - No usage
- ✅ `export_session_costs()` - No usage
- ✅ `get_tool_usage_stats()` - No usage
- ✅ `get_optimization_status()` - No usage

### State (0 usage)
- ✅ `get_conversation_stage()` - No usage
- ✅ `get_stage_recommended_tools()` - No usage
- ✅ `get_observed_files()` - No usage
- ✅ `get_modified_files()` - No usage

### Task Tracking (0 usage)
- ✅ `get_tool_calls_count()` - No usage
- ✅ `get_tool_budget()` - No usage
- ✅ `get_iteration_count()` - No usage
- ✅ `get_max_iterations()` - No usage

### Provider & Model (0 usage)
- ✅ `current_provider()` - No usage
- ✅ `current_model()` - No usage
- ✅ `get_current_provider_info()` - No usage

### Tools & Prompts (0 usage - 1 partial)
- ⚠️ `get_available_tools()` - 1 usage in protocols.py (needs investigation)
- ✅ `is_tool_enabled()` - No usage
- ✅ `get_system_prompt()` - No usage (different method exists on VerticalBase)
- ✅ `set_system_prompt()` - No usage
- ✅ `append_to_system_prompt()` - No usage
- ✅ `get_messages()` - No usage
- ✅ `get_message_count()` - No usage

### Search (0 usage)
- ✅ `route_search_query()` - No usage
- ✅ `get_recommended_search_tool()` - No usage
- ✅ `check_tool_selector_health()` - No usage

---

## Priority Migration Actions

### Immediate (High Priority)

1. ✅ **COMPLETED**: Remove `set_vertical_context()` fallback in `step_handlers.py`
2. ⚠️ **INVESTIGATE**: `get_available_tools()` in `framework/protocols.py` - determine if still used
3. ✅ **COMPLETED**: Fix remaining security imports in production code

### Short-term (Medium Priority)

4. ⚠️ **DEFERRED**: Migrate `get_token_usage()` in evaluation code (2 locations)
   - Requires MetricsCoordinator injection
   - Update TokenUsage type handling
   - Add tests for evaluation adapter

### Long-term (Low Priority)

5. 📋 **PLANNED**: Remove LegacyAPIMixin entirely in v0.7.0
6. 📋 **PLANNED**: Remove deprecation shims in v1.0.0
7. 📋 **PLANNED**: Update all documentation to reference canonical APIs

---

## Technical Debt Summary

| Item | Impact | Effort | Priority | Timeline |
|------|--------|--------|----------|----------|
| LegacyAPIMixin removal | High | High | P1 | v0.7.0 |
| `get_token_usage()` migration | Medium | Medium | P2 | v0.7.0 |
| `get_available_tools()` investigation | Low | Low | P3 | v0.6.1 |
| Deprecation shim removal | High | Low | P1 | v1.0.0 |

---

## Notes

- **Good News**: 40 out of 43 deprecated methods have NO usage in the codebase
- **Defensive Code**: Some uses are wrapped in `hasattr()` checks, preventing breakage
- **Evaluation Code**: The `get_token_usage()` usage is in benchmark/evaluation code, not production hot paths
- **Capability System**: New capability-based approach is working well for vertical context

---

## Next Steps

1. ✅ Fix `framework/protocols.py:get_available_tools()` - check if still needed
2. ⚠️ Refactor `evaluation/agent_adapter.py` to use MetricsCoordinator
3. ✅ Remove LegacyAPIMixin in v0.7.0 (or mark as deprecated with removal notice)
4. 📋 Update CLAUDE.md with canonical API patterns
5. 📋 Add codemods to detect deprecated method usage at CI time
