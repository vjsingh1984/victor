# Deprecated API Call Sites Migration Plan

This document tracks all deprecated API call sites that need migration to canonical APIs.

**Last Updated**: 2025-01-25
**Status**: In Progress

---

## Migration Summary

| Category | Total Files | Migrated | Remaining | Priority |
|----------|-------------|----------|-----------|----------|
| Security Infrastructure (auth/audit) | 6 | 3 | 3 | High |
| Security Analysis (safety → patterns) | 15 | 10 | 5 | High |
| Optimization (optimizations → runtime) | 4 | 2 | 2 | Medium |
| Deprecated Aliases | 20+ | 0 | 20+ | Low |

---

## Phase 1: Security Infrastructure (High Priority)

### Files Requiring Migration

| File | Line | Old Import | New Import | Status |
|------|------|------------|------------|--------|
| `victor/agent/tool_executor.py` | TBD | `victor.security.auth.rbac` | `victor.core.security.auth` | ✅ Done |
| `victor/config/api_keys.py` | TBD | `victor.security.audit` | `victor.core.security.audit` | ✅ Done |
| `victor/config/secure_paths.py` | TBD | `victor.security.audit` | `victor.core.security.audit` | ✅ Done |
| `victor/tools/audit_tool.py` | TBD | `victor.security.audit` | `victor.core.security.audit` | ✅ Done |

### Remaining Files (Need Migration)

| File | Current Import | Action Required |
|------|----------------|-----------------|
| `victor/security/__init__.py` | `victor.security.auth.*` | Deprecation shim - keep as-is |
| `victor/core/security/auth/rbac.py` | `victor.security.auth.rbac` | Re-export - keep as-is |
| `victor/core/verticals/protocols/safety_provider.py` | `victor.security.safety.types` | Update to `victor.security_analysis.patterns.types` |
| `victor/core/verticals/protocols/__init__.py` | `victor.security.safety.types` | Update to `victor.security_analysis.patterns.types` |

---

## Phase 2: Security Analysis (High Priority)

### Pattern Module Files

These files are in the new location but may have old imports in their content:

| File | Status | Notes |
|------|--------|-------|
| `victor/security_analysis/patterns/pii.py` | ✅ Verified | Re-exports from old location |
| `victor/security_analysis/patterns/content_patterns.py` | ⚠️ Check | May have old imports |
| `victor/security_analysis/patterns/source_credibility.py` | ⚠️ Check | May have old imports |
| `victor/security_analysis/patterns/infrastructure.py` | ⚠️ Check | May have old imports |
| `victor/security_analysis/patterns/code_patterns.py` | ⚠️ Check | May have old imports |
| `victor/security_analysis/patterns/secrets.py` | ⚠️ Check | May have old imports |
| `victor/security_analysis/patterns/registry.py` | ⚠️ Check | May have old imports |
| `victor/security_analysis/patterns/types.py` | ⚠️ Check | May have old imports |

### Old Location Files (Legacy)

| File | Status | Action |
|------|--------|--------|
| `victor/security/safety/pii.py` | 🔴 Deprecated | Replaced by `victor/security_analysis/patterns/pii.py` |
| `victor/security/safety/code_patterns.py` | 🔴 Deprecated | Replaced by `victor/security_analysis/patterns/code_patterns.py` |
| `victor/security/safety/infrastructure.py` | 🔴 Deprecated | Replaced by `victor/security_analysis/patterns/infrastructure.py` |
| `victor/security/safety/registry.py` | 🔴 Deprecated | Replaced by `victor/security_analysis/patterns/registry.py` |
| `victor/security/safety/__init__.py` | 🔴 Deprecated | Replaced by deprecation shim |

---

## Phase 3: Optimization (Medium Priority)

### Runtime Files

| File | Status | Action |
|------|--------|--------|
| `victor/optimizations/lazy_loader.py` | ✅ Copied | Now in `victor/optimization/runtime/lazy_loader.py` |
| `victor/optimizations/parallel_executor.py` | ✅ Copied | Now in `victor/optimization/runtime/parallel_executor.py` |
| `victor/optimizations/memory.py` | ✅ Copied | Now in `victor/optimization/runtime/memory.py` |
| `victor/optimizations/database.py` | ✅ Copied | Now in `victor/optimization/runtime/database.py` |
| `victor/optimizations/network.py` | ✅ Copied | Now in `victor/optimization/runtime/network.py` |
| `victor/optimizations/algorithms.py` | ✅ Copied | Now in `victor/optimization/runtime/algorithms.py` |
| `victor/optimizations/concurrency.py` | ✅ Copied | Now in `victor/optimization/runtime/concurrency.py` |

### Files Still Importing from Old Location

Check for remaining imports from `victor.optimizations` in production code.

---

## Phase 4: Deprecated Aliases (Low Priority)

### Type Aliases

| File | Alias | Canonical | Priority |
|------|-------|-----------|----------|
| `victor/core/security/__init__.py` | `Severity` | `CVESeverity` | Low |
| `victor/core/security/__init__.py` | `Dependency` | `SecurityDependency` | Low |
| `victor/security/__init__.py` | `EnhancedUser` | `User` | Low |

### Protocol Re-exports

These are backward compatibility re-exports and can remain until removal:

- Various `__init__.py` files re-export protocol types
- These provide convenience and don't need immediate migration
- Documented in `DEPRECATION_INVENTORY.md`

---

## Phase 5: LegacyAPIMixin Methods (43 methods)

All methods in `victor/agent/mixins/legacy_api.py` are deprecated and need migration:

### Vertical Context & Middleware

| Method | Replacement | Files Using It |
|--------|-------------|----------------|
| `set_vertical_context()` | `VerticalContext.set_context()` | TBD |
| `apply_vertical_middleware()` | `VerticalIntegrationAdapter.apply_middleware()` | TBD |
| `set_middleware()` | `VerticalIntegrationAdapter` | TBD |
| `get_middleware()` | `VerticalIntegrationAdapter` | TBD |
| `set_safety_patterns()` | `VerticalIntegrationAdapter` | TBD |
| `get_safety_patterns()` | `VerticalIntegrationAdapter` | TBD |

### Team & Configuration

| Method | Replacement | Files Using It |
|--------|-------------|----------------|
| `set_team_specs()` | `TeamCoordinator.set_team_specs()` | TBD |
| `get_team_specs()` | `TeamCoordinator.get_team_specs()` | TBD |
| `set_tiered_tool_config()` | `ConfigurationManager.set_tiered_tool_config()` | TBD |

### Metrics & Analytics

| Method | Replacement | Files Using It |
|--------|-------------|----------------|
| `get_tool_usage_stats()` | `MetricsCoordinator.get_tool_usage_stats()` | TBD |
| `get_token_usage()` | `MetricsCoordinator.get_token_usage()` | TBD |
| `reset_token_usage()` | `MetricsCoordinator.reset_token_usage()` | TBD |
| `get_last_stream_metrics()` | `MetricsCoordinator.get_last_stream_metrics()` | TBD |
| `get_streaming_metrics_summary()` | `MetricsCoordinator.get_streaming_metrics_summary()` | TBD |
| `get_streaming_metrics_history()` | `MetricsCoordinator.get_streaming_metrics_history()` | TBD |
| `get_session_cost_summary()` | `MetricsCoordinator.get_session_cost_summary()` | TBD |
| `get_session_cost_formatted()` | `MetricsCoordinator.get_session_cost_formatted()` | TBD |
| `export_session_costs()` | `MetricsCoordinator.export_session_costs()` | TBD |
| `get_optimization_status()` | `AnalyticsCoordinator.get_optimization_status()` | TBD |

### State & Conversation

| Method | Replacement | Files Using It |
|--------|-------------|----------------|
| `get_conversation_stage()` | `StateCoordinator.get_stage()` | TBD |
| `get_stage_recommended_tools()` | `StateCoordinator.get_stage_tools()` | TBD |
| `get_observed_files()` | `StateCoordinator.observed_files` (property) | TBD |
| `get_modified_files()` | `conversation_state.state.modified_files` | TBD |

### Task Tracking

| Method | Replacement | Files Using It |
|--------|-------------|----------------|
| `get_tool_calls_count()` | `unified_tracker.tool_calls_used` | TBD |
| `get_tool_budget()` | `unified_tracker.tool_budget` | TBD |
| `get_iteration_count()` | `unified_tracker.iteration_count` | TBD |
| `get_max_iterations()` | `unified_tracker.max_iterations` | TBD |

### Provider & Model

| Method | Replacement | Files Using It |
|--------|-------------|----------------|
| `current_provider()` | `orchestrator.provider_name` | TBD |
| `current_model()` | `orchestrator.model` | TBD |
| `get_current_provider_info()` | `ProviderManager.get_info()` | TBD |

### Tools & Prompts

| Method | Replacement | Files Using It |
|--------|-------------|----------------|
| `get_available_tools()` | `tools.list_tools()` | TBD |
| `is_tool_enabled()` | `ToolAccessConfigCoordinator.is_tool_enabled()` | TBD |
| `get_system_prompt()` | `prompt_builder.build()` | TBD |
| `set_system_prompt()` | `prompt_builder.set_custom_prompt()` | TBD |
| `append_to_system_prompt()` | `prompt_builder.append_content()` | TBD |
| `get_messages()` | `conversation.messages` | TBD |
| `get_message_count()` | `len(conversation.messages)` | TBD |

### Search & Discovery

| Method | Replacement | Files Using It |
|--------|-------------|----------------|
| `route_search_query()` | `SearchCoordinator.route_search_query()` | TBD |
| `get_recommended_search_tool()` | `SearchCoordinator.get_recommended_search_tool()` | TBD |
| `check_tool_selector_health()` | `tool_selector.get_health()` | TBD |

### Workspace & Initialization

| Method | Replacement | Files Using It |
|--------|-------------|----------------|
| `set_workspace()` | `set_project_root()` from settings | TBD |
| `get_middleware_chain()` | `StateCoordinator` or properties | TBD |

---

## Migration Order

1. **Phase 1**: Security Infrastructure (High) - Complete remaining 3 files
2. **Phase 2**: Security Analysis (High) - Check and fix pattern files
3. **Phase 3**: Optimization (Medium) - Find and fix remaining old imports
4. **Phase 4**: Deprecated Aliases (Low) - Update code using aliases
5. **Phase 5**: LegacyAPIMixin (High Impact) - Find and migrate all 43 method calls

---

## Next Steps

1. Search for all usages of LegacyAPIMixin methods
2. Update deprecated imports in remaining files
3. Test all migrations
4. Update documentation
5. Mark old files for removal in v1.0.0
