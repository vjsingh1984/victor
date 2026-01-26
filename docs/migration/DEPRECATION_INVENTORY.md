# Victor Deprecation Inventory

Comprehensive inventory of all deprecation notices, shims, aliases, and legacy exports in the Victor codebase.

**Last Updated**: 2025-01-25
**Coverage**: Full codebase scan with automated agents

---

## Summary Statistics

- **Total module-level deprecations**: 7 major modules
- **Total deprecated methods**: 43 methods in LegacyAPIMixin
- **Total deprecated classes**: 8+ classes
- **Total backward compatibility aliases**: 50+ re-exports
- **Files with `__getattr__` shims**: 2 (security, optimizations)
- **Removal timeline**: v0.7.0, v1.0.0, v2.0.0

---

## Module-Level Deprecations

### Security Module Reorganization

| Old Location | New Location | Type | Deprecated | Removal | Notes |
|--------------|--------------|------|------------|---------|-------|
| `victor.security.auth` | `victor.core.security.auth` | module | 0.6.0 | 1.0.0 | Infrastructure moved to core |
| `victor.security.audit` | `victor.core.security.audit` | module | 0.6.0 | 1.0.0 | Infrastructure moved to core |
| `victor.security.safety` | `victor.security_analysis.patterns` | module | 0.6.0 | 1.0.0 | Analysis tools to vertical |
| `victor.security.penetration_testing` | `victor.security_analysis.tools` | module | 0.6.0 | 1.0.0 | Analysis tools to vertical |
| `victor.security.authorization_enhanced` | `victor.core.security.authorization` | module | 0.6.0 | 1.0.0 | Infrastructure moved to core |

### Optimization Module Consolidation

| Old Location | New Location | Type | Deprecated | Removal | Notes |
|--------------|--------------|------|------------|---------|-------|
| `victor.optimizations` | `victor.optimization.runtime` | module | 0.6.0 | 1.0.0 | Runtime optimizations |
| `victor.core.optimizations` | `victor.optimization.core` | module | 0.6.0 | 1.0.0 | Hot path utilities |

---

## Deprecation Shims and Warnings

### Module-Level `__getattr__` Shims

| File | Deprecated Item | Redirect | Warning Message |
|------|-----------------|----------|-----------------|
| `victor/security/__init__.py:249` | `auth` | `victor.core.security.auth` | "victor.security.auth is deprecated. Use victor.core.security.auth" |
| `victor/security/__init__.py:267` | `safety` | `victor.security_analysis.patterns` | "victor.security.safety is deprecated. Use victor.security_analysis.patterns" |
| `victor/security/__init__.py:279` | `audit` | `victor.core.security.audit` | "victor.security.audit is deprecated. Use victor.core.security.audit" |
| `victor/security/__init__.py:291` | `penetration_testing` | `victor.security_analysis.tools` | "victor.security.penetration_testing is deprecated..." |
| `victor/security/__init__.py:305` | `authorization_enhanced` | `victor.core.security.authorization` | "victor.security.authorization_enhanced is deprecated..." |
| `victor/optimizations/__init__.py:61` | Module | `victor.optimization.runtime` | "victor.optimizations is deprecated and will be removed in v1.0.0" |
| `victor/core/optimizations/__init__.py:34` | Module | `victor.optimization.core` | "victor.core.optimizations is deprecated and will be removed in v1.0.0" |

---

## Legacy Class and Type Aliases

### Security Infrastructure

| File | Line | Deprecated Item | Replacement | Purpose |
|------|------|-----------------|-------------|---------|
| `victor/security/__init__.py` | 178 | `EnhancedUser` | `User` | Legacy class name alias |
| `victor/core/security/__init__.py` | 145 | `Severity` (enum) | `CVESeverity` | Enum alias |
| `victor/core/security/__init__.py` | 150 | `Dependency` | `SecurityDependency` | Type alias |
| `victor/security/protocol.py` | 68 | `CVE` | Re-exported | Protocol re-export |
| `victor/security/protocol.py` | 140 | `SecurityScanResult` | Re-exported | Protocol re-export |
| `victor/security/audit/protocol.py` | 92 | `AuditEvent` | Re-exported | Protocol re-export |
| `victor/security/safety/code_patterns.py` | 408 | `SafetyPattern` | Re-exported | Protocol re-export |

### Protocol and Framework Aliases

| File | Line | Alias | Original | Purpose |
|------|------|-------|----------|---------|
| `victor/protocols/grounding.py` | 104 | `GroundingRuleProtocol` | Multiple | Backward compatibility |
| `victor/deps/protocol.py` | 161 | `Dependency` | Protocol type | Convenience |
| `victor/deps/parsers.py` | 83 | Multiple | Parser types | Convenience |
| `victor/iac/protocol.py` | 68, 241 | Multiple | IaC types | Convenience |
| `victor/tools/selection/protocol.py` | 163 | Multiple | Tool selection types | Convenience |
| `victor/integrations/api/fastapi_server.py` | 186 | `WebSocket` | Protocol type | Convenience |

### Agent and Coordinator Aliases

| File | Line | Alias | Original | Purpose |
|------|------|-------|----------|---------|
| `victor/agent/resilience.py` | 294 | `RetryConfig` | Protocol type | Convenience |
| `victor/agent/protocols.py` | 274 | Multiple | Agent protocol types | Convenience |
| `victor/agent/recovery/protocols.py` | 97 | Multiple | Recovery types | Convenience |
| `victor/agent/recovery/fallback.py` | 134 | `FallbackStrategy` | Protocol type | Convenience |
| `victor/agent/unified_task_tracker.py` | 130 | Multiple | Tracker types | Convenience |
| `victor/agent/grounding_verifier.py` | 828 | `GroundingVerifier` | Protocol type | Convenience |
| `victor/agent/error_recovery.py` | 55 | Multiple | Error recovery types | Convenience |
| `victor/agent/adaptive_mode_controller.py` | 105 | Multiple | Mode types | Convenience |
| `victor/agent/conversation_export.py` | 700 | Multiple | Export types | Convenience |
| `victor/agent/safety.py` | 141 | Multiple | Safety types | Convenience |

### Provider and Storage Aliases

| File | Line | Alias | Original | Purpose |
|------|------|-------|----------|---------|
| `victor/providers/base.py` | 233 | Error classes | `victor.core.errors` | Backward compatibility |
| `victor/providers/resilience.py` | 286 | Multiple | Resilience types | Convenience |
| `victor/storage/unified/protocol.py` | 238 | Multiple | Storage types | Convenience |
| `victor/storage/state/__init__.py` | 87 | Multiple | State types | Conversation-specific |

### Framework and Workflow Aliases

| File | Line | Alias | Original | Purpose |
|------|------|-------|----------|---------|
| `victor/workflows/compiled_executor.py` | 137 | Multiple | Executor types | Convenience |
| `victor/agent/orchestrator_recovery.py` | 141 | Multiple | Recovery types | Convenience |
| `victor/framework/rl/learners/mode_transition.py` | 59 | Multiple | RL types | Convenience |
| `victor/coding/review/protocol.py` | 42 | Multiple | Review types | Convenience |
| `victor/coding/codebase/indexer.py` | 940 | Multiple | Indexer types | Convenience |
| `victor/ui/tui/widgets.py` | 1719 | Multiple | Widget types | Convenience |

---

## Deprecated Classes

### Legacy Mixins and Adapters

| File | Line | Class | Replacement | Removal | Notes |
|------|------|-------|-------------|---------|-------|
| `victor/agent/mixins/legacy_api.py` | 49 | `LegacyAPIMixin` | Individual coordinators | 0.7.0 | 500+ lines consolidated |
| `victor/agent/creation_strategies.py` | 185 | `LegacyStrategy` | `FrameworkStrategy` | Future | For troubleshooting |
| `victor/core/verticals/prompt_adapter.py` | 69 | `LegacyTaskHint` | `TaskTypeHint` | Future | Dict-based format |
| `victor/core/verticals/capability_adapter.py` | 82 | `LegacyWriteMode` | Enum values | Future | Migration phases |
| `victor/coding/codebase/unified_extractor.py` | 184 | `LegacyTierConfig` | `UnifiedLanguageCapability` | Future | Adapter interface |

### Tool Dependency Providers (Vertical-Specific)

| File | Class | Replacement | Notes |
|------|--------|-------------|-------|
| `victor/coding/tool_dependencies.py` | `CodingToolDependencyProvider` | `create_vertical_tool_dependency_provider("coding")` | Factory function |
| `victor/devops/tool_dependencies.py` | `DevOpsToolDependencyProvider` | `create_vertical_tool_dependency_provider("devops")` | Factory function |
| `victor/rag/tool_dependencies.py` | `RAGToolDependencyProvider` | `create_vertical_tool_dependency_provider("rag")` | Factory function |
| `victor/research/tool_dependencies.py` | `ResearchToolDependencyProvider` | `create_vertical_tool_dependency_provider("research")` | Factory function |
| `victor/dataanalysis/tool_dependencies.py` | `DataAnalysisToolDependencyProvider` | `create_vertical_tool_dependency_provider("dataanalysis")` | Factory function |
| `victor/core/tool_dependency_backward_compat.py` | `DeprecatedToolDependencyProvider` | Wrapper class | Centralized deprecation |

---

## Deprecated Methods (LegacyAPIMixin)

All 43 methods in `victor/agent/mixins/legacy_api.py` with `@deprecated` decorator:

### Vertical Context and Middleware

| Line | Method | Replacement | Removal | Reason |
|------|--------|-------------|---------|--------|
| 129 | `set_vertical_context()` | `VerticalContext.set_context()` | 0.7.0 | Protocol violation |
| 178 | `set_workspace()` | `set_project_root()` from settings | 0.7.0 | Global management |
| 210 | `apply_vertical_middleware()` | `VerticalIntegrationAdapter.apply_middleware()` | 0.7.0 | Centralization |
| 227 | `apply_vertical_safety_patterns()` | `VerticalIntegrationAdapter.apply_safety_patterns()` | 0.7.0 | Centralization |
| 260 | `set_middleware()` | `VerticalIntegrationAdapter` | 0.7.0 | DIP violation |
| 276 | `get_middleware()` | `VerticalIntegrationAdapter` | 0.7.0 | DIP violation |
| 292 | `set_safety_patterns()` | `VerticalIntegrationAdapter` | 0.7.0 | DIP violation |
| 308 | `get_safety_patterns()` | `VerticalIntegrationAdapter` | 0.7.0 | DIP violation |
| 244 | `get_middleware_chain()` | `StateCoordinator` or properties | 0.7.0 | Encapsulation |

### Team and Configuration

| Line | Method | Replacement | Removal | Reason |
|------|--------|-------------|---------|--------|
| 324 | `set_team_specs()` | `TeamCoordinator.set_team_specs()` | 0.7.0 | Centralization |
| 345 | `get_team_specs()` | `TeamCoordinator.get_team_specs()` | 0.7.0 | Centralization |
| 155 | `set_tiered_tool_config()` | `ConfigurationManager.set_tiered_tool_config()` | 0.7.0 | Centralization |

### Metrics and Analytics

| Line | Method | Replacement | Removal | Reason |
|------|--------|-------------|---------|--------|
| 372 | `get_tool_usage_stats()` | `MetricsCoordinator.get_tool_usage_stats()` | 0.7.0 | Centralization |
| 392 | `get_token_usage()` | `MetricsCoordinator.get_token_usage()` | 0.7.0 | Centralization |
| 413 | `reset_token_usage()` | `MetricsCoordinator.reset_token_usage()` | 0.7.0 | Centralization |
| 428 | `get_last_stream_metrics()` | `MetricsCoordinator.get_last_stream_metrics()` | 0.7.0 | Centralization |
| 446 | `get_streaming_metrics_summary()` | `MetricsCoordinator.get_streaming_metrics_summary()` | 0.7.0 | Centralization |
| 464 | `get_streaming_metrics_history()` | `MetricsCoordinator.get_streaming_metrics_history()` | 0.7.0 | Centralization |
| 485 | `get_session_cost_summary()` | `MetricsCoordinator.get_session_cost_summary()` | 0.7.0 | Centralization |
| 503 | `get_session_cost_formatted()` | `MetricsCoordinator.get_session_cost_formatted()` | 0.7.0 | Centralization |
| 521 | `export_session_costs()` | `MetricsCoordinator.export_session_costs()` | 0.7.0 | Centralization |
| 585 | `get_optimization_status()` | `AnalyticsCoordinator.get_optimization_status()` | 0.7.0 | New coordinator |

### State and Conversation

| Line | Method | Replacement | Removal | Reason |
|------|--------|-------------|---------|--------|
| 544 | `get_conversation_stage()` | `StateCoordinator.get_stage()` | 0.7.0 | Centralization |
| 567 | `get_stage_recommended_tools()` | `StateCoordinator.get_stage_tools()` | 0.7.0 | Centralization |
| 612 | `get_observed_files()` | `StateCoordinator.observed_files` | 0.7.0 | Property access |
| 630 | `get_modified_files()` | `conversation_state.state.modified_files` | 0.7.0 | Direct access |

### Task Tracking

| Line | Method | Replacement | Removal | Reason |
|------|--------|-------------|---------|--------|
| 653 | `get_tool_calls_count()` | `unified_tracker.tool_calls_used` | 0.7.0 | Protocol |
| 671 | `get_tool_budget()` | `unified_tracker.tool_budget` | 0.7.0 | Protocol |
| 689 | `get_iteration_count()` | `unified_tracker.iteration_count` | 0.7.0 | Protocol |
| 707 | `get_max_iterations()` | `unified_tracker.max_iterations` | 0.7.0 | Protocol |

### Provider and Model

| Line | Method | Replacement | Removal | Reason |
|------|--------|-------------|---------|--------|
| 730 | `current_provider()` | `orchestrator.provider_name` | 0.7.0 | Property |
| 746 | `current_model()` | `orchestrator.model` | 0.7.0 | Property |
| 762 | `get_current_provider_info()` | `ProviderManager.get_info()` | 0.7.0 | Manager |

### Tools and Prompts

| Line | Method | Replacement | Removal | Reason |
|------|--------|-------------|---------|--------|
| 792 | `get_available_tools()` | `tools.list_tools()` | 0.7.0 | Protocol |
| 810 | `is_tool_enabled()` | `ToolAccessConfigCoordinator.is_tool_enabled()` | 0.7.0 | Coordinator |
| 836 | `get_system_prompt()` | `prompt_builder.build()` | 0.7.0 | Protocol |
| 854 | `set_system_prompt()` | `prompt_builder.set_custom_prompt()` | 0.7.0 | Protocol |
| 871 | `append_to_system_prompt()` | `prompt_builder.append_content()` | 0.7.0 | Protocol |
| 893 | `get_messages()` | `conversation.messages` | 0.7.0 | Protocol |
| 911 | `get_message_count()` | `len(conversation.messages)` | 0.7.0 | Protocol |

### Search and Discovery

| Line | Method | Replacement | Removal | Reason |
|------|--------|-------------|---------|--------|
| 934 | `route_search_query()` | `SearchCoordinator.route_search_query()` | 0.7.0 | Coordinator |
| 955 | `get_recommended_search_tool()` | `SearchCoordinator.get_recommended_search_tool()` | 0.7.0 | Coordinator |
| 981 | `check_tool_selector_health()` | `tool_selector.get_health()` | 0.7.0 | Direct access |

---

## Framework/Vertical Integration Deprecations

| File | Line | Type | Deprecated Item | Replacement | Removal | Notes |
|------|------|------|-----------------|-------------|---------|-------|
| `victor/framework/vertical_integration.py` | 182 | function | `apply_middleware_from_extensions()` | `VerticalIntegrationAdapter` | 0.7.0 | Function deprecated |
| `victor/framework/vertical_integration.py` | 223 | function | `apply_safety_patterns_from_extensions()` | `VerticalIntegrationAdapter` | 0.7.0 | Function deprecated |
| `victor/framework/step_handlers.py` | 199 | notice | `register_tool_dependency_handler()` | Built-in handler | 0.7.0 | Manual registration deprecated |
| `victor/framework/step_handlers.py` | 236 | notice | `register_capability_handler()` | Built-in handler | 0.7.0 | Manual registration deprecated |

---

## Configuration/Settings Deprecations

| File | Line | Type | Deprecated Item | Replacement | Removal | Notes |
|------|------|------|-----------------|-------------|---------|-------|
| `victor/config/settings.py` | 621 | setting | Legacy setting | New setting | 2.0.0 | Kept for backward compat |
| `victor/config/settings.py` | 752 | setting | `use_semantic_tool_selection` | New approach | 2.0.0 | With deprecation warning |

---

## Team/Persona Deprecations

| File | Line | Type | Deprecated Item | Replacement | Removal | Notes |
|------|------|------|-----------------|-------------|---------|-------|
| `victor/coding/teams/specs.py` | 25 | class | `CodingTeamSpec` | New spec system | Future | Backward compat maintained |
| `victor/coding/teams/specs.py` | 197 | method | Legacy method | New method | Future | Wrapper provided |
| `victor/research/teams/__init__.py` | 38 | class | `ResearchTeamSpec` | New spec system | Future | Backward compat maintained |
| `victor/research/teams/__init__.py` | 148 | method | Legacy method | New method | Future | Wrapper provided |
| `victor/teams/RELEASE_NOTES.md` | 118 | class | Team spec wrapper | Direct usage | 0.5.0 | Deprecated |

---

## Additional Backward Compatibility Files

Module-level redirects for debugging and analytics:

| File | Purpose | Type |
|------|---------|------|
| `victor/debug/manager.py` | Backward compatibility redirect | Module redirect |
| `victor/debug/adapters/__init__.py` | Backward compatibility redirect | Module redirect |
| `victor/debug/protocol.py` | Backward compatibility redirect | Module redirect |
| `victor/debug/adapter.py` | Backward compatibility redirect | Module redirect |
| `victor/debug/registry.py` | Backward compatibility redirect | Module redirect |
| `victor/analytics/streaming_metrics.py` | Backward compatibility redirect | Module redirect |
| `victor/analytics/logger.py` | Backward compatibility redirect | Module redirect |
| `victor/analytics/enhanced_logger.py` | Backward compatibility redirect | Module redirect |

---

## Migration Patterns

### Security Infrastructure → Core
```python
# OLD
from victor.security.auth import RBACManager
from victor.security.audit import AuditManager

# NEW
from victor.core.security.auth import RBACManager
from victor.core.security.audit import AuditManager
```

### Security Analysis → Vertical
```python
# OLD
from victor.security.safety.types import SafetyPattern
from victor.security.penetration_testing import SecurityTestSuite

# NEW
from victor.security_analysis.patterns.types import SafetyPattern
from victor.security_analysis.tools import SecurityTestSuite
```

### Optimization Consolidation
```python
# OLD
from victor.optimizations import LazyComponentLoader
from victor.core.optimizations import json_dumps

# NEW
from victor.optimization.runtime import LazyComponentLoader
from victor.optimization.core import json_dumps
```

### Coordinator Pattern (Instead of Orchestrator Methods)
```python
# OLD (via LegacyAPIMixin)
token_usage = orchestrator.get_token_usage()

# NEW (direct coordinator access)
from victor.agent.coordinators import MetricsCoordinator
metrics = container.get(MetricsCoordinator)
token_usage = metrics.get_token_usage()
```

### Tool Dependencies (Factory Pattern)
```python
# OLD
from victor.coding.tool_dependencies import CodingToolDependencyProvider
provider = CodingToolDependencyProvider()

# NEW
from victor.core.tool_dependency_factory import create_vertical_tool_dependency_provider
provider = create_vertical_tool_dependency_provider("coding")
```

---

## Removal Timeline

### v0.7.0 (Next Release)
- Remove all 43 LegacyAPIMixin methods
- Remove framework vertical integration functions
- Remove manual handler registration functions

### v1.0.0 (Major Release)
- Remove `victor.security.*` deprecation shims
- Remove `victor.optimizations` module
- Remove `victor.core.optimizations` module
- Remove legacy team spec classes

### v2.0.0 (Future)
- Remove legacy configuration settings
- Remove backward compatibility aliases

---

## Verification

To check for remaining deprecated imports in your code:

```bash
# Check for old security imports
grep -r "from victor\.security\.\(auth\|audit\|safety\)" --include="*.py"

# Check for old optimization imports
grep -r "from victor\.optimizations" --include="*.py"
grep -r "from victor\.core\.optimizations" --include="*.py"

# Run with deprecation warnings as errors
python -W error::DeprecationWarning your_code.py
```

---

## References

- Main Migration Guide: [SECURITY_OPTIMIZATION_MIGRATION.md](SECURITY_OPTIMIZATION_MIGRATION.md)
- Architecture Refactoring: [../architecture/REFACTORING_OVERVIEW.md](../architecture/REFACTORING_OVERVIEW.md)
- Best Practices: [../architecture/BEST_PRACTICES.md](../architecture/BEST_PRACTICES.md)
