# Victor TDI Roadmap — Technical Debt & Implementation Items

**Last Updated**: 2026-04-14 (**37/37 items DONE** — TDI roadmap COMPLETE)
**Strategy**: Foundational/critical first, high impact + low effort, layered TDD

---

## TDI Items by Category

### Category 1: SDK Boundary Enforcement (FOUNDATIONAL)

| ID | Item | Severity | Effort | Impact | Status |
|----|------|----------|--------|--------|--------|
| SDK-1 | Promote `OrchestratorProtocol` to `victor_sdk` | CRITICAL | Low | Unblocks all vertical SDK-only imports | DONE (already in victor_sdk/protocols.py) |
| SDK-2 | Promote `victor.core.verticals.protocols` → `victor_sdk` | CRITICAL | Low | Same | DONE (promoted.py + promoted_types.py) |
| SDK-3 | Promote `tool_dependency_loader` types → `victor_sdk` | HIGH | Low | Fixes victor-research import violation | DONE (exported from victor_sdk) |
| SDK-4 | Promote `StageDefinition`, `VerticalBase` references → SDK | HIGH | Low | Fixes victor-rag, victor-dataanalysis | DONE (exported from victor_sdk) |
| SDK-5 | Fix victor-rag dependency: `victor-ai` → `victor-sdk` | HIGH | Medium | Removes tight coupling | DONE (already depends on victor-sdk; victor-ai is optional runtime dep) |
| SDK-6 | Fix victor-dataanalysis dependency: `victor-ai` → `victor-sdk` | HIGH | Medium | Same | DONE (already depends on victor-sdk; victor-ai is optional runtime dep) |
| SDK-7 | Add CI linter rule: core cannot import external verticals | HIGH | Low | Prevents regression | DONE (check_imports.py Rule 4 + guard tests) |
| SDK-8 | Create `victor_invest/plugin.py` (missing VictorPlugin) | MEDIUM | Low (30min) | Completes vertical alignment | DONE (plugin.py + entry point already exist) |

### Category 2: Core Decoupling (FOUNDATIONAL)

| ID | Item | Severity | Effort | Impact | Status |
|----|------|----------|--------|--------|--------|
| CORE-1 | Remove `from victor_coding` imports from core (runtime + docstring) | CRITICAL | Medium | Core functions without victor-coding | DONE |
| CORE-2 | Replace tree-sitter import with CapabilityRegistry in proximadb_multi.py | HIGH | Medium | Protocol-based discovery | DONE |
| CORE-3 | Replace importlib.import_module in indexer.py with CapabilityRegistry | HIGH | Low | Same | DONE |
| CORE-4 | Replace `VERTICAL_CORES` hardcoded dict with ConfigProvider | HIGH | Medium | OCP compliance for tool sets | DONE (already empty) |
| CORE-5 | Replace `VERTICAL_READONLY_DEFAULTS` with ConfigProvider | MEDIUM | Low | Same | DONE (already empty) |
| CORE-6 | Replace `GroundingRules.for_vertical()` with ConfigProvider | MEDIUM | Low | Same | DONE (already uses dynamic registry + register_addendum()) |
| CORE-7 | Remove hardcoded `_provider_hints` vertical entries | MEDIUM | Low | Only "default" remains in core | DONE |
| CORE-8 | Remove hardcoded `_evaluation_criteria` vertical entries | MEDIUM | Low | Only "default" remains in core | DONE |
| CORE-9 | Delete `compatibility_matrix.json` (use manifest negotiation) | LOW | Low | Removes static coupling | DONE (file already deleted, docstring updated) |

### Category 3: Service Layer Maturation (HIGH IMPACT)

| ID | Item | Severity | Effort | Impact | Status |
|----|------|----------|--------|--------|--------|
| SVC-1 | Service layer structural validation (bootstrap, delegation, health) | HIGH | Medium | Validates all 6 services + 16 delegation points | DONE (13 tests) |
| SVC-2 | Runtime performance benchmark (deferred — needs real workloads) | HIGH | Medium | Validates <5% overhead | DEFERRED |
| SVC-3 | Add delegation for ContextService (2 methods) | MEDIUM | Low | Extends service coverage | DONE (check_context_overflow, get_context_metrics) |
| SVC-4 | Add delegation for ProviderService (2 methods) | MEDIUM | Low | Same | DONE (get_current_provider_info, switch_provider) |
| SVC-5 | Resolve ProviderService + RecoveryService in orchestrator | MEDIUM | Low | Same | DONE (added to _initialize_services) |
| SVC-6 | Remove coordinator fallback paths (major version) | LOW | High | Achieves <2,000 LOC target | FUTURE |

### Category 4: State-Passed Architecture (INCREMENTAL)

| ID | Item | Severity | Effort | Impact | Status |
|----|------|----------|--------|--------|--------|
| SPA-1 | Foundation (ContextSnapshot, StateTransition, etc.) | - | - | - | COMPLETE |
| SPA-2 | Migrate ExplorationCoordinator to state-passed (simplest first) | LOW | Medium | Validates pattern at scale | DONE (exploration_state_passed.py + 11 tests) |
| SPA-3 | Migrate SystemPromptCoordinator to state-passed | LOW | Medium | Improves testability | DONE (system_prompt_state_passed.py + 13 tests) |
| SPA-4 | Migrate SafetyCoordinator to state-passed | LOW | Medium | Same | DONE (safety_state_passed.py + 9 tests) |

**Note**: ChatCoordinator uses excellent protocol-based design — no migration needed.

### Category 5: Global State Elimination (ARCHITECTURAL)

| ID | Item | Severity | Effort | Impact | Status |
|----|------|----------|--------|--------|--------|
| GS-1 | Design `ExecutionContext` dataclass | MEDIUM | Medium | Foundation for context passing | DONE (victor/runtime/context.py + 15 tests) |
| GS-2 | Wire ExecutionContext into orchestrator + guard test | MEDIUM | Medium | Explicit state, testability | DONE (orchestrator wiring + 3 guard tests) |
| GS-3 | Cap `get_container()` calls + guard tests | MEDIUM | Medium | Prevents proliferation | DONE (capped at 25, 3 guard tests) |
| GS-4 | Add cleanup hooks for long-running sessions | MEDIUM | Medium | Memory leak prevention | DONE (ExecutionContext.cleanup() + 7 tests) |
| GS-5 | Singleton guard (cap at 68 files) + conftest reset check | LOW | Low | Prevents proliferation | DONE (guard test + 3 checks) |

### Category 6: Observability & Production (HARDENING)

| ID | Item | Severity | Effort | Impact | Status |
|----|------|----------|--------|--------|--------|
| OBS-1 | Unified cache invalidation strategy | MEDIUM | Medium | Reduces stale data bugs | DONE (CacheRegistry + category invalidation + 14 tests) |
| OBS-2 | Trace context propagation across core↔vertical boundary | MEDIUM | Medium | Debuggability | DONE (TraceContext + contextvars + 19 tests) |
| OBS-3 | Automated cross-repo integration tests (core HEAD + verticals) | HIGH | Medium | Catches breaking changes early | DONE (external-vertical-compat.yml dispatches to 6 repos) |
| OBS-4 | Tool embedding cache size limit | LOW | Low | Prevents disk growth | DONE (TieredCache has 1GB disk limit + 1000 item memory limit) |
| OBS-5 | Event fanout: async subscribers + selective filtering | LOW | Medium | Performance under load | DONE (InMemoryEventBackend has async dispatch + wildcard filtering) |

---

## Strategic Implementation Order

### Layer 0: Foundation (Weeks 1-2) — Unblock Everything

**Focus**: SDK boundary + core decoupling. These are the critical path items that block all downstream work.

```
SDK-1 → SDK-2 → SDK-3 → SDK-4   (Promote protocols to SDK)
  │
  ├── TDD: test_sdk_protocol_exports.py — verify all promoted protocols importable from victor_sdk
  ├── TDD: test_vertical_sdk_only_imports.py — verify verticals compile with SDK-only deps
  └── Regression: make test-definition-boundaries must pass
  │
CORE-1 → CORE-2 → CORE-3        (Remove core→vertical imports)
  │
  ├── TDD: test_core_no_vertical_imports.py — grep guard as unit test
  └── Regression: full test suite without victor-coding installed
```

**Exit criteria**: `grep -r "from victor.core\|from victor.agent" victor-*/` = 0 matches AND `grep -r "victor_coding" victor/` = 0 functional matches.

### Layer 1: High Impact / Low Effort (Weeks 2-3)

**Focus**: Quick wins that improve architecture quality with minimal risk.

```
CORE-4 through CORE-9            (ConfigProvider protocol replaces hardcoded dicts)
  │
  ├── TDD: test_config_provider_protocol.py — verify protocol contract
  ├── TDD: test_vertical_config_registration.py — verify verticals can register own configs
  └── Regression: existing vertical behavior unchanged
  │
SDK-5 → SDK-6                    (Fix rag/dataanalysis SDK dependency)
  │
  ├── TDD: test_vertical_without_framework.py — import vertical with only SDK installed
  └── Regression: existing integration tests pass
  │
SDK-7                             (CI linter rule)
  │
  └── TDD: scripts/check_imports.py extended, runs in CI
  │
SDK-8                             (victor-invest plugin)
  │
  └── TDD: test_invest_plugin_discovery.py
```

### Layer 2: Validation & Service Maturation (Weeks 3-4)

**Focus**: Validate service layer, extend coverage.

```
SVC-1 → SVC-2                    (Performance + integration validation)
  │
  ├── TDD: test_service_layer_performance.py — latency within 5%
  ├── TDD: test_service_layer_integration.py — feature parity
  └── Regression: full test suite with USE_SERVICE_LAYER=true
  │
SVC-3 → SVC-4 → SVC-5            (Extend delegation to remaining services)
  │
  ├── TDD: test_context_service_delegation.py
  ├── TDD: test_provider_service_delegation.py
  └── TDD: test_recovery_service_delegation.py
  │
OBS-3                             (Cross-repo integration tests)
  │
  └── TDD: CI job that tests core HEAD + latest vertical releases
```

### Layer 3: Architectural Improvements (Weeks 5-8)

**Focus**: Global state elimination, state-passed migration, production hardening.

```
GS-1 → GS-2 → GS-3              (ExecutionContext + global state removal)
  │
  ├── TDD: test_execution_context.py — context creation, passing, cleanup
  ├── TDD: test_no_global_state.py — grep guard
  └── Regression: full test suite, memory profiling
  │
SPA-2 → SPA-3 → SPA-4            (State-passed coordinator migration)
  │
  ├── TDD: test_sync_chat_state_passed.py — snapshot in, transitions out
  └── Regression: existing coordinator tests still pass
  │
OBS-1 → OBS-2                    (Cache + tracing)
  │
  ├── TDD: test_cache_invalidation.py
  └── TDD: test_cross_boundary_tracing.py
```

### Layer 4: Optimization (Weeks 8-12)

**Focus**: Remove dual paths, achieve LOC targets.

```
SVC-6                             (Remove coordinator fallback → major version)
  │
  ├── TDD: test_service_only_path.py
  └── Regression: full suite, performance benchmark
  │
GS-4 → GS-5                      (Cleanup hooks, reduce singletons)
  │
  └── TDD: test_long_running_cleanup.py, test_parallel_test_isolation.py
  │
OBS-4 → OBS-5                    (Cache limits, async event bus)
```

---

## Summary Matrix

| Layer | Items | Effort | Risk | Value |
|-------|-------|--------|------|-------|
| **L0: Foundation** | SDK-1..4, CORE-1..3 | 2 weeks | MEDIUM | Unblocks all vertical extraction |
| **L1: Quick Wins** | CORE-4..9, SDK-5..8 | 1-2 weeks | LOW | Architecture quality + OCP |
| **L2: Validation** | SVC-1..5, OBS-3 | 1-2 weeks | MEDIUM | Service layer confidence |
| **L3: Architecture** | GS-1..3, SPA-2..4, OBS-1..2 | 3-4 weeks | HIGH | Clean architecture |
| **L4: Optimization** | SVC-6, GS-4..5, OBS-4..5 | 2-3 weeks | HIGH | Performance + LOC reduction |

**Total**: 37 TDI items across 6 categories, 4 implementation layers, ~10-12 weeks.

---

## TDD Discipline

Every TDI item follows this cycle:

1. **Write failing test** — Captures the desired behavior or guard
2. **Implement minimal fix** — Just enough to pass the test
3. **Refactor** — Clean up while tests stay green
4. **Run regression suite** — `make test` + `make lint` + `make test-definition-boundaries`
5. **Update this tracker** — Mark item status

Guard tests (grep-based, import-based) prevent regression of boundary fixes. They run as part of the standard test suite.
