# Framework + Vertical Integration Re-Analysis

Generated: 2026-02-20 04:34:27Z  
Baseline commit: `baa309109`
Updated: 2026-02-20 14:54:42Z (post-decision-closure verification)

## Scope
- Framework core: API, orchestration, tool system, events/observability.
- Vertical system: registration, pipelines, adapters, extensions, workflows, RL, teams.
- Framework/vertical boundary quality and where generic capabilities should live.

## Verification Addendum (2026-02-20 14:54Z)
- Strict-mode scope expanded beyond capability-config helpers to protocol fallback probes in capability runtime and integration snapshot getters (`victor/framework/strict_mode.py`, `victor/framework/capability_runtime.py`, `victor/framework/vertical_integration.py`, `victor/framework/_internal.py`, `victor/config/settings.py`).
- Added fast-CI strict protocol-fallback lane to prevent regression of duck-typed fallback probes (`.github/workflows/ci-fast.yml`).
- Capability-config model finalized as DI-global service with scoped session keys resolved via protocol port (`victor/framework/capability_config_service.py`, `victor/framework/capability_config_helpers.py`, `victor/framework/protocols.py`, `victor/agent/orchestrator.py`, `victor/framework/step_handlers.py`).
- Observability overflow defaults now codified per topic (`block_with_timeout` for critical topics, `drop_oldest` for metrics-heavy topics) with settings validation + backend enforcement (`victor/config/settings.py`, `victor/core/events/backends.py`).
- Orchestrator decomposition progressed with lazy coordination runtime for recovery/planner/task/chunk components (`victor/agent/runtime/coordination_runtime.py`, `victor/agent/orchestrator.py`).
- Non-MLX local provider decision closed: `vllm`/`llamacpp` stay lazy-only without hardware preflight (`victor/providers/registry.py`, `tests/unit/providers/test_providers_registry.py`).
- Regression evidence after latest changes:
  - `../.venv/bin/pytest tests/unit/agent/test_coordination_runtime.py tests/unit/agent/test_provider_runtime.py tests/unit/agent/test_metrics_runtime.py tests/unit/agent/test_workflow_runtime.py -q` -> `6 passed`.
  - `../.venv/bin/pytest tests/unit/agent/test_orchestrator_core.py -k "tool_planner or recovery_coordinator or task_coordinator" -q` -> `3 passed`.
  - `../.venv/bin/pytest tests/unit/providers/test_providers_registry.py -q` -> `23 passed`.
- Startup KPI + activation probe guardrails revalidated:
  - `import_victor.cold_ms=1458.54`, `agent_create.cold_ms=881.55`, `activation.cold_ms=974.74`
  - `activation.warm_mean_ms=261.51`, framework registry totals `attempted=19`, `applied=57`
  - Required runtime flags all true (`generic_result_cache_enabled`, `http_connection_pool_enabled`, `framework_preload_enabled`).

## 1) Architecture Map (Framework <-> Verticals)

### Core modules and roles
| Layer | Key modules | Current role |
|---|---|---|
| Public framework API | `victor/framework/agent.py:112` | `Agent.create()` entrypoint and SDK-facing lifecycle. |
| API -> runtime bridge | `victor/framework/_internal.py:49` | Creates orchestrator, bootstraps container, applies vertical + observability integration. |
| Shared vertical application service | `victor/framework/vertical_service.py:40`, `victor/framework/vertical_service.py:92` | Singleton `VerticalIntegrationPipeline` and unified `apply_vertical_configuration()`. |
| Vertical activation | `victor/core/verticals/vertical_loader.py:732`, `victor/core/bootstrap.py:550`, `victor/core/bootstrap.py:668` | Canonical `activate_vertical_services()` used by bootstrap and switch/step-handler paths. |
| Integration pipeline | `victor/framework/vertical_integration.py:872` | Builds context, runs step handlers, caches plans, supports dependency-driven parallel execution. |
| Step handlers | `victor/framework/step_handlers.py:2303` | Ordered application of capability config/tools/tiered config/extensions/framework/context. |
| Orchestrator runtime facade | `victor/agent/orchestrator.py:430` | Runtime host; now partially decomposed into provider/memory/metrics/workflow/coordination runtime modules. |
| Runtime decomposition modules | `victor/agent/runtime/provider_runtime.py:71`, `victor/agent/runtime/memory_runtime.py:31`, `victor/agent/runtime/metrics_runtime.py:36`, `victor/agent/runtime/workflow_runtime.py:32`, `victor/agent/runtime/coordination_runtime.py:26` | Lazy boundaries for heavy subsystems and coordinator creation. |
| Observability/event transport | `victor/observability/integration.py:54`, `victor/core/events/backends.py:136` | Orchestrator wiring + queue/backpressure policies + optional durable overflow sink. |

### End-to-end data flow
1. `Agent.create()` calls `create_orchestrator_from_options()` (`victor/framework/agent.py:112`, `victor/framework/_internal.py:49`).
2. Bootstrap ensures container + vertical activation path (`victor/core/bootstrap.py:640`, `victor/core/bootstrap.py:668`, `victor/core/verticals/vertical_loader.py:732`).
3. Factory builds orchestrator and runtime boundaries (`victor/framework/_internal.py:134`, `victor/agent/orchestrator.py:452`, `victor/agent/orchestrator.py:492`, `victor/agent/orchestrator.py:518`, `victor/agent/orchestrator.py:525`).
4. Vertical integration runs via shared service + pipeline (`victor/framework/_internal.py:169`, `victor/framework/vertical_service.py:92`, `victor/framework/vertical_integration.py:1042`).
5. Step handlers apply vertical artifacts into context + orchestrator ports (`victor/framework/step_handlers.py:2303`).
6. Framework-level registries/workflows/RL/teams/chains/personas/handlers are populated from vertical providers (`victor/framework/step_handlers.py:1158`, `victor/framework/step_handlers.py:1207`, `victor/framework/step_handlers.py:1283`, `victor/framework/step_handlers.py:1364`, `victor/framework/step_handlers.py:1414`, `victor/framework/step_handlers.py:1604`).
7. Observability integration binds event bus to orchestrator via public setter path (`victor/observability/integration.py:241`, `victor/framework/_internal.py:187`).

## 2) Gaps: Generic Capabilities Still Embedded in Verticals

| Gap ID | Gap | Evidence | What should move to framework | Concrete suggestion |
|---|---|---|---|---|
| G11 | Resolved: capability helper logic is now centralized in framework runtime helper | `victor/framework/capability_runtime.py:10`, `victor/framework/step_handlers.py:148`, `victor/framework/vertical_integration.py:160`, `victor/framework/_internal.py:42` | Single `framework.capability_runtime` helper surface | Keep this as the only capability helper import path and remove any future local helper re-introductions in reviews. |
| G12 | Resolved for current scope: adapter private writes removed and snapshot reads are protocol/getter-first | `victor/agent/vertical_integration_adapter.py:304`, `victor/framework/vertical_integration.py:1383`, `victor/agent/capability_registry.py:138`, `victor/agent/orchestrator.py:1636` | Strict protocol-only framework/orchestrator contract | Keep strict-mode guards expanding to other framework probes and block regressions in CI. |
| G13 | Resolved: framework registry inversion now includes versioned idempotence + metrics | `victor/framework/framework_integration_registry_service.py:48`, `victor/framework/framework_integration_registry_service.py:309`, `victor/framework/vertical_integration.py:1549` | Framework-owned integration registry facade/service | Keep metrics surfaced in observability and add KPI thresholds for registration churn regressions. |
| G14 | Resolved for built-ins: per-vertical `get_tiered_tools()` wrappers removed | `victor/coding/assistant.py:369`, `victor/research/assistant.py:188`, `victor/rag/assistant.py:173`, `victor/devops/assistant.py:216`, `victor/dataanalysis/assistant.py:192`, `victor/core/verticals/extension_loader.py:721` | Single migration adapter in framework/core | Keep compatibility only in extension-loader adapter and set explicit removal deadline. |
| G15 | Resolved: capability config storage now uses DI-global scoped service contract | `victor/framework/capability_config_service.py:13`, `victor/framework/capability_config_helpers.py:37`, `victor/framework/protocols.py:609`, `victor/agent/orchestrator.py:1218`, `victor/framework/step_handlers.py:898` | Strict service-backed capability config store with session scoping | Keep using `get_capability_config_scope_key()` as the single scope source and block attr-based fallback regressions in strict mode CI. |
| G16 | Partially addressed: generic runtime infra is now wired into active web/preload paths behind flags | `victor/tools/web_search_tool.py:362`, `victor/tools/http_pool.py:121`, `victor/agent/orchestrator.py:1237`, `victor/tools/base.py:119` | Framework runtime services for caching/http preloading | Add benchmark/CI KPI gates proving value and retire unused generic modules. |

## 3) SOLID Evaluation (Violations and Fixes)

| Principle | Status | Evidence | Violation summary | Fix |
|---|---|---|---|---|
| SRP | Partial | `victor/agent/orchestrator.py` (4262 LOC), `wc -l` | Orchestrator still owns too many concerns despite runtime slices. | Continue P4 decomposition: split tool/runtime + chat/runtime + session/runtime modules with facade-only orchestration. |
| OCP | Improved | Metadata-driven handlers (`victor/framework/step_handlers.py:399`, `victor/framework/vertical_integration.py:1943`) plus service-facade registry writes (`victor/framework/framework_integration_registry_service.py:13`, `victor/framework/step_handlers.py:1082`) | Extension points now avoid direct global registry imports in step handlers; remaining work is service-level idempotence/versioning policy. | Keep expanding service-backed boundaries and implement registration policy hooks inside service. |
| LSP | Partial | Deprecated tiered adapter (`victor/core/verticals/extension_loader.py:721`), remaining duck-typed compatibility probes (`victor/framework/step_handlers.py:1060`) | Substitutability risk is now mostly isolated to explicit compatibility adapters. | Remove `get_tiered_tools()` adapter after 2026-06-30 and keep protocol conformance tests mandatory. |
| ISP | Improved | Focused infrastructure ports (`victor/framework/protocols.py:597`, `victor/framework/protocols.py:606`, `victor/framework/protocols.py:623`) | Vertical/framework integration still depends on broad orchestrator surface via `hasattr` in places. | Define focused `VerticalRuntimePort` (context/tools/middleware/safety only) and pass into pipeline. |
| DIP | Partial | Port usage exists (`victor/agent/orchestrator.py:1201`, `victor/agent/orchestrator.py:1214`, `victor/agent/orchestrator.py:1226`) but private/global fallback remains (`victor/framework/vertical_integration.py:1539`, `victor/framework/step_handlers.py:1062`) | Framework still depends on concrete internals in fallback paths. | Eliminate fallback internals and route all framework writes through protocol/DI services only. |

## 4) Scalability + Performance Risks

| Hot path | Risk | Evidence | Mitigation |
|---|---|---|---|
| Agent startup | Large orchestrator still initializes many components early | `victor/agent/orchestrator.py:430` (file size 4262 LOC), constructor body starts at `victor/agent/orchestrator.py:537` | Continue lazy boundaries for chat/tool/session coordinators; gate via first-use materialization. |
| Vertical application on repeated sessions | Registry writes now centralized, but idempotence/version policy is still implicit | `victor/framework/framework_integration_registry_service.py:13`, `victor/framework/step_handlers.py:1082` | Add explicit dedup/version keying in `FrameworkIntegrationRegistryService` and expose registration metrics for cache-hit paths. |
| Integration cache safety checks | Reduced risk: snapshot now has strict-mode fallback blocking for protocol probes | `victor/framework/vertical_integration.py:1383`, `victor/framework/vertical_integration.py:1401`, `tests/unit/framework/test_vertical_integration.py:1964` | Add CI strict-mode lane and keep snapshot probe tests mandatory to prevent regression. |
| Extension loading saturation | Shared pool is bounded but statically configured and globally shared | `victor/core/verticals/extension_loader.py:137`, `victor/core/verticals/extension_loader.py:186`, `victor/core/verticals/extension_loader.py:253` | Make worker/queue limits adaptive (CPU count + workload), expose autoscaling policy. |
| Event delivery under load | Topic defaults now reduce critical-event loss, but pressure tuning remains static | `victor/config/settings.py:957`, `victor/core/events/backends.py:337`, `victor/core/events/backends.py:503`, `tests/integration/framework/test_vertical_reliability_stress.py:110` | Add env/profile presets per deployment tier and emit alert thresholds when topic override hits exceed baseline. |
| Startup/import overhead creep | New generic modules exported but not integrated, increasing surface without measurable ROI | `victor/storage/cache/__init__.py:61`, `victor/tools/__init__.py:119`, `victor/framework/preload.py:184`, `scripts/benchmark_startup_kpi.py:173` | Use startup+activation KPI guardrails (runtime flags + registry totals) and remove/park modules that remain off active path. |

## 5) Competitive Comparison (5-7 dimensions, 1-10 with rationale)

Weights: Boundary Clarity 20, Orchestration Determinism 20, Observability/Reliability 15, Runtime Performance 15, Vertical Extensibility 15, Multi-agent/Workflow Depth 15.

| Dimension (Weight) | Victor | LangGraph | CrewAI | LangChain | LlamaIndex | AutoGen |
|---|---|---|---|---|---|---|
| Boundary clarity (20) | 8 - Strong protocol/DI direction but fallback internals remain. | 8 - Clear graph runtime boundary, fewer vertical conventions. | 6 - Team abstractions are clear, core boundary less strict. | 6 - Broad abstractions but mixed historical layering. | 7 - Clear data/RAG boundary, less full-stack orchestration separation. | 6 - Agent abstractions clear, framework boundary is less strict. |
| Orchestration determinism (20) | 8 - Metadata dependency levels + handler plan cache. | 9 - StateGraph/DAG execution is mature and explicit. | 6 - Good role flows, weaker deterministic dependency model. | 7 - Flexible chains/agents, determinism varies by stack. | 6 - Strong pipelines for retrieval, less general orchestration control. | 7 - Strong conversation orchestration, weaker static dependency controls. |
| Observability/reliability (15) | 8 - Queue policies + pressure metrics + durable sink hooks. | 7 - Good tracing integrations, fewer built-in queue policy controls. | 5 - Basic telemetry; reliability policy surface is smaller. | 6 - Good ecosystem observability, less unified runtime policy. | 6 - Good pipeline observability, less system-wide event policy. | 5 - Core telemetry exists, backpressure policy is less structured. |
| Runtime performance/startup (15) | 7 - Provider/memory/metrics/workflow/coordination lazy slices added, constructor still heavy. | 7 - Lean runtime core, user integrations can add overhead. | 6 - Moderate runtime footprint; team overhead in larger runs. | 6 - Rich but heavier import/runtime footprint. | 7 - Efficient for RAG pipelines; broader runtime still moderate. | 6 - Multi-agent orchestration can be resource heavy. |
| Vertical extensibility (15) | 8 - Rich vertical protocol surface (workflows/RL/teams/tools). | 6 - Extensible graph nodes, less vertical package convention. | 7 - Team-centric extension model is straightforward. | 8 - Very broad extension ecosystem and integrations. | 9 - Best-in-class retrieval/data extension ecosystem. | 6 - Extensible agents, less verticalized domain packaging. |
| Multi-agent/workflow depth (15) | 8 - Teams/workflows/RL integrated in one vertical pipeline. | 7 - Strong workflow graphing, less built-in team policy. | 8 - Strong team collaboration primitives. | 6 - Multi-agent available but less cohesive by default. | 5 - Focus is retrieval/data more than teams. | 9 - Deep multi-agent conversation and delegation patterns. |
| **Overall weighted score** | **7.85** | **7.45** | **6.30** | **6.50** | **6.65** | **6.50** |

## 6) Roadmap to Best-in-Class

### Phase A (0-2 weeks): Contract hardening
- Remove private fallback reads/writes in framework integration paths.
- Completed: merged duplicated capability helper logic into `victor.framework.capability_runtime`.
- Completed: strict-mode fallback blocking now covers capability config, capability runtime fallback, and integration snapshot getter fallback paths.
- Completed: strict CI tests for protocol-only integration are active in fast CI.

### Phase B (2-5 weeks): Registry inversion
- Completed: introduced `FrameworkIntegrationRegistryService` and migrated `FrameworkStepHandler` global registry writes to service calls.
- Completed: added idempotence/versioning policy and exposed registry metrics in `vertical.applied` payloads.

### Phase C (5-8 weeks): Legacy cleanup completion
- Completed: removed framework-level tiered-tools fallback and built-in `get_tiered_tools()` wrappers; remaining adapter lives only in extension-loader compatibility layer with explicit removal warning (`after 2026-06-30`).
- Enforce canonical `get_tiered_tool_config()` in static checks.

### Phase D (8-12 weeks): Runtime and startup optimization
- Continue orchestrator decomposition (tool/runtime, chat/runtime, recovery/runtime).
- Completed: integrated KPI thresholds from `scripts/benchmark_startup_kpi.py` into CI fast-check guardrails (`.github/workflows/ci-fast.yml`).
- Completed: added coordination runtime lazy boundaries for recovery/planner/task/chunk components.
- Add import/startup flamegraph checks for regression triage.

### Phase E (12-16 weeks): Generic infrastructure assimilation
- Integrate `GenericResultCache`, `HttpConnectionPool`, and `PreloadManager` into active framework services behind flags.
- Completed: startup KPI harness now supports activation probe telemetry + guardrails for runtime flags and framework registry totals (`scripts/benchmark_startup_kpi.py`, `tests/unit/test_benchmark_startup_kpi.py`).
- Keep only features with measured startup/latency wins; remove dead generic modules if unused.

## 7) Session Handoff

Use `docs/analysis_reports/09_framework_vertical_integration_remediation_plan.md` as the source of truth plan, and treat this report as the refreshed architecture baseline for post-refactor decisions.
