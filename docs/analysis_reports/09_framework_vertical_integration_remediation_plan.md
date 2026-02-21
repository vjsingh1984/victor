# Framework + Vertical Integration Remediation Plan

**Generated**: 2026-02-19  
**Status**: IN PROGRESS  
**Horizon**: 6 months (phased)

---

## Goal

Address architecture gaps between framework core and vertical system so Victor has:
- a single integration path,
- protocol-first boundaries (no private-attribute coupling),
- scalable startup/load behavior,
- and a stable plugin/capability contract.

---

## Scope

In scope:
- Framework API, orchestration, tool system, events/observability.
- Vertical registration, pipelines, adapters, extensions, workflows, RL, teams.
- Promotion of generic behaviors from vertical modules into framework services.

Out of scope:
- New end-user vertical features.
- Model/provider quality tuning not related to architecture boundaries.

---

## Gap Register (Actionable)

| ID | Gap | Evidence (code) | Suggested Fix | Target Phase |
|---|---|---|---|---|
| G1 | Capability config still split between context model and direct orchestrator mutation | `victor/framework/step_handlers.py:856`, `victor/coding/capabilities.py:94`, `victor/research/capabilities.py:79`, `victor/rag/capabilities.py:82`, `victor/devops/capabilities.py:80`, `victor/dataanalysis/capabilities.py:79` | Create framework `CapabilityConfigService` and route reads/writes through protocol methods only; verticals provide defaults only | P1 |
| G2 | Duplicate service registration paths (bootstrap vs step handler) | `victor/core/bootstrap.py:508`, `victor/core/bootstrap.py:542`, `victor/framework/step_handlers.py:1920`, `victor/framework/step_handlers.py:1939` | Introduce single `VerticalActivationService` with idempotent registration and explicit source-of-truth lifecycle | P0 |
| G3 | Legacy compatibility still leaks into active path (`get_tiered_tools`, deprecated registries) | `victor/agent/orchestrator.py:3924`, `victor/framework/vertical_integration.py:323`, `victor/core/verticals/extension_loader.py:433` | Remove deprecated accessors after migration window; enforce `get_tiered_tool_config()` contract | P2 |
| G4 | Parallel step execution classification hard-coded by class name | `victor/framework/vertical_integration.py:1532`, `victor/framework/vertical_integration.py:1535` | Add handler metadata (`depends_on`, `side_effects`, `parallel_safe`) and execute via DAG scheduler | P2 |
| G5 | Private/internal orchestrator access in framework integration | `victor/framework/step_handlers.py:1419`, `victor/framework/step_handlers.py:1929`, `victor/observability/integration.py:254` | Add explicit orchestrator ports/interfaces and inject dependencies through protocol/DI only | P0 |
| G6 | Extension async loading creates per-call thread pools | `victor/core/verticals/extension_loader.py:827` | Use shared bounded executor and queue; collect loader metrics | P3 |
| G7 | Integration cache hit still replays handlers with full side effects | `victor/framework/vertical_integration.py:1023`, `victor/framework/vertical_integration.py:1024` | Compile immutable integration plans and apply delta/no-op detection | P3 |
| G8 | Orchestrator remains monolithic and eagerly initializes many subsystems | `victor/agent/orchestrator.py:430`, `victor/agent/orchestrator.py:514`, `victor/agent/orchestrator.py:664`, `victor/agent/orchestrator.py:672` | Split into runtime modules and lazy-init non-critical services | P4 |
| G9 | Observability path is intentionally lossy under load without policy controls | `victor/core/events/backends.py:214`, `victor/core/events/backends.py:719` | Add configurable drop/backpressure policies and optional durable sink | P3 |
| G10 | Vertical wrappers duplicate generic provider plumbing | `victor/coding/assistant.py:342`, `victor/research/assistant.py:199`, `victor/rag/assistant.py:252`, `victor/devops/assistant.py:227`, `victor/dataanalysis/assistant.py:203` | Move to manifest/convention registration for workflows/RL/teams/handlers/capability defaults | P1 |

---

## Phased Plan

## P0: Boundary Hardening (Weeks 1-3)
**Objective**: Remove fragile coupling and unify activation flow.

Work items:
- Implement `VerticalActivationService` in framework and route both SDK/CLI + bootstrap through it.
- Add explicit orchestrator ports for:
  - container access,
  - capability loader access,
  - observability binding.
- Replace private attribute writes/reads in step handlers and observability integration with protocol calls.

Likely touch points:
- `victor/framework/vertical_service.py`
- `victor/core/bootstrap.py`
- `victor/framework/step_handlers.py`
- `victor/observability/integration.py`
- `victor/framework/protocols.py`

Acceptance criteria:
- No framework writes to `orchestrator._*` for vertical/observability integration.
- Vertical services registered exactly once per activation (idempotent check with tests).
- CLI and SDK paths share one activation code path.

---

## P1: Promote Generic Capability State (Weeks 4-8)
**Objective**: Remove per-vertical orchestrator mutation as runtime mechanism.

Work items:
- Add `CapabilityConfigService` in framework (typed getters/setters + merge policies).
- Make vertical capability modules emit defaults only.
- Route runtime consumers to `VerticalContext` + new config service APIs.
- Replace duplicated assistant wrapper patterns with framework-driven conventions.

Likely touch points:
- `victor/framework/step_handlers.py`
- `victor/agent/vertical_context.py`
- `victor/*/capabilities.py`
- `victor/*/assistant.py`

Acceptance criteria:
- No direct config mutation pattern like `orchestrator.<x>_config = ...` in vertical capability modules.
- `get_capability_config()` is used by runtime consumers beyond docs/examples.
- A single shared helper handles workflows/RL/teams/handlers registration hooks.

---

## P2: Contract Cleanup + Scheduler Upgrade (Weeks 9-12)
**Objective**: Eliminate legacy shims and improve OCP behavior.

Work items:
- Deprecate and remove `get_tiered_tools()` fallback path after migration.
- Remove obsolete/deprecated extension registry artifacts from active integration flow.
- Replace class-name based parallel grouping with metadata-driven DAG execution.

Likely touch points:
- `victor/framework/vertical_integration.py`
- `victor/framework/step_handlers.py`
- `victor/agent/orchestrator.py`
- `victor/core/verticals/extension_loader.py`

Acceptance criteria:
- Handler execution ordering is metadata-based, not type-name checks.
- Legacy tiered-tools compatibility is fully removed or isolated behind explicit adapter with deprecation flag.
- Integration tests cover deterministic ordering and parallel safety.

---

## P3: Performance + Reliability (Weeks 13-18)
**Objective**: Reduce startup overhead and improve runtime resilience.

Work items:
- Replace per-call executor creation with shared bounded pool for extension loading.
- Add compiled integration plan cache with delta application to avoid full replay.
- Add observability delivery policies:
  - drop-oldest,
  - block-with-timeout,
  - durable optional backend.

Likely touch points:
- `victor/core/verticals/extension_loader.py`
- `victor/framework/vertical_integration.py`
- `victor/core/events/backends.py`

Acceptance criteria:
- Measured reduction in vertical apply latency and thread churn.
- Cache hit path avoids unnecessary handler side effects where safe.
- Observability drop rate and queue pressure are externally measurable.

---

## P4: Orchestrator Decomposition (Weeks 19-24)
**Objective**: Finish SRP/DIP cleanup by reducing orchestrator monolith.

Work items:
- Extract runtime modules (tool/runtime, workflow/runtime, memory/runtime, metrics/runtime).
- Convert eager initialization to lazy activation where feasible.
- Keep orchestration interface stable via facade/adapter.

Likely touch points:
- `victor/agent/orchestrator.py`
- `victor/agent/orchestrator_factory.py`
- `victor/agent/coordinators/*`

Acceptance criteria:
- Core orchestrator file size and constructor responsibilities reduced materially.
- Startup path initializes only required components for first request.
- Existing public APIs remain backward compatible.

---

## SOLID-Focused Refactor Targets

| Principle | Current Risk | Refactor Target |
|---|---|---|
| SRP | `Orchestrator` is oversized and multi-concern | Runtime module extraction + lazy init |
| OCP | Legacy compatibility shims still present in tiered-tools and extension adapters | Remove shims after migration window; keep explicit adapter boundaries only |
| LSP | Protocol fallback logic allows weak substitutes | Strict protocol boundary + adapters |
| ISP | Capability surface too broad | Smaller capability interfaces by domain |
| DIP | Framework reads/writes internals (`_container`, `_capability_loader`, `_observability`) | Explicit ports + DI wiring |

---

## Tracking and Session Handoff

Use this section in every follow-up session.

**Current Phase**: `P4 in progress + startup/activation KPI hardening`  
**Last Updated**: `2026-02-21 03:38:41Z`  
**Owner**: `framework-architecture`

### Open Decisions
- [x] Decide strict migration timeline for removing `get_tiered_tools()` compatibility path (`remove after 2026-06-30`).
- [x] Choose observability default policy (`drop-oldest` vs `block-with-timeout`) -> `block_with_timeout` for critical topics, `drop_oldest` for metrics-heavy topics.
- [x] Decide whether capability-config service is context-local or DI-global with scoped session key -> `DI-global service + scoped session key`.
- [x] Decide whether non-MLX local providers should get runtime preflight checks (or remain lazy-only) -> `remain lazy-only (server-adapter providers)`.

### Next 3 Tasks
- [x] Continue P4 decomposition by extracting recovery handler/integration wiring into dedicated runtime boundaries with lazy materialization.
- [x] Extend startup KPI guardrails to enforce activation cold/warm thresholds and runtime lazy expectations in CI.
- [ ] Re-execute full framework+vertical architecture re-analysis with updated scoring after slice 6 runtime decomposition.

### Progress Log
| Date | Phase | Update | Evidence/PR |
|---|---|---|---|
| 2026-02-19 | P0 | Initial remediation plan created from architecture gap analysis | `docs/analysis_reports/09_framework_vertical_integration_remediation_plan.md` |
| 2026-02-19 | P0 | Unified activation path via `activate_vertical_services`; bootstrap and step handlers now share vertical service activation flow; explicit orchestrator ports added for container/capability-loader/observability | `victor/core/verticals/vertical_loader.py`, `victor/core/bootstrap.py`, `victor/framework/step_handlers.py`, `victor/agent/orchestrator.py`, `victor/framework/protocols.py`, `victor/observability/integration.py` |
| 2026-02-19 | P0 | Tightened protocol-first boundaries and added CLI/SDK idempotence parity tests for shared activation helper behavior | `victor/framework/step_handlers.py`, `victor/framework/_internal.py`, `tests/unit/framework/test_vertical_service.py`, `tests/unit/framework/test_framework_step_handler.py`, `tests/unit/framework/test_framework_internal.py` |
| 2026-02-19 | P1 | Added framework `CapabilityConfigService`, wired `CapabilityConfigStepHandler` to persist defaults into framework service, and migrated Research capability handlers/getters to service-first config storage with compatibility fallback | `victor/framework/capability_config_service.py`, `victor/core/bootstrap.py`, `victor/framework/step_handlers.py`, `victor/research/capabilities.py`, `tests/unit/framework/test_capability_config_service.py`, `tests/unit/research/test_research_capabilities.py` |
| 2026-02-19 | P1 | Migrated Coding/RAG/DevOps/DataAnalysis capability handlers/getters to service-first config access, hardened service resolution for mock/legacy compatibility, aligned DataAnalysis default config keys, and added e2e defaults-flow test (`vertical defaults -> step handler -> service -> runtime getter`) | `victor/coding/capabilities.py`, `victor/rag/capabilities.py`, `victor/devops/capabilities.py`, `victor/dataanalysis/capabilities.py`, `victor/research/capabilities.py`, `tests/unit/framework/test_framework_step_handler.py`, `tests/unit/coding/test_coding_capabilities.py`, `tests/unit/rag/test_rag_capabilities.py`, `tests/unit/devops/test_devops_capability_config_service.py`, `tests/unit/dataanalysis/test_dataanalysis_capability_config_service.py` |
| 2026-02-19 | P1 | Completed remaining capability-module migration for Benchmark + framework Privacy with service-first config storage, removed benchmark runtime errors (`field(...)` default and hard dependency on missing benchmark safety extension), and validated compatibility fallback behavior | `victor/benchmark/capabilities.py`, `victor/framework/capabilities/privacy.py`, `tests/unit/benchmark/test_benchmark_capability_config_service.py`, `tests/unit/framework/capabilities/test_privacy_capability_config_service.py` |
| 2026-02-19 | P1 | Extracted shared capability-config helper utilities into framework and removed duplicated service-resolution/load/store logic across migrated vertical capability modules | `victor/framework/capability_config_helpers.py`, `victor/research/capabilities.py`, `victor/coding/capabilities.py`, `victor/devops/capabilities.py`, `victor/dataanalysis/capabilities.py`, `victor/rag/capabilities.py`, `victor/benchmark/capabilities.py`, `victor/framework/capabilities/privacy.py`, `tests/unit/framework/test_capability_config_helpers.py` |
| 2026-02-19 | P1 | Expanded integration coverage for service-backed runtime getter flow across Coding/RAG/DevOps/DataAnalysis capability modules | `tests/integration/framework/test_vertical_capability_integration.py` |
| 2026-02-19 | P2 | Moved tiered tool integration to canonical `get_tiered_tool_config()` contract: orchestrator now uses canonical method, framework legacy fallback is disabled by default and only enabled via explicit env adapter flag, and built-in vertical assistants now implement canonical method with compatibility wrappers | `victor/framework/step_handlers.py`, `victor/agent/orchestrator.py`, `victor/coding/assistant.py`, `victor/research/assistant.py`, `victor/rag/assistant.py`, `victor/devops/assistant.py`, `victor/dataanalysis/assistant.py`, `tests/unit/framework/test_tiered_tool_config.py` |
| 2026-02-19 | P2 | Completed metadata-driven scheduler upgrade: parallel execution now uses dependency levels (`depends_on`) plus explicit safety invariants (`parallel_safe` + `side_effects`), with deterministic ordering and cycle fallback; added unit coverage for dependency ordering, cycle fallback, and side-effect parallel safety invariants | `victor/framework/vertical_integration.py`, `tests/unit/framework/test_vertical_integration.py` |
| 2026-02-19 | P2 | Removed legacy extension registry artifacts from active integration path by disabling pipeline-level legacy registry wiring while retaining explicit compatibility adapter APIs | `victor/framework/vertical_integration.py`, `tests/unit/framework/test_vertical_integration.py` |
| 2026-02-19 | P3 | Replaced per-call async extension loader executors with shared bounded infrastructure (shared thread pool + bounded async semaphore) and introduced extension-loader queue/in-flight metrics with unit coverage | `victor/core/verticals/extension_loader.py`, `tests/unit/core/test_vertical_base.py`, `tests/unit/core/verticals/test_extension_cache.py` |
| 2026-02-19 | P3 | Wired extension-loader telemetry into `vertical.applied` observability payload to expose queue/in-flight metrics during framework integration events | `victor/framework/vertical_integration.py`, `tests/unit/framework/test_vertical_integration.py` |
| 2026-02-19 | P3 | Added extension-loader pressure controls (warn/error thresholds, optional pressure event emission, periodic metrics reporter) with saturation-focused unit tests | `victor/core/verticals/extension_loader.py`, `tests/unit/core/test_vertical_base.py` |
| 2026-02-19 | P3 | Implemented integration-plan cache + per-orchestrator delta/no-op handler skipping on cache hits, with handler fingerprinting, state-safety checks, and observability payload stats | `victor/framework/vertical_integration.py`, `tests/unit/framework/test_vertical_integration.py` |
| 2026-02-19 | P3 | Added observability queue overflow policy controls (`drop_newest`, `drop_oldest`, `block_with_timeout`) and optional durable sink path with measurable queue/drop stats | `victor/core/events/backends.py`, `victor/core/events/protocols.py`, `tests/unit/core/events/test_event_backends.py` |
| 2026-02-19 | P3 | Added settings-level defaults/validation for queue overflow + extension-loader pressure/reporter controls and wired runtime bootstrap configuration for those policies | `victor/config/settings.py`, `victor/core/events/backends.py`, `victor/core/bootstrap.py`, `tests/unit/core/test_settings.py`, `tests/unit/core/test_bootstrap_runtime_config.py`, `tests/unit/observability/test_event_registry.py` |
| 2026-02-19 | P3 | Added queue-pressure and extension saturation stress/integration coverage validating `block_with_timeout` semantics and loader pressure metrics | `tests/integration/framework/test_vertical_reliability_stress.py` |
| 2026-02-19 | P3 | Completed warm-cache benchmark harness validation (`iterations=200`): sync gain `79.86%`, async gain `76.52%`, side-effect skip ratio `100%` | `scripts/benchmark_vertical_integration_cache.py` |
| 2026-02-19 | P3/P4 | Fixed MLX/MPS native abort path by moving provider stack to lazy materialization, adding MLX subprocess preflight guard, and broadening provider registry to lazy-load local/cloud providers for startup consistency | `victor/providers/registry.py`, `victor/providers/mlx_provider.py`, `tests/unit/providers/test_providers_registry.py` |
| 2026-02-19 | P4 | Completed orchestrator decomposition slice 1 by extracting provider runtime boundary and lazy coordinator materialization from `AgentOrchestrator.__init__` | `victor/agent/orchestrator.py`, `victor/agent/runtime/provider_runtime.py`, `victor/agent/runtime/__init__.py`, `tests/unit/agent/test_provider_runtime.py` |
| 2026-02-19 | P4 | Added startup KPI harness for `import victor` and `Agent.create()` cold/warm baseline tracking | `scripts/benchmark_startup_kpi.py` |
| 2026-02-19 | P4 | Documented provider lazy-loading behavior and MLX preflight controls for operators | `docs/reference/environment-variables.md`, `docs/reference/providers/index.md` |
| 2026-02-19 | P4 | Completed orchestrator decomposition slice 2 by extracting memory/session runtime initialization into dedicated runtime module and routing embedding-store init through runtime boundary | `victor/agent/orchestrator.py`, `victor/agent/runtime/memory_runtime.py`, `victor/agent/runtime/__init__.py`, `tests/unit/agent/test_memory_runtime.py` |
| 2026-02-19 | P4 | Fixed SDK `Agent.create()` provider resolution regression (`get_provider_class` import drift) and validated startup KPI harness run (`ollama/qwen3-coder:30b`, iterations=2): import cold `1502.74ms` / warm mean `0.00ms`, create cold `1305.17ms` / warm mean `245.43ms` | `victor/framework/_internal.py`, `scripts/benchmark_startup_kpi.py`, `tests/unit/framework/test_agent.py` |
| 2026-02-20 | P4 | Completed orchestrator decomposition slice 3 by extracting metrics/runtime and workflow/runtime lazy boundaries from orchestrator constructor and adding lazy workflow registry materialization | `victor/agent/orchestrator.py`, `victor/agent/runtime/metrics_runtime.py`, `victor/agent/runtime/workflow_runtime.py`, `victor/agent/runtime/__init__.py` |
| 2026-02-20 | P4 | Added startup KPI threshold guardrails (`--max-*`) with structured failure reporting and non-zero exit on regression | `scripts/benchmark_startup_kpi.py` |
| 2026-02-20 | P4 | Added focused unit coverage for runtime decomposition + KPI threshold evaluation | `tests/unit/agent/test_metrics_runtime.py`, `tests/unit/agent/test_workflow_runtime.py`, `tests/unit/test_benchmark_startup_kpi.py` |
| 2026-02-20 | Cross-phase | Re-executed framework + vertical integration architecture assessment after major refactors and refreshed gap/roadmap/competitive scoring baseline | `docs/analysis_reports/10_framework_vertical_reanalysis_20260220.md` |
| 2026-02-20 | P1/P4 | Introduced `FrameworkIntegrationRegistryService` and migrated `FrameworkStepHandler` registry side effects (workflows/triggers/RL/team/chains/personas/tool-graphs/handlers) behind DI-resolvable framework service boundary with bootstrap registration | `victor/framework/framework_integration_registry_service.py`, `victor/framework/step_handlers.py`, `victor/core/bootstrap.py`, `tests/unit/framework/test_framework_integration_registry_service.py`, `tests/unit/framework/test_framework_step_handler.py` |
| 2026-02-20 | P0/P1 | Completed framework capability-helper consolidation path by switching framework internals to shared `capability_runtime` helpers and fixing `prompt_builder` capability mapping regression | `victor/framework/_internal.py`, `victor/framework/capability_runtime.py`, `victor/framework/capability_registry.py`, `tests/unit/framework/test_framework_internal.py` |
| 2026-02-20 | P4 | Wired startup KPI threshold guardrails into CI fast checks (`imports` job) with bounded thresholds for `import victor` and `Agent.create()` startup regressions | `.github/workflows/ci-fast.yml`, `scripts/benchmark_startup_kpi.py` |
| 2026-02-20 | P0/P1 | Added strict private-fallback guardrails for capability config helpers, removed remaining adapter private setter usage via public storage ports (`set_middleware`, `set_safety_patterns`, `set_middleware_chain`), and aligned orchestrator storage APIs with protocol-first usage | `victor/framework/strict_mode.py`, `victor/framework/capability_config_helpers.py`, `victor/agent/vertical_integration_adapter.py`, `victor/agent/orchestrator.py`, `tests/unit/framework/test_capability_config_helpers.py`, `tests/unit/agent/test_orchestrator_core.py` |
| 2026-02-20 | P1/P3 | Hardened framework registry replay policy with versioned dedupe keys and exposed framework registry metrics in `vertical.applied` observability payloads for cache-hit/replay visibility | `victor/framework/framework_integration_registry_service.py`, `victor/framework/vertical_integration.py`, `tests/unit/framework/test_framework_integration_registry_service.py`, `tests/unit/framework/test_vertical_integration.py` |
| 2026-02-20 | P2 | Removed built-in vertical `get_tiered_tools()` compatibility wrappers so legacy tiered-tool API remains isolated to core compatibility adapter paths only | `victor/coding/assistant.py`, `victor/research/assistant.py`, `victor/rag/assistant.py`, `victor/devops/assistant.py`, `victor/dataanalysis/assistant.py`, `tests/unit/framework/test_tiered_tool_config.py` |
| 2026-02-20 | P2/P0 | Removed framework-level `get_tiered_tools()` fallback from step handlers, set explicit adapter deprecation deadline warning (`remove after 2026-06-30`), and hardened integration-plan snapshots to protocol/getter-only reads (no direct `enabled_tools` attribute probe) with regression tests | `victor/framework/step_handlers.py`, `victor/core/verticals/extension_loader.py`, `victor/framework/vertical_integration.py`, `victor/agent/capability_registry.py`, `victor/agent/orchestrator.py`, `tests/unit/framework/test_tiered_tool_config.py`, `tests/unit/framework/test_vertical_integration.py`, `tests/unit/agent/test_capability_registry.py` |
| 2026-02-20 | P3/P4 | Extended startup KPI benchmark with optional vertical activation probe telemetry (runtime infra flags + framework registry totals) and added validation guardrails (`max activation`, `min registry totals`, `required runtime flags`) with focused unit coverage | `scripts/benchmark_startup_kpi.py`, `tests/unit/test_benchmark_startup_kpi.py` |
| 2026-02-20 | Docs | Updated vertical documentation examples to canonical `get_tiered_tool_config()` API usage | `docs/verticals/coding.md`, `docs/verticals/research.md`, `docs/verticals/rag.md`, `docs/verticals/devops.md`, `docs/verticals/data-analysis.md` |
| 2026-02-20 | Cross-phase | Fixed integration regressions in bootstrap/shim paths: typed coercion for extension-loader runtime settings, robust `default_vertical` fallback, and built-in vertical re-registration in shim helpers; aligned observability mock setter semantics in integration tests | `victor/core/bootstrap.py`, `victor/framework/shim.py`, `tests/unit/core/test_bootstrap_runtime_config.py`, `tests/integration/verticals/test_cli_framework_integration.py`, `tests/unit/framework/test_framework_shim.py` |
| 2026-02-20 | P0/P1 | Extended strict-mode coverage beyond capability-config helpers by adding protocol-fallback blocking for capability runtime + integration snapshot getter fallbacks, with settings/env wiring and focused regression tests | `victor/framework/strict_mode.py`, `victor/framework/capability_runtime.py`, `victor/framework/vertical_integration.py`, `victor/framework/_internal.py`, `victor/config/settings.py`, `tests/unit/agent/test_capability_registry.py`, `tests/unit/framework/test_vertical_integration.py`, `tests/unit/core/test_settings.py` |
| 2026-02-20 | Cross-phase | Revalidated post-hardening matrices and KPI baseline: framework/agent unit slices (`226 passed`), framework/vertical integration slice (`76 passed`), and startup+activation KPI guardrails (all thresholds/requirements passing) | `../.venv/bin/pytest tests/unit/agent/test_capability_registry.py tests/unit/framework/test_vertical_integration.py tests/unit/core/test_settings.py -q`, `../.venv/bin/pytest tests/integration/verticals/test_cli_framework_integration.py tests/integration/framework/test_vertical_capability_integration.py tests/integration/framework/test_vertical_reliability_stress.py -q`, `../.venv/bin/python scripts/benchmark_startup_kpi.py ... --json` |
| 2026-02-20 | P0/P1 | Added focused CI fast-check lane enforcing protocol-fallback strict mode on framework compatibility probes | `.github/workflows/ci-fast.yml`, `tests/unit/agent/test_capability_registry.py`, `tests/unit/framework/test_vertical_integration.py` |
| 2026-02-20 | P1 | Finalized capability-config scoping model as DI-global service with scoped session keys and protocol port resolution (`get_capability_config_scope_key`) across helpers/step-handler/orchestrator | `victor/framework/capability_config_service.py`, `victor/framework/capability_config_helpers.py`, `victor/framework/protocols.py`, `victor/framework/step_handlers.py`, `victor/agent/orchestrator.py`, `tests/unit/framework/test_capability_config_service.py`, `tests/unit/framework/test_capability_config_helpers.py`, `tests/unit/framework/test_framework_step_handler.py` |
| 2026-02-20 | P3 | Codified per-topic observability overflow defaults (critical topics `block_with_timeout`, metrics topics `drop_oldest`) with settings validation, backend policy resolution, and stress/integration coverage | `victor/config/settings.py`, `victor/core/events/backends.py`, `victor/core/events/protocols.py`, `docs/reference/configuration-options.md`, `tests/unit/core/test_settings.py`, `tests/unit/core/events/test_event_backends.py`, `tests/unit/observability/test_event_registry.py`, `tests/integration/framework/test_vertical_reliability_stress.py` |
| 2026-02-20 | P3/P4 | Closed non-MLX preflight decision by keeping `vllm`/`llamacpp` lazy-only (no hardware preflight) and adding regression coverage for MLX-flag independence | `victor/providers/registry.py`, `tests/unit/providers/test_providers_registry.py` |
| 2026-02-20 | P4 | Completed orchestrator decomposition slice 4 by introducing coordination runtime boundaries with lazy `recovery_coordinator`, `chunk_generator`, `tool_planner`, and `task_coordinator` materialization | `victor/agent/runtime/coordination_runtime.py`, `victor/agent/runtime/__init__.py`, `victor/agent/orchestrator.py`, `tests/unit/agent/test_coordination_runtime.py` |
| 2026-02-20 | P4 | Completed orchestrator decomposition slice 5 by extracting interaction runtime boundaries with lazy `chat_coordinator`, `tool_coordinator`, and `session_coordinator` materialization; added comprehensive unit tests validating lazy initialization for all runtime boundaries (coordination, interaction, provider, metrics, workflow); removed redundant `hasattr` compatibility probes from framework step handlers, enforcing protocol-only boundaries | `victor/agent/runtime/interaction_runtime.py`, `victor/agent/runtime/__init__.py`, `victor/agent/orchestrator.py`, `victor/framework/step_handlers.py`, `tests/unit/agent/test_runtime_lazy_init.py` |
| 2026-02-21 | P4 | Completed orchestrator decomposition slice 6 by introducing resilience runtime boundaries with lazy `recovery_handler` and `recovery_integration` materialization, while preserving property-level compatibility for callers expecting concrete instances | `victor/agent/runtime/resilience_runtime.py`, `victor/agent/runtime/__init__.py`, `victor/agent/orchestrator.py`, `tests/unit/agent/test_resilience_runtime.py` |
| 2026-02-21 | P4/P3 | Hardened startup KPI guardrails: activation probe now measures cold+warm timing, resolves framework registry totals directly from service metrics, and enforces lazy runtime expectations (`coordination` + `interaction`) through script flags wired into fast CI | `scripts/benchmark_startup_kpi.py`, `.github/workflows/ci-fast.yml`, `tests/unit/test_benchmark_startup_kpi.py` |

---

## Minimal KPI Set

- Vertical activation p95 latency.
- Extension loading thread count and pool queue depth.
- Integration cache hit rate and replay/no-op ratio.
- Observability queue drop rate.
- Startup time to first token in CLI and SDK paths.
