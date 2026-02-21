# Framework + Vertical Integration Re-Analysis

Generated: 2026-02-20 04:34:27Z  
Baseline commit: `59ad5aae8`  
Updated: 2026-02-21 04:04:58Z (full architecture re-analysis + scoring refresh)

## Scope
- Framework core: API, orchestration, tool system, events/observability.
- Vertical system: registration, pipelines, adapters, extensions, workflows, RL, teams.
- Framework/vertical boundary quality and where generic capabilities should live.

## Verification Addendum (2026-02-21 04:05Z)
- Fresh KPI strict-guard run (same flags as fast CI) failed only one expectation:
  - `activation_probe.runtime_flags.interaction_runtime_lazy: False != True`.
  - Evidence: `scripts/benchmark_startup_kpi.py:469`, `scripts/benchmark_startup_kpi.py:471`, `.github/workflows/ci-fast.yml:121`, `.github/workflows/ci-fast.yml:122`.
- Fresh KPI baseline run (without `--require-interaction-runtime-lazy`) passed all thresholds:
  - `import_victor.cold_ms=1572.92`, `agent_create.cold_ms=902.72`, `activation.cold_ms=872.35`.
  - `activation.warm_mean_ms=379.71`, framework registry totals `attempted=19`, `applied=57`.
  - Runtime flags: cache/http/preload/coordination lazy=true, interaction lazy=false.
- Fresh validation suite run in `../.venv`:
  - `../.venv/bin/pytest ... test_runtime_lazy_init.py ... test_benchmark_startup_kpi.py -q` -> `28 passed in 71.48s`.
- KPI script activation probe is now class-safe via shim resolution (raw vertical string bug fixed):
  - `scripts/benchmark_startup_kpi.py:203`, `scripts/benchmark_startup_kpi.py:209`, `scripts/benchmark_startup_kpi.py:298`.

## 1) Architecture Map (Framework <-> Verticals)

### Core modules and roles
| Layer | Key modules | Current role |
|---|---|---|
| Public framework API | `victor/framework/agent.py:112` | `Agent.create()` entrypoint and SDK lifecycle. |
| API -> runtime bridge | `victor/framework/_internal.py:53` | Creates orchestrator, bootstraps container, applies vertical + observability integration. |
| Shared vertical application service | `victor/framework/vertical_service.py:92` | Unified `apply_vertical_configuration()` backed by shared `VerticalIntegrationPipeline`. |
| Vertical activation | `victor/core/bootstrap.py:633`, `victor/core/bootstrap.py:747`, `victor/core/verticals/vertical_loader.py:732` | Canonical vertical activation and switch path. |
| Integration pipeline | `victor/framework/vertical_integration.py:745`, `victor/framework/vertical_integration.py:889` | Builds context, executes step handlers, caches integration plans, emits `vertical.applied`. |
| Step handlers | `victor/framework/step_handlers.py:987` | Ordered handlers for tools/tiered config/extensions/framework registries/context. |
| Framework registry facade | `victor/framework/framework_integration_registry_service.py:36`, `victor/framework/framework_integration_registry_service.py:307` | Versioned/idempotent framework registration + per-kind metrics snapshot. |
| Orchestrator runtime facade | `victor/agent/orchestrator.py:430` | Large facade with provider/memory/metrics/workflow/coordination/interaction/resilience runtime wiring. |
| Runtime decomposition modules | `victor/agent/orchestrator.py:452`, `victor/agent/orchestrator.py:492`, `victor/agent/orchestrator.py:518`, `victor/agent/orchestrator.py:525`, `victor/agent/orchestrator.py:537`, `victor/agent/orchestrator.py:550`, `victor/agent/runtime/coordination_runtime.py:26`, `victor/agent/runtime/interaction_runtime.py:25`, `victor/agent/runtime/resilience_runtime.py:24` | Lazy runtime proxies and subsystem boundary extraction. |
| Observability/event transport | `victor/observability/integration.py:239`, `victor/core/events/backends.py:321`, `victor/config/settings.py:954` | EventBus wiring, overflow policies, topic-level backpressure controls. |

### End-to-end data flow
1. `Agent.create()` enters framework API (`victor/framework/agent.py:112`) and bridges through internal factory/bootstrap (`victor/framework/_internal.py:53`).
2. Bootstrap ensures container + vertical activation (`victor/core/bootstrap.py:633`, `victor/core/bootstrap.py:747`, `victor/core/verticals/vertical_loader.py:732`).
3. Orchestrator is instantiated and runtime boundaries are attached (`victor/agent/orchestrator.py:452`, `victor/agent/orchestrator.py:525`, `victor/agent/orchestrator.py:550`).
4. Vertical pipeline applies step handlers and cached plan decisions (`victor/framework/vertical_service.py:92`, `victor/framework/vertical_integration.py:889`, `victor/framework/vertical_integration.py:1871`).
5. Framework artifacts are registered via service facade (`victor/framework/step_handlers.py:1148`, `victor/framework/step_handlers.py:1313`, `victor/framework/step_handlers.py:1614`, `victor/framework/framework_integration_registry_service.py:307`).
6. Observability emits final integration payload with extension + registry metrics (`victor/framework/vertical_integration.py:1556`, `victor/framework/vertical_integration.py:1614`, `victor/framework/vertical_integration.py:1642`).

## 2) Gaps: Generic Capabilities Still Embedded in Verticals

| Gap ID | Gap | Evidence | Promote to framework | Specific suggestion |
|---|---|---|---|---|
| G11 | Resolved: capability helper logic centralized. | `victor/framework/capability_runtime.py:14`, `victor/framework/step_handlers.py:148`, `victor/framework/vertical_integration.py:166` | Keep one capability runtime helper surface. | Block new local helper copies in review checks. |
| G12 | Mostly resolved: strict fallback guards are active but fallback paths still exist for compatibility. | `victor/framework/strict_mode.py:12`, `victor/framework/vertical_integration.py:1401`, `.github/workflows/ci-fast.yml:144` | Protocol-only framework/orchestrator contract. | Continue removing fallback code paths and keep strict lanes required in CI. |
| G13 | Resolved: framework registry writes are service-owned and measurable. | `victor/framework/framework_integration_registry_service.py:36`, `victor/framework/framework_integration_registry_service.py:307`, `victor/framework/step_handlers.py:1148` | Registry idempotence/metrics should remain framework-owned. | Add alert thresholds on registry churn deltas in KPI reporting. |
| G14 | Partial: legacy tiered-tools adapter still exists in extension loader compatibility path. | `victor/core/verticals/extension_loader.py:737`, `victor/core/verticals/extension_loader.py:774`, `victor/rag/assistant.py:173` | Canonical `get_tiered_tool_config()` only. | Enforce hard removal after 2026-06-30 and add static check for banned `get_tiered_tools()`. |
| G15 | Resolved: capability-config storage is DI-scoped service with scope-key resolution. | `victor/framework/capability_config_service.py:13`, `victor/framework/capability_config_helpers.py:39`, `victor/framework/capability_config_helpers.py:69` | Service-backed capability config as framework primitive. | Keep `resolve_capability_config_scope_key()` as the only scope source. |
| G16 | Partial: generic cache/http/preload capabilities are integrated but still uneven in adoption/ROI evidence. | `victor/tools/base.py:137`, `victor/tools/web_search_tool.py:123`, `victor/framework/preload.py:184`, `victor/agent/orchestrator.py:2372` | Generic runtime services should be first-class framework infra. | Keep only paths with measured KPI benefit; retire non-adopted generic modules. |
| G17 | Active: interaction runtime laziness breaks during activation path. | `victor/agent/orchestrator.py:550`, `victor/agent/orchestrator.py:570`, `victor/agent/orchestrator.py:4059`, `victor/framework/step_handlers.py:438`, `scripts/benchmark_startup_kpi.py:471` | Lazy-boundary semantics should be framework-defined and testable. | Split `set_enabled_tools` into lightweight state update + deferred coordinator sync on first tool execution. |

## 3) SOLID Evaluation (Violations and Fixes)

| Principle | Status | Evidence | Violation | Fix |
|---|---|---|---|---|
| SRP | Partial | `victor/agent/orchestrator.py` (4274 LOC), runtime init spread across constructor path | Orchestrator still mixes lifecycle, routing, tooling, session, recovery, observability. | Extract `tool_runtime`, `chat_runtime`, `session_runtime` facades and keep orchestrator as coordinator shell. |
| OCP | Improved | Metadata-driven plan/handler execution (`victor/framework/vertical_integration.py:1871`, `victor/framework/step_handlers.py:295`) | Extension points improved, but legacy compatibility branches still alter behavior. | Move compatibility adapters behind explicit plugin policies and keep core path closed to legacy conditionals. |
| LSP | Partial | Deprecated tiered adapter remains (`victor/core/verticals/extension_loader.py:737`) | Old adapter contracts allow non-canonical substitution behavior. | Remove adapter deadline path and enforce canonical contract-only vertical base tests. |
| ISP | Partial | Broad orchestrator port remains required by framework steps (`victor/framework/step_handlers.py:438`, `victor/agent/orchestrator.py:4050`) | Step handlers depend on large orchestrator surface. | Introduce narrow `VerticalRuntimePort` with only context/tools/middleware/safety operations. |
| DIP | Partial | Service inversion present (`victor/framework/framework_integration_registry_service.py:36`) but method-mapping fallback still used (`victor/framework/capability_registry.py:25`) | Framework still relies on concrete method naming in fallback mode. | Remove fallback mapping from active path; require protocol registry capabilities for framework-owned writes. |

## 4) Scalability + Performance Risks

| Hot path | Risk | Evidence | Mitigation |
|---|---|---|---|
| Agent startup | Constructor remains heavy despite runtime slicing. | `victor/agent/orchestrator.py:430`, `victor/agent/orchestrator.py:970`, `victor/agent/orchestrator.py:1013` | Continue constructor slimming; instantiate only immutable/lightweight coordinators in `__init__`. |
| Activation lazy-boundary regression | Vertical tool application currently forces interaction runtime materialization. | `victor/framework/step_handlers.py:438`, `victor/agent/orchestrator.py:4059`, KPI failure on `interaction_runtime_lazy` | Decouple activation-time tool state from `ToolCoordinator` object creation. |
| Registry churn in repeated integrations | Metrics exist but no enforced drift thresholds per vertical/version. | `victor/framework/framework_integration_registry_service.py:307`, `victor/framework/vertical_integration.py:1578` | Add baseline envelopes and fail KPI if attempted/applied drift exceeds tolerance. |
| Extension loading saturation | Shared executor is bounded but static. | `victor/core/verticals/extension_loader.py:137`, `victor/core/verticals/extension_loader.py:180`, `victor/core/verticals/extension_loader.py:225` | Make worker/queue sizing adaptive to CPU + workload profile. |
| Event pipeline under pressure | Topic-level policies exist but tuning remains static. | `victor/config/settings.py:956`, `victor/core/events/backends.py:465`, `victor/core/events/backends.py:689` | Add deployment presets and alerting on sustained queue-pressure transitions. |
| Generic infra surface creep | Cache/preload/http exports exceed proven active-path ROI. | `victor/storage/cache/__init__.py:61`, `victor/framework/preload.py:184`, `victor/tools/http_pool.py:121` | Keep KPI-gated features only; prune dormant generic modules. |

## 5) Competitive Comparison (6 dimensions, 1-10 + one-line rationale)

Weights: Boundary Clarity 20, Orchestration Determinism 20, Observability/Reliability 15, Runtime Performance 15, Vertical Extensibility 15, Multi-agent/Workflow Depth 15.

| Dimension (Weight) | Victor | LangGraph | CrewAI | LangChain | LlamaIndex | AutoGen |
|---|---|---|---|---|---|---|
| Boundary clarity (20) | 8 - Protocol/DI trajectory is strong, with remaining fallback compatibility paths. | 8 - Graph runtime boundaries are explicit and clean. | 6 - Crew abstraction is clear but framework internals are less rigid. | 6 - Broad APIs with mixed-era layering patterns. | 7 - Clear data/retrieval boundary with narrower orchestration focus. | 6 - Agent abstractions are strong, boundaries less strict. |
| Orchestration determinism (20) | 8 - Metadata dependency levels + plan cache make execution predictable. | 9 - Mature DAG/state graph determinism. | 6 - Role flow is good but less dependency-explicit. | 7 - Determinism varies by chain/agent mix. | 6 - Strong retrieval pipeline control, less general orchestration depth. | 7 - Conversation orchestration is strong, static dependency control is lighter. |
| Observability/reliability (15) | 8 - Topic overflow policy + pressure metrics + vertical.applied payload are integrated. | 7 - Good tracing integrations, less built-in queue policy depth. | 5 - Basic telemetry/reliability controls. | 6 - Strong integrations, less unified runtime reliability policy. | 6 - Good pipeline telemetry, limited runtime-wide policy controls. | 5 - Core telemetry exists, policy model is lighter. |
| Runtime performance/startup (15) | 7 - Runtime slices exist, but activation still eagerly materializes interaction runtime in current path. | 7 - Lean core with integration-dependent overhead. | 6 - Moderate overhead grows with crew complexity. | 6 - Feature-rich but heavier runtime/import profile. | 7 - Efficient for RAG-centric workloads. | 6 - Multi-agent patterns can be heavy at runtime. |
| Vertical extensibility (15) | 8 - Rich vertical protocol surface across workflows/RL/teams/tools. | 6 - Extensible nodes, fewer vertical packaging conventions. | 7 - Team-centric extension model is practical. | 8 - Very broad extension ecosystem. | 9 - Strongest retrieval/data extension ecosystem. | 6 - Extensible agents with less vertical packaging depth. |
| Multi-agent/workflow depth (15) | 8 - Teams/workflows/RL integrated in one vertical pipeline path. | 7 - Strong workflow modeling, less team-policy depth. | 8 - Strong collaboration primitives. | 6 - Multi-agent available but less cohesive by default. | 5 - Focus remains retrieval/data over teams. | 9 - Deep agent-to-agent delegation patterns. |

## 6) Roadmap to Best-in-Class

### Phase A (0-2 weeks): Runtime lazy-boundary correctness
- Refactor `set_enabled_tools()` to avoid forcing `ToolCoordinator` materialization during activation (`victor/agent/orchestrator.py:4050`).
- Add explicit runtime-lazy contract test at activation boundary and align KPI guardrail semantics (`scripts/benchmark_startup_kpi.py:469`, `.github/workflows/ci-fast.yml:121`).
- Decide contract: either keep strict lazy requirement and fix code, or intentionally relax requirement and update CI/KPI accordingly.

### Phase B (2-5 weeks): Contract hardening + fallback retirement
- Continue removal of compatibility fallbacks from framework integration paths (`victor/framework/vertical_integration.py:1401`, `victor/framework/capability_runtime.py:38`).
- Keep strict fallback CI lanes mandatory and expand to framework step-handler probes.

### Phase C (5-8 weeks): Orchestrator decomposition completion
- Extract `tool_runtime`, `chat_runtime`, `session_runtime` modules and leave orchestrator as facade/composer.
- Reduce constructor work and side effects in `AgentOrchestrator.__init__` (`victor/agent/orchestrator.py:537` onward).

### Phase D (8-12 weeks): Adaptive scaling + registry economics
- Make extension loader queue/worker settings adaptive (`victor/core/verticals/extension_loader.py:137`, `victor/core/verticals/extension_loader.py:225`).
- Add KPI drift budgets for framework registry attempted/applied totals per vertical/version.

### Phase E (12-16 weeks): Generic infra consolidation
- Retain only cache/http/preload features with measured startup/latency improvements.
- Remove or park non-adopted generic exports to reduce import surface and maintenance load.

## 7) Weighted Scoring Table (frameworks as columns)

| Dimension (Weight) | Victor | LangGraph | CrewAI | LangChain | LlamaIndex | AutoGen |
|---|---:|---:|---:|---:|---:|---:|
| Boundary clarity (20) | 8 | 8 | 6 | 6 | 7 | 6 |
| Orchestration determinism (20) | 8 | 9 | 6 | 7 | 6 | 7 |
| Observability/reliability (15) | 8 | 7 | 5 | 6 | 6 | 5 |
| Runtime performance/startup (15) | 7 | 7 | 6 | 6 | 7 | 6 |
| Vertical extensibility (15) | 8 | 6 | 7 | 8 | 9 | 6 |
| Multi-agent/workflow depth (15) | 8 | 7 | 8 | 6 | 5 | 9 |
| **Overall weighted score** | **7.85** | **7.45** | **6.30** | **6.50** | **6.65** | **6.50** |

## Session Handoff

Source-of-truth execution plan remains `docs/analysis_reports/09_framework_vertical_integration_remediation_plan.md`.  
This report is now the refreshed architecture/scoring baseline for subsequent phases.
