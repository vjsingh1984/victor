# Victor Roadmap (Canonical)

**Status**: Single source of truth for all roadmap priorities
**Last Updated**: 2026-03-26
**Next Review**: 2026-03-30 (weekly)

> ⚠️ **All other planning documents are considered non-canonical** unless explicitly
> referenced below as supporting documentation. This roadmap overrides any conflicting
> priorities or timelines found in other documents.

## Document Governance

### Canonical Hierarchy
1. **This file** (`roadmap.md`) - Active 90-day priorities and directional horizons
2. [`docs/planning/RANKED_90_DAY_EXECUTION_PLAN.md`](docs/planning/RANKED_90_DAY_EXECUTION_PLAN.md) - Detailed 90-day execution plan with ranked initiatives
3. [`docs/planning/`](docs/planning/) - Supporting planning documents and templates

### Archived Plans (Historical Reference Only)
- [`docs/roadmap/improvement-plan-v1.md`](docs/roadmap/improvement-plan-v1.md) - Superseded by 90-day plan
- [`docs/roadmap/framework-vertical-foundation-plan.md`](docs/roadmap/framework-vertical-foundation-plan.md) - Active foundation work (tracked separately)
- [`docs/P0-P4_IMPLEMENTATION_STATUS.md`](docs/P0-P4_IMPLEMENTATION_STATUS.md) - Completed (2025-02-21)

### Maintenance Process
- **Weekly updates**: Monday of each week (or at milestone cuts)
- **Review cadence**: At each M1/M2/M3 cut
- **Conflict resolution**: This roadmap takes precedence over all other planning documents
- **Change process**: Edit this file directly, then update linked documents as needed

## Governance

- Owner role: Product/Program Lead
- Update cadence: Weekly (Monday) and at each milestone cut
- Scope: Active 90-day execution plan + directional horizons

Related documents:
- Product vision: [`VISION.md`](VISION.md)
- Detailed strategy archive: [`docs/roadmap/improvement-plan-v1.md`](docs/roadmap/improvement-plan-v1.md)
- Tracker template: [`docs/planning/GITHUB_PROJECT_90_DAY_TEMPLATE.md`](docs/planning/GITHUB_PROJECT_90_DAY_TEMPLATE.md)
- Consolidated implementation reference: [`docs/planning/consolidated-implementation-plan-2026-03-27.md`](docs/planning/consolidated-implementation-plan-2026-03-27.md)
- Release review logs: [`docs/roadmap/release-reviews/2026q2/README.md`](docs/roadmap/release-reviews/2026q2/README.md)
- Evidence-based assessment: [`docs/tech-debt/codebase-assessment-2026-03-26.md`](docs/tech-debt/codebase-assessment-2026-03-26.md)
- Issue-sized execution backlog: [`docs/planning/tiered-execution-backlog-2026-03-26.md`](docs/planning/tiered-execution-backlog-2026-03-26.md)

## 2026-03-26 Hindsight Refresh

- The previous 2026-03-15 assessment is now partially stale against the live tree and has been superseded.
- Current concentration risk is broader than orchestration alone: `victor/integrations/api/server.py`, `victor/framework/vertical_integration.py`, `victor/agent/conversation_memory.py`, and `victor/storage/vector_stores/proximadb_multi.py` now deserve first-class roadmap attention alongside `victor/agent/orchestrator.py`.
- Security posture improved materially, but generic and server-side secret fields still need the same redaction-safe treatment already used for provider-specific keys.
- The strongest product differentiators remain the external vertical contract model, event bridge reliability work, and the integrated ProximaDB-backed semantic/graph/metrics indexing path.
- The largest product-execution gaps are now benchmark publication, onboarding clarity, and deciding whether the observability dashboard remains a prototype or becomes a supported surface.

## Current Execution Order (Tiered)

1. **Foundational alignment**
   - Keep roadmap dates current.
   - Keep one active assessment and one durable execution backlog.
   - Maintain a concise vision statement in-repo.
2. **Security hardening**
   - Normalize secret handling across provider, server, and session settings.
   - Keep scanner/SBOM operator guidance close to the implemented baseline.
3. **Architecture reduction**
   - Remove deprecated sync orchestration seams.
   - Complete the real workflow compiler boundary.
   - Decompose the API server, vertical integration, conversation memory, and ProximaDB provider hotspots.
4. **Product proof**
   - Publish benchmark results.
   - Clarify onboarding and happy-path docs.
   - Productize or explicitly archive the observability dashboard prototype.

## Current 90-Day Priorities (2026Q2)

| Epic | Focus | Current Milestone |
|------|-------|-------------------|
| `E1` | Orchestration tech-debt burn-down | `M3` in progress (orchestrator 3,940 LOC; 37 properties + callbacks + session state extracted; protocol injection pending) |
| `E2` | Roadmap governance consolidation | `M3` in progress (audit corrections + drift guardrails) |
| `E3` | Type-safety + quality gates | `M3` complete |
| `E4` | Event bridge reliability | `M3` complete |
| `E5` | Legacy compatibility debt reduction | `M3` in progress (9/13 = 69% removed; migration notes still open) |
| `E6` | Competitive benchmark ground-truth | `M2` in progress |

Milestone targets:
- `M1: Foundation Cut` due 2026-03-31
- `M2: Midpoint Cut` due 2026-04-28
- `M3: Quarter Exit` due 2026-06-01

## Directional Horizons

| Horizon | Focus | Example Outcomes |
|---------|-------|------------------|
| **0-3 months** | Stability + execution rigor | Coordinator extraction, roadmap governance, baseline quality gates |
| **3-6 months** | Reliability + scale | Event delivery SLOs, global strict CI enforcement, midpoint benchmark runs |
| **6-12 months** | Platform maturity | Broader ecosystem growth, advanced multi-agent and workflow capabilities |

## Recently Delivered

- Provider switching with context independence
- Workflow DSL with graph execution
- Multi-agent team formations
- Provider/tool/vertical registries and validation

## Active Work (M1-M2)

### E1: Orchestration Tech-Debt Burn-Down
**Status**: M3 in progress
**Owner**: Architecture Lead
**Progress**:
- ✅ M1: Extract execution coordinator (ExecutionCoordinator: 415 lines)
- ✅ M1: Extract tool coordinator (ToolCoordinator: 1565 lines)
- ✅ M1: Extract chat coordinator (ChatCoordinator: 1464 lines)
- ✅ M1: Extract planning coordinator (PlanningCoordinator: 520 lines)
- ✅ M2: Sync/streaming coordinator paths split
  - SyncChatCoordinator: 228 lines (clean)
  - StreamingChatCoordinator: 281 lines (clean)
- ✅ M2: Remove dead legacy code (-196 lines from chat_coordinator)
- ✅ M2: Reduce coordinator sizes (ChatCoordinator: 1464 → 1372 LOC, -92 lines)
- ✅ M3: ToolCoordinator reduced from 1565 → ~1185 LOC via extraction
  - Extracted `ToolObservabilityHandler` (tool_observability.py): on_tool_complete, preview helpers, stats
  - Extracted `ToolRetryExecutor` (tool_retry.py): execute_tool_with_retry with full retry/cache logic
  - Thin delegation methods preserved on ToolCoordinator for backward compatibility
- ⏳ M3: Protocol-based injection complete

**Current Coordinator LOC**:
| Coordinator | LOC | Target | Status |
|-------------|-----|--------|--------|
| ChatCoordinator | 1372 | 1400 | ✅ Below M2 target (was 1464) |
| ToolCoordinator | ~1185 | 1200 | ✅ Below target (was 1565) |
| ExecutionCoordinator | 415 | 1200 | ✅ Below target |
| SyncChatCoordinator | 228 | 1200 | ✅ Below target |
| StreamingChatCoordinator | 281 | 1200 | ✅ Below target |
| PlanningCoordinator | 520 | 1200 | ✅ Below target |

**M2 Complete**: ChatCoordinator reduced from 1464 → 1372 LOC (-92 lines)
- Removed duplicate planning logic in `chat_with_planning`
- Simplified recovery integration methods (thin wrappers)
- Simplified intelligent response validation method

**M3 Complete**: ToolCoordinator reduced from 1565 → ~1185 LOC (-380 lines)
- Extracted observability (on_tool_complete, preview helpers, stats) into ToolObservabilityHandler
- Extracted retry logic (execute_tool_with_retry) into ToolRetryExecutor
- Backward-compatible thin delegation methods preserved

### E2: Roadmap Governance Consolidation
**Status**: M1 complete, M2 complete, M3 in progress
**Owner**: Product/Program Lead (assigned 2026-03-10)
**Progress**:
- ✅ M1: Canonical roadmap established (roadmap.md)
- ✅ M1: Single source of truth defined with governance hierarchy
- ✅ M1: E3 (Type-Safety) M1 complete - 11 strict modules, CI-blocking
- ✅ M1: E4 (Event Bridge) M1 complete - async subscribe path merged
- ✅ M2: Duplicate sources inventory - No conflicts found
- ✅ M2: Active work mapping to owner/date/KPI - 100% complete (6/6 epics)
- ✅ M2: Weekly update cadence setup - Template ready, tracking defined
- ✅ M2: GitHub labels and milestones documented
- ✅ 2026-03-15: Workflow validation/governance drift corrected in-repo
- ✅ 2026-03-15: Automation added for workflow syntax, repo-link, roadmap-link, archive-banner, and local lint-gate drift checks

### E3: Type-Safety + Quality Gates
**Status**: M3 complete
**Owner**: DevEx/Quality Lead (assigned 2026-03-10)
**Progress**:
- ✅ M1: MyPy baseline captured (11 strict modules, 0 findings)
- ✅ M1: Strict mode expanded to 11 modules (exceeds target of >= 6)
- ✅ M1: Strict-package mypy made CI-blocking
- ✅ M2: Expanded to 15 modules (+4: victor.state, victor.workflows, victor.teams, victor.integrations.api)
- ✅ M2: All new modules pass strict mypy checks
- ✅ M2: CI workflow updated to enforce strict checking on all 15 modules
- ✅ M3: Global strict mode enabled across the full `victor/` tree
- ✅ M3: Zero mypy findings maintained
- ✅ M3: CI updated to enforce global strict (`mypy victor --strict`)
- ✅ 2026-03-15: `make lint` realigned with the current mypy gate after local verification (`mypy victor`)
- 📄 Baseline report: [`docs/quality/mypy_baseline_report.md`](docs/quality/mypy_baseline_report.md)

**Strict Module Timeline**:
- M1 baseline (6 modules): `victor.config.*`, `victor.storage.cache.*`, `victor.telemetry.*`, `victor.analytics.*`, `victor.profiler.*`, `victor.debug.*`
- M1 expansion (+5): `victor.agent.services.*`, `victor.core.container`, `victor.core.protocols`, `victor.providers.base`, `victor.framework.*`
- M2 expansion (+4): `victor.state.*`, `victor.workflows.*`, `victor.teams.*`, `victor.integrations.api.*`
- **M3 achievement**: Global strict mode enabled for entire codebase

**Technical Debt**: 15 modules use `ignore_errors = true` for gradual remediation (documented in baseline report)

### E4: Event Bridge / Observability Reliability
**Status**: M3 complete
**Owner**: Observability Lead (assigned 2026-03-10)
**Progress**:
- ✅ M1: Async subscribe path merged (primary API)
- ✅ M1: Sync subscribe deprecated (RuntimeError if used)
- ✅ M1: EventBusAdapter.connect_async() added
- ✅ M1: EventBusAdapter.disconnect_async() added
- ✅ M1: EventBridge.async_start() and async_stop() added
- ✅ M1: 11 new async path tests added
- ✅ M2: Integration tests for loss/ordering (12 tests)
- ✅ M2: High-volume burst, slow consumer, reconnect scenarios
- ✅ M2: Event ordering tests (single source, concurrent, varying latency)
- ✅ M2: SLO validation tests (delivery rate, p95 latency, skipped subscriptions)
- ✅ M3: SLO dashboards published
- ✅ M3: Prometheus metrics exporter created
- ✅ M3: Alerting thresholds documented
- ✅ M3: Runbook for common issues

**M2 Integration Tests Added**:
- `test_no_event_loss_during_high_volume_burst` - 100 events, no loss
- `test_no_event_loss_during_slow_consumer` - Queued delivery with 50ms delay
- `test_event_delivery_with_client_reconnect` - Reconnect scenario
- `test_multiple_clients_no_loss` - 5 concurrent clients
- `test_events_arrive_in_order_from_single_source` - Sequential ordering
- `test_event_ordering_with_concurrent_sources` - Multi-source ordering
- `test_event_ordering_with_varying_processing_times` - Variable latency ordering
- `test_delivery_success_rate_slo` - Validates 99.9% SLO
- `test_dispatch_latency_p95_slo` - Validates < 200ms SLO
- `test_zero_skipped_subscriptions` - All event types subscribed
- `test_sustained_load_no_loss` - 10 batches of 20 events
- `test_reliability_metrics_under_load` - SLO validation under load

**M3 Deliverables**:
- SLO dashboard: `docs/observability/slo-dashboard.md`
- Metrics exporter: `victor/integrations/api/metrics_exporter.py`
- CLI access: `python -m victor.integrations.api.metrics_exporter`

### E6: Competitive Benchmark Ground-Truth
**Status**: M1 complete, M2 code-complete (runtime execution remaining)
**Owner**: Platform Lead (assigned 2026-03-10)
**Progress**:
- ✅ M1: Benchmark rubric frozen (v1.0)
- ✅ M1: 22 tasks defined across 5 categories (exceeds target of 20)
- ✅ M1: Scoring methodology documented (100-point scale)
- ✅ M1: Competitor matrix complete (6 frameworks)
- ✅ M1: 3 example tasks created (C1, R2, W2)
- ✅ M2: Benchmark execution script created
- ✅ M2: All 22 task definitions in TASK_REGISTRY (was 8, added R3-R4, T2-T5, A1-A4, W2-W4)
- ✅ M2: Task definition markdown files for all categories (22 files in docs/benchmarking/tasks/)
- ✅ M2: Victor adapter implemented
- ✅ M2: Competitor stub adapters created (LangGraph, CrewAI)
- ✅ M2: Statistical significance analysis module (analyze_results.py)
  - Per-task metrics: mean, std dev, 95% CI across runs
  - High-variance flagging (std dev > 20% of mean)
  - Weighted composite scores (6 dimensions per scoring methodology)
  - Framework comparison with statistical tie detection (overlapping CIs)
  - CLI + JSON output
- 🔄 M2: Execute benchmarks on Victor + 2 competitors (runtime, requires API keys)
- ✅ M2: P0-P1 benchmark fixes applied (2026-03-11):
  - R4 (Debug investigation): Fixed — error recovery fallback wired into tool pipeline; `semantic_code_search` now falls back to `code_search`/`grep` on dependency errors
  - T1 (File operations): Per-tool timeout added (prevents individual tool hangs); overall 90s task timeout still exceeded due to LLM inference latency on local Ollama
  - Memory: First-task spike reduced from +694MB to +111MB by deferring embedding preload (`preload_embeddings=False`)
  - Per-tool timeout: `asyncio.wait_for()` wraps all serial tool calls (15-60s based on complexity)
- ⏳ M3: Report published, action items identified

**Benchmark Suite** (22 tasks):
- Code Generation (5): Single-file, multi-file refactor, bug fix, code review, docs
- Multi-Step Reasoning (4): Research synthesis, architecture design, migration planning, debug investigation
- Tool Usage (5): File ops, git workflow, web research, database operations, command execution
- Analysis (4): Security audit, performance analysis, dependency analysis, test coverage
- Workflow (4): Sequential, parallel, human-in-the-loop, error recovery

**Competitors**:
- LangGraph (workflow/state machines)
- CrewAI (role-based agents)
- AutoGPT (autonomous agents)
- OpenAI Swarm (multi-agent patterns)
- Semantic Kernel (enterprise)

**Scoring**:
- Task Success Rate (40%)
- Output Quality (20%)
- Execution Speed (10%)
- Resource Efficiency (15%)
- Reliability (10%)
- Developer Experience (5%)

**Documentation**: [`docs/benchmarking/`](docs/benchmarking/)

## Deep Audit Findings (2026-03-15)

Full assessment: [`docs/tech-debt/codebase-assessment-2026-03-15.md`](docs/tech-debt/codebase-assessment-2026-03-15.md)

### Completed in This Tranche
- ✅ Architecture strengthening: orchestrator 4,514→3,940 LOC, extension loader 2,049→1,897 LOC
- ✅ SDK contract: ExtensionManifest, CapabilityNegotiator, API versioning (v2)
- ✅ ProviderPool deduplicated and wired with `use_provider_pooling` feature flag
- ✅ Contrib verticals emit DeprecationWarning; external packages preferred
- ✅ Victor-devops: 0 forbidden imports (migrated to `victor.framework.extensions`)
- ✅ 32 documentation files reconciled with current metrics (24 providers, 34 tools)
- ✅ Bare `except:` fixed in `experiments.py`
- ✅ `eval()`/`exec()` audit: only 2 real calls (both sandboxed with `__builtins__: {}`)

### Remaining Priority Queue

| Priority | ID | Item | Status |
|----------|-----|------|--------|
| P0 | E1 | Reduce orchestrator.py below 3,800 LOC | ✅ Done (`c6e63df`) — 3,940→3,783 LOC |
| P0 | E6 | Execute Victor benchmark baseline (22/22 tasks, 100% success, claude-sonnet-4-20250514) | ✅ Done (`84bc900`) |
| P1 | S-02 | Baseline Semgrep exclusions so SAST can become merge-blocking | Backlog |
| P1 | S-04 | Audit provider/API-key fields for SecretStr adoption | ✅ Done — no gaps found |
| P1 | D-02 | Decompose codebase_analyzer.py (2,988 LOC) into 5 modules | ✅ Done (`d070486`) |
| P1 | D-03 | Decompose orchestrator_factory.py (2,773→719 LOC) into 4 builder modules | ✅ Done (`d070486`) |
| P2 | F-01 | Convert the 19 actionable TODO/FIXME markers into tracked issues | Backlog |
| P2 | X-01 | Document high-churn settings fields with examples and migration notes | Backlog |
| P3 | F-04 | Reduce `Any` type annotations: factory 102→11, service_provider 69→32, adapters 31→11 | ✅ Done (top 3 offenders) |
| P3 | D-03 | Decompose `indexer.py` (3,555→package) and `native/__init__.py` (3,112→297 LOC) | ✅ Done (`4e8b8c0`, `d8c0b4b`) |

## How to Influence the Roadmap

- Open an issue/discussion with a concrete use case and measurable impact.
- Submit a [FEP](feps/fep-0000-template.md) for framework-level changes.
