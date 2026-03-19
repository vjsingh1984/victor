# Victor Roadmap (Canonical)

**Status**: Single source of truth for all roadmap priorities
**Last Updated**: 2026-03-10
**Next Review**: 2026-03-17 (weekly)

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
- Detailed strategy archive: [`docs/roadmap/improvement-plan-v1.md`](docs/roadmap/improvement-plan-v1.md)
- Tracker template: [`docs/planning/GITHUB_PROJECT_90_DAY_TEMPLATE.md`](docs/planning/GITHUB_PROJECT_90_DAY_TEMPLATE.md)
- Release review logs: [`docs/roadmap/release-reviews/2026q2/README.md`](docs/roadmap/release-reviews/2026q2/README.md)

## Current 90-Day Priorities (2026Q2)

| Epic | Focus | Current Milestone |
|------|-------|-------------------|
| `E1` | Orchestration tech-debt burn-down | `M3` in progress (ToolCoordinator ✅) |
| `E2` | Roadmap governance consolidation | `M2` complete |
| `E3` | Type-safety + quality gates | `M3` complete |
| `E4` | Event bridge reliability | `M3` complete |
| `E5` | Legacy compatibility debt reduction | `M3` complete (9/13 = 69% removed) |
| `E6` | Competitive benchmark ground-truth | `M2` in progress |

Milestone targets:
- `M1: Foundation Cut` due 2026-03-31
- `M2: Midpoint Cut` due 2026-04-28
- `M3: Quarter Exit` due 2026-06-01

## Directional Horizons

| Horizon | Focus | Example Outcomes |
|---------|-------|------------------|
| **0-3 months** | Stability + execution rigor | Coordinator extraction, roadmap governance, baseline quality gates |
| **3-6 months** | Reliability + scale | Event delivery SLOs, strict-package CI enforcement, midpoint benchmark runs |
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
**Status**: M1 complete, M2 complete
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
- ✅ M3: Global strict mode enabled (all 1,456 files)
- ✅ M3: Zero mypy findings maintained
- ✅ M3: CI updated to enforce global strict (`mypy victor --strict`)
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
**Status**: M1 complete, M2 in progress
**Owner**: Platform Lead (assigned 2026-03-10)
**Progress**:
- ✅ M1: Benchmark rubric frozen (v1.0)
- ✅ M1: 22 tasks defined across 5 categories (exceeds target of 20)
- ✅ M1: Scoring methodology documented (100-point scale)
- ✅ M1: Competitor matrix complete (6 frameworks)
- ✅ M1: 3 example tasks created (C1, R2, W2)
- ✅ M2: Benchmark execution script created
- ✅ M2: Task definition files created (C1, R2, T1, W1)
- ✅ M2: Victor adapter implemented
- ✅ M2: Competitor stub adapters created (LangGraph, CrewAI)
- 🔄 M2: Execute benchmarks on Victor + 2 competitors
- ⏳ M2: Statistical significance analysis
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

## How to Influence the Roadmap

- Open an issue/discussion with a concrete use case and measurable impact.
- Submit a [FEP](feps/fep-0000-template.md) for framework-level changes.
