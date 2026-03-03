# Ranked 90-Day Execution Plan (Q2 2026)

Plan window: **March 3, 2026 to June 1, 2026**

## Milestones

- **M1: Foundation Cut (2026-03-31)**: architecture seams defined, baseline metrics captured, roadmap source finalized.
- **M2: Midpoint Cut (2026-04-28)**: major refactors landed, quality gates expanded, reliability tests in CI.
- **M3: Quarter Exit (2026-06-01)**: KPI targets met, benchmark report published, deprecation reductions complete.

## Ranked Initiatives

| Rank | Initiative | Owner | Milestones | Success Metrics |
|---|---|---|---|---|
| 1 | Orchestration Tech-Debt Burn-Down | Architecture Lead | M1: extract execution coordinator; M2: split sync/streaming coordinator paths; M3: protocol-based injection complete | `orchestrator.py <= 3800 LOC`; no coordinator > `1200 LOC`; no behavior regressions in coordinator tests |
| 2 | Roadmap Governance Consolidation | Product/Program Lead | M1: canonical roadmap published; M2: active work mapped to owner/date/KPI; M3: 2 release review cycles complete | 100% active work mapped; weekly update adherence >= 90%; no conflicting roadmap sources |
| 3 | Type-Safety + Quality Gates | DevEx/Quality Lead | M1: mypy baseline by package; M2: strict mode expanded to 4 more packages; M3: strict-package mypy CI-blocking | strict mypy packages >= 6; mypy findings reduced >= 30% in priority modules |
| 4 | Event Bridge / Observability Reliability | Observability Lead | M1: async subscribe path merged; M2: integration tests for loss/ordering; M3: SLO dashboards published | event delivery success >= 99.9%; p95 dispatch latency < 200ms; zero known skipped-subscription paths |
| 5 | Legacy Compatibility Debt Reduction | Verticals Lead | M1: deprecation inventory + policy; M2: remove first 30%; M3: remove 60% + migration notes | deprecated symbol count reduced >= 60%; all remaining deprecations include removal date/version |
| 6 | Competitive Benchmark Ground-Truth | Platform Lead | M1: benchmark rubric frozen; M2: Victor + 2 competitors measured; M3: report + action list published | reproducible suite with >= 20 tasks; benchmark results drive next-quarter priorities |

## First 14 Days (Execution Start)

1. Launch project board and seed epics/tasks from `scripts/planning/bootstrap_90_day_project.sh`.
2. Lock canonical roadmap location and archive/supersede duplicates.
3. Capture baseline metrics (LOC, mypy counts, event bridge reliability, deprecated symbol inventory).
4. Assign named owners for each initiative and confirm weekly operating cadence.
