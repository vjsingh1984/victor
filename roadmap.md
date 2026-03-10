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
| `E1` | Orchestration tech-debt burn-down | `M1` completed, `M2` in planning |
| `E2` | Roadmap governance consolidation | `M1` in progress |
| `E3` | Type-safety + quality gates | `M1` in progress |
| `E4` | Event bridge reliability | `M3` completed |
| `E5` | Legacy compatibility debt reduction | `M2` completed, `M3` in planning |
| `E6` | Competitive benchmark ground-truth | `M1` in progress |

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

### E2: Roadmap Governance Consolidation
**Status**: M1 in progress
**Owner**: Product/Program Lead (assigned 2026-03-10)
**Progress**:
- ✅ Canonical roadmap established (roadmap.md)
- 🔄 Duplicate sources inventory in progress
- ⏳ Active work mapping to owner/date/KPI (target: 100%)
- ⏳ Weekly update cadence setup (target: >= 90% adherence)

### E3: Type-Safety + Quality Gates
**Status**: M1 complete, M2 in progress
**Owner**: DevEx/Quality Lead (assigned 2026-03-10)
**Progress**:
- ✅ M1: MyPy baseline captured (11 strict modules, 0 findings)
- ✅ M1: Strict mode expanded to 11 modules (exceeds target of >= 6)
- ✅ M1: Strict-package mypy made CI-blocking
- 🔄 M2: Expand to 15+ modules (4+ additional packages)
- ⏳ M2: Reduce mypy findings by 30% in priority modules
- 📄 Baseline report: [`docs/quality/mypy_baseline_report.md`](docs/quality/mypy_baseline_report.md)

**New Strict Modules (added 2026-03-10)**:
- `victor.agent.services.*` (22 files)
- `victor.core.container` (1 file)
- `victor.core/protocols` (1 file)
- `victor.providers.base` (1 file)
- `victor.framework.*` (190+ files)

## How to Influence the Roadmap

- Open an issue/discussion with a concrete use case and measurable impact.
- Submit a [FEP](feps/fep-0000-template.md) for framework-level changes.
