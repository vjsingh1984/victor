# Duplicate Roadmap Sources Inventory

**Date**: 2026-03-10
**Epic**: E2 Roadmap Governance Consolidation - M2
**Purpose**: Identify all potential roadmap/planning documents and classify them according to canonical roadmap governance

---

## Canonical Source

**File**: `roadmap.md` (repository root)
**Status**: ✅ Active - Single source of truth
**Last Updated**: 2026-03-10
**Review Cadence**: Weekly (Monday) and at milestone cuts

---

## Document Inventory

### 1. Planning Documents (Canonical Hierarchy)

| File | Status | Notes | Action |
|------|--------|-------|--------|
| `docs/planning/RANKED_90_DAY_EXECUTION_PLAN.md` | ✅ Active | Ranked initiatives with milestones | Keep - Ranked execution plan supporting canonical roadmap |
| `docs/planning/GITHUB_PROJECT_90_DAY_TEMPLATE.md` | ✅ Active | Template for GitHub project setup | Keep - Supporting documentation |
| `docs/planning/` | ✅ Active | Supporting planning templates/docs | Keep - Templates and governance docs |

### 2. Historical Roadmaps (Archived)

| File | Status | Notes | Action |
|------|--------|-------|--------|
| `docs/roadmap/improvement-plan-v1.md` | 📦 Archived | Superseded by 90-day plan | ✅ Already marked as archived in roadmap.md |
| `docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md` | ⚠️ Duplicate | Comprehensive historical roadmap | 🔄 Add header noting archived status |
| `docs/ROLLOUT_PLAN.md` | ⚠️ Duplicate | Rollout plan | 🔄 Add header noting archived status |
| `docs/competitive-analysis-comprehensive.md` | ℹ️ Related | Market analysis, not roadmap | ✅ Not conflicting - different purpose |
| `docs/architecture-analysis-phase3.md` | ℹ️ Related | Architecture deep-dive, not roadmap | ✅ Not conflicting - different purpose |

### 3. Active Foundation Plans (Tracked Separately)

| File | Status | Notes | Action |
|------|--------|-------|--------|
| `docs/roadmap/framework-vertical-foundation-plan.md` | 🔄 Active | Foundation work tracked separately | ✅ Already noted as active in roadmap.md |

### 4. Implementation Status (Completed)

| File | Status | Notes | Action |
|------|--------|-------|--------|
| `docs/P0-P4_IMPLEMENTATION_STATUS.md` | ✅ Complete | Historical P0-P4 work (completed 2025-02-21) | ✅ Already marked as completed in roadmap.md |

### 5. Testing Plans (Specialized Domain)

| File | Status | Notes | Action |
|------|--------|-------|--------|
| `docs/development/testing/test-migration-plan.md` | ✅ Active | Testing-specific migration plan | ✅ Not conflicting - domain-specific |
| `docs/development/testing/all-verticals-migration-plan.md` | ✅ Active | Verticals migration plan | ✅ Not conflicting - domain-specific |
| `MIGRATED_TESTS.md` (repo root) | 📝 Reference | Test migration status | ✅ Not conflicting - reference document |

### 6. Release Reviews

| File | Status | Notes | Action |
|------|--------|-------|--------|
| `docs/roadmap/release-reviews/2026q2/README.md` | ✅ Active | Release review tracking | Keep - Scheduled milestone reviews |
| `docs/roadmap/release-reviews/2026q2/2026-03-03-cycle-01.md` | ✅ Active | Cycle 1 review | Keep - Historical record |
| `docs/roadmap/release-reviews/2026q2/2026-03-03-cycle-02.md` | ✅ Active | Cycle 2 review | Keep - Historical record |

### 7. Strategy/Status Documents

| File | Status | Notes | Action |
|------|--------|-------|--------|
| `docs/development/testing/strategy.md` | ✅ Active | Testing strategy | ✅ Not conflicting - domain-specific |
| `docs/guides/TASK_PLANNER.md` | 📖 Guide | Task planning guide | ✅ Not conflicting - instructional |
| `docs/DEPLOYMENT.md` | 📖 Guide | Deployment guide | ✅ Not conflicting - operational |

### 8. Platform Convergence Plans

| File | Status | Notes | Action |
|------|--------|-------|--------|
| `docs/roadmap/vertical-platform-convergence-plan.md` | 🔄 Active | Platform convergence strategy | ✅ Already noted as active in roadmap.md |

---

## Conflict Analysis

### No Conflicts Detected ✅

All identified documents fall into clear categories:
1. **Canonical roadmap** - Single source of truth
2. **Supporting hierarchy** - RANKED_90_DAY_EXECUTION_PLAN, templates
3. **Archived** - Marked as superseded/historical
4. **Domain-specific** - Testing, deployment, architectural analysis
5. **Specialized tracking** - Release reviews, foundation plans

### Key Insight

**The codebase already has good separation**: Most "roadmap-like" documents serve different purposes:
- Strategic planning (execution plan)
- Historical reference (implementation status)
- Domain-specific work (testing, foundations)
- Operational guidance (deployment, task planning)

**No duplicate/conflicting roadmap sources found** - The `roadmap.md` file is truly the single source of truth for 90-day priorities.

---

## Recommendations

### Immediate Actions (M2)

1. ✅ **Duplicate sources inventory** - Complete (this document)
2. **Add archival headers** to potentially confusing files:
   - `docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md` - Add header noting archived status
   - `docs/ROLLOUT_PLAN.md` - Add header noting archived status

3. **Active work mapping** (next step):
   - Create GitHub labels for each epic
   - Assign owner to each epic
   - Set milestone due dates in issues
   - Define KPIs for each epic

4. **Weekly update cadence**:
   - Establish Monday update template
   - Create reminder system for weekly reviews
   - Track adherence to cadence

### Maintenance

- **Weekly**: Review and update `roadmap.md` on Mondays
- **At milestone cuts**: Update all milestone statuses
- **Quarterly**: Review and archive completed work

---

## Governance Compliance

All documents comply with the canonical hierarchy defined in `roadmap.md`:

1. **This file** (`roadmap.md`) - Active 90-day priorities
2. [`docs/planning/RANKED_90_DAY_EXECUTION_PLAN.md`](docs/planning/RANKED_90_DAY_EXECUTION_PLAN.md) - Detailed execution plan
3. [`docs/planning/`](docs/planning/) - Supporting planning documents

**No conflicting roadmap sources detected.** ✅
