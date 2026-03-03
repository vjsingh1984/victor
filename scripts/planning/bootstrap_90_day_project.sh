#!/usr/bin/env bash
set -euo pipefail

# Bootstrap labels, milestones, and seed issues for the 90-day execution plan.
#
# Usage:
#   bash scripts/planning/bootstrap_90_day_project.sh <owner> <repo>
# Example:
#   bash scripts/planning/bootstrap_90_day_project.sh vjsingh1984 victor

OWNER="${1:-}"
REPO="${2:-}"

if [[ -z "${OWNER}" || -z "${REPO}" ]]; then
  echo "Usage: $0 <owner> <repo>"
  exit 1
fi

command -v gh >/dev/null 2>&1 || { echo "gh CLI is required"; exit 1; }
gh auth status >/dev/null 2>&1 || {
  echo "gh is not authenticated. Run: gh auth login"
  exit 1
}

REPO_SLUG="${OWNER}/${REPO}"

ensure_label() {
  local name="$1"
  local color="$2"
  local description="$3"
  gh label create "${name}" --repo "${REPO_SLUG}" --color "${color}" --description "${description}" --force >/dev/null
  echo "label: ${name}"
}

ensure_milestone() {
  local title="$1"
  local due_on="$2"
  local existing
  existing="$(gh api "repos/${REPO_SLUG}/milestones" --jq ".[] | select(.title==\"${title}\") | .number" || true)"
  if [[ -n "${existing}" ]]; then
    echo "milestone exists: ${title}"
  else
    gh api -X POST "repos/${REPO_SLUG}/milestones" \
      -f "title=${title}" \
      -f "due_on=${due_on}T23:59:59Z" >/dev/null
    echo "milestone created: ${title}"
  fi
}

issue_exists() {
  local title="$1"
  local count
  count="$(gh issue list --repo "${REPO_SLUG}" --search "in:title \"${title}\"" --json number --jq 'length')"
  [[ "${count}" -gt 0 ]]
}

create_issue() {
  local title="$1"
  local body="$2"
  local labels="$3"
  local milestone="$4"

  if issue_exists "${title}"; then
    echo "issue exists: ${title}"
    return 0
  fi

  gh issue create \
    --repo "${REPO_SLUG}" \
    --title "${title}" \
    --body "${body}" \
    --label "${labels}" \
    --milestone "${milestone}" >/dev/null
  echo "issue created: ${title}"
}

echo "== Creating labels =="
ensure_label "priority:p0" "B60205" "Highest priority initiative"
ensure_label "priority:p1" "D93F0B" "High priority initiative"
ensure_label "priority:p2" "FBCA04" "Medium priority initiative"

ensure_label "track:architecture" "0052CC" "Architecture and orchestration track"
ensure_label "track:roadmap" "1D76DB" "Roadmap and program governance track"
ensure_label "track:quality" "5319E7" "Quality and type-safety track"
ensure_label "track:observability" "0E8A16" "Observability and reliability track"
ensure_label "track:verticals" "006B75" "Vertical migration and compatibility track"
ensure_label "track:benchmark" "A2EEEF" "Competitive benchmarking track"

ensure_label "type:epic" "3E4B9E" "Top-level initiative"
ensure_label "type:milestone-task" "5319E7" "Milestone-scoped task"
ensure_label "type:risk" "E11D21" "Risk tracking issue"
ensure_label "status:blocked" "E11D21" "Work is currently blocked"

ensure_label "owner:architecture" "C5DEF5" "Owned by Architecture Lead role"
ensure_label "owner:program" "C2E0C6" "Owned by Product/Program Lead role"
ensure_label "owner:devex" "F9D0C4" "Owned by DevEx/Quality Lead role"
ensure_label "owner:observability" "BFDADC" "Owned by Observability Lead role"
ensure_label "owner:verticals" "D4C5F9" "Owned by Verticals Lead role"
ensure_label "owner:platform" "FEF2C0" "Owned by Platform Lead role"

echo "== Creating milestones =="
ensure_milestone "M1: Foundation Cut" "2026-03-31"
ensure_milestone "M2: Midpoint Cut" "2026-04-28"
ensure_milestone "M3: Quarter Exit" "2026-06-01"

echo "== Creating epics =="
create_issue "[90D][E1] Orchestration Tech-Debt Burn-Down" \
"Owner Role: Architecture Lead

Quarter KPI:
- orchestrator.py <= 3800 LOC
- No coordinator module > 1200 LOC
- >=80% test coverage for new coordinator modules

Milestone Tasks:
- [ ] [90D][E1][M1] Extract ExecutionCoordinator
- [ ] [90D][E1][M2] Split ChatCoordinator (sync/streaming)
- [ ] [90D][E1][M3] Complete protocol-based coordinator injection" \
"type:epic,priority:p0,track:architecture,owner:architecture" \
"M3: Quarter Exit"

create_issue "[90D][E2] Roadmap Governance Consolidation" \
"Owner Role: Product/Program Lead

Quarter KPI:
- One canonical roadmap source
- 100% active work mapped to owner/date/KPI
- Weekly roadmap updates >=90% adherence

Milestone Tasks:
- [ ] [90D][E2][M1] Publish canonical roadmap
- [ ] [90D][E2][M2] Map all active work to owners/dates
- [ ] [90D][E2][M3] Complete 2 release review cycles" \
"type:epic,priority:p0,track:roadmap,owner:program" \
"M3: Quarter Exit"

create_issue "[90D][E3] Type-Safety + Quality Gate Hardening" \
"Owner Role: DevEx/Quality Lead

Quarter KPI:
- Strict mypy packages increased from 2 to >=6
- Strict-package mypy is CI-blocking
- >=30% reduction in mypy findings across victor.framework + victor.agent

Milestone Tasks:
- [ ] [90D][E3][M1] Publish mypy baseline inventory
- [ ] [90D][E3][M2] Expand strict mypy to 4 additional packages
- [ ] [90D][E3][M3] Enforce CI failure on strict-package mypy issues" \
"type:epic,priority:p1,track:quality,owner:devex" \
"M3: Quarter Exit"

create_issue "[90D][E4] Event Bridge / Observability Reliability" \
"Owner Role: Observability Lead

Quarter KPI:
- Event delivery success >=99.9%
- p95 bridge dispatch latency <200ms
- Zero known skipped-subscription paths

Milestone Tasks:
- [ ] [90D][E4][M1] Implement async subscribe path in event bridge
- [ ] [90D][E4][M2] Add integration tests + event loss checks
- [ ] [90D][E4][M3] Publish reliability dashboard and SLO checks" \
"type:epic,priority:p1,track:observability,owner:observability" \
"M3: Quarter Exit"

create_issue "[90D][E5] Legacy Compatibility Debt Reduction" \
"Owner Role: Verticals Lead

Quarter KPI:
- Deprecated symbol count reduced by >=60%
- All remaining deprecations include removal version/date
- Zero undocumented breaking changes

Milestone Tasks:
- [ ] [90D][E5][M1] Publish deprecation inventory + removal policy
- [ ] [90D][E5][M2] Remove 30% deprecated shims/constants
- [ ] [90D][E5][M3] Remove 60% and publish migration notes" \
"type:epic,priority:p1,track:verticals,owner:verticals" \
"M3: Quarter Exit"

create_issue "[90D][E6] Competitive Benchmark Ground-Truth" \
"Owner Role: Platform Lead

Quarter KPI:
- Reproducible benchmark suite with >=20 tasks
- Victor baseline + at least 2 competitors measured
- Published findings drive next roadmap priorities

Milestone Tasks:
- [ ] [90D][E6][M1] Finalize benchmark spec and scoring
- [ ] [90D][E6][M2] Run baselines for Victor + 2 competitors
- [ ] [90D][E6][M3] Publish benchmark report and prioritized action list" \
"type:epic,priority:p2,track:benchmark,owner:platform" \
"M3: Quarter Exit"

echo "== Creating milestone-scoped tasks =="

# E1
create_issue "[90D][E1][M1] Extract ExecutionCoordinator" \
"Definition of done:
- Extract _handle_tool_calls path into ExecutionCoordinator
- Add unit tests for execution coordinator
- No functional regressions in unit test suite" \
"type:milestone-task,priority:p0,track:architecture,owner:architecture" \
"M1: Foundation Cut"
create_issue "[90D][E1][M2] Split ChatCoordinator (sync/streaming)" \
"Definition of done:
- Separate sync and streaming coordinator responsibilities
- Shared abstractions reduced to clear interfaces
- Streaming behavior benchmarked before/after" \
"type:milestone-task,priority:p0,track:architecture,owner:architecture" \
"M2: Midpoint Cut"
create_issue "[90D][E1][M3] Complete protocol-based coordinator injection" \
"Definition of done:
- Coordinator constructors use protocol interfaces instead of concrete orchestrator
- Dependency inversion checks documented
- All coordinator tests green" \
"type:milestone-task,priority:p0,track:architecture,owner:architecture" \
"M3: Quarter Exit"

# E2
create_issue "[90D][E2][M1] Publish canonical roadmap" \
"Definition of done:
- Single roadmap file designated as source of truth
- Other roadmap docs linked or archived with status notes
- Owner and update cadence documented" \
"type:milestone-task,priority:p0,track:roadmap,owner:program" \
"M1: Foundation Cut"
create_issue "[90D][E2][M2] Map all active work to owners/dates/KPIs" \
"Definition of done:
- Every active initiative includes owner, date, KPI
- Project board updated with priority and confidence fields" \
"type:milestone-task,priority:p0,track:roadmap,owner:program" \
"M2: Midpoint Cut"
create_issue "[90D][E2][M3] Complete two release review cycles" \
"Definition of done:
- Two release reviews completed with action logs
- Decision notes published for carry-over and cut items" \
"type:milestone-task,priority:p0,track:roadmap,owner:program" \
"M3: Quarter Exit"

# E3
create_issue "[90D][E3][M1] Publish mypy baseline inventory" \
"Definition of done:
- Package-level mypy error baseline captured
- Hotspots prioritized with remediation owners" \
"type:milestone-task,priority:p1,track:quality,owner:devex" \
"M1: Foundation Cut"
create_issue "[90D][E3][M2] Expand strict mypy to 4 additional packages" \
"Definition of done:
- Strict mode enabled for >=4 additional packages
- New strict checks pass in CI" \
"type:milestone-task,priority:p1,track:quality,owner:devex" \
"M2: Midpoint Cut"
create_issue "[90D][E3][M3] Enforce CI fail on strict-package mypy issues" \
"Definition of done:
- CI blocks merges when strict-package mypy fails
- Developer docs updated with local validation commands" \
"type:milestone-task,priority:p1,track:quality,owner:devex" \
"M3: Quarter Exit"

# E4
create_issue "[90D][E4][M1] Implement async subscribe in event bridge" \
"Definition of done:
- Async subscribe path implemented and wired
- No skipped subscription fallback remains in primary path" \
"type:milestone-task,priority:p1,track:observability,owner:observability" \
"M1: Foundation Cut"
create_issue "[90D][E4][M2] Add integration tests and event loss checks" \
"Definition of done:
- Event bridge integration test coverage added
- Event loss and ordering checks automated" \
"type:milestone-task,priority:p1,track:observability,owner:observability" \
"M2: Midpoint Cut"
create_issue "[90D][E4][M3] Publish reliability dashboard and SLOs" \
"Definition of done:
- Dashboard includes delivery success and p95 dispatch latency
- SLO alert thresholds documented and enabled" \
"type:milestone-task,priority:p1,track:observability,owner:observability" \
"M3: Quarter Exit"

# E5
create_issue "[90D][E5][M1] Publish deprecation inventory and removal policy" \
"Definition of done:
- Deprecated APIs/symbols inventoried with owner and target removal version
- Public migration policy documented" \
"type:milestone-task,priority:p1,track:verticals,owner:verticals" \
"M1: Foundation Cut"
create_issue "[90D][E5][M2] Remove 30% deprecated shims/constants" \
"Definition of done:
- 30% of deprecated compatibility artifacts removed
- Release notes include migration guidance" \
"type:milestone-task,priority:p1,track:verticals,owner:verticals" \
"M2: Midpoint Cut"
create_issue "[90D][E5][M3] Remove 60% and publish migration notes" \
"Definition of done:
- Total >=60% deprecated artifacts removed
- Migration notes validated by examples/tests" \
"type:milestone-task,priority:p1,track:verticals,owner:verticals" \
"M3: Quarter Exit"

# E6
create_issue "[90D][E6][M1] Finalize benchmark spec and scoring rubric" \
"Definition of done:
- Task set and scoring rubric approved
- Reproducibility checklist published" \
"type:milestone-task,priority:p2,track:benchmark,owner:platform" \
"M1: Foundation Cut"
create_issue "[90D][E6][M2] Run Victor baseline + 2 competitor baselines" \
"Definition of done:
- At least 20 tasks executed
- Results archived with reproducible run metadata" \
"type:milestone-task,priority:p2,track:benchmark,owner:platform" \
"M2: Midpoint Cut"
create_issue "[90D][E6][M3] Publish benchmark report + prioritized bets" \
"Definition of done:
- Comparative report published
- Top-5 roadmap bets linked to benchmark findings" \
"type:milestone-task,priority:p2,track:benchmark,owner:platform" \
"M3: Quarter Exit"

echo "Done. Next steps:"
echo "1) Create Project v2 board: Victor 90-Day Execution (2026Q2)"
echo "2) Add project fields from docs/planning/GITHUB_PROJECT_90_DAY_TEMPLATE.md"
echo "3) Add created issues to the project and group by Status/Milestone"
