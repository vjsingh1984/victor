# GitHub Project Template: 90-Day Execution Plan

Use this template to operationalize the Q2 2026 plan (March 3, 2026 to June 1, 2026).

## 1) Project Setup (GitHub Projects v2)

- Project name: `Victor 90-Day Execution (2026Q2)`
- Views:
  - `Roadmap` (table)
  - `Execution Board` (board by `Status`)
  - `Milestones` (group by `Milestone`)
  - `Risks` (filter `status:blocked` or `risk:high`)

### Recommended Fields

- `Status` (single select): `Backlog`, `Ready`, `In Progress`, `Blocked`, `In Review`, `Done`
- `Priority` (single select): `P0`, `P1`, `P2`
- `Owner Role` (single select): `Platform Lead`, `Architecture Lead`, `DevEx/Quality Lead`, `Observability Lead`, `Verticals Lead`, `Product/Program Lead`
- `Track` (single select): `Architecture`, `Roadmap`, `Quality`, `Observability`, `Verticals`, `Benchmark`
- `Milestone Date` (date)
- `KPI` (text)
- `Confidence` (single select): `High`, `Medium`, `Low`

## 2) Milestones (Dates)

- `M1: Foundation Cut` due `2026-03-31`
- `M2: Midpoint Cut` due `2026-04-28`
- `M3: Quarter Exit` due `2026-06-01`

## 3) Label Taxonomy

- Priority: `priority:p0`, `priority:p1`, `priority:p2`
- Tracks: `track:architecture`, `track:roadmap`, `track:quality`, `track:observability`, `track:verticals`, `track:benchmark`
- Type: `type:epic`, `type:milestone-task`, `type:risk`
- Owner-role tags: `owner:architecture`, `owner:platform`, `owner:devex`, `owner:observability`, `owner:verticals`, `owner:program`
- State signal: `status:blocked`

## 4) Ranked Initiatives (Epics)

1. `E1` Orchestration Tech-Debt Burn-Down (`Architecture Lead`, `track:architecture`, `priority:p0`)
2. `E2` Roadmap Governance Consolidation (`Product/Program Lead`, `track:roadmap`, `priority:p0`)
3. `E3` Type-Safety + Quality Gate Hardening (`DevEx/Quality Lead`, `track:quality`, `priority:p1`)
4. `E4` Event Bridge / Observability Reliability (`Observability Lead`, `track:observability`, `priority:p1`)
5. `E5` Legacy Compatibility Debt Reduction (`Verticals Lead`, `track:verticals`, `priority:p1`)
6. `E6` Competitive Benchmark Ground-Truth (`Platform Lead`, `track:benchmark`, `priority:p2`)

## 5) Seed Issue Naming Convention

- Epic: `[90D][E#] <initiative>`
- Milestone task: `[90D][E#][M1|M2|M3] <deliverable>`

## 6) KPI Targets (Quarter Exit)

- `E1`: `orchestrator.py <= 3800 LOC`, no coordinator > `1200 LOC`
- `E2`: single canonical roadmap, 100% active work mapped with owner/date/KPI
- `E3`: strict mypy packages >= `6`, and strict-package mypy is CI-blocking
- `E4`: event delivery success >= `99.9%`, p95 bridge dispatch < `200ms`
- `E5`: deprecated symbol count reduced by >= `60%`
- `E6`: reproducible benchmark suite with >= `20` tasks and published results

## 7) Bootstrap Automation

Run:

```bash
bash scripts/planning/bootstrap_90_day_project.sh <owner> <repo>
```

Example:

```bash
bash scripts/planning/bootstrap_90_day_project.sh vjsingh1984 victor
```

This script creates labels, milestones, and seed epic+task issues aligned to this template.
