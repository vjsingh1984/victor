# Mypy Baseline Inventory (2026-03-03)

Owner: DevEx/Quality Lead  
Related tracker item: [90D][E3][M1] Publish mypy baseline inventory (#42)

## Method

Baseline was captured package-by-package with a temporary permissive mypy config
(`ignore_missing_imports = true`, `follow_imports = silent`) to measure current
error volume, not strict-gate compliance.

Command pattern:

```bash
mypy --config-file /tmp/mypy_baseline.ini victor/<package>
```

## Package Baseline

| Package | Errors | Files |
|---|---:|---:|
| `verticals` | 745 | 147 |
| `ui` | 657 | 126 |
| `workflows` | 636 | 144 |
| `agent` | 568 | 92 |
| `tools` | 552 | 124 |
| `framework` | 422 | 103 |
| `native` | 373 | 95 |
| `providers` | 367 | 98 |
| `observability` | 340 | 95 |
| `storage` | 329 | 91 |
| `teams` | 304 | 89 |
| `core` | 237 | 61 |
| `protocols` | 232 | 78 |
| `state` | 218 | 73 |
| `deps` | 216 | 54 |
| `evaluation` | 164 | 51 |
| `coordination` | 148 | 32 |
| `context` | 139 | 29 |
| `config` | 112 | 29 |

Totals: **6,759 errors** across **1,611 files** in **19 packages**.

## Prioritized Hotspots And Owners

1. `verticals` + `ui`: Verticals Lead and Product/Program Lead (highest error volume user-facing paths).
2. `workflows` + `agent`: Architecture Lead (orchestration and execution-path reliability).
3. `tools` + `framework`: Architecture Lead with DevEx/Quality Lead support (core abstractions).
4. `providers` + `storage` + `native`: Platform Lead (integration/type boundary cleanup).
5. `observability`: Observability Lead (typed telemetry/event surfaces).

## M2 Prep Targets (Draft)

- Reduce combined errors in top 5 packages by **>=15%** before M2 cut.
- Promote at least **2 additional packages** to strict-friendly readiness (low-Any boundaries, typed returns).
- Track weekly deltas in this inventory until M2.
