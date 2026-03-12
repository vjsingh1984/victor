# Startup and Discovery Baseline (2026-03-12)

Context:

- Roadmap task: `VPC-T5.14`
- Goal: capture a reusable local baseline for startup import cost and vertical
  discovery cost before any discovery-path optimization work.
- Scope of this snapshot: provider-independent only. `Agent.create()` was
  intentionally skipped so the measurement is not coupled to local model or
  network availability.

Command:

```bash
../.venv/bin/python scripts/benchmark_startup_kpi.py \
  --json \
  --skip-agent-create \
  --discovery-probe \
  --iterations 5
```

Environment:

- Date: `2026-03-12`
- Interpreter: `/Users/vijaysingh/code/.venv/bin/python`
- Working tree: local development checkout

Measured results:

| Metric | Value |
|---|---:|
| `import_victor.cold_ms` | `1800.67` |
| `import_victor.warm_mean_ms` | `0.00097` |
| `import_victor.warm_p95_ms` | `0.00258` |
| `discovery_probe.cold_ms` | `127.94` |
| `discovery_probe.warm_mean_ms` | `0.00103` |
| `discovery_probe.warm_p95_ms` | `0.00279` |
| `discovery_probe.discovered_count` | `7` |
| `discovery_probe.cache_hit_total` | `5` |
| `discovery_probe.call_total` | `6` |
| `discovery_probe.scan_total` | `1` |
| `discovery_probe.entry_point_groups_cached` | `1` |
| `discovery_probe.entry_point_vertical_entries` | `7` |

Discovered vertical entry points:

- `benchmark`
- `coding`
- `dataanalysis`
- `devops`
- `investment`
- `rag`
- `research`

Interpretation:

- The forced discovery scan is currently about `128 ms` in this environment.
- Warm discovery is effectively free after the loader cache is populated.
- The entry-point cache currently resolves `7` vertical packages in this checkout.
- Future discovery/resolver changes should rerun the same command and append a
  comparison block instead of replacing this baseline.
