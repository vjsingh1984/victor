# MyPy Strict CI Gate (M3, 2026-03-04)

Related tracker item: [90D][E3][M3] Enforce CI fail on strict-package mypy issues (#44)

## Summary

Added a blocking strict-package mypy gate in fast CI and documented the matching
local validation command for contributors.

## CI Change

File updated:
- `.github/workflows/ci-fast.yml`

New blocking job:
- `strict-typecheck` (`Strict Type Gate (MyPy)`)

Command enforced in CI:

```bash
mypy --strict \
  victor/config \
  victor/storage/cache \
  victor/telemetry \
  victor/analytics \
  victor/profiler \
  victor/debug
```

The existing whole-repo mypy job remains advisory (`continue-on-error: true`) to
avoid blocking merges on non-strict-package backlog outside the M3 scope.

## Developer Docs Updated

- `docs/development/README.md`
- `docs/development/index.md`
- `docs/development/code-style.md`

These docs now include the exact local command that mirrors the blocking CI gate.
