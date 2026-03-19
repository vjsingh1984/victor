# Deprecation And Migration Policy

This policy defines how Victor deprecates public APIs and compatibility shims.

## Policy Rules

1. Every deprecation must include:
- replacement API/path
- owner role
- target removal version
- target removal date

2. Minimum support window:
- at least 2 minor releases, or
- at least 90 calendar days,
whichever is longer.

3. Warning behavior:
- runtime use should emit `DeprecationWarning` where feasible
- docs must include migration guidance for affected APIs

4. Removal gates:
- no removal without an inventory entry
- no removal without migration notes in release documentation
- no release may remove a deprecated API or shim unless the release notes call out
  the replacement path, target removal version/date, and migration guidance

## Release Note Requirements

For every release that introduces, carries forward, or removes a deprecation:

- update `CHANGELOG.md` or the release notes with:
  - deprecated API/shim name
  - replacement API/path
  - target removal version and target removal date
  - migration guide or migration snippet
- if the release keeps a temporary compatibility shim, note that the shim remains
  supported only through its published removal milestone
- if the release removes a deprecated surface, include the removal in the
  `Breaking Changes` section and link the migration guidance

For `VerticalBase.create_agent()` and legacy config-only vertical activation shims:

- deprecated in `Unreleased` on `2026-03-10`
- earliest removal release remains `v0.8.0`
- target removal date remains `2026-12-31`
- every release before removal must restate the migration path to
  `Agent.create(vertical=MyVertical, ...)`

## Current Removal Targets

- `v0.7.0` target date: `2026-06-30`
- `v0.8.0` target date: `2026-12-31`

## Source Of Truth

- Inventory: `docs/development/deprecation-inventory-2026-03-03.md`
- Tracker epic: `[90D][E5] Legacy Compatibility Debt Reduction` (#34)
