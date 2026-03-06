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

## Current Removal Targets

- `v0.7.0` target date: `2026-06-30`
- `v0.8.0` target date: `2026-12-31`

## Source Of Truth

- Inventory: `docs/development/deprecation-inventory-2026-03-03.md`
- Tracker epic: `[90D][E5] Legacy Compatibility Debt Reduction` (#34)
