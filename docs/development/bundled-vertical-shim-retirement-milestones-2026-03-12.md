# Bundled Vertical Shim Retirement Milestones (2026-03-12)

Purpose: define `VPC-T4.3` so bundled contrib copies stop being treated as
parallel authorities and instead follow one consistent retirement policy after a
vertical is extracted.

## Scope

- Included: first-party bundled contrib shims in `victor-ai`
- Included: support-window milestones, release gates, and removal criteria
- Included: wave-relative retirement rules for the extraction order defined in
  `vertical-extraction-order-and-dependency-graph-2026-03-12.md`
- Excluded: actual shim deletions (`VPC-T4.9` through `VPC-T4.12`)

## Baseline Policy

Bundled contrib copies must never remain ambiguous once a vertical flips to an
authoritative external package.

From the extraction release onward:

1. the external package is authoritative
2. the bundled contrib copy is a compatibility shim only
3. the shim inherits the deprecation policy support window:
   - at least 2 minor releases, and
   - at least 90 calendar days

Both conditions must be met before removal.

## Required Milestones Per Vertical

Each extracted vertical moves through the same four milestones:

| Milestone | Release timing | Bundled contrib state | Required user-facing signal |
|---|---|---|---|
| M0 | Before extraction | Bundled copy may still be authoritative | None beyond existing docs |
| M1 | Extraction release `N` | Bundled copy becomes a forwarding shim or migration adapter | Release notes declare the external package authoritative |
| M2 | Carry-forward release `N+1` | Shim remains supported but no longer authoritative | Release notes restate shim status and removal target |
| M3 | Earliest removal candidate `N+2` and `>=90` days after `N` | Shim may be removed if all gates pass | Breaking-change notes plus migration guidance |

## Removal Gates

No bundled shim may be removed until all of these are true:

1. The authoritative external package has been published successfully.
2. Clean-environment install/discovery smoke tests pass for the external package.
3. `victor-ai` docs and CLI output no longer describe the bundled copy as
   authoritative.
4. Entry points and install guidance point users at the external package.
5. Upgrade and downgrade compatibility notes exist for the affected vertical.
6. Tests no longer rely on bundled business logic as the primary implementation.
7. The deprecation inventory and release notes include:
   - the replacement package/path
   - target removal version
   - target removal date
   - migration guidance

## Wave-Relative Retirement Plan

The exact release numbers stay deferred until extraction releases are scheduled,
but the relative retirement rule is fixed now:

| Extraction wave | Verticals | Earliest shim-removal milestone |
|---|---|---|
| Wave 1 | `research`, `devops` | `N+2` minor releases and at least 90 days after their extraction release |
| Wave 2 | `dataanalysis` | `N+2` minor releases and at least 90 days after its extraction release |
| Wave 3 | `rag` | `N+2` minor releases and at least 90 days after its extraction release |
| Wave 4 | `coding` | `N+2` minor releases and at least 90 days after its extraction release |

This keeps the policy stable even if the specific release numbers move.

## What Counts As A Shim

Allowed shim behavior:

- forwarding imports to the external authoritative package
- compatibility wrappers that preserve an old import path
- temporary migration adapters that emit deprecation warnings

Disallowed behavior after extraction:

- independent bundled business logic that can drift from the external package
- bundled entry points being described as the canonical implementation source
- feature additions landing in the bundled copy without landing in the
  authoritative external package

## Required Signals At Each Milestone

### Extraction release `N`

- changelog/release notes identify the authoritative package
- bundled shim status is explicit
- install command examples use the external package
- compatibility tests verify external install/discovery

### Carry-forward release `N+1`

- release notes restate the earliest removal milestone
- deprecation inventory stays current
- parity/upgrade coverage remains green

### Earliest removal release `N+2`

- removal only proceeds if the 90-day floor has also passed
- breaking changes section includes migration guidance
- stale shim imports/tests are removed in the same release train

## Consequences For Follow-On Tasks

### Unblocks `VPC-T4.4`

- release checklists can now ask whether a release is `M1`, `M2`, or `M3` for
  any extracted vertical

### Unblocks `VPC-T4.9` through `VPC-T4.12`

- code deletions now have explicit timing and gating rules

## Summary Decision

Bundled contrib copies may remain only as temporary compatibility shims after an
extraction flip. The earliest removal point for any extracted vertical is:

- 2 minor releases after extraction, and
- 90 days after extraction,

with both conditions required and release-note/deprecation gates satisfied.
