# Vertical Extraction Order And Dependency Graph (2026-03-12)

Purpose: define `VPC-T4.1` using the measured post-`VPC-E3` migration state so
packaging work can proceed in a deliberate order instead of extracting whichever
vertical happens to be "ready enough" first.

## Scope

- Included: bundled contrib verticals currently published from `victor-ai`
- Included: release-order dependencies, packaging-risk dependencies, and wave
  sequencing for extraction
- Included: the current import-resolver and entry-point baseline that the
  extraction plan must converge from
- Excluded: final package naming/versioning policy (`VPC-T4.2`)
- Excluded: bundled-shim retirement milestones (`VPC-T4.3`)
- Excluded: release checklist updates (`VPC-T4.4`)

## Inputs Used

- `docs/architecture/adr/007-vertical-distribution-and-sdk-boundary.md`
- `pyproject.toml`
- `victor/core/verticals/import_resolver.py`
- `docs/development/vertical-module-layer-classification-2026-03-10.md`
- `docs/development/vertical-package-layout-target-2026-03-10.md`
- per-vertical inventories for `coding`, `rag`, `devops`, `dataanalysis`, and
  `research`
- the external vertical clean-install/discovery regression already added in
  `tests/integration/verticals/test_external_vertical_install_discovery.py`

## Current Packaging Baseline

Three current facts drive the extraction order:

1. `victor-ai` still publishes bundled contrib vertical entry points for
   `coding`, `rag`, `devops`, `dataanalysis`, and `research`.
2. The import resolver still prefers external package namespaces first
   (`victor_<vertical>` / `victor_dataanalysis`) before falling back to bundled
   contrib packages.
3. All five bundled verticals now have SDK-clean definition layers, but their
   runtime blast radius is not equal.

Important consequence:

- There is no hard runtime dependency from one vertical onto another vertical.
- The dependency graph for extraction is therefore a release/operational graph,
  not a Python import graph between vertical packages.

## Shared Preconditions

No vertical should be flipped to an authoritative external package until all of
these are true:

1. ADR-007 is accepted for breaking packaging changes.
2. Package naming, ownership, and versioning rules are defined (`VPC-T4.2`).
3. Bundled-shim retirement milestones are defined (`VPC-T4.3`).
4. Release checklists are updated for the extracted-vertical model (`VPC-T4.4`).
5. The vertical has clean-environment install/discovery coverage and migration
   parity coverage.

Planning, documentation, smoke tests, and shim-first preparation can continue
before ADR acceptance, but entry-point flips and bundled-package removals should
not.

## Measured Packaging-Risk Baseline

| Vertical | Runtime/core/tool import count | Runtime-layer import count | Packaging-specific risk |
|---|---:|---:|---|
| `research` | 11 | 9 | Lowest measured coupling; no special resolver aliasing |
| `devops` | 12 | 10 | Operational shell/git/runtime surface, but limited package-shape complexity |
| `dataanalysis` | 12 | 10 | Special resolver/package alias handling for `dataanalysis` |
| `rag` | 15 | 13 | Broader document/vector/retrieval runtime surface and optional dependency story |
| `coding` | 20 | 17 | Highest coupling plus the widest repo-wide external-package references and fallbacks |

Source notes:

- Counts are taken from the per-vertical inventory documents after the `VPC-E3`
  migration tranche.
- `coding` was the first definition-boundary proving ground, but that does not
  make it the safest first packaging flip.

## Dependency Graph

### Hard prerequisite graph

All vertical extractions depend on a shared packaging baseline:

```text
Shared packaging baseline
  (ADR-007 accepted, T4.2, T4.3, T4.4 complete)
    -> research
    -> devops
    -> dataanalysis
    -> rag
    -> coding
```

### Recommended operational sequence

The extraction order should still be staged so each wave proves a different part
of the packaging story before the next wave increases blast radius:

```text
research -> devops -> dataanalysis -> rag -> coding
```

This is a sequencing recommendation, not a runtime dependency between those
verticals.

## Per-Vertical Extraction Dependencies

| Vertical | Extraction readiness dependencies | Why it sits where it does |
|---|---|---|
| `research` | Shared packaging baseline only | Smallest measured runtime surface and no special resolver naming case make it the safest pilot |
| `devops` | Shared baseline plus one successful low-blast pilot extraction | Reuses the same runtime split patterns as `research`, but introduces more operational capability/safety surface |
| `dataanalysis` | Shared baseline plus final alias/package-name policy for `dataanalysis` vs. `data_analysis` | Similar coupling to `devops`, but the resolver override makes packaging policy correctness more important |
| `rag` | Shared baseline plus a proven extras/dependency pattern for data/document-style verticals | Broader retrieval/vector/document runtime surface makes it a better later-wave extraction |
| `coding` | Shared baseline plus lessons from all prior waves and repo-wide fallback cleanup | Highest coupling and the largest number of repo-wide references to `victor-coding` make it the last safe flip |

## Recommended Extraction Waves

### Wave 0: Shared packaging decisions only

Do not change authoritative package ownership yet. Finish:

- `VPC-T4.2`
- `VPC-T4.3`
- `VPC-T4.4`

### Wave 1: Low-blast pilot

1. `research`
2. `devops`

Why:

- Both have definition-layer parity in place.
- Both avoid the special `dataanalysis` alias problem.
- Both are substantially lower risk than `rag` and `coding`.

### Wave 2: Naming-sensitive analytics package

1. `dataanalysis`

Why:

- It is not especially runtime-heavy, but it does have the resolver override:
  `dataanalysis -> victor_dataanalysis`.
- That makes it the best vertical to validate the final naming/version policy
  after at least one simpler extraction has succeeded.

### Wave 3: Data/document platform package

1. `rag`

Why:

- `rag` has a broader runtime surface and more optional dependency pressure than
  `research`, `devops`, or `dataanalysis`.
- It benefits from having the packaging/extras model validated on earlier waves.

### Wave 4: Highest-blast-radius extraction

1. `coding`

Why:

- `coding` still has the highest measured runtime-boundary count.
- `coding` has the widest repo-level external-package assumptions across docs,
  fallbacks, capability loaders, code intelligence paths, and tests.
- It should be the last package moved to authoritative external ownership.

## Release Rule For Bundled Shims

For every wave:

1. Move authoritative entry points and packaging metadata to the external
   package.
2. Keep bundled contrib modules as compatibility shims only.
3. Do not remove bundled shims until the retirement milestone for that wave is
   defined in `VPC-T4.3`.

This keeps `VPC-T4.1` operationally useful without pre-deciding removal timing.

## Consequences For Follow-On Tasks

### Unblocks `VPC-T4.2`

- Package naming/version policy can now be written against a concrete wave order.
- `dataanalysis` is explicitly called out as the naming-policy special case.

### Unblocks `VPC-T4.3`

- Shim retirement can now be defined by extraction wave instead of by ad hoc
  package choice.

### Unblocks `VPC-T4.4`

- Release checklists can distinguish pilot-wave requirements from
  high-blast-radius releases such as `coding`.

## Summary Decision

Recommended extraction order:

1. `research`
2. `devops`
3. `dataanalysis`
4. `rag`
5. `coding`

Rationale:

- extract the lowest-blast packages first
- validate naming policy before `dataanalysis`
- validate richer extras/runtime packaging before `rag`
- leave the broadest repo-level dependency surface (`coding`) for last
