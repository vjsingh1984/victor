# Vertical Package Naming, Ownership, And Versioning Policy (2026-03-12)

Purpose: define `VPC-T4.2` so extracted first-party vertical packages follow one
predictable naming and release policy instead of drifting between docs, resolver
fallbacks, entry points, and PyPI metadata.

## Scope

- Included: first-party extracted vertical packages for `coding`, `rag`,
  `devops`, `dataanalysis`, and `research`
- Included: PyPI distribution names, import roots, entry-point names, ownership
  rules, and versioning rules
- Included: the special alias policy for `dataanalysis`
- Excluded: third-party package naming outside the reserved Victor namespace
- Excluded: actual `pyproject.toml` and entry-point edits (`VPC-F4.2`)

## Policy Summary

Victor will use one canonical identity per first-party vertical across four
surfaces:

1. vertical ID used by CLI/runtime: lower-case slug such as `coding`
2. PyPI distribution name: `victor-<vertical>`
3. Python import root: `victor_<vertical>`
4. entry-point name in `victor.verticals`: same as the vertical ID

This keeps package installs human-readable while matching Python module naming
rules and the current import resolver behavior.

## Canonical Naming Matrix

| Vertical | Canonical vertical ID | PyPI distribution | Python import root | Entry-point name | Notes |
|---|---|---|---|---|---|
| Coding | `coding` | `victor-coding` | `victor_coding` | `coding` | Matches current docs and optional-fallback language |
| RAG | `rag` | `victor-rag` | `victor_rag` | `rag` | Short name remains canonical |
| DevOps | `devops` | `victor-devops` | `victor_devops` | `devops` | Preserve current vertical slug |
| Data Analysis | `dataanalysis` | `victor-dataanalysis` | `victor_dataanalysis` | `dataanalysis` | `data_analysis` stays an alias only, not a package identity |
| Research | `research` | `victor-research` | `victor_research` | `research` | Straight mapping |

## Alias Rules

Alias handling is intentionally narrow:

- `data_analysis` and `data-analysis` may remain accepted as CLI/import-resolver
  normalization aliases during the transition.
- The canonical package identity remains `dataanalysis` /
  `victor-dataanalysis` / `victor_dataanalysis`.
- No first-party package should be published under `victor-data-analysis` or
  `victor_data_analysis`.

This keeps one authoritative package identity while preserving
backwards-friendly lookup behavior.

## Reserved First-Party Namespace

The following distribution names and import roots are reserved for first-party
Victor packages:

- `victor-ai`
- `victor-sdk`
- `victor-coding` / `victor_coding`
- `victor-rag` / `victor_rag`
- `victor-devops` / `victor_devops`
- `victor-dataanalysis` / `victor_dataanalysis`
- `victor-research` / `victor_research`

Third-party extensions should continue using distinct package names and import
roots, even though they still register through the shared `victor.verticals`
entry-point group.

## Ownership Model

Ownership is package-by-package and role-specific:

| Package family | Authoritative owner | Responsibilities |
|---|---|---|
| `victor-sdk` | Platform architecture maintainers | SDK contracts, capability IDs, tool IDs, definition schemas |
| `victor-ai` | Runtime/platform maintainers | Discovery, runtime adaptation, CLI install UX, compatibility shims |
| First-party vertical packages | Verticals maintainers under the Victor project | Vertical definition modules, runtime add-ons, package metadata, parity tests |

Rules:

1. Once a vertical is extracted, its external package becomes the only
   authoritative implementation source.
2. Bundled contrib copies in `victor-ai` may remain temporarily, but only as
   compatibility shims or migration adapters.
3. Ownership transfer for a first-party vertical package requires an explicit ADR
   or maintainer decision and must be reflected in release docs.

## Versioning Policy

### First-party Victor suite packages

Until the suite reaches a later explicit versioning ADR, first-party extracted
verticals remain lockstep-versioned with `victor-ai` and `victor-sdk`.

If the Victor suite release is `X.Y.Z`, then:

- `victor-ai` uses `X.Y.Z`
- `victor-sdk` uses `X.Y.Z`
- every published first-party extracted vertical for that release train also uses
  `X.Y.Z`

This matches the current exact version lock already used by `victor-ai ->
victor-sdk` and keeps upgrade diagnostics simple during extraction.

### Dependency rules for first-party extracted verticals

At initial extraction time, use exact suite matching:

- authoring/definition dependency: `victor-sdk==X.Y.Z`
- runtime integration dependency or `runtime` extra: `victor-ai==X.Y.Z`

Relaxing that policy later requires a follow-on architecture decision after the
packaging model has stabilized.

### Third-party vertical packages

Third-party verticals are not required to use Victor suite lockstep versions.
They should instead declare a compatibility range against `victor-sdk` and any
optional runtime packages they rely on.

## Entry-Point Policy

For first-party extracted verticals:

- entry-point group stays `victor.verticals`
- entry-point name is always the canonical vertical ID
- the entry point should point at the authoritative external package once that
  vertical flips

Example target shape:

```toml
[project.entry-points."victor.verticals"]
research = "victor_research:ResearchAssistant"
```

The bundled contrib entry point in `victor-ai` should then remain only as a shim
or be removed later under the `VPC-T4.3` retirement policy.

## Why Lockstep Versioning Wins For Now

This policy intentionally favors clarity over package-level independence:

- it matches the current `victor-ai` to `victor-sdk` exact pin
- it avoids ambiguous mixed-version support during the first extraction waves
- it makes release notes, compatibility matrices, and bug reports easier to
  reason about while bundled shims still exist

Independent versioning can be reconsidered later, but not while the project is
still converging on one authoritative implementation path.

## Consequences For Follow-On Work

### Unblocks `VPC-T4.3`

- shim retirement milestones can assume a single suite version line
- `dataanalysis` alias handling is now explicit

### Unblocks `VPC-T4.5` through `VPC-T4.8`

- entry-point flips can target canonical package names and import roots
- extras/install UX can refer to one stable package identity per vertical

## Summary Decision

For first-party extracted verticals:

- use `victor-<vertical>` as the PyPI distribution name
- use `victor_<vertical>` as the Python import root
- use the bare vertical slug as the entry-point name
- keep ownership inside the Victor project
- keep versions lockstep with `victor-ai` and `victor-sdk` until a later ADR
  intentionally relaxes that rule
