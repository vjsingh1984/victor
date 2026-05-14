# Contracts Naming and Native Efficiency Plan

This plan captures the migration from the current `victor-sdk` naming to a
semantically accurate contracts surface, while keeping Python as the framework
runtime and Rust as the optional native acceleration layer.

## Target Vocabulary

- `victor-ai`: the Python framework/runtime host.
- `victor-contracts`: the zero/low-dependency protocol, definition, manifest,
  validator, and test-fixture package currently published as `victor-sdk`.
- `victor_contracts`: the preferred import namespace for contract definitions.
- `victor-sdk` / `victor_sdk`: compatibility names for the current contracts
  package during the transition.
- `victor-client`: reserved for a future actual client SDK that talks to a
  Victor API/server/cloud runtime.
- `victor.extension.protocols`: preferred entry-point group for protocol
  providers.
- `victor.extension.capabilities`: preferred entry-point group for capability
  providers.
- `victor.sdk.protocols` and `victor.sdk.capabilities`: legacy entry-point
  groups supported until the next major cleanup.

## Package Rename Todos

- [x] Document the naming target and rollout order.
- [x] Add `victor_contracts` as an import-compatible alias for `victor_sdk`.
- [x] Add framework discovery support for `victor.extension.protocols`.
- [x] Add framework discovery support for `victor.extension.capabilities`.
- [x] Update new vertical scaffolds to import from `victor_contracts`.
- [x] Update contract dependency auditing to treat `victor_contracts` as first-party.
- [ ] Publish a `victor-contracts` distribution that contains the current
  contract package.
- [ ] Make `victor-sdk` a compatibility distribution depending on
  `victor-contracts`.
- [ ] Update sibling package dependencies from `victor-sdk` to
  `victor-contracts`, while keeping old import compatibility.
- [ ] Update sibling package entry points from `victor.sdk.*` to
  `victor.extension.*`.
- [ ] Rename `victor-dataanalysis` to `victor-data-analysis` with package import
  compatibility.
- [ ] Reserve `victor-client` for actual client APIs instead of protocol
  definitions.

## Sibling Package Rollout

Update first-party sibling packages in this order:

1. `victor-rag`
2. `victor-devops`
3. `victor-research`
4. `victor-coding`
5. `victor-invest`
6. `victor-dataanalysis` to `victor-data-analysis`
7. `victor-registry` metadata after packages expose the new names

Each package should:

- Depend on `victor-contracts>=0.8,<1.0` once published.
- Keep compatibility with `victor-sdk` for one minor release if needed.
- Prefer imports from `victor_contracts`.
- Register providers in `victor.extension.protocols` and
  `victor.extension.capabilities`.
- Keep `victor.plugins` as the canonical plugin bootstrap entry point.

## Native Acceleration Strategy

Use Rust as the default native language. Keep C/C++ usage indirect through
mature dependencies such as tree-sitter, SQLite, NumPy/BLAS, Arrow, and
LanceDB. Do not add custom C++ unless a required upstream library forces it.

Native Rust is appropriate for:

- Token counting and context fitting.
- JSON/YAML repair and parsing.
- Tool-call extraction and argument normalization.
- Streaming filters and thinking-token stripping.
- Secret scanning and multi-pattern matching.
- Classification hot paths.
- Batch text chunking.
- Content hashing and deduplication.
- Graph algorithms and trace scanning.
- Code identifier extraction where batching beats Python overhead.

Keep Python authoritative for:

- Provider calls and network I/O.
- Agent service ownership and orchestration policy.
- Plugin discovery and lifecycle.
- CLI/TUI/API surfaces.
- Database migrations and schema ownership.
- High-change framework abstractions.

## Native Integration Todos

- [x] Keep native acceleration optional via `victor_native`.
- [x] Preserve Python fallbacks for every native operation.
- [x] Avoid Rust for operations where benchmarked Python/NumPy is faster.
- [ ] Add benchmark gates before moving any additional Python path to Rust.
- [ ] Release the GIL for long Rust batch operations where not already done.
- [ ] Add feature-level parity tests for each native/fallback pair.
- [ ] Publish platform wheels for the native package only after parity and
  benchmark gates pass.
- [ ] Keep `victor-ai` functional when native wheels are unavailable.

## Phase Gates

- Phase 1 is complete when `victor_contracts` imports and new entry-point groups
  work without breaking legacy `victor_sdk` and `victor.sdk.*` users.
- Phase 2 is complete when all sibling packages publish both old and new
  metadata.
- Phase 3 is complete when docs and examples prefer `victor-contracts`.
- Phase 4 is complete when `victor-sdk` is a compatibility package only.
- Phase 5 is complete when native acceleration has benchmark-backed coverage and
  no required runtime dependency on Rust wheels.
