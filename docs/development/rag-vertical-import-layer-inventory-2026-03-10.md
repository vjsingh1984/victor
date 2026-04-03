# RAG Vertical Import Layer Inventory (2026-03-10)

Purpose: complete `VPC-T3.10` by recording the current import-boundary state of
`victor/verticals/contrib/rag` so the next migration tasks can target concrete
definition/runtime blockers instead of re-scanning the package each session.

## Scope

- Included: Python modules under `victor/verticals/contrib/rag`
- Included: imports from `victor.framework`, `victor.core`, `victor.tools`, and
  `victor_sdk`
- Excluded: YAML assets, notebooks, and non-RAG vertical packages

## Method

- Runtime-boundary scan:
  - `rg -l "^(from|import) victor(\\.framework|\\.core|\\.tools)" victor/verticals/contrib/rag -g '*.py' | sort`
- SDK-adoption scan:
  - `rg -l "victor_sdk" victor/verticals/contrib/rag -g '*.py' | sort`
- Focused blocker inspection:
  - `assistant.py`
  - `prompts.py`
  - `__init__.py`
  - `tool_dependencies.py`

## Measured Status

| Metric | Count | Notes |
|---|---:|---|
| Python files in `rag` package | 39 | Includes the growing `runtime/` package plus tools, UI, runtime helpers, demos, and prompt metadata |
| Python files with `victor.framework` / `victor.core` / `victor.tools` imports | 15 | This is the active runtime-boundary surface for `rag` |
| Python files with `victor_sdk` imports | 1 | Only `assistant.py` currently imports the SDK |
| Definition-layer targets still importing runtime/core | 0 / 2 | `assistant.py` and `prompt_metadata.py` are now definition-layer clean |
| Shim candidates still depending on runtime-heavy package or core loaders | 2 / 2 | `__init__.py`, `tool_dependencies.py` |
| Runtime-layer files with direct runtime/core/tool imports | 13 | Includes `prompts.py`, which now acts as a runtime adapter |

Key findings:

- `rag` is earlier in the migration than `coding`. SDK adoption is still
  limited to `assistant.py`, but the definition layer is now clean: that file
  uses the SDK base plus SDK-owned tool, stage, and tier contracts.
- `assistant.py` no longer imports runtime/core modules. Runtime compatibility
  for package consumers is now preserved at the package boundary via a
  host-owned shim exported from `__init__.py`.
- prompt/task-hint data now lives in `prompt_metadata.py`, while `prompts.py`
  has become a thin runtime adapter over that shared metadata.
- `rag` now declares SDK capability requirements for file operations, document
  ingestion, retrieval, vector indexing, and optional web access.
- `runtime/mode_config.py` and `runtime/safety_enhanced.py` now exist as the
  first runtime-owned modules moved out of the package root, with root shims
  preserving import compatibility.
- `runtime/capabilities.py` now owns the capability provider/config runtime
  logic, while the package-root `capabilities.py` is reduced to a compatibility
  shim. This validates the shared mixed-mode capability resolution path against
  a real vertical.
- `runtime/safety.py` now owns the primary RAG safety extension and framework
  safety-rule factories, while the package-root `safety.py` is reduced to a
  compatibility shim. That keeps the external `victor.rag.safety` import path
  stable while exercising the shared safety-extension loader against a
  runtime-owned module.
- `runtime/enrichment.py` now owns the query-enrichment strategy and singleton
  helpers, while the package-root `enrichment.py` is reduced to a compatibility
  shim for direct consumers such as the query tool.
- `runtime/workflows.py`, `runtime/teams.py`, and `runtime/rl.py` now own the
  workflow, team, and RL implementations. Their root packages are reduced to
  compatibility shims, and the shared runtime loader resolves those runtime
  modules through the mixed-mode candidate path.
- `rag` now has explicit migration parity coverage for discovery, bootstrap
  activation, and runtime-wrapper behavior in
  `tests/integration/verticals/test_rag_migration_parity.py`.
- The root package and tool-dependency module still act as transitional shims,
  with `__init__.py` now explicitly acting as the runtime export shim.

## Definition-Layer Status

`rag` no longer has a definition-layer import blocker:

- `assistant.py` is now SDK-only and relies on host-owned runtime wrapping.
- `prompt_metadata.py` remains data-only.

Remaining definition-layer work is semantic rather than import-boundary work:

- keep future definition changes out of the package-root runtime shim
- preserve parity while runtime helpers move behind narrower shims

## Shim-Layer Blockers

| File | Shim role | Current boundary imports | Why it stays a shim for now | Target end state |
|---|---|---|---|---|
| `__init__.py` | Package-root export shim | `victor.framework.vertical_runtime_adapter` plus runtime-heavy re-exports | Exports a runtime-compatible `RAGAssistant` shim for backward compatibility while still aggregating document store, tools, prompt contributor, mode config, capabilities, and enhanced runtime helpers | Narrow re-export surface centered on runtime `RAGAssistant` plus documented compatibility aliases |
| `tool_dependencies.py` | Root compatibility shim | `victor.core.tool_dependency_loader`, `victor.core.tool_types`, `victor.tools.tool_graph` | Provides the current YAML-backed dependency provider plus legacy constants and graph helpers | Delegate to `runtime.tool_dependencies` while preserving the root import path until packaging convergence |

## Runtime-Layer Inventory

These modules are runtime-owned and should move under `runtime/` during later
`rag` migration steps. They are grouped by migration concern.

### Capability, safety, and enrichment runtime

| Files | Current runtime/core imports | Migration note |
|---|---|---|
| `runtime/capabilities.py` + root `capabilities.py` shim | `victor.framework.protocols`, `victor.framework.capability_loader`, `victor.framework.capability_config_helpers`, `victor.framework.capabilities` in runtime module only | Capability implementations now follow the runtime-module pattern with a root compatibility shim |
| `runtime/enrichment.py` + root `enrichment.py` shim | `victor.framework.enrichment` in runtime module only | Runtime enrichment strategy now follows the runtime-module pattern with a root compatibility shim |
| `runtime/mode_config.py` + root `mode_config.py` shim | `victor.core.mode_config` in runtime module only | First runtime-module extraction slice is complete; package-root import remains as a compatibility shim |
| `prompts.py` | `victor.core.verticals.prompt_adapter` | Runtime adapter over shared prompt metadata; no longer a definition blocker |
| `runtime/safety.py` + root `safety.py` shim | `victor.core.verticals.protocols`, `victor.framework.config` in runtime module only | Primary safety extension now follows the runtime-module pattern with a root compatibility shim |
| `runtime/safety_enhanced.py` + root `safety_enhanced.py` shim | `victor.core.verticals.protocols`, `victor.framework.config` in runtime module only | Enhanced safety integration now follows the runtime-module pattern with a root compatibility shim |

### Workflow, team, and RL runtime

| Files | Current runtime/core imports | Migration note |
|---|---|---|
| `runtime/workflows.py` + root `workflows/__init__.py` shim | `victor.framework.workflows` in runtime module only | Workflow provider now follows the runtime-module pattern with a root compatibility shim |
| `runtime/teams.py` + root `teams/__init__.py` shim | `victor.framework.teams` in runtime module only | Team specs now follow the runtime-module pattern with a root compatibility shim |
| `runtime/rl.py` + root `rl/__init__.py` shim | `victor.framework.rl`, `victor.framework.rl.config`, `victor.framework.tool_naming` in runtime module only | RL hooks/config now follow the runtime-module pattern; framework `ToolNames` usage remains a follow-on cleanup item |

### Tool runtime entry points

| Files | Current runtime/core imports | Migration note |
|---|---|---|
| `tools/ingest.py`, `tools/management.py`, `tools/query.py`, `tools/search.py` | `victor.tools.base`; `tools/query.py` also imports `victor.framework.enrichment` | Tool implementations are runtime entry points, not definition metadata; they should remain runtime-owned and eventually sit behind a cleaner package boundary |

## Low-Friction Runtime Moves

The following modules are still runtime-owned, but they mostly depend on
package-local code and third-party libraries rather than broad framework/core
surfaces:

- `chunker.py`
- `conversation_enhanced.py`
- `demo_docs.py`
- `demo_sec_filings.py`
- `document_store.py`
- `entity_resolver.py`
- `escape_hatches.py`
- `handlers.py`
- `query_enhancer.py`
- `ui/__init__.py`
- `ui/panel.py`
- `ui/widgets.py`

These are good candidates for mechanical relocation under `runtime/` once the
package-root shims are in place, because most of the work is import-path
plumbing rather than contract redesign.

## Immediate Implications For Next Tasks

### `VPC-T3.11` Replace definition-layer imports with SDK contracts in `rag`

Primary targets:

- `assistant.py`
- prompt/task-hint metadata slice already moved into `prompt_metadata.py`

Expected first moves:

- `assistant.py` runtime/base split is complete; protect it with parity coverage
- keep `prompts.py` as a runtime adapter over shared metadata instead of letting
  definition data drift back into protocol-specific classes
- avoid reintroducing runtime-owned imports into the definition layer

### `VPC-T3.12` Express retrieval/vector/document needs through SDK capability identifiers

Current status:

- `assistant.py` now declares SDK capability requirements for:
  - file operations
  - document ingestion
  - retrieval
  - vector indexing
  - optional web access
- the runtime registry resolves those requirements against the current RAG tool
  bundle, so activation does not emit unknown-capability warnings for the
  current surface

Follow-on capability buckets to consider later:

- enrichment / reranking
- source verification / citation enforcement

### `VPC-T3.13` Move runtime integrations out of `rag` definition modules

Current status:

- complete for the definition-layer migration goal
- runtime-owned modules now sit behind root compatibility shims for:
  - capabilities
  - enrichment
  - mode config
  - safety
  - enhanced safety
  - workflows
  - teams
  - RL

Follow-on runtime packaging work that is no longer required to satisfy the
definition-layer acceptance gate:

1. `tool_dependencies.py`
2. tool implementations and lower-level runtime helpers
3. broader package-topology cleanup under later packaging epics

Root-package transition rules that still apply:

- keep `__init__.py` as a narrow compatibility shim
- keep `tool_dependencies.py` at the root as a delegating shim until packaging work starts
- do not move `assistant.py` or prompt metadata out of the definition layer

### `VPC-T3.14` Add RAG migration tests and parity checks

Current status:

- complete
- parity coverage now exists for:
  - discovery and runtime binding
  - bootstrap activation of RAG extensions
  - runtime-wrapper behavior after the module split

## Conclusion

`rag` has completed the same migration pattern first proven on `coding`:

1. `assistant.py` is SDK-only and remains cleanly in the definition layer
2. prompt/task-hint metadata and capability requirements live on the SDK
   definition contract
3. the major runtime integrations now sit behind runtime modules plus root
   compatibility shims
4. discovery, activation, and runtime-wrapper behavior all have explicit parity
   coverage
