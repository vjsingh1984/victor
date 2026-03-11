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
| Python files in `rag` package | 30 | Includes tools, UI, runtime helpers, demos, and prompt metadata |
| Python files with `victor.framework` / `victor.core` / `victor.tools` imports | 15 | This is the active runtime-boundary surface for `rag` |
| Python files with `victor_sdk` imports | 1 | Only `assistant.py` currently imports the SDK |
| Definition-layer targets still importing runtime/core | 1 / 2 | `assistant.py` is still blocked; `prompt_metadata.py` is data-only |
| Shim candidates still depending on runtime-heavy package or core loaders | 2 / 2 | `__init__.py`, `tool_dependencies.py` |
| Runtime-layer files with direct runtime/core/tool imports | 13 | Includes `prompts.py`, which now acts as a runtime adapter |

Key findings:

- `rag` is earlier in the migration than `coding`. SDK adoption is limited to
  `assistant.py`, and even there it only covers shared `ToolNames`.
- `assistant.py` still imports the runtime base class, runtime `StageDefinition`,
  and runtime `TieredToolConfig`, so the definition layer is still coupled to
  the host implementation.
- prompt/task-hint data now lives in `prompt_metadata.py`, while `prompts.py`
  has become a thin runtime adapter over that shared metadata.
- `rag` has not yet adopted SDK capability requirements. Retrieval, indexing,
  storage, and enrichment needs are still implicit in runtime modules.
- The root package and tool-dependency module still act as transitional shims,
  with `__init__.py` eagerly exporting runtime-heavy surfaces.

## Definition-Layer Blockers

| File | Current layer target | Current boundary imports | Why it is still blocked | Required next move |
|---|---|---|---|---|
| `assistant.py` | Definition | `victor.core.verticals.base`, `victor.core.verticals.protocols`, `victor_sdk` | Uses runtime `VerticalBase`, `StageDefinition`, and `TieredToolConfig`; does not declare typed capability requirements; still mixes SDK tool names with runtime-owned config types | Replace runtime base/protocol dependencies with SDK definition contracts and move runtime tool-tier config behind adapters or runtime metadata |

Definition-layer implication:

- `rag` is no longer blocked on prompt/task-hint data modeling. The remaining
  definition-layer blocker is `assistant.py` plus missing SDK capability
  declarations for retrieval/indexing concerns.

## Shim-Layer Blockers

| File | Shim role | Current boundary imports | Why it stays a shim for now | Target end state |
|---|---|---|---|---|
| `__init__.py` | Package-root export shim | No direct `victor.framework` / `victor.core` imports, but eagerly re-exports runtime-heavy modules from the root package | Aggregates assistant, document store, tools, prompt contributor, mode config, capabilities, and enhanced runtime helpers into one root import surface | Narrow re-export surface centered on `RAGAssistant` plus documented compatibility aliases |
| `tool_dependencies.py` | Root compatibility shim | `victor.core.tool_dependency_loader`, `victor.core.tool_types`, `victor.tools.tool_graph` | Provides the current YAML-backed dependency provider plus legacy constants and graph helpers | Delegate to `runtime.tool_dependencies` while preserving the root import path until packaging convergence |

## Runtime-Layer Inventory

These modules are runtime-owned and should move under `runtime/` during later
`rag` migration steps. They are grouped by migration concern.

### Capability, safety, and enrichment runtime

| Files | Current runtime/core imports | Migration note |
|---|---|---|
| `capabilities.py` | `victor.framework.protocols`, `victor.framework.capability_loader`, `victor.framework.capability_config_helpers`, `victor.framework.capabilities` | Capability implementations are fully runtime-owned; later definition work should declare needs instead of importing implementations |
| `enrichment.py` | `victor.framework.enrichment` | Runtime enrichment strategy and query-enhancement integration; keep out of the definition package |
| `mode_config.py` | `victor.core.mode_config` | Runtime mode registration/provider; candidate for `runtime.mode_config` |
| `prompts.py` | `victor.core.verticals.prompt_adapter` | Runtime adapter over shared prompt metadata; no longer a definition blocker |
| `safety.py`, `safety_enhanced.py` | `victor.core.verticals.protocols`, `victor.framework.config` | Runtime safety extensions and rule registration; do not keep them on the definition path |

### Workflow, team, and RL runtime

| Files | Current runtime/core imports | Migration note |
|---|---|---|
| `workflows/__init__.py` | `victor.framework.workflows` | Strong runtime workflow provider; move behind shared workflow resolution |
| `teams/__init__.py` | `victor.framework.teams` | Runtime team specs and formations; should move under `runtime.teams` |
| `rl/__init__.py` | `victor.framework.rl`, `victor.framework.rl.config`, `victor.framework.tool_naming` | Runtime-only RL hooks/config; still uses framework `ToolNames`, which becomes a follow-on cleanup item |

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

- continue moving `assistant.py` away from runtime `VerticalBase`,
  `StageDefinition`, and `TieredToolConfig`
- keep `prompts.py` as a runtime adapter over shared metadata instead of letting
  definition data drift back into protocol-specific classes
- remove the remaining runtime-owned definition imports before broader package
  moves

### `VPC-T3.12` Express retrieval/vector/document needs through SDK capability identifiers

Current gap:

- `rag` exposes no SDK capability requirements today
- retrieval, indexing, document ingestion, storage, and enrichment dependencies
  are still implicit in runtime modules and tool implementations

Likely capability buckets to formalize:

- document access / ingestion
- retrieval / search
- vector storage / indexing
- enrichment / reranking
- optional web fetch / shell support

### `VPC-T3.13` Move runtime integrations out of `rag` definition modules

Recommended move order:

1. `capabilities.py`, `enrichment.py`, `mode_config.py`, `safety*.py`
2. `workflows/`, `teams/`, `rl/`
3. `tool_dependencies.py`
4. tool implementations and lower-level runtime helpers

Root-package transition rules:

- keep `__init__.py` as a narrow compatibility shim
- keep `tool_dependencies.py` at the root as a delegating shim
- do not move `assistant.py` or prompt metadata out of the definition layer

## Conclusion

`rag` is ready for the same migration pattern now proven on `coding`, but its
definition layer is less mature:

1. `assistant.py` is the only current SDK adopter, and it still depends on
   runtime definition types
2. prompt/task-hint metadata is now split cleanly, but `assistant.py` still
   needs the remaining definition/runtime boundary work
3. the runtime-heavy surface is already well-bounded enough to migrate in
   grouped layers once `VPC-T3.11` and `VPC-T3.12` establish the definition
   contract
