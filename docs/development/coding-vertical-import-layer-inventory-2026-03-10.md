# Coding Vertical Import Layer Inventory (2026-03-10)

Purpose: complete `VPC-T3.5` by recording the current import-boundary state of
`victor/verticals/contrib/coding` so the next migration tasks can target
concrete files and blockers instead of re-scanning the package each session.

## Scope

- Included: Python modules under `victor/verticals/contrib/coding`
- Included: imports from `victor.framework`, `victor.core`, `victor.tools`, and
  `victor_sdk`
- Excluded: YAML/SVG assets and non-coding vertical packages

## Method

- Runtime-boundary scan:
  - `rg -l "^(from|import) victor(\\.framework|\\.core|\\.tools)" victor/verticals/contrib/coding -g '*.py' | sort`
- SDK-adoption scan:
  - `rg -l "victor_sdk" victor/verticals/contrib/coding -g '*.py' | sort`
- Focused blocker inspection:
  - `assistant.py`
  - `prompts.py`
  - `__init__.py`
  - `tool_dependencies.py`

## Measured Status

| Metric | Count | Notes |
|---|---:|---|
| Python files with `victor.framework` / `victor.core` / `victor.tools` imports | 20 | Down from the `VPC-T3.1` baseline of 22 |
| Python files with `victor_sdk` imports | 2 | `assistant.py` and `rl/config.py` currently import the SDK |
| Definition-layer targets still importing runtime/core | 1 / 2 | `prompts.py` remains the only root-level definition blocker |
| Shim candidates still importing runtime/core | 2 / 2 | `__init__.py`, `tool_dependencies.py` |
| Runtime-layer files with direct runtime/core/tool imports | 17 | See grouped inventory below; `teams/__init__.py` is intentionally runtime-owned |

Key findings:

- The `coding` package has now removed runtime/core imports from
  `assistant.py`, but the package is still overwhelmingly runtime-owned. The
  next refactors should focus on converting `prompts.py` into a thin runtime
  adapter over serializable prompt metadata and narrowing the two package-root
  shims.
- Canonical tool identifiers are no longer sourced from
  `victor.framework.tool_naming` or `victor.tools.tool_names` anywhere under
  `coding`.
- `CodingAssistant` now imports only SDK-owned definition contracts and exposes
  coding-specific metadata through `get_metadata()`, with the package root
  exporting the runtime wrapper separately.

## Definition-Layer Blockers

| File | Current layer target | Current boundary imports | Why it is still blocked | Required next move |
|---|---|---|---|---|
| `prompts.py` | Definition | `victor.core.verticals.protocols` | Implements `PromptContributorProtocol` and `TaskTypeHint` runtime objects inside a definition module | Convert prompt/task-hint data to plain serializable SDK-facing metadata and let runtime prompt contributors be optional adapters |

Definition-layer implication:

- `assistant.py` is no longer a definition-layer blocker. The remaining root
  definition work is isolated to `prompts.py`.

## Shim-Layer Blockers

| File | Shim role | Current boundary imports | Why it stays a shim for now | Target end state |
|---|---|---|---|---|
| `__init__.py` | Package-root export shim | `victor.core.tool_dependency_loader` plus broad imports of runtime modules from the package root | Aggregates assistant, middleware, safety, prompts, service provider, capability providers, and tool dependencies in one import surface | Narrow re-export surface centered on `CodingAssistant` plus explicitly documented compatibility aliases |
| `tool_dependencies.py` | Root compatibility shim | `victor.core.tool_dependency_loader` | Provides the current entry-point provider and YAML provider surface | Delegate to `runtime.tool_dependencies` while preserving the root import path until `VPC-E4` |

## Runtime-Layer Inventory

These modules are runtime-owned and should move under `runtime/` during
`VPC-T3.8`. They are grouped by migration concern.

### Capability, middleware, safety, and DI runtime

| Files | Current runtime/core imports | Migration note |
|---|---|---|
| `capabilities.py` | `victor.framework.protocols`, `victor.framework.capability_loader`, `victor.framework.capability_config_helpers`, `victor.framework.capabilities` | Remains runtime-owned; `assistant.py` should request capabilities declaratively instead of importing implementations |
| `middleware.py` | `victor.core.verticals.protocols`, `victor.framework.middleware` | The assistant override is now gone; shared loader defaults resolve this runtime module via `get_middleware()` |
| `mode_config.py` | `victor.core.mode_config` | Move to `runtime.mode_config` behind shared extension resolution |
| `safety.py`, `safety_enhanced.py` | `victor.core.verticals.protocols`, `victor.framework.config`, `victor.framework.safety` | Keep runtime-only; root package should stop importing these eagerly |
| `service_provider.py` | `victor.core.verticals.protocols` | The assistant override is now gone; shared loader defaults resolve this runtime module through the generic service-provider path |

### Workflow, team, and RL runtime

| Files | Current runtime/core imports | Migration note |
|---|---|---|
| `workflows/provider.py` | `victor.framework.workflows` | Strong runtime-only module; move early because shared mixed-mode workflow resolution now exists |
| `teams/__init__.py`, `teams/personas.py`, `teams/specs.py` | `victor.framework.team_schema`, `victor.framework.multi_agent`, `victor.framework.teams` | Move under `runtime.teams`; root package should stop re-exporting teams by default |
| `rl/config.py`, `rl/hooks.py` | `victor.framework.rl`, `victor.framework.tool_naming` | Runtime-only; `rl/config.py` is the remaining `ToolNames` boundary hotspot for `VPC-T3.6` |

### Enrichment, tool composition, and code intelligence helpers

| Files | Current runtime/core imports | Migration note |
|---|---|---|
| `enrichment.py` | `victor.framework.enrichment` | Runtime enrichment strategy, not definition metadata |
| `composed_chains.py` | `victor.tools.composition` | Runtime composition helper, should move with the service/runtime stack |
| `codebase/indexer.py`, `codebase/unified_extractor.py` | `victor.core.utils.ast_helpers` | Runtime domain engine; relocation is mostly package-structure work |
| `codebase/query_expander.py` | `victor.framework.search.query_expansion` | Runtime helper; keep out of the definition package entirely |

## Low-Friction Runtime Moves

The following subpackages are still runtime-owned, but they currently depend
mostly on package-local modules and third-party libraries rather than on broad
framework/core surfaces:

- `completion/`
- `coverage/`
- `docgen/`
- `editing/`
- `languages/`
- `lsp/`
- `refactor/`
- `review/`
- `testgen/`
- `codebase/graph/`
- `codebase/embeddings/`

These are good candidates for mechanical relocation under `runtime/` once the
package-root import shims are in place, because most of the work is import-path
plumbing rather than contract redesign.

## Immediate Implications For Next Tasks

### `VPC-T3.6` Replace tool-name runtime paths with SDK identifiers

Status:

- completed in the same migration tranche
- `rl/config.py` now imports `ToolNames` from `victor_sdk`
- no remaining `victor.framework.tool_naming` or `victor.tools.tool_names`
  imports remain anywhere under `coding`

### `VPC-T3.7` Replace direct framework capability coupling in the definition layer

Status:

- completed for capability requirement declaration
- `assistant.py` now expresses core host/runtime needs through SDK capability
  requirements:
  - required: `file_ops`
  - optional: `git`, `lsp`
- direct framework capability implementations remain isolated to
  `capabilities.py`, which is correctly classified as runtime-only

Remaining definition-layer work that is still outside `VPC-T3.7`:

- `prompts.py` is still a runtime prompt-contributor module rather than pure
  serializable metadata

### `VPC-T3.8` Move runtime-specific middleware, workflows, and helpers out of the definition layer

Status:

- completed for the `coding` assistant definition layer
- `assistant.py` no longer defines:
  - `get_middleware()`
  - `get_service_provider()`
  - `get_composed_chains()`
  - `get_personas()`
- shared loader defaults now resolve those runtime hooks from:
  - `middleware.py`
  - `service_provider.py`
  - `composed_chains.py`
  - `teams/__init__.py`

Remaining definition-layer work after `VPC-T3.8`:

- `prompts.py` still implements a runtime prompt contributor rather than pure
  serializable metadata

### `VPC-T3.9` Add coding vertical migration tests and parity checks

Status:

- completed
- added `tests/integration/verticals/test_coding_migration_parity.py`
- coverage now locks:
  - discovery parity through `VerticalLoader.load("coding")`
  - activation parity through `bootstrap_container(..., vertical="coding")`
    plus DI-visible `VerticalExtensions` and coding service registrations
  - behavior parity for shared runtime-helper defaults after the helper hooks
    were removed from `assistant.py`

Remaining definition-layer work after `VPC-T3.9`:

- `prompts.py` still implements a runtime prompt contributor rather than pure
  serializable metadata

### `VPC-T3.36` Remove direct framework-registry access from migrated definition modules

Status:

- completed for the `coding` assistant definition module
- `assistant.py` now imports only SDK-owned definition contracts
- coding-specific runtime metadata moved from `customize_config()` into
  `get_metadata()`
- `victor.verticals.contrib.coding` now exports a host-owned runtime wrapper
  while `assistant.py` remains the SDK definition source

Remaining definition-layer work after `VPC-T3.36`:

- `prompts.py` still implements a runtime prompt contributor rather than pure
  serializable metadata
- the package-root shims still import runtime/core modules, which is tracked
  separately under `VPC-E4`

### `VPC-T3.8` Move runtime-specific behavior behind adapters or `runtime/`

Recommended move order:

1. `capabilities.py`, `middleware.py`, `mode_config.py`, `safety*.py`, `service_provider.py`
2. `workflows/`, `teams/`, `rl/`
3. `enrichment.py`, `composed_chains.py`
4. deeper domain engines and helper subpackages

Root-package transition rules:

- keep `__init__.py` as a narrow compatibility shim
- keep `tool_dependencies.py` at the root as a delegating shim
- do not move `assistant.py` or `prompts.py` under `runtime/`

## Conclusion

`coding` is still the right first vertical to migrate, but the package is now
well enough inventoried to proceed surgically:

1. the remaining root-level definition blocker is `prompts.py`
2. the package-root shim problem is limited to `__init__.py` and
   `tool_dependencies.py`
3. the runtime-heavy surface is well-bounded and already aligns with the
   `runtime/` package layout introduced in `VPC-T3.2` and `VPC-T3.4`
