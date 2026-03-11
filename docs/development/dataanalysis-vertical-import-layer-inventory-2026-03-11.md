# Data Analysis Vertical Import Layer Inventory (2026-03-11)

Purpose: complete `VPC-T3.20` by recording the current import-boundary state of
`victor/verticals/contrib/dataanalysis` so the next migration tasks can target
measured definition/runtime blockers instead of re-scanning the package.

## Scope

- Included: Python modules under `victor/verticals/contrib/dataanalysis`
- Included: imports from `victor.framework`, `victor.core`, `victor.tools`, and
  `victor_sdk`
- Excluded: YAML assets and non-Data Analysis vertical packages

## Method

- Runtime-boundary scan:
  - `rg -l "^(from|import) victor(\\.framework|\\.core|\\.tools)" victor/verticals/contrib/dataanalysis -g '*.py' | sort`
- SDK-adoption scan:
  - `rg -l "victor_sdk" victor/verticals/contrib/dataanalysis -g '*.py' | sort`
- Focused blocker inspection:
  - `assistant.py`
  - `prompts.py`
  - `__init__.py`
  - `tool_dependencies.py`

## Measured Status

| Metric | Count | Notes |
|---|---:|---|
| Python files in `dataanalysis` package | 16 | Includes runtime helpers, teams, workflows, RL, and package exports |
| Python files with `victor.framework` / `victor.core` / `victor.tools` imports | 12 | This is the active runtime-boundary surface for `dataanalysis` |
| Python files with `victor_sdk` imports | 1 | Only `assistant.py` currently imports the SDK |
| Definition-layer targets still importing runtime/core | 2 / 2 | `assistant.py` and `prompts.py` are both still coupled to core/runtime APIs |
| Shim candidates still depending on runtime-heavy package or core loaders | 2 / 2 | `__init__.py`, `tool_dependencies.py` |
| Runtime-layer files with direct runtime/core/tool imports | 10 | Runtime integrations are still package-root modules today |

Key findings:

- `assistant.py` already uses SDK-owned tool identifiers, but it still
  subclasses the core `VerticalBase` and imports `StageDefinition` from
  `victor.core.verticals.base`.
- `assistant.py` is the primary definition blocker:
  - core vertical base import
  - core stage type import
  - no runtime wrapper exported at the package boundary yet
- `prompts.py` is the second definition blocker:
  - imports `PromptContributorProtocol` and `TaskTypeHint` from core
  - stores prompt/task-hint data as runtime protocol objects rather than
    serializable metadata
- `__init__.py` still re-exports the root assistant directly instead of a
  runtime wrapper, so package-root callers are still coupled to the current
  mixed definition/runtime class.
- `tool_dependencies.py` is already narrower than `devops`, but it still stays
  in the root shim layer because it depends on the core YAML dependency loader.

## Definition-Layer Status

Current definition-layer blockers:

- `assistant.py`
  - core `VerticalBase` import
  - core `StageDefinition` import
- `prompts.py`
  - core prompt protocol imports
  - runtime prompt object declarations instead of serializable metadata

Definition-layer targets to migrate first in `VPC-T3.21`:

1. `assistant.py`
2. `prompts.py`

Expected first moves:

- migrate the assistant to SDK `VerticalBase` and SDK `StageDefinition`
- split prompt/task-hint data into serializable metadata plus a runtime adapter
- export a runtime wrapper `DataAnalysisAssistant` from the package root

## Shim-Layer Blockers

| File | Shim role | Current boundary imports | Why it stays a shim for now | Target end state |
|---|---|---|---|---|
| `__init__.py` | Package-root export shim candidate | package-local runtime re-exports | Preserves package-root public imports for assistant, prompts, safety, capabilities, and conversation helpers | Runtime wrapper export plus documented compatibility aliases |
| `tool_dependencies.py` | Root compatibility shim | `victor.core.tool_dependency_loader` | Already reduced to a narrow YAML-backed provider factory, but still belongs at the shim boundary until runtime packaging work starts | Delegate to `runtime.tool_dependencies` while preserving the root import path |

## Runtime-Layer Inventory

These modules are runtime-owned and should move behind `runtime/` shims after
the definition contract is cleaned up.

### Capability, safety, config, and prompt runtime

| Files | Current runtime/core imports | Migration note |
|---|---|---|
| `capabilities.py` | `victor.framework.protocols`, `victor.framework.capability_loader`, `victor.framework.capabilities` | Capability provider/config module; strong candidate for the runtime-module pattern |
| `enrichment.py` | `victor.framework.enrichment` | Runtime prompt-enrichment strategy and analysis context injection |
| `mode_config.py` | `victor.core.mode_config` | Registry-backed runtime config provider |
| `prompts.py` | `victor.core.verticals.protocols` | Needs the same metadata/runtime-adapter split used in `rag` and `devops` |
| `safety.py` | `victor.core.verticals.protocols`, `victor.framework.config` | Primary safety extension and framework safety-rule factories |
| `safety_enhanced.py` | `victor.core.verticals.protocols`, `victor.framework.config` | Enhanced safety integration with runtime coordinators |

### Workflow, team, and RL runtime

| Files | Current runtime/core imports | Migration note |
|---|---|---|
| `workflows/__init__.py` | `victor.framework.workflows` | Runtime workflow provider; should follow the `runtime/workflows.py` pattern |
| `teams/__init__.py`, `teams/personas.py` | `victor.framework.teams` in `teams/__init__.py` | Team specs and personas should move behind a runtime team shim |
| `rl/__init__.py` | `victor.framework.rl`, `victor.framework.rl.config`, `victor.framework.tool_naming` | Runtime-only RL hooks/config; SDK `ToolNames` cleanup is likely needed here too |

### Remaining root runtime helpers

| Files | Current runtime/core imports | Migration note |
|---|---|---|
| `tool_dependencies.py` | `victor.core.tool_dependency_loader` | Keep as a later runtime shim once definition-layer work is complete |

## Low-Friction Runtime Moves

The following modules are still runtime-owned but mostly package-local:

- `conversation_enhanced.py`
- `escape_hatches.py`
- `handlers.py`

These can move later under `runtime/` once the core definition/runtime split is
established.

## Immediate Implications For Next Tasks

### `VPC-T3.21` Replace definition-layer imports with SDK contracts in `dataanalysis`

Primary targets:

- `assistant.py`
- `prompts.py`

Expected first moves:

- migrate the assistant to SDK `VerticalBase` and SDK `StageDefinition`
- replace runtime prompt objects with serializable task-hint/prompt metadata
- preserve current behavior via runtime adapters and a package-root runtime
  wrapper

### `VPC-T3.22` Express data/file/notebook needs through SDK capability identifiers in `dataanalysis`

Likely capability buckets:

- file operations
- shell execution
- analysis/validation
- optional web/data-source access

### `VPC-T3.23` Move runtime integrations out of `dataanalysis` definition modules

Recommended move order:

1. prompt metadata/runtime adapter split and package-root runtime wrapper
2. `capabilities.py`, `mode_config.py`, `safety.py`, `safety_enhanced.py`
3. `workflows/`, `teams/`, `rl/`
4. `tool_dependencies.py`

## Conclusion

`dataanalysis` is positioned similarly to where `devops` was before its
definition-layer migration:

1. the assistant already uses SDK tool naming
2. the remaining blockers are concentrated in `assistant.py`, `prompts.py`, and
   the package root
3. the next useful move is the SDK definition/runtime wrapper split
