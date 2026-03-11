# DevOps Vertical Import Layer Inventory (2026-03-11)

Purpose: complete `VPC-T3.15` by recording the current import-boundary state of
`victor/verticals/contrib/devops` so the next migration tasks can target
measured definition/runtime blockers instead of re-scanning the package.

## Scope

- Included: Python modules under `victor/verticals/contrib/devops`
- Included: imports from `victor.framework`, `victor.core`, `victor.tools`, and
  `victor_sdk`
- Excluded: YAML assets and non-DevOps vertical packages

## Method

- Runtime-boundary scan:
  - `rg -l "^(from|import) victor(\\.framework|\\.core|\\.tools)" victor/verticals/contrib/devops -g '*.py' | sort`
- SDK-adoption scan:
  - `rg -l "victor_sdk" victor/verticals/contrib/devops -g '*.py' | sort`
- Focused blocker inspection:
  - `assistant.py`
  - `prompts.py`
  - `__init__.py`
  - `tool_dependencies.py`

## Measured Status

| Metric | Count | Notes |
|---|---:|---|
| Python files in `devops` package | 16 | Includes runtime helpers, teams, workflows, RL, and package exports |
| Python files with `victor.framework` / `victor.core` / `victor.tools` imports | 12 | This is the active runtime-boundary surface for `devops` |
| Python files with `victor_sdk` imports | 1 | Only `assistant.py` currently imports the SDK |
| Definition-layer targets still importing runtime/core | 2 / 2 | `assistant.py` and `prompts.py` are both still coupled to core/runtime APIs |
| Shim candidates still depending on runtime-heavy package or core loaders | 2 / 2 | `__init__.py`, `tool_dependencies.py` |
| Runtime-layer files with direct runtime/core/tool imports | 10 | Runtime integrations are still package-root modules today |

Key findings:

- `devops` is earlier in migration than `rag`. The assistant already uses
  SDK-owned tool identifiers, but it still subclasses the core `VerticalBase`
  and imports `StageDefinition` from `victor.core.verticals.base`.
- `assistant.py` is the primary definition blocker:
  - imports the core vertical base and stage type
  - imports `MiddlewareProtocol` from core
  - still defines `get_middleware()` directly, which reaches into
    `victor.framework.middleware.MiddlewareComposer`
- `prompts.py` is the second definition blocker:
  - imports `PromptContributorProtocol` and `TaskTypeHint` from core
  - stores prompt/task-hint data as runtime protocol objects rather than
    serializable metadata
- `__init__.py` already behaves like a package-root runtime export shim, but it
  still re-exports root runtime modules directly rather than preferring a
  `runtime/` package.
- `tool_dependencies.py` remains the main compatibility shim candidate because
  it still owns YAML-backed tool-graph logic plus legacy helpers at the package
  root.

## Definition-Layer Status

Current definition-layer blockers:

- `assistant.py`
  - core `VerticalBase` import
  - core `StageDefinition` import
  - core `MiddlewareProtocol` import
  - direct framework middleware composition in the assistant class
- `prompts.py`
  - core prompt protocol imports
  - runtime prompt object declarations instead of serializable metadata

Definition-layer targets to migrate first in `VPC-T3.16`:

1. `assistant.py`
2. `prompts.py`

Expected first moves:

- move `StageDefinition`/base-class usage to SDK contracts
- express prompt/task-hint data as serializable metadata, then keep `prompts.py`
  as a runtime adapter if needed
- remove assistant-owned middleware composition in favor of runtime-module
  defaults or a runtime shim

## Shim-Layer Blockers

| File | Shim role | Current boundary imports | Why it stays a shim for now | Target end state |
|---|---|---|---|---|
| `__init__.py` | Package-root export shim | package-local runtime re-exports | Preserves package-root public imports for assistant, prompts, safety, capabilities, and conversation helpers | Narrow runtime export shim centered on the runtime-wrapped `DevOpsAssistant` and documented aliases |
| `tool_dependencies.py` | Root compatibility shim candidate | `victor.core.tool_dependency_loader`, `victor.framework.tool_naming`, `victor.tools.tool_graph` | Still owns YAML-backed tool-graph logic and helper exports used by current consumers | Delegate to `runtime.tool_dependencies` while preserving the root import path until packaging convergence |

## Runtime-Layer Inventory

These modules are runtime-owned and should move behind `runtime/` shims after
the definition contract is cleaned up.

### Capability, safety, config, and prompt runtime

| Files | Current runtime/core imports | Migration note |
|---|---|---|
| `capabilities.py` | `victor.framework.protocols`, `victor.framework.capability_loader`, `victor.framework.capabilities` | Capability provider/config module; good candidate for the runtime-module pattern already used in `rag` |
| `enrichment.py` | `victor.framework.enrichment` | Runtime prompt-enrichment strategy and infra context injection |
| `mode_config.py` | `victor.core.mode_config` | Registry-backed runtime config provider |
| `prompts.py` | `victor.core.verticals.protocols` | Split needed: metadata to SDK-facing data, runtime adapter stays at the runtime layer |
| `safety.py` | `victor.core.verticals.protocols`, `victor.framework.config` | Primary safety extension and framework safety-rule factories |
| `safety_enhanced.py` | `victor.core.verticals.protocols`, `victor.framework.config` | Enhanced safety integration with runtime coordinators |

### Workflow, team, and RL runtime

| Files | Current runtime/core imports | Migration note |
|---|---|---|
| `workflows/__init__.py` | `victor.framework.workflows` | Runtime workflow provider; should follow the `runtime/workflows.py` pattern |
| `teams/__init__.py`, `teams/personas.py` | `victor.framework.teams` in `teams/__init__.py` | Team specs and personas should move behind a runtime team shim |
| `rl/__init__.py` | `victor.framework.rl`, `victor.framework.rl.config`, `victor.framework.tool_naming` | Runtime-only RL hooks/config; SDK `ToolNames` follow-on cleanup is likely needed here |

### Remaining root runtime helpers

| Files | Current runtime/core imports | Migration note |
|---|---|---|
| `tool_dependencies.py` | `victor.core.tool_dependency_loader`, `victor.framework.tool_naming`, `victor.tools.tool_graph` | Keep as a later runtime shim once definition-layer work is complete |

## Low-Friction Runtime Moves

The following modules are still runtime-owned but mostly package-local:

- `conversation_enhanced.py`
- `escape_hatches.py`
- `handlers.py`

These can move later under `runtime/` once the core definition/runtime split is
established.

## Immediate Implications For Next Tasks

### `VPC-T3.16` Replace definition-layer imports with SDK contracts in `devops`

Primary targets:

- `assistant.py`
- `prompts.py`

Expected first moves:

- migrate the assistant to SDK `VerticalBase` and SDK `StageDefinition`
- replace runtime prompt objects with serializable task-hint/prompt metadata
- preserve current behavior via runtime adapters and package-root shims

### `VPC-T3.17` Express shell/git/infra needs through SDK capability identifiers in `devops`

Likely capability buckets:

- file operations
- shell execution
- git operations
- container/runtime tooling
- optional web/documentation access

### `VPC-T3.18` Move runtime integrations out of `devops` definition modules

Recommended move order:

1. `prompts.py` runtime adapter cleanup and assistant-owned middleware extraction
2. `capabilities.py`, `mode_config.py`, `safety.py`, `safety_enhanced.py`
3. `workflows/`, `teams/`, `rl/`
4. `tool_dependencies.py`

## Conclusion

`devops` is ready for the same migration sequence already proven on `coding`
and `rag`:

1. clean the assistant and prompt definition layer first
2. express runtime needs through SDK capability requirements
3. move runtime providers behind `runtime/` modules plus root compatibility
   shims
