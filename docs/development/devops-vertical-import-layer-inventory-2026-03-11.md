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
| Python files in `devops` package | 17 | Includes runtime helpers, teams, workflows, RL, prompt metadata, and package exports |
| Python files with `victor.framework` / `victor.core` / `victor.tools` imports | 12 | This is the active runtime-boundary surface for `devops` |
| Python files with `victor_sdk` imports | 1 | Only `assistant.py` currently imports the SDK |
| Definition-layer targets still importing runtime/core | 0 / 2 | `assistant.py` and `prompt_metadata.py` are now definition-layer clean |
| Shim candidates still depending on runtime-heavy package or core loaders | 2 / 2 | `__init__.py`, `tool_dependencies.py` |
| Runtime-layer files with direct runtime/core/tool imports | 10 | Runtime integrations are still mostly package-root modules today |

Key findings:

- `assistant.py` now subclasses SDK `VerticalBase`, uses SDK `StageDefinition`,
  and exposes serializable prompt metadata through the definition contract.
- `assistant.py` now declares SDK capability requirements for file operations,
  shell access, git access, container runtime access, validation, and optional
  web access.
- `prompt_metadata.py` now owns the serializable task-hint and prompt-section
  data, while `prompts.py` has been reduced to a runtime adapter using
  `PromptContributorAdapter`.
- `__init__.py` now behaves like the RAG package boundary: it exports a runtime
  wrapper `DevOpsAssistant` built from the SDK definition class and keeps the
  root public import path stable.
- `middleware.py` now owns the DevOps middleware stack, so `assistant.py` is no
  longer carrying runtime helper implementations.
- `devops` now has explicit migration parity coverage for discovery, bootstrap
  activation, and runtime-wrapper behavior in
  `tests/integration/verticals/test_devops_migration_parity.py`.
- `tool_dependencies.py` remains the main compatibility shim candidate because
  it still owns YAML-backed tool-graph logic plus legacy helpers at the package
  root.

## Definition-Layer Status

`VPC-T3.16` is now complete:

- `assistant.py` is SDK-only at the import boundary
- `prompt_metadata.py` is the definition-layer prompt data source
- `prompts.py` is now a runtime adapter instead of a definition blocker
- the package root exports a runtime wrapper while preserving the public
  `victor.devops` import surface

Remaining semantic follow-on for later runtime extraction:

- move additional runtime providers behind `runtime/` shims once packaging work
  reaches `devops`

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
| `prompts.py` | `victor.core.verticals.prompt_adapter` | Prompt runtime adapter over shared metadata; no longer a definition blocker |
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

Current status:

- complete
- `assistant.py` now uses SDK base/stage/prompt contracts
- `prompt_metadata.py` owns serializable prompt metadata
- `prompts.py` is a runtime adapter over shared metadata
- `victor.verticals.contrib.devops` now exports a runtime wrapper around the SDK
  definition class

### `VPC-T3.17` Express shell/git/infra needs through SDK capability identifiers in `devops`

Current status:

- complete
- current DevOps definition requirements are:
  - file operations
  - shell access
  - git access
  - container runtime access
  - validation
  - optional web access

### `VPC-T3.18` Move runtime integrations out of `devops` definition modules

Current status:

- complete for the definition-layer migration goal
- `middleware.py` now owns the middleware stack and the package-root runtime
  wrapper resolves it through shared loader defaults
- remaining runtime packaging work now sits outside the definition-layer
  acceptance gate

### `VPC-T3.19` Add DevOps migration tests and parity checks

Current status:

- complete
- parity coverage now exists for:
  - discovery and runtime binding
  - bootstrap activation of DevOps extensions
  - runtime-wrapper behavior after the definition/runtime split

## Conclusion

`devops` is ready for the same migration sequence already proven on `coding`
and `rag`:

1. the assistant and prompt definition layer are now cleaned up
2. runtime needs are now declared through SDK capability requirements
3. move the remaining runtime providers behind `runtime/` modules plus root
   compatibility shims
