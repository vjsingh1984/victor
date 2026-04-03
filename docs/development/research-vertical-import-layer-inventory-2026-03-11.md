# Research Vertical Import Layer Inventory (2026-03-11)

Purpose: record the measured import-boundary state of
`victor/verticals/contrib/research`, starting with the `VPC-T3.25` baseline
and then updating the package after `VPC-T3.26` and `VPC-T3.27` so the next
migration tasks can target current runtime-boundary blockers instead of
re-scanning the package.

## Scope

- Included: Python modules under `victor/verticals/contrib/research`
- Included: imports from `victor.framework`, `victor.core`, `victor.tools`, and
  `victor_sdk`
- Excluded: YAML assets and non-Research vertical packages

## Method

- Runtime-boundary scan:
  - `rg -l "^(from|import) victor(\\.framework|\\.core|\\.tools)" victor/verticals/contrib/research -g '*.py' | sort`
- SDK-adoption scan:
  - `rg -l "victor_sdk" victor/verticals/contrib/research -g '*.py' | sort`
- Focused blocker inspection:
  - `assistant.py`
  - `prompts.py`
  - `__init__.py`
  - `tool_dependencies.py`

## Measured Status

| Metric | Count | Notes |
|---|---:|---|
| Python files in `research` package | 16 | Includes runtime helpers, teams, workflows, RL, package exports, and prompt metadata |
| Python files with `victor.framework` / `victor.core` / `victor.tools` imports | 11 | This is the active runtime-boundary surface for `research` |
| Python files with `victor_sdk` imports | 1 | `assistant.py` is the current SDK-backed definition entrypoint |
| Definition-layer targets still importing runtime/core | 0 / 2 | `assistant.py` and prompt metadata are now definition-layer clean after `VPC-T3.26` |
| Shim candidates still depending on runtime-heavy package or core loaders | 2 / 2 | `__init__.py`, `tool_dependencies.py` |
| Runtime-layer files with direct runtime/core/tool imports | 9 | Runtime integrations are still package-root modules today |

Key findings:

- `assistant.py` now subclasses the SDK `VerticalBase`, uses SDK
  `StageDefinition`, exposes serializable prompt metadata, and declares
  runtime capabilities through SDK capability identifiers.
- `prompt_metadata.py` now holds the prompt templates, task hints, grounding
  rules, and priority as serializable definition-layer data with no runtime
  imports.
- `prompts.py` is no longer a definition blocker; it now acts as a runtime
  adapter over shared metadata using `PromptContributorAdapter`.
- `__init__.py` now exports a runtime wrapper
  (`VerticalRuntimeAdapter.as_runtime_vertical_class(...)`) so package-root
  callers keep runtime helper behavior while the definition class stays SDK-only.
- `tool_dependencies.py` is already narrow, but it still stays in the root shim
  layer because it depends on the core YAML dependency loader.

## Definition-Layer Status

`VPC-T3.26` completed the definition-layer split:

- `assistant.py`
  - migrated to SDK `VerticalBase`
  - migrated to SDK `StageDefinition`
  - now exposes SDK prompt metadata hooks
  - now exports SDK capability requirements
- `prompt_metadata.py`
  - added as pure definition-layer prompt/task-hint storage
- `prompts.py`
  - reduced to a runtime adapter over shared metadata
- package root
  - now exports a runtime wrapper `ResearchAssistant` plus
    `ResearchAssistantDefinition`

Remaining definition-layer blockers:

- none in the current `research` definition entrypoints

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
| `mode_config.py` | `victor.core.mode_config` | Registry-backed runtime config provider |
| `prompts.py` | `victor.core.verticals.protocols` | Needs the same metadata/runtime-adapter split used in other migrated verticals |
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

These can move later under `runtime/` once the core definition/runtime split is
established.

## Immediate Implications For Next Tasks

### `VPC-T3.26` Replace definition-layer imports with SDK contracts in `research`

Completed:

- migrated the assistant to SDK `VerticalBase` and SDK `StageDefinition`
- moved prompt/task-hint data into serializable `prompt_metadata.py`
- preserved runtime behavior through a runtime prompt adapter and package-root
  runtime wrapper

### `VPC-T3.27` Express web/search/document needs through SDK capability identifiers in `research`

Completed:

- declared file operations via `CapabilityIds.FILE_OPS`
- declared web retrieval via `CapabilityIds.WEB_ACCESS`
- declared source verification via `CapabilityIds.SOURCE_VERIFICATION`
- declared result validation via `CapabilityIds.VALIDATION`

### `VPC-T3.28` Move runtime integrations out of `research` definition modules

Recommended move order:

1. prompt metadata/runtime adapter split and package-root runtime wrapper
2. `capabilities.py`, `mode_config.py`, `safety.py`, `safety_enhanced.py`
3. `workflows/`, `teams/`, `rl/`
4. `tool_dependencies.py`

## Conclusion

`research` is positioned similarly to where `dataanalysis` and `devops` were
before their runtime extraction work:

1. the definition layer now follows the SDK-only pattern
2. declarative capability requirements are now in place
3. the remaining work is concentrated in runtime module extraction under
   `VPC-T3.28`

## Completion Update (Post `VPC-T3.28` / `VPC-T3.29`)

Measured after the full `research` migration tranche:

| Metric | Count | Notes |
|---|---:|---|
| Python files in `research` package | 26 | Includes the new `runtime/` package plus root compatibility shims |
| Python files with `victor.framework` / `victor.core` / `victor.tools` imports | 11 | Runtime-heavy imports are now concentrated in runtime-owned modules |
| Python files with `victor_sdk` imports | 1 | `assistant.py` remains the sole definition-layer SDK entrypoint |

Completed runtime extraction:

- `capabilities.py` -> `runtime/capabilities.py`
- `mode_config.py` -> `runtime/mode_config.py`
- `safety.py` -> `runtime/safety.py`
- `safety_enhanced.py` -> `runtime/safety_enhanced.py`
- `workflows/__init__.py` -> `runtime/workflows.py`
- `rl/__init__.py` -> `runtime/rl.py`
- `tool_dependencies.py` -> `runtime/tool_dependencies.py`
- `teams/__init__.py` -> `runtime/teams.py`
- `teams/personas.py` -> `runtime/team_personas.py`

Legacy import paths preserved through compatibility shims:

- package root shims for capabilities, mode config, safety, workflows, RL, and
  tool dependencies
- `teams/__init__.py` shim delegating to `runtime.teams`
- `teams/personas.py` shim delegating to `runtime.team_personas`

Parity coverage now exists for the completed migration shape:

- `tests/unit/core/verticals/test_runtime_helper_defaults.py`
- `tests/integration/verticals/test_research_migration_parity.py`
- `tests/unit/framework/test_team_registry.py`
- `tests/integration/framework/test_persona_integration.py`

Current status:

- `VPC-T3.28` complete
- `VPC-T3.29` complete
- `VPC-F3.6` complete

Next resume point:

- `VPC-T3.30` rewrite `examples/external_vertical` to use SDK-only definition
  imports
