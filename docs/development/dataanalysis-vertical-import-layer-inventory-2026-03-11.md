# Data Analysis Vertical Import Layer Inventory (2026-03-11)

Purpose: record the measured import-boundary state of
`victor/verticals/contrib/dataanalysis`, starting with the `VPC-T3.20`
baseline and then updating the package through the completed `VPC-F3.5`
migration so later tasks can resume from current facts instead of re-scanning
the package.

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
  - `prompt_metadata.py`
  - `prompts.py`
  - `__init__.py`
  - `tool_dependencies.py`

## Measured Status

| Metric | Count | Notes |
|---|---:|---|
| Python files in `dataanalysis` package | 27 | Includes runtime helpers, teams, workflows, RL, package exports, prompt metadata, and the expanded `runtime/` package |
| Python files with `victor.framework` / `victor.core` / `victor.tools` imports | 12 | This is the active runtime-boundary surface for `dataanalysis` |
| Python files with `victor_sdk` imports | 1 | `assistant.py` is the current SDK-backed definition entrypoint |
| Definition-layer targets still importing runtime/core | 0 / 2 | `assistant.py` and prompt metadata are now definition-layer clean after `VPC-T3.21` |
| Shim candidates still present at the package boundary | 8 | Root compatibility shims preserve the historical import surface while runtime ownership moved under `runtime/` |
| Runtime-layer files with direct runtime/core/tool imports | 10 | Direct runtime imports are now concentrated under `dataanalysis/runtime/`, plus `prompts.py`, `enrichment.py`, and the package root |

Key findings:

- `assistant.py` now subclasses the SDK `VerticalBase`, uses SDK
  `StageDefinition`, and exposes serializable prompt metadata through the SDK
  definition contract.
- `assistant.py` now also declares SDK capability requirements for file
  operations, shell-backed analysis, validation, and optional web access.
- `prompt_metadata.py` now holds the prompt templates, task hints, grounding
  rules, and priority as serializable definition-layer data with no runtime
  imports.
- `prompts.py` is no longer a definition blocker; it now acts as a runtime
  adapter over shared metadata using `PromptContributorAdapter`.
- `__init__.py` now exports a runtime wrapper
  (`VerticalRuntimeAdapter.as_runtime_vertical_class(...)`) so package-root
  callers keep runtime helper behavior while the definition class stays SDK-only.
- `runtime/capabilities.py`, `runtime/safety.py`, `runtime/mode_config.py`,
  `runtime/safety_enhanced.py`, `runtime/workflows.py`, `runtime/rl.py`,
  `runtime/tool_dependencies.py`, `runtime/teams.py`, and
  `runtime/team_personas.py` now own the extracted runtime modules.
- Root compatibility shims are now preserved in `capabilities.py`,
  `safety.py`, `mode_config.py`, `safety_enhanced.py`, `workflows/__init__.py`,
  `rl/__init__.py`, `tool_dependencies.py`, `teams/__init__.py`, and
  `teams/personas.py`.
- The main remaining root runtime package with direct framework imports is
  no longer `teams/`; the remaining package-local runtime helpers are
  `enrichment.py`, `conversation_enhanced.py`, `escape_hatches.py`, and
  `handlers.py`.

## Definition-Layer Status

`VPC-T3.21` completed the definition-layer split:

- `assistant.py`
  - migrated to SDK `VerticalBase`
  - migrated to SDK `StageDefinition`
  - now exposes SDK prompt metadata hooks
- `prompt_metadata.py`
  - added as pure definition-layer prompt/task-hint storage
- `prompts.py`
  - reduced to a runtime adapter over shared metadata
- package root
  - now exports a runtime wrapper `DataAnalysisAssistant` plus
    `DataAnalysisAssistantDefinition`

Remaining definition-layer blockers:

- none in the current `dataanalysis` definition entrypoints

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
| `capabilities.py` | compatibility shim only | Root shim now delegates to `runtime.capabilities` |
| `enrichment.py` | `victor.framework.enrichment` | Runtime prompt-enrichment strategy and analysis context injection |
| `mode_config.py` | compatibility shim only | Root shim now delegates to `runtime.mode_config` |
| `prompts.py` | `victor.core.verticals.prompt_adapter` | Runtime adapter is already split from definition-layer metadata; keep as runtime-owned helper |
| `safety.py` | compatibility shim only | Root shim now delegates to `runtime.safety` |
| `safety_enhanced.py` | compatibility shim only | Root shim now delegates to `runtime.safety_enhanced` |
| `runtime/capabilities.py` | `victor.framework.protocols`, `victor.framework.capability_loader`, `victor.framework.capabilities` | Runtime-owned capability provider/config module is now extracted under `runtime/` |
| `runtime/mode_config.py` | `victor.core.mode_config` | Runtime-owned mode-config provider is now extracted under `runtime/` |
| `runtime/safety.py` | `victor.core.verticals.protocols`, `victor.framework.config` | Runtime-owned primary safety extension is now extracted under `runtime/` |
| `runtime/safety_enhanced.py` | `victor.core.verticals.protocols`, `victor.framework.config` | Runtime-owned enhanced safety provider is now extracted under `runtime/` |

### Workflow, team, and RL runtime

| Files | Current runtime/core imports | Migration note |
|---|---|---|
| `workflows/__init__.py` | compatibility shim only | Root workflow package now delegates to `runtime.workflows` |
| `teams/__init__.py`, `teams/personas.py` | compatibility shims only | Root teams package now delegates to `runtime.teams` and `runtime.team_personas` |
| `rl/__init__.py` | compatibility shim only | Root RL package now delegates to `runtime.rl` |
| `runtime/workflows.py` | `victor.framework.workflows` | Runtime-owned workflow provider is now extracted under `runtime/` |
| `runtime/rl.py` | `victor.framework.rl`, `victor.framework.rl.config`, `victor.framework.tool_naming` | Runtime-owned RL hooks/config are now extracted under `runtime/`; SDK `ToolNames` cleanup is still a follow-up |
| `runtime/teams.py`, `runtime/team_personas.py` | `victor.framework.teams` and `victor.framework.multi_agent` | Runtime-owned team specs and personas are now extracted under `runtime/` |

### Remaining root runtime helpers

| Files | Current runtime/core imports | Migration note |
|---|---|---|
| `tool_dependencies.py` | compatibility shim only | Root shim now delegates to `runtime.tool_dependencies` while preserving the package-root provider factory |
| `runtime/tool_dependencies.py` | `victor.core.tool_dependency_loader` | Runtime-owned provider now resolves the package-root YAML path explicitly from `runtime/` |

## Low-Friction Runtime Moves

The following modules are still runtime-owned but mostly package-local:

- `conversation_enhanced.py`
- `escape_hatches.py`
- `handlers.py`

These can move later under `runtime/` once the core definition/runtime split is
established.

## Immediate Implications For Next Tasks

### `VPC-T3.21` Replace definition-layer imports with SDK contracts in `dataanalysis`

Completed:

- migrated the assistant to SDK `VerticalBase` and SDK `StageDefinition`
- moved prompt/task-hint data into serializable `prompt_metadata.py`
- preserved runtime behavior through a runtime prompt adapter and package-root
  runtime wrapper

### `VPC-T3.22` Express data/file/notebook needs through SDK capability identifiers in `dataanalysis`

Completed:

- declared file operations via `CapabilityIds.FILE_OPS`
- declared notebook-like/statistical execution via `CapabilityIds.SHELL_ACCESS`
- declared output verification via `CapabilityIds.VALIDATION`
- declared optional remote context via `CapabilityIds.WEB_ACCESS`

### `VPC-T3.23` Move runtime integrations out of `dataanalysis` definition modules

Recommended move order:

1. `mode_config.py`, `safety_enhanced.py`, `capabilities.py`, `safety.py`,
   `workflows/__init__.py`, `rl/__init__.py`, and `tool_dependencies.py` are
   now behind `runtime/` shims
2. evaluate whether `enrichment.py`, `conversation_enhanced.py`,
   `escape_hatches.py`, and `handlers.py` should follow into `runtime/`

## Conclusion

`dataanalysis` is positioned similarly to where `devops` was before its
definition-layer migration:

1. the definition layer now follows the SDK-only pattern
2. declarative capability requirements are now in place
3. the migration scope is complete for `VPC-F3.5`; any remaining package-local
   runtime helpers are cleanup candidates outside the finished tranche
