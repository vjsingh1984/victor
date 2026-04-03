# Vertical Module Layer Classification (2026-03-10)

Purpose: record the `VPC-T3.1` baseline classification for bundled contrib
vertical modules so later migration work can proceed without re-inventorying the
same boundaries.

## Scope

- Included: vertical-facing packages under `victor/verticals/contrib/*`
- Included: entrypoint modules, prompt/task-hint modules, runtime integration
  modules, workflow/team/tool-dependency packages, and domain subpackages shipped
  with each vertical
- Excluded: `__pycache__` artifacts and non-vertical framework/core packages

## Method

- File inventory was taken from `victor/verticals/contrib/{coding,rag,research,devops,dataanalysis}`
- Import-boundary scan checked for `victor.framework`, `victor.core`,
  `victor.tools`, and `victor_sdk`
- Classification uses three buckets:
  - `Definition-layer target`: should remain in the extracted vertical package and
    be rewritten to depend on `victor-sdk` only
  - `Runtime-layer`: belongs behind host/runtime services, adapters, or bundled
    runtime integrations
  - `Shim candidate`: exists mainly to preserve current import paths or legacy
    activation while the definition/runtime split is in flight

## Measured Baseline

| Vertical | Python files with `victor.framework` / `victor.core` / `victor.tools` imports |
|---|---:|
| `coding` | 22 |
| `rag` | 16 |
| `research` | 12 |
| `devops` | 13 |
| `dataanalysis` | 13 |

Additional boundary finding:

- `victor/verticals/contrib/*` currently contains zero Python modules importing
  `victor_sdk`

## Cross-Cutting Classification Rules

| Module family | Classification | Why | Current blocker |
|---|---|---|---|
| `assistant.py` | Definition-layer target | Owns vertical identity, tools, stages, and prompt/workflow metadata | Currently mixes definition hooks with `victor.core` bases/protocols and framework capabilities/tool registries |
| `prompts.py` | Definition-layer target | Mostly serializable task hints and prompt templates | Still implemented as runtime `PromptContributorProtocol` objects |
| `victor-vertical.toml` | Definition-layer target | Already serializable package metadata | Needs to align with the new SDK manifest schema over time |
| `__init__.py` | Shim candidate | Preserves current import surface and mixed re-exports | Re-exports assistant plus runtime helpers from one package root |
| `tool_dependencies.py` | Shim candidate | Carries backward-compatible provider wrappers and raw exports | Currently runtime-owned dependency providers with legacy exports |
| `capabilities.py` | Runtime-layer | Registers framework capability providers/loaders | Depends directly on runtime capability loader and orchestrator types |
| `middleware.py` | Runtime-layer | Runtime execution middleware | Depends on framework middleware contracts |
| `mode_config.py` | Runtime-layer | Registers runtime mode providers | Depends on central mode registry/services |
| `safety.py`, `safety_enhanced.py` | Runtime-layer | Runtime safety enforcement and coordinators | Depend on framework safety/config services |
| `service_provider.py` | Runtime-layer | DI/service-container integration | Depends on runtime container and provider protocols |
| `teams/*` | Runtime-layer | Multi-agent team schema, registry, and persona runtime | Depends on framework team/persona systems |
| `workflows/*` | Runtime-layer | YAML workflow providers, escape hatches, workflow execution | Depends on framework workflow runtime |
| `handlers.py`, `escape_hatches.py`, `enrichment.py`, `conversation_enhanced.py`, `rl/*` | Runtime-layer | Execution helpers and runtime-specific orchestration | Depend on workflow/runtime services |
| Deep domain engines (`codebase/*`, `completion/*`, `lsp/*`, `tools/*`, etc.) | Runtime-layer | Actual operational behavior, tools, storage, indexing, UI, or search engines | Not part of the serializable SDK definition contract |

## Evidence Snapshots

- `coding/assistant.py` imports `victor.core.verticals.base`,
  `victor.core.verticals.protocols`, and `victor.framework.capabilities`, so it is
  a split target rather than a clean SDK definition module.
- `rag/assistant.py` imports `victor.core.verticals.base` and
  `victor.framework.tool_naming.ToolNames`, showing the same mixed responsibility.
- `coding/prompts.py` and `research/prompts.py` are mostly serializable hints, but
  they still depend on `victor.core.verticals.protocols`.
- `coding/workflows/provider.py` and similar workflow providers inherit
  `victor.framework.workflows.BaseYAMLWorkflowProvider`, making them runtime-only.
- `rag/tool_dependencies.py` explicitly documents itself as backward-compatible and
  deprecated, so it should remain a shim until packaging convergence lands.

## Per-Vertical Classification

### `coding`

Definition-layer targets:

- `assistant.py`
- `prompts.py`
- `victor-vertical.toml`

Runtime-layer modules:

- `capabilities.py`
- `middleware.py`
- `mode_config.py`
- `safety.py`
- `safety_enhanced.py`
- `service_provider.py`
- `conversation_enhanced.py`
- `composed_chains.py`
- `enrichment.py`
- `escape_hatches.py`
- `handlers.py`
- `workflows/`
- `teams/`
- `rl/`
- `codebase/`
- `completion/`
- `coverage/`
- `docgen/`
- `editing/`
- `languages/`
- `lsp/`
- `refactor/`
- `review/`
- `testgen/`

Shim candidates:

- `__init__.py`
- `tool_dependencies.py`
- `tool_dependencies.yaml`

### `rag`

Definition-layer targets:

- `assistant.py`
- `prompts.py`
- `victor-vertical.toml`

Runtime-layer modules:

- `capabilities.py`
- `mode_config.py`
- `safety.py`
- `safety_enhanced.py`
- `conversation_enhanced.py`
- `enrichment.py`
- `escape_hatches.py`
- `handlers.py`
- `chunker.py`
- `document_store.py`
- `entity_resolver.py`
- `query_enhancer.py`
- `tools/`
- `ui/`
- `teams/`
- `workflows/`
- `rl/`
- `demo_docs.py`
- `demo_sec_filings.py`

Shim candidates:

- `__init__.py`
- `tool_dependencies.py`
- `tool_dependencies.yaml`

### `research`

Definition-layer targets:

- `assistant.py`
- `prompts.py`
- `victor-vertical.toml`

Runtime-layer modules:

- `capabilities.py`
- `mode_config.py`
- `safety.py`
- `safety_enhanced.py`
- `conversation_enhanced.py`
- `escape_hatches.py`
- `handlers.py`
- `teams/`
- `workflows/`
- `rl/`

Shim candidates:

- `__init__.py`
- `tool_dependencies.py`
- `tool_dependencies.yaml`

### `devops`

Definition-layer targets:

- `assistant.py`
- `prompts.py`
- `victor-vertical.toml`

Runtime-layer modules:

- `capabilities.py`
- `mode_config.py`
- `safety.py`
- `safety_enhanced.py`
- `conversation_enhanced.py`
- `enrichment.py`
- `escape_hatches.py`
- `handlers.py`
- `teams/`
- `workflows/`
- `rl/`

Shim candidates:

- `__init__.py`
- `tool_dependencies.py`
- `tool_dependencies.yaml`

### `dataanalysis`

Definition-layer targets:

- `assistant.py`
- `prompts.py`
- `victor-vertical.toml`

Runtime-layer modules:

- `capabilities.py`
- `mode_config.py`
- `safety.py`
- `safety_enhanced.py`
- `conversation_enhanced.py`
- `enrichment.py`
- `escape_hatches.py`
- `handlers.py`
- `teams/`
- `workflows/`
- `rl/`

Shim candidates:

- `__init__.py`
- `tool_dependencies.py`
- `tool_dependencies.yaml`

## Migration Implications

1. The first split in each vertical should be `assistant.py` plus `prompts.py`,
   because those are the only modules that belong in the definition package.
2. `__init__.py` and `tool_dependencies.py` should be preserved as compatibility
   shims until packaging/source-of-truth work in `VPC-E4`.
3. Workflow/team/capability/middleware stacks should not be forced into the SDK;
   they need runtime adapters or host-owned extension registration.
4. `coding` remains the highest-risk migration because it has the deepest runtime
   surface area and the highest measured coupling count.

## Next Tasks Unblocked

- `VPC-T3.2` define the target package layout for migrated verticals
- `VPC-T3.3` extract common runtime-only helpers out of definition modules
- `VPC-T3.5` inventory and classify `coding` imports by definition/runtime/shim layer
