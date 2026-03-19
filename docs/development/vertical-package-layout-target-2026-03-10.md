# Target Package Layout For Migrated Verticals (2026-03-10)

Purpose: define the target file/package layout for `VPC-T3.2` so vertical
migrations follow one repeatable structure instead of inventing a custom split per
vertical.

## Scope

This layout applies to:

- extracted vertical packages such as `victor_coding` and `victor_research`
- bundled contrib verticals while they are being split into definition and runtime
  layers
- compatibility shims that remain temporarily in `victor-ai`

It does not require packaging removals yet. That remains `VPC-E4`.

## Design Goals

1. Keep the definition layer importable with `victor-sdk` only.
2. Minimize import-path churn for existing vertical users.
3. Make runtime-only modules obvious by location.
4. Preserve temporary compatibility surfaces without letting them remain the source
   of truth.

## Chosen Layout

The definition layer stays at the package root. Runtime-heavy modules move under a
dedicated `runtime/` package.

This is preferred over adding a new `definition/` package because the current
ecosystem already treats `assistant.py` as the canonical vertical entrypoint. Keeping
that module at the root reduces migration churn for imports, entry points, docs, and
tests.

## Authoritative Extracted Package Layout

```text
src/victor_<vertical>/
    __init__.py
    assistant.py
    prompts.py
    victor-vertical.toml
    tool_dependencies.py
    runtime/
        __init__.py
        capabilities.py
        middleware.py
        mode_config.py
        safety.py
        safety_enhanced.py
        service_provider.py
        conversation_enhanced.py
        enrichment.py
        handlers.py
        escape_hatches.py
        tool_dependencies.py
        workflows/
            __init__.py
            *.yaml
            provider.py
        teams/
            __init__.py
            personas.py
            specs.py
        rl/
            __init__.py
        domain/
            ...
```

Notes:

- `assistant.py` is the SDK-only `VerticalBase` subclass and the package entry point.
- `prompts.py` contains serializable prompt templates, task hints, and related
  helpers used by `assistant.py`.
- `victor-vertical.toml` remains package metadata and may later converge with the
  manifest schema.
- `tool_dependencies.py` at the package root is a temporary shim when old imports
  must remain stable. New runtime code should import from `runtime.tool_dependencies`.
- `runtime/` is the only place where `victor-ai` runtime imports are expected.

## Layer Responsibilities

| Path | Layer | Allowed dependencies | Notes |
|---|---|---|---|
| `assistant.py` | Definition | `victor_sdk`, stdlib, package-local definition modules | No `victor.framework`, no `victor.core.verticals`, no runtime providers |
| `prompts.py` | Definition | `victor_sdk`, stdlib | Prefer plain data/serializable helpers over runtime contributor objects |
| `victor-vertical.toml` | Definition metadata | None | Keep package identity/version metadata here until manifest convergence |
| `runtime/*` | Runtime | `victor-ai`, `victor_sdk`, stdlib, vertical-local modules | Owns workflow providers, teams, safety, middleware, handlers, and domain engines |
| `__init__.py` | Shim/export surface | package-local imports only | Re-export canonical symbols; avoid side effects and registry work |
| `tool_dependencies.py` | Shim | package-local imports only | Thin wrapper over `runtime.tool_dependencies` until removals are scheduled |

## Transition Layout Inside `victor-ai`

Until `VPC-E4` removes bundled contrib copies, the in-repo contrib package should
move toward a shim-first shape:

```text
victor/verticals/contrib/<vertical>/
    __init__.py
    assistant.py
    prompts.py
    tool_dependencies.py
    runtime/
        ...
```

Transition rules:

- `assistant.py` and `prompts.py` should become the first SDK-only modules.
- Runtime-heavy current modules should move under `runtime/` even before extraction
  to external packages.
- Root `__init__.py` should become a narrow compatibility export, not a runtime
  assembly point.
- Root `tool_dependencies.py` should delegate to `runtime.tool_dependencies`.

## Mixed-Mode Runtime Resolution

During migration, runtime loaders should treat package-root modules as temporary
shims, not the source of truth.

For runtime-owned module families such as `capabilities`, `handlers`, `teams`,
`workflows`, `safety`, and `escape_hatches`, the shared resolver now uses this
lookup order within each package namespace:

1. `runtime.<module>`
2. package-root `<module>` shim

Applied across namespaces, the effective order is:

1. `victor_<vertical>.runtime.<module>`
2. `victor_<vertical>.<module>`
3. `victor.<vertical>.runtime.<module>`
4. `victor.<vertical>.<module>`
5. `victor.verticals.contrib.<vertical>.runtime.<module>`
6. `victor.verticals.contrib.<vertical>.<module>`

Definition-layer modules such as `assistant.py` and `prompts.py` do not use the
runtime-first lookup and should remain rooted at the package top level.

## Module Family Mapping

| Current family | Target location | Rationale |
|---|---|---|
| `assistant.py` | root `assistant.py` | Preserve canonical entrypoint and entry-point references |
| `prompts.py` | root `prompts.py` | Keep definition metadata adjacent to the assistant |
| `capabilities.py` | `runtime/capabilities.py` | Runtime capability registration |
| `middleware.py` | `runtime/middleware.py` | Runtime execution concern |
| `mode_config.py` | `runtime/mode_config.py` | Runtime registration concern |
| `safety.py`, `safety_enhanced.py` | `runtime/` | Runtime enforcement/coordinator concern |
| `service_provider.py` | `runtime/service_provider.py` | DI/runtime-only |
| `conversation_enhanced.py`, `enrichment.py`, `handlers.py`, `escape_hatches.py` | `runtime/` | Runtime orchestration/helpers |
| `workflows/*` | `runtime/workflows/*` | YAML providers and execution glue are runtime concerns |
| `teams/*` | `runtime/teams/*` | Multi-agent runtime concern |
| `rl/*` | `runtime/rl/*` | Runtime optimization/config concern |
| Deep domain engines (`codebase/*`, `tools/*`, `ui/*`, etc.) | `runtime/domain/...` or `runtime/<existing-subpackage>/...` | Operational behavior, not SDK manifest data |
| `tool_dependencies.py` | root shim + `runtime/tool_dependencies.py` | Preserve imports while moving ownership to runtime |

## Package-Root Rules

Package-root modules should be boring:

- `__init__.py` should re-export the vertical class and a minimal supported surface.
- Do not perform registry mutation, team registration, or workflow loading in
  `__init__.py`.
- Do not import runtime-heavy modules from `assistant.py`.
- Keep all package-root imports acyclic and cheap enough for definition-only smoke
  tests.

## Testing Layout

Use the same split in tests:

```text
tests/
    unit/
        definition/
            test_<vertical>_assistant.py
            test_<vertical>_prompts.py
        runtime/
            test_<vertical>_workflows.py
            test_<vertical>_teams.py
            test_<vertical>_tool_dependencies.py
    integration/
        test_<vertical>_activation.py
        test_<vertical>_parity.py
```

Required verification for a migrated vertical:

1. Definition-only import test for `assistant.py`.
2. Forbidden-import scan over `assistant.py` and `prompts.py`.
3. Runtime activation test through Victor discovery/application.
4. Parity check that key tools/prompts/stages remain equivalent.

## Why This Unblocks The Next Work

This layout gives later tasks a concrete destination:

- `VPC-T3.3` can move shared runtime-only helpers into `runtime/`
- `VPC-T3.4` can define temporary adapters around the root shim files
- `VPC-T3.5+` can migrate each vertical against the same package contract

## Follow-On Changes Expected

- Update docs/examples/templates to teach this layout consistently.
- Migrate `coding` first because it has the highest measured runtime coupling.
- Once extraction is authoritative, reduce bundled contrib packages to shims and
  eventually remove them in `VPC-E4`.
