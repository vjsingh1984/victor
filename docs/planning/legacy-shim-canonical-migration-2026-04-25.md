# Legacy Shim And Canonical API Migration Plan

Date: 2026-04-25

## Goal

Reduce internal dependence on deprecated or compatibility-only APIs while
preserving public compatibility shims where they still serve external callers.

The target state is:
- internal runtime code depends only on canonical service/state-passed/SDK APIs
- compatibility modules remain thin public adapters with minimal internal usage
- legacy field names remain accepted at configuration boundaries, but runtime
  code reads canonical nested configuration

## Audit Table

| Surface | Current role | Canonical API / module | Active internal usage | Priority | Action |
|---|---|---|---|---|---|
| Flat `Settings.use_semantic_tool_selection` | Legacy flat field mapped into nested config | `settings.tool_selection.use_semantic_tool_selection` | Runtime reads in `victor/agent/service_provider.py`, `victor/agent/factory/tool_builders.py`, `victor/agent/factory/coordination_builders.py` | P0 | Migrate internal reads to canonical nested access while leaving legacy input mapping at config boundary |
| `victor.core.tool_dependency_base` | SDK promotion shim | `victor_sdk.verticals.tool_dependencies` | Internal imports in `victor/core/tool_dependency_loader.py`, `victor/core/tool_types.py`, `victor/core/entry_points/tool_dependency_provider.py` | P0 | Migrate internal imports to SDK-owned canonical types; keep shim for public imports |
| `victor.tools` package-root re-exports | Deprecated convenience surface | Specific submodules like `victor.tools.registry`, `victor.tools.base` | Minor internal import in `victor/agent/response_processor.py` | P1 | Move internal imports to concrete tool modules |
| `victor.agent.provider_coordinator` | Deprecated root-level provider compatibility wrapper | `victor.agent.services.provider_service.ProviderService` plus provider runtime/service wiring | Active production imports in `victor/agent/orchestrator.py`, `victor/agent/runtime/provider_runtime.py`; unit tests still target wrapper directly | P0 | Remove internal dependence on the wrapper and collapse provider switching onto canonical service/runtime seams |
| `victor.agent.coordinators.provider_coordinator` | Deprecated coordinator-path provider wrapper | `ProviderService` / provider runtime | Compatibility surface with real logic, but not the canonical runtime path | P1 | After root provider wrapper migration, shrink or replace with a thin public shim |
| `victor.agent.provider_switch_coordinator` | Legacy provider-switch orchestration surface | Provider runtime/service-owned switching boundary | Still used in `victor/agent/factory/runtime_builders.py`, `victor/agent/orchestrator_factory.py`, `victor/agent/provider/__init__.py`, `victor/agent/post_switch_hooks.py` | P0 | Introduce a canonical service-owned switch boundary, then migrate runtime/factory imports off the coordinator module |
| `victor.agent.coordinators.recovery_controller` | Deprecated recovery controller | `victor.agent.services.recovery_service.RecoveryService` | Compatibility surface with lightweight real logic | P1 | Keep as external shim only; remove any remaining internal dependence on controller semantics |
| `victor.agent.coordinators` package exports | Deprecated coordinator compatibility package | `victor.agent.services.*`, `victor.agent.services.protocols.chat_runtime`, state-passed coordinators, `OrchestrationFacade.chat_stream_runtime` | Mostly tests/examples/docs; no meaningful production import dependence found | P1 | Leave public shim, migrate remaining non-test examples/docs and any future internal imports |
| `victor.agent.coordinators.chat_protocols` | Deprecated chat protocol shim | `victor.agent.services.protocols.chat_runtime` | Tests/compat only | P1 | Leave shim; no production callers should import it |
| Service-owned `*_compat.py` shims | Deprecated coordinator/sync adapter layer | `ChatService`, `ToolService`, `SessionService`, `PromptRuntimeProtocol`, `StateRuntimeProtocol`, `ServiceStreamingRuntime` | Still surfaced through orchestrator properties and facades | P1 | Reduce internal dependence on compat accessors; keep shim layer thin |
| Orchestrator/facade compatibility properties | Deprecated shim accessors for old coordinator surfaces | Direct DI/service/runtime access | Active in `victor/agent/orchestrator_properties.py`, `victor/agent/orchestrator.py`, `victor/agent/facades/orchestration_facade.py`, `victor/agent/runtime/bootstrapper.py` | P1 | Move internal callers to direct service/runtime seams; retain properties only for external compatibility |
| Legacy `_stream_chat_runtime` hook | Old fallback hook for older integrations | `ChatService.stream_chat` / `ServiceStreamingRuntime.stream_chat` | Only fallback path in `victor/agent/services/chat_compat.py` and orchestrator hook | P2 | Keep isolated as terminal fallback; do not allow new internal callers |
| `victor.workflows.executor` | Deprecated DAG executor module | `victor.workflows.unified_executor`, `victor.workflows.context`, `victor_sdk.workflows` runtime types | Heavy internal import usage across workflow runtime, framework, benchmark, and CLI modules | P0 | Multi-step migration: first move shared runtime types/helpers, then executor consumers |
| `victor.workflows.graph_compiler` | Legacy compiler surface | `victor.workflows.compiler.boundary`, `victor.workflows.unified_compiler` | Heavy internal import usage | P0 | Migrate call sites through compiler boundary/unified compiler; keep graph compiler as backend adapter only |
| `victor.workflows.yaml_to_graph_compiler` | Legacy compiler adapter surface | Compiler boundary / unified compiler entrypoints | Medium internal import usage | P1 | Move callers to boundary/unified surfaces after compiler API normalization |
| `victor.core.types` | Backward-compatible type re-export | `victor.core.vertical_types` | Low internal usage | P2 | Migrate any internal imports; keep public re-export |
| `victor.teams.protocols` | Re-export shim | `victor.protocols.team` | Tests/docs only | P2 | Leave shim for compatibility, avoid new internal imports |
| `victor.verticals.contrib.*` fallback namespace | Legacy plugin/import fallback | Standalone packages plus canonical entry points via import resolver | Fallback logic remains active in resolver/loader code | P1 | Keep fallback for external compatibility; avoid new direct imports and narrow loader ownership |
| `create_default_validated_session_truth_service(...)` | Older evaluation service factory alias | `victor.evaluation.services.create_validated_session_truth_service(...)` | Production code already migrated; tests still cover compatibility | Done | Keep alias only as compatibility coverage surface |

## Phase Plan

### Phase 1
- Migrate flat semantic-selection setting reads to canonical nested config.
- Migrate internal `victor.core.tool_dependency_base` imports to SDK-owned types.
- Remove remaining internal `victor.tools` package-root re-export usage.

### Phase 2
- Reduce internal dependence on coordinator compatibility exports and
  orchestrator compatibility properties.
- Keep `*_compat.py` modules public, but ensure runtime code reaches services
  and state-passed boundaries directly.
- Collapse the provider migration seam in the right order:
  1. migrate internal orchestrator switching off the coordinator fallback and
     onto `ProviderService` only
  2. migrate internal imports off `victor.agent.provider_coordinator`
  3. introduce a canonical provider-switch boundary to replace direct
     `provider_switch_coordinator` construction
  4. only then retire the coordinator-path provider shims

### Phase 3
- Migrate workflow runtime imports off `victor.workflows.executor` legacy types.
- Move compiler consumers to boundary/unified compiler APIs.

### Phase 4
- Clean up low-volume re-export shims (`victor.core.types`,
  `victor.teams.protocols`) and review remaining public-only shims for
  retirement milestones.

## TDD Strategy

For each batch:
1. Add focused tests that fail if runtime code reaches through the shim or
   legacy flat field.
2. Migrate internal call sites to the canonical module/API.
3. Run targeted regression for the touched seam.
4. Run a broader surrounding regression suite before commit.

## Current Batch

Phase 1 is complete. Phase 2.1, Phase 2.2, Phase 2.3, Phase 2.4, Phase 2.5, Phase 2.6, Phase 2.7, and Phase 2.8 are now complete:
- internal provider switching is service-first in `AgentOrchestrator`
- provider-switch hook contracts now live in a canonical provider contract
  module instead of being sourced from the legacy switch coordinator
- `ProviderSwitchCoordinator` behavior now lives in the canonical provider
  package, with the root module reduced to a public compatibility shim
- `ProviderCoordinator` behavior now lives in the canonical provider package
  module, with the root module reduced to a public compatibility shim and
  internal runtime imports moved to the canonical module path
- `victor.agent.coordinators.provider_coordinator` is now a thin
  compatibility adapter over the canonical provider coordinator rather than a
  second behavior owner
- simple `AgentOrchestrator` compatibility properties now resolve from direct
  canonical backing attributes instead of facades, so facades are no longer
  behavior owners for those runtime accessors
- lazy sync/streaming/unified deprecated chat coordinator wiring is now bound
  directly from bootstrapper to compatibility constructors, removing the last
  internal-only wrapper methods from `AgentOrchestrator` for those shims
- lazy deprecated chat/tool/session coordinator access in bootstrapper now
  binds directly to the backing compatibility slots, removing the remaining
  internal wrapper methods from `AgentOrchestrator` for those shims
- secondary compatibility facades (`provider`, `session`, `metrics`,
  `resilience`, `workflow`) now bootstrap lazily via `LazyRuntimeProxy`,
  reducing eager runtime compatibility construction while keeping the public
  facade attributes available
- `OrchestrationFacade` now also bootstraps lazily via `LazyRuntimeProxy`,
  so state-passed compatibility surfaces and deprecated coordinator getters
  are materialized only on demand instead of during orchestrator startup
- `ChatFacade` and `ToolFacade` now also bootstrap lazily via
  `LazyRuntimeProxy`, so every orchestrator facade is now a compatibility-only
  object that materializes on demand rather than at orchestrator startup
- internal runtime code no longer depends on any facade as a behavior owner;
  facades are now reduced to thin compatibility groupings over canonical
  service/runtime/state-passed seams

- Phase 3.1 compute handler registry extraction:
  - `victor.workflows.compute_registry` is now the canonical module for
    `ComputeHandler`, `register_compute_handler`, `get_compute_handler`,
    `list_compute_handlers`; executor.py re-exports from it with a shared
    `_compute_handlers` dict
  - NodeResult/ExecutorNodeStatus imported directly from victor_sdk.workflows
    in all benchmark callers
  - Boundary tests enforce canonical paths for all migrated consumers

- Phase 3.2 workflow context/result/temporal canonicalization:
  - `victor.workflows.context` is now the canonical module for
    `WorkflowContext`, `WorkflowResult`, and `TemporalContext`
  - executor.py re-exports those types for compatibility but no longer defines
    them, and `TemporalContext.get_date_range()` behavior is preserved in the
    canonical module
  - internal callers that previously imported those types directly from
    executor.py now import them from `victor.workflows.context`
  - workflow package exports now source those types from the canonical module
  - boundary and workflow regression tests enforce the canonical import paths

- Phase 3.3 workflow graph-compiler caller migration:
  - `victor.framework.coordinators.graph_coordinator` now compiles
    `WorkflowGraph` and `WorkflowDefinition` execution through the canonical
    `UnifiedWorkflowCompiler` path instead of importing legacy
    `victor.workflows.graph_compiler` execution-time entrypoints directly
  - `GraphTurnExecutor` now caches canonical unified compiler instances keyed
    by node-runner usage, preserving the `use_node_runners` execution contract
    without reviving direct legacy compiler ownership
  - `victor.framework.workflow_engine` no longer carries dead legacy
    `WorkflowGraphCompiler` / `WorkflowDefinitionCompiler` state and type
    imports; the workflow engine now treats `UnifiedWorkflowCompiler` as the
    only compiler boundary it owns directly

- Phase 3.4 workflow executor canonical boundary migration:
  - `victor.workflows.unified_executor.StateGraphExecutor` now compiles
    `WorkflowDefinition` execution through `NativeWorkflowGraphCompiler` from
    `victor.workflows.compiler.boundary` instead of routing through
    `victor.workflows.yaml_to_graph_compiler`
  - executor-local compiler caching is preserved for the default path, while
    per-call custom checkpointers now flow through a temporary canonical
    boundary compiler instead of mutating a legacy compiler config object
  - runtime behavior for iteration/timeout overlays and HITL interruption is
    preserved by overlaying executor config onto the definition before
    canonical compilation

Next resume point:
- Continue Phase 3 by migrating remaining internal
  `victor.workflows.yaml_to_graph_compiler` callers onto compiler boundary /
  unified compiler entrypoints, starting with the workflow UI command
  validation and dry-run call sites rather than public compatibility exports.
