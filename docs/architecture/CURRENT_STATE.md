# Victor Agent Runtime — Current State

**Last Updated**: 2026-05-05

**Scope**: This is the authoritative current-state document for the
`victor/agent` runtime architecture. Historical ADRs, migration guides, and
seam-by-seam audits remain useful for context, but they should not be treated
as the steady-state design.

## Executive Summary

- The runtime is service-first. The canonical effectful surfaces are
  `ChatService`, `ToolService`, `SessionService`, `ContextService`,
  `ProviderService`, and `RecoveryService`.
- `AgentOrchestrator` remains the composition root, session boundary, and
  compatibility hotspot. It is still large at 4,720 LOC and should continue to
  shrink.
- State-passed is a selective architectural pattern, not a blanket rewrite
  mandate. It is canonical for read-heavy, policy, and decision seams such as
  exploration, safety, system-prompt classification, and coordination
  recommendation.
- Facades remain as grouping and compatibility surfaces. They must not become
  behavior-owning layers.
- The parallel `victor/agent/services/chat/*` and
  `victor/agent/services/tools/*` trees were removed on 2026-05-01. The only
  canonical owners in those domains are `chat_service.py` and
  `tool_service.py`.
- Deprecated coordinator/property accessors still exist in a few places as
  explicit compatibility shims. Internal production code should not expand
  those seams.

## Verified Runtime Shape (2026-05-05)

```text
Agent / Public API
  -> AgentOrchestrator
     - composition root
     - lifecycle + session boundary
     - compatibility hotspot
     - not the canonical owner of business logic
       ->
       OrchestrationFacade
         - service access
         - deprecated coordinator shims
       Canonical Services
         - ChatService
         - ToolService
         - SessionService
         - ContextService
         - ProviderService
         - RecoveryService
       Selective State-Passed Boundaries
         - ExplorationStatePassedCoordinator
         - SystemPromptStatePassedCoordinator
         - SafetyStatePassedCoordinator
         - CoordinationStatePassedCoordinator
```

## Canonical Ownership

| Concern | Canonical owner | Notes |
|---------|-----------------|-------|
| Chat execution | `victor.agent.services.chat_service.ChatService` | Service-owned runtime |
| Tool execution | `victor.agent.services.tool_service.ToolService` | Service-owned runtime |
| Session lifecycle | `victor.agent.services.session_service.SessionService` | Service-owned runtime |
| Context management | `victor.agent.services.context_service.ContextService` | Service-owned runtime |
| Provider management | `victor.agent.services.provider_service.ProviderService` via `provider_management_runtime.py` | Provider runtime no longer owns provider coordinators |
| Recovery and resilience | `victor.agent.services.recovery_service.RecoveryService` | Service-owned runtime |
| Live conversation-stage state | `victor.agent.services.state_runtime.StateRuntimeAdapter` over `ConversationController` + `ConversationStateMachine` | No concrete `StateCoordinator` shim remains; only the deprecated protocol alias remains for compatibility |
| Conversation stage-transition batching | `victor.agent.services.stage_transition_runtime.StageTransitionCoordinator` | Internal service-owned runtime helper; coordinator path is compatibility only |
| Exploration decisions | `victor.agent.coordinators.ExplorationStatePassedCoordinator` | Selective state-passed seam; `TurnExecutor` now prefers the shared facade-owned state-passed surface and uses direct `ExplorationCoordinator` only as a no-orchestrator fallback |
| System-prompt / task classification | `victor.agent.coordinators.SystemPromptStatePassedCoordinator` | Selective state-passed seam |
| Safety checks | `victor.agent.coordinators.SafetyStatePassedCoordinator` | Selective state-passed seam |
| Coordination recommendation | `victor.agent.coordinators.CoordinationStatePassedCoordinator` | Selective state-passed seam |

## Compatibility and Historical Surfaces

- `victor/agent/facades/` is a grouping and property-access layer. It is not a
  canonical behavior layer.
- `OrchestrationFacade` now exposes only canonical services plus state-passed
  surfaces. The old `chat_coordinator`, `tool_coordinator`,
  `session_coordinator`, and related chat-shim properties were removed on
  2026-05-05 and must not be reintroduced.
- The deprecated chat-compat telemetry/reporting surface was also removed on
  2026-05-05. New code should not depend on shim-usage diagnostics for already
  deleted chat coordinator surfaces.
- `victor/agent/coordinators/` is a mixed package. Treat the state-passed
  modules as live architecture. Treat most remaining files there as
  compatibility seams, examples, protocols, or historical helpers unless
  verified otherwise.
- Stage-transition batching was moved to `victor/agent/services/` on
  2026-05-04. The coordinator modules remain compatibility re-export paths,
  but internal production code should import the service-owned runtime.
- Turn-level parallel exploration now prefers the shared
  `exploration_state_passed` surface when an orchestrator-backed runtime is
  available. The direct `ExplorationCoordinator` service runtime still exists,
  but only as a fallback for contexts that do not have an orchestrator
  snapshot to pass through the state-passed seam.
- `CoordinatorFactory.create_safety_coordinator()` and
  `CoordinatorFactory.create_conversation_coordinator()` are deprecated
  SDK-owned compatibility helpers. They are not canonical `victor/agent`
  runtime ownership surfaces.
- Package-level `victor.agent.coordinators` re-exports of SDK-owned safety and
  conversation symbols are also compatibility-only and now warn explicitly.
  New code should import those symbols from `victor_sdk.safety` or
  `victor_sdk.conversation` directly.
- The old concrete state shim modules are gone. `victor.agent.state_coordinator`
  and `victor.agent.services.state_compat` should not be reintroduced. The
  only remaining state-side compatibility surface is the deprecated
  `StateCoordinatorProtocol` alias in `victor.agent.protocols`.
- All deprecated prompt compatibility wrappers were removed on 2026-05-05.
  Internal runtime assembly no longer wires any separate prompt-support or
  prompt-coordinator object. New runtime code should use
  `UnifiedPromptPipeline` directly, and state-passed prompt classification
  should use `SystemPromptStatePassedCoordinator`.
- `PromptRuntimeAdapter` remains the canonical DI/runtime surface for
  `PromptRuntimeProtocol`, but its system-prompt build path now delegates to
  `UnifiedPromptPipeline`. The adapter still owns only the narrow mutable
  protocol state: task hints, extra sections, grounding mode, and safety
  rules.
- Live prompt optimization remains canonical on
  `victor.agent.prompt_pipeline.UnifiedPromptPipeline` for **all** provider
  tiers. GEPA/MiPROv2/CoT-distillation guidance, experiment-memory guidance,
  failure hints, and credit/reputation guidance should flow through the
  pipeline-backed per-turn prefix path; KV support only changes cache-stable
  prefix strategy and observability, not whether those optimizations run.
- Frozen prompt management is now explicit rather than session-blind. The live
  prompt runtime refreshes the stable system prompt when provider/model,
  mode-guidance, stable core-tool set, or `.victor/init.md` change. Query
  classification guidance for frozen tiers is injected through the per-turn
  user-prefix path instead of forcing a system-prompt rebuild.
- Tool prompting is split intentionally. Stable core-tool guidance stays in the
  system prompt; infrequently used or long-tail tool hints belong in the
  per-turn user-prefix path when the current request makes them relevant. This
  prompt-side split is separate from provider tool schemas and from KV/API
  cache mechanics.
- For cache-friendly/session-locked providers, dynamic long-tail tool hints
  should prefer the current planned tool sequence when available, then fall
  back to keyword-selected or selected-tool heuristics. The stable prefix stays
  fixed; only the per-turn hint block reflects the current plan. When planner
  goals or an active intent guard are available, the hint block should surface
  that compact rationale rather than emitting a generic tool list. When
  existing tool descriptions or metadata use-cases are available, the hint
  block may also surface concise per-tool reasons without triggering a second
  planning pass.
- Active runtime protocols now have service-owned canonical hosts under
  `victor.agent.services.protocols.runtime_support` and
  `victor.agent.services.protocols.infrastructure_runtime`, including
  coordination, state, prompt, chunk generation, streaming recovery, RL,
  intent classification, reminder management, response sanitization, and
  streaming handler/metrics/confidence seams. Legacy names in
  `victor.agent.protocols` such as `ToolPlannerProtocol`,
  `TaskCoordinatorProtocol`, `StateCoordinatorProtocol`,
  `PromptCoordinatorProtocol`, `ChunkGeneratorProtocol`,
  `StreamingRecoveryCoordinatorProtocol`, `RLCoordinatorProtocol`,
  `IntentClassifierProtocol`, `ReminderManagerProtocol`,
  `ResponseSanitizerProtocol`, `StreamingHandlerProtocol`,
  `StreamingMetricsCollectorProtocol`, and
  `StreamingConfidenceMonitorProtocol` are deprecated compatibility aliases
  only.
- Provider runtime ownership is singular. `provider_runtime.py` no longer owns
  `provider_coordinator` or `provider_switch_coordinator`, and the removed root
  shim modules must stay absent. `ProviderService` is the canonical provider
  authority.
- The older `USE_SERVICE_LAYER` rollout story is obsolete for the agent
  runtime. The remaining major runtime feature flag in this area is
  `USE_STATEGRAPH_AGENTIC_LOOP`, which belongs to the framework-side agentic
  loop path rather than service ownership in `victor/agent`.

## State-Passed Role

- Use state-passed when immutable snapshots and explicit transitions improve
  correctness, debuggability, and unit-test isolation.
- Keep effectful runtime operations service-owned. Chat, tool, provider,
  session, context, and recovery flows should not be rewritten into a second
  parallel abstraction layer.
- Do not move logic from the orchestrator into facades just to claim
  decomposition.
- Do not treat every legacy coordinator as a mandatory state-passed migration
  candidate. If a seam is already cleanly owned by a service, keep it there.

## Active Migration Priorities

1. Continue shrinking `AgentOrchestrator` toward composition, lifecycle, and
   compatibility only.
2. Demote remaining facades to grouping-only and compatibility-only roles.
3. Retire remaining internal dependencies on deprecated coordinator shims.
4. Preserve the distinction between live runtime state
   (`victor/agent/services/state_runtime.py`) and persisted vertical/project
   state (`victor/agent/state_service.py`).
5. Keep teams as formations on `StateGraph`; do not create a separate
   multi-agent graph abstraction.

## Explicit Non-Goals

- Reintroducing nested `services/chat/*` or `services/tools/*` trees
- Moving business logic into facades
- Creating a second first-class service hierarchy for the same concern
- Treating teams as a distinct graph runtime instead of `StateGraph` formations

## Historical References

- `docs/architecture/adr/001-agent-orchestration.md`: original
  coordinator-centric ADR, now historical
- `docs/architecture/state-passed-architecture.md`: reference for the selective
  state-passed pattern
- `docs/development/agent-facade-service-migration-audit-2026-04-26.md`:
  seam-by-seam migration evidence
