# Victor Agent Runtime — Current State

**Last Updated**: 2026-05-04

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

## Verified Runtime Shape (2026-05-04)

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
| Exploration decisions | `victor.agent.coordinators.ExplorationStatePassedCoordinator` | Selective state-passed seam |
| System-prompt / task classification | `victor.agent.coordinators.SystemPromptStatePassedCoordinator` | Selective state-passed seam |
| Safety checks | `victor.agent.coordinators.SafetyStatePassedCoordinator` | Selective state-passed seam |
| Coordination recommendation | `victor.agent.coordinators.CoordinationStatePassedCoordinator` | Selective state-passed seam |

## Compatibility and Historical Surfaces

- `victor/agent/facades/` is a grouping and property-access layer. It is not a
  canonical behavior layer.
- `victor/agent/coordinators/` is a mixed package. Treat the state-passed
  modules as live architecture. Treat most remaining files there as
  compatibility seams, examples, protocols, or historical helpers unless
  verified otherwise.
- Provider compatibility remains explicit. `provider_runtime.py` no longer owns
  `provider_coordinator` or `provider_switch_coordinator`; deprecated accessors
  now materialize compatibility shims on demand.
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
