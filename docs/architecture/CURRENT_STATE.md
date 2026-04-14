# Victor Architecture — Current State & Active Initiatives

**Last Updated**: 2026-04-14
**Supersedes**: `post-extraction-analysis.md` (Mar 7), `post-extraction-architecture-review-2026-03-16.md` (Mar 16), `post-extraction-analysis-2026-04-10.md` (Apr 10), `victor-post-extraction-analysis.md` (Apr 14)

---

## System Architecture

```
User Code
  │  agent = Agent.create(vertical="coding")
  │  result = await agent.run("fix the bug")
  ▼
victor-ai (Framework Core)
  ├── Agent (1,090 LOC) → Orchestrator (3,519 LOC) → 8 Domain Facades
  ├── Runtime Layer
  │   ├── ComponentAssembler (tools→conv→intelligence)
  │   ├── AgentRuntimeBootstrapper (facades→lifecycle)
  │   └── OrchestratorFactory (mixin builders)
  ├── CapabilityRegistry (stub/enhance pattern)
  ├── VerticalLoader (entry point discovery, cached)
  └── EventSystem (async + backpressure)
      ▼
victor-sdk (Contract Layer — zero framework dependencies)
  ├── VerticalBase (ABC) │ ProtocolRegistry │ Discovery API
      ▼  entry_points["victor.plugins"]
External Verticals
  ├── victor-coding │ victor-research │ victor-devops
  ├── victor-invest │ victor-rag      │ victor-dataanalysis
```

### Vertical Loading Sequence

```
1. Agent.create(vertical="coding")
2. → VerticalLoader.discover_verticals()          [entry_points scan, cached]
3. → VerticalLoader.load("coding")                [dynamic import]
4. → VerticalRuntimeAdapter.create(CodingVertical) [SDK→runtime bridge]
5. → _negotiate_manifest()                        [version + capability check]
6. → VerticalIntegrationPipeline                  [inject tools/prompts/middleware]
7. → CapabilityRegistry.register()                [enhanced providers replace stubs]
8. → Agent ready
```

### Runtime Assembly

```
OrchestratorFactory
  → ComponentAssembler.assemble_tools()         [cache/graph/registry/executor/selector]
  → ComponentAssembler.assemble_conversation()  [controller/ledger/pipeline/streaming]
  → ComponentAssembler.assemble_intelligence()  [RL/compactor/analytics/resilience]
  → AgentRuntimeBootstrapper.prepare_components() [checkpoint/workflow/vertical context]
  → AgentRuntimeBootstrapper.finalize()          [8 facades/lifecycle/protocol assert]
```

---

## Coordinator Decoupling (Current State)

Victor has THREE complementary patterns for coordinator design:

| Pattern | Used By | Best For |
|---------|---------|----------|
| **Dependency Injection** | ToolCoordinator, MetricsCoordinator, ExplorationCoordinator | Simple coordinators with clear deps |
| **Protocol-Based** | ChatCoordinator (via ChatOrchestratorProtocol + 3 sub-protocols) | Complex coordinators needing many orchestrator capabilities |
| **State-Passed** | ExampleStatePassedCoordinator (foundation) | New coordinators, high testability, pure functions |

All existing coordinators are already well-decoupled. State-passed is an additional pattern for new development.

---

## SDK Boundary

**Rule**: SDK MUST NOT import from `victor/` (framework). Verticals MUST only import from `victor-sdk`.

### Current Violations (from Apr 10 analysis)

| Vertical | Violation | Impact |
|----------|-----------|--------|
| victor-coding | `from victor.core.protocols import OrchestratorProtocol` | Core refactors break vertical |
| victor-research | `from victor.core.tool_dependency_loader import ...` | Same |
| victor-devops | `from victor.core.protocols import OrchestratorProtocol` | Same |
| victor-invest | `from victor.core.verticals.base import VerticalRegistry` | Same |
| victor-rag | `from victor.core.verticals.base import StageDefinition, VerticalBase` | Same + depends on victor-ai (not just SDK) |
| victor-dataanalysis | `from victor.core.verticals.protocols import ...` | Same + depends on victor-ai |

**Core→Vertical Violations**: 11 files in core import from `victor_coding` (extension_loader, coding_support, indexer, vertical_integration_adapter).

### Enforcement

- `scripts/check_imports.py` — layer rule: `config/ ← providers/ ← tools/ ← agent/ ← ui/`
- `tests/unit/sdk/test_contrib_import_boundaries.py` — contrib directory guard
- `tests/unit/sdk/test_sdk_contract_shapes.py` — SDK contract stability
- `make test-definition-boundaries` — SDK import boundary validation

---

## Service Layer (Strangler Fig Migration)

**Feature Flag**: `USE_SERVICE_LAYER`

### Status

| Phase | Status | Details |
|-------|--------|---------|
| Foundation | COMPLETE | Services created/registered in DI container by default |
| Delegation | COMPLETE | 16 delegation points: 4 chat + 5 tool + 3 session + 2 context + 2 provider |
| Service Resolution | COMPLETE | All 6 services resolved: Chat, Tool, Session, Context, Provider, Recovery |
| Validation | PENDING | Performance benchmarking (<5% impact), integration testing |
| Optimization | FUTURE | Remove coordinator fallback paths, achieve 2,000 LOC target |

### Delegation Pattern

```python
async def chat(self, user_message: str) -> CompletionResponse:
    if self._use_service_layer and self._chat_service:
        return await self._chat_service.chat(user_message)
    return await self._chat_coordinator.chat(user_message)
```

### Service-to-Coordinator Mapping

| Service | Coordinator(s) | Status |
|---------|----------------|--------|
| ChatService | ChatCoordinator (via adapter) | Delegating (3 methods) |
| ToolService | ToolCoordinator | Delegating (5 methods) |
| SessionService | SessionCoordinator | Delegating (4 methods) |
| ContextService | ContextCompactor | Delegating (2 methods) |
| ProviderService | ProviderCoordinator | Delegating (2 methods) |
| RecoveryService | RecoveryController | Resolved, delegation pending |

---

## Plugin-Based Tool Registration (COMPLETE)

SDK protocols for dynamic tool registration:

- **`ToolFactory`** — Lazy tool creation protocol
- **`ToolFactoryAdapter`** — Converts factories/instances to VictorPlugin
- **`ToolPluginHelper`** — Convenience: `from_instances()`, `from_factories()`, `from_module()`
- **ToolRegistry** gained `register_plugin()`, `discover_plugins()` methods

All 5 aligned external verticals use `VictorPlugin` from SDK. `victor-invest` still missing plugin registration.

---

## External Vertical SDK Alignment

| Package | VictorPlugin | SDK-Only Imports | Import Violations |
|---------|-------------|------------------|-------------------|
| victor-coding | YES | NO (uses victor.core) | OrchestratorProtocol |
| victor-devops | YES | NO (uses victor.core) | OrchestratorProtocol |
| victor-research | YES | NO (uses victor.core) | tool_dependency_loader |
| victor-rag | YES | NO (depends on victor-ai) | StageDefinition, VerticalBase |
| victor-dataanalysis | YES | NO (depends on victor-ai) | verticals.protocols |
| victor-invest | NO (missing plugin.py) | N/A | VerticalRegistry |

---

## Key Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Orchestrator LOC | 3,519-3,973 | <2,000 |
| SDK import violations (verticals) | 6/6 verticals (external repos) | 0 |
| Core→vertical imports | **0** (was 11 files) | 0 |
| Hardcoded vertical configs in core | **0** (was 5 dicts) | 0 |
| Service delegation points | **16/16** (was 12) | 16+ |
| Services resolved in orchestrator | **6/6** (was 4) | 6 |
| ExecutionContext | **Wired** (orchestrator + cleanup hooks) | Pass to all components |
| Global state guard | **Capped** (get_global_manager: state/ + runtime/ only) | Eliminate |
| Container singleton guard | **Capped** (25 non-infra calls) | Reduce via DI |
| Singleton guard | **Capped** (68 files) | Reduce incrementally |
| Cache registry | **Unified** (CacheRegistry with category invalidation) | All caches registered |
| Trace context | **Implemented** (contextvars propagation) | Wire into vertical loader |
| State-passed coordinators | **3** (Exploration, SystemPrompt, Safety) | Incremental |
| TDI completion | **37/37 COMPLETE** | All done |

---

## Scoring

### Decoupling Assessment

| Dimension | Score | Evidence |
|-----------|-------|----------|
| Package Separation | 9/10 | All 6 verticals in separate repos |
| Logic Isolation | 5/10 | Hardcoded config dicts in 5+ core files |
| Protocol Maturity | 8/10 | ISP-compliant focused protocols |
| Inversion of Control | 6/10 | Core→vertical and vertical→core imports |
| SDK Boundary | 6/10 | Defined but violated by all 6 verticals |
| Failure Isolation | 9/10 | Nested try/except, graceful degradation |
| **Overall** | **7.2/10** | Good structure; contract enforcement is the gap |

### Competitive Positioning

| Dimension (Weight) | Victor | LangGraph | CrewAI | LangChain | AutoGen |
|-------------------|--------|-----------|--------|-----------|---------|
| Orchestration (0.20) | 9 | 10 | 7 | 6 | 8 |
| Extensibility (0.20) | 9 | 7 | 6 | 8 | 7 |
| Multi-Agent (0.15) | 8 | 9 | 10 | 5 | 10 |
| Tooling (0.15) | 10 | 8 | 7 | 9 | 7 |
| Dev Experience (0.15) | 9 | 8 | 9 | 5 | 6 |
| Production (0.15) | 7 | 9 | 8 | 7 | 6 |
| **Weighted** | **8.7** | **8.7** | **7.7** | **6.8** | **7.5** |

**Victor's edge**: Vertical SDK plugin system + native Rust hot paths + 34 tool modules. Tied with LangGraph; stronger on extensibility, weaker on production cloud story.
