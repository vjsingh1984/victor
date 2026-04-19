# Victor Post-Extraction Architecture Analysis

**Date**: 2026-04-14
**Scope**: Core framework (victor-ai) + SDK (victor-sdk) + 6 external verticals
**Method**: Static analysis of source code across all repos

---

## Section A: Current-State Architecture Map

### A.1 Repository Topology

```
victor-ai (core)        ~57K LOC   Framework, orchestration, tools, providers, workflows
victor-sdk (bundled)     ~8K LOC   Zero-dep contract layer (VerticalBase, protocols, discovery)
victor-coding           45,433 LOC  Coding vertical (45+ tools, LSP, indexing, RL)
victor-invest           27,779 LOC  Investment research (multi-agent, XBRL, valuations)
victor-rag              10,798 LOC  RAG pipeline (ChromaDB, LanceDB, vector ops)
victor-devops            6,520 LOC  DevOps vertical (Docker, K8s, Terraform)
victor-dataanalysis      6,145 LOC  Data analysis (pandas, visualization)
victor-research          5,450 LOC  Web research (search, synthesis)
                       ─────────
                       ~162K LOC total
```

### A.2 Dependency Hierarchy

```
External Verticals (all 6)
    │ pip install dependency
    ├── victor-sdk>=0.7.0,<1.0  ← REQUIRED (zero-dep contract layer)
    │
    └── [optional-dependencies.runtime]
        └── victor-ai>=0.7.0,<1.0  ← OPTIONAL (rich framework features)

victor-ai (core)
    │ bundled
    └── victor-sdk/  ← mono-repo, independent semver
```

All verticals depend on `victor-sdk` as base; `victor-ai` is an optional runtime extra. This enables SDK-only testing and definition-layer validation without the full framework.

### A.3 Vertical Loading Sequence

```
1. CLI/API entry
     │
2. ensure_bootstrapped(settings, vertical)     [core/bootstrap.py]
     │
3. bootstrap_container()                        [core/bootstrap.py]
     ├── Register 19+ services in DI container  [core/bootstrap_services.py]
     ├── PluginRegistry.discover()              [core/plugins/registry.py]
     │     └── scan entry_points("victor.plugins")
     │           └── CodingPlugin.register(context)
     │                 └── context.register_vertical(CodingAssistant)
     └── _register_vertical_services()
           ├── VerticalLoader.load(name)        [core/verticals/vertical_loader.py]
           │     ├── VerticalRegistry.get(name)  (built-in first)
           │     ├── _import_from_entrypoint()   (fallback)
           │     ├── _negotiate_manifest()       (API version + capability check)
           │     └── _activate()                 (set as active)
           └── activate_vertical_services()
                 └── VerticalExtensionLoader.get_extensions()
4. OrchestratorFactory → ComponentAssembler
     ├── assemble_tools()
     ├── assemble_conversation()
     └── assemble_intelligence()
5. AgentRuntimeBootstrapper.prepare_components()
     └── _initialize_services()  [orchestrator.py:636]
           ├── _register_coordinators_for_services()
           ├── _bootstrap_service_layer()
           ├── Resolve 6 services (Chat/Tool/Session/Context/Provider/Recovery)
           └── _create_execution_context()  → ExecutionContext
6. Agent ready — 16 delegation points active
```

### A.4 Entry Point Groups (22 registered)

| Group | Purpose | Used By |
|-------|---------|---------|
| `victor.plugins` | Canonical plugin bootstrap | All 6 verticals |
| `victor.tool_dependencies` | Tool execution patterns | coding, research, devops, rag, dataanalysis |
| `victor.safety_rules` | Safety rule providers | coding, research, devops, dataanalysis |
| `victor.prompt_contributors` | System prompt sections | coding, research, devops |
| `victor.mode_configs` | Mode configuration | coding, research, devops |
| `victor.workflow_providers` | Workflow definitions | coding, research, devops |
| `victor.capability_providers` | Capability configs | coding, research, devops, rag, dataanalysis |
| `victor.team_spec_providers` | Team formations | coding, research, devops, rag, dataanalysis |
| `victor.sdk.protocols` | Protocol implementations | All except invest |
| `victor.sdk.capabilities` | SDK capabilities | coding |
| `victor.sandbox.providers` | Sandboxed execution | coding, devops |
| `victor.permission.providers` | Permission models | coding, devops |
| `victor.hook.providers` | Lifecycle hooks | coding, devops |
| `victor.compaction.providers` | Context compaction | coding |
| `victor.commands` | CLI commands | coding |
| `victor.api_routers` | API routes | coding |
| `victor.service_providers` | Service providers | coding |
| `victor.escape_hatches` | Escape hatch defs | (none currently) |
| `victor.capabilities` | Capability defs | (none currently) |
| `victor.chunking_strategies` | Chunking providers | (none currently) |

**Integration richness gradient**: coding (23 entry points) → devops/research (12-15) → rag/dataanalysis (4-8) → invest (1)

### A.5 Service Layer Architecture

```
AgentOrchestrator (4,077 LOC)
  │
  ├── USE_SERVICE_LAYER flag (Strangler Fig)
  │
  ├── 16 delegation points:
  │   ├── ChatService:     chat(), stream_chat(), chat_with_planning()
  │   ├── ToolService:     execute_tool_with_retry(), get_available/enabled_tools(),
  │   │                    set_enabled_tools(), is_tool_enabled()
  │   ├── SessionService:  save/restore_checkpoint(), get_recent_sessions(), get_session_stats()
  │   ├── ContextService:  check_context_overflow(), get_context_metrics()
  │   └── ProviderService: get_current_provider_info(), switch_provider()
  │
  └── ExecutionContext (created post-bootstrap):
      ├── ServiceAccessor (lazy resolution for 6 services)
      ├── CacheRegistry (unified invalidation by category)
      ├── TraceContext (contextvars propagation for cross-boundary debugging)
      └── Cleanup hooks (LIFO for long-running session resources)
```

---

## Section B: Findings (Ordered by Severity)

### B.1 MEDIUM: Orchestrator Still 4,077 LOC

**Location**: `victor/agent/orchestrator.py`

The orchestrator maintains dual code paths (service layer + coordinator fallback) during the Strangler Fig migration. All 16 delegation points are wired but the coordinator paths remain. This doubles the maintenance surface.

**Impact**: Higher cognitive load, risk of path drift where service and coordinator behaviors diverge.

**Remedy**: Once service layer is validated via SVC-2 (runtime benchmarks), remove coordinator fallback paths. Target: <2,000 LOC. Gated by major version bump.

### B.2 MEDIUM: victor-invest Architectural Divergence

**Location**: `victor-invest/` — 1 entry point vs. 23 for coding

victor-invest uses an older architecture pattern:
- Minimal plugin integration (only `victor.plugins`)
- Own `framework_bootstrap.py` for Agent creation
- `src/investigator/` namespace alongside `victor_invest/`
- Multi-agent system with dedicated handlers, not using standard entry points
- Missing: tool_dependencies, safety_rules, capability_providers, team_spec_providers

**Impact**: invest cannot leverage framework tool selection, safety enforcement, capability negotiation, or team coordination.

**Remedy**: Phased migration (see Section D, Phase 3).

### B.3 LOW: Framework Imports in Verticals (Deferred, Not Module-Level)

All 6 verticals import from `victor.framework.*` (55+ in coding, 28+ in rag). These are:
- **Deferred**: Inside method bodies, not module-level
- **Guarded**: Only execute when `victor-ai` is installed
- **Through shim**: `victor.framework.extensions` re-exports SDK types

**Impact**: Functional but creates implicit coupling. If `victor.framework` API changes, all verticals break at runtime.

**Remedy**: Promote the most-used framework re-exports to `victor_sdk` proper (teams, safety types, workflow base classes). Track import counts per release.

### B.4 LOW: 68 Singleton Files in Core

**Location**: `victor/` — 68 files with `_instance: Optional` pattern

Guard test caps at 68. Five autouse test fixtures reset the critical ones. But singleton proliferation creates hidden state dependencies and complicates parallel test execution.

**Impact**: Test isolation risk, memory leak potential in long-running processes.

**Remedy**: The `ExecutionContext` pathway is in place (GS-1/GS-2 complete). Incrementally migrate singletons to DI as code is touched. Guard test prevents new singletons.

### B.5 INFO: Config Registry Fully Decoupled

**Location**: `victor/core/verticals/config_registry.py:85-110`

`_provider_hints` and `_evaluation_criteria` contain only `"default"` fallback. Vertical-specific configs registered dynamically via `register_vertical_config()`. `GroundingRules` uses `register_addendum()` with empty `_grounding_addendums`. `VERTICAL_CORES`/`VERTICAL_READONLY_DEFAULTS` are empty deprecated dicts.

**Status**: CLEAN. No hardcoded vertical knowledge in core.

### B.6 INFO: SDK Boundary Enforced via AST Guards

**Location**: `scripts/check_imports.py` Rule 4 + `tests/unit/sdk/test_core_vertical_import_boundary.py`

Core has zero `from victor_coding`/`import victor_coding` statements (static or dynamic). Enforced by AST-walking guard tests that fail CI if violations introduced.

**Status**: CLEAN. Zero violations.

---

## Section C: Target Architecture

### C.1 Contract-First Plugin Model (Current State = Target)

The current architecture already implements the target model:

```
External Vertical
  │ depends ONLY on victor-sdk
  │
  ├── entry_points["victor.plugins"] → VictorPlugin.register(context)
  ├── entry_points["victor.sdk.protocols"] → protocol implementations
  ├── entry_points["victor.tool_dependencies"] → tool specs
  ├── entry_points["victor.safety_rules"] → safety rules
  ├── entry_points["victor.capability_providers"] → capabilities
  └── ExtensionManifest → declares API version + deps
  │
victor-sdk (stable contract — zero deps)
  │ defines: VerticalBase, protocols, discovery, types
  │
victor-ai (runtime)
  │ consumes: entry points, manifests, protocols
  │ provides: orchestration, tools, events, RL
  └── NEVER imports from external verticals (AST-enforced)
```

### C.2 Remaining Architecture Targets

| Target | Current State | Gap |
|--------|--------------|-----|
| Orchestrator <2,000 LOC | 4,077 LOC (dual paths) | Remove coordinator fallback after validation |
| Zero framework imports in verticals | 55+ deferred imports in coding | Promote top-20 types to SDK |
| All verticals on new architecture | invest divergent (1 entry point) | Migrate invest to full entry point model |
| Singleton count <40 | 68 files | Migrate to DI as code is touched |
| SDK v1.0 stability guarantee | v0.7.0 | Finalize protocol shapes, publish changelog |

### C.3 Stable Extension SDK Shape

The SDK (`victor-sdk`) provides:

**Protocols** (26 protocol definition files in `victor_sdk/verticals/protocols/`):
- `VerticalBase` (ABC) — 4 required methods: `get_name`, `get_description`, `get_tools`, `get_system_prompt`
- `VictorPlugin` — `name`, `register(context)`, lifecycle hooks, health check
- 13 focused provider protocols: Tool, Safety, Prompt, Workflow, Team, Middleware, Mode, RL, Enrichment, Service, Handler, Capability, MCP
- Promoted types from core: `promoted.py` + `promoted_types.py` (zero-dep re-exports)
- Storage protocols: Graph, Vector, Embedding
- Config protocols: Settings, ApiKey

**Lifecycle**: Bootstrap (plugin.register) → Activation (get_extensions) → Runtime (protocol methods) → Teardown (on_deactivate)

**Discovery**: `ProtocolRegistry` with single-pass `importlib.metadata.entry_points()` scan, cached globally.

---

## Section D: Phased Roadmap

### Phase 0: Foundation — COMPLETE ✅

All items delivered:
- SDK boundary enforcement (AST guard tests, check_imports.py Rule 4)
- Config registry decoupled (default-only, dynamic registration)
- Core→vertical imports eliminated (0 static, 0 dynamic)
- Service layer foundation (6 services, 16 delegation points)
- ExecutionContext + CacheRegistry + TraceContext in `victor/runtime/`
- 3 state-passed coordinator migrations
- Guard tests preventing regression (6 test files, 37 TDI items complete)

### Phase 1: Service Layer Validation (1-2 weeks)

**Goal**: Validate service layer parity with coordinator path.

| Step | Action | Exit Criteria |
|------|--------|---------------|
| 1a | Runtime performance benchmark (SVC-2) | <5% overhead vs coordinator |
| 1b | Integration test with real workloads | Feature parity confirmed |
| 1c | Monitor production metrics for 1 week | Error rate unchanged |

### Phase 2: Coordinator Path Removal (2-3 weeks)

**Goal**: Remove dual paths, achieve <2,000 LOC orchestrator.

| Step | Action | Exit Criteria |
|------|--------|---------------|
| 2a | Remove coordinator fallback in 16 delegation methods | Single code path |
| 2b | Remove `USE_SERVICE_LAYER` flag | Flag deleted |
| 2c | Simplify error handling (single path) | LOC < 2,000 |
| 2d | Major version bump (breaking change) | v1.0.0 |

**Compatibility**: External verticals unaffected (they don't reference orchestrator internals).

### Phase 3: victor-invest Migration — COMPLETE ✅

**Delivered**:
- Created `tool_dependencies.py` + `tool_dependencies.yaml` (YAML-based, matching other verticals)
- Created `prompts/contributor.py` with `InvestmentPromptContributor` (4 task type hints)
- Added `create_investment_safety_rules()` factory to `safety_enhanced.py`
- Wired 5 entry point groups in `pyproject.toml` (was 1, now 5: plugins, tool_dependencies, safety_rules, prompt_contributors, workflow_providers)
- Created `test_sdk_boundary_contract.py` with 6 boundary validation tests (all passing)

### Phase 4: SDK v1.0 Stabilization (ongoing)

**Goal**: Publish SDK v1.0.0 with stability guarantee.

| Step | Action | Exit Criteria |
|------|--------|---------------|
| 4a | Audit all `victor_sdk` public API for stability | Breaking changes documented |
| 4b | Promote top-20 framework re-exports to SDK | `from victor.framework` usage reduced 50% |
| 4c | Publish migration guide for external developers | Guide published |
| 4d | Tag `sdk-v1.0.0` | Semver stability contract |

---

## Section E: Score Tables

### E.1 Decoupling Assessment

| Dimension | Score (1-10) | Evidence |
|-----------|-------------|----------|
| Package Separation | 9 | 6 verticals in separate repos, independent versioning |
| Import Direction | 9 | Zero core→vertical imports (AST-enforced); vertical→core only via deferred/shim |
| Protocol Maturity | 8 | 26 protocol definitions in SDK; ISP-compliant focused interfaces |
| Configuration Decoupling | 10 | Zero hardcoded vertical configs in core; dynamic registration only |
| Failure Isolation | 9 | Nested try/except on all vertical/plugin loading; graceful degradation |
| Inversion of Control | 8 | Entry points + CapabilityRegistry; slight gap from framework shim imports |
| Test Isolation | 7 | 6 guard test files, 5 autouse reset fixtures; 68 singletons still present |
| SDK Boundary | 9 | Contract shape tests, boundary AST tests, zero-dep SDK, promoted types |
| **Overall** | **8.6** | Strong post-extraction architecture; main gaps are orchestrator LOC and invest divergence |

### E.2 Comparative Positioning

**Scoring Weights**: Orchestration (0.20), Extensibility (0.20), Multi-Agent (0.15), Tooling (0.15), Dev Experience (0.15), Production Readiness (0.15)

| Dimension (Weight) | Victor | LangGraph | CrewAI | LangChain | LlamaIndex | AutoGen |
|---------------------|--------|-----------|--------|-----------|------------|---------|
| **Orchestration** (0.20) | 9 — StateGraph + workflow + service layer | 10 — purpose-built graph | 7 — sequential/hierarchical | 6 — LCEL chains | 7 — pipeline | 8 — conversation |
| **Extensibility** (0.20) | 9 — SDK + 22 entry point groups + 26 protocols | 7 — hooks/callbacks | 6 — limited | 8 — large ecosystem | 6 — data-focused | 7 — skills |
| **Multi-Agent** (0.15) | 8 — Teams API, 4 formations, team specs | 9 — hierarchical + parallel | 10 — native crew | 5 — manual | 5 — agent-focused | 10 — native |
| **Tooling** (0.15) | 10 — 34 modules + tree-sitter + LSP + Rust | 8 — MCP | 7 — standard | 9 — huge ecosystem | 7 — standard | 7 — standard |
| **Dev Experience** (0.15) | 9 — CLI + TUI + VS Code + 46 commands | 8 — LangSmith | 9 — simple API | 5 — complex abstractions | 7 — good docs | 6 — research |
| **Production** (0.15) | 8 — circuit breaker + RL + CacheRegistry + TraceContext | 9 — LangSmith Cloud | 8 — reliable | 7 — bloated deps | 7 — good indexing | 6 — fragile |
| **Weighted Total** | **8.8** | **8.7** | **7.7** | **6.8** | **6.5** | **7.5** |

**Victor's edge**: Deepest extensibility model (22 entry point groups vs. hooks/callbacks elsewhere) + strongest native tooling (34 modules, Rust hot paths). Production story improved significantly with CacheRegistry, TraceContext, and guard tests. Tied/ahead of LangGraph; clear lead on extensibility.

**Victor's gap**: No managed cloud service (LangGraph has LangSmith). Orchestrator LOC is high (being addressed via Phase 2). Multi-agent lags behind CrewAI/AutoGen on ease-of-use (mitigated by Teams API).
