# Victor Post-Extraction Architecture Analysis

**Date:** 2026-03-07
**Scope:** Core framework + 4 external verticals (coding, research, devops, invest)
**Methodology:** Static analysis of source code across all repos

---

## Section A: Current-State Architecture Map

### A.1 Repository Topology

```
victor (core)          ~45K LOC   Framework, orchestration, tools, providers, workflows
victor-coding          ~42K LOC   Coding assistant vertical (108 files)
victor-research        ~3.2K LOC  Research assistant vertical
victor-devops          ~8K LOC    DevOps assistant vertical
victor-invest          ~60K LOC   Investment analysis vertical (incl. legacy engine)
```

### A.2 Core Framework Module Map

```
victor/
  framework/           Public API surface (Agent, Task, State, Events, ToolSet)
    __init__.py         680 lines; 164 lazy-loadable names via PEP 562 __getattr__
    agent.py            Agent class (facade for AgentOrchestrator)
    config.py           AgentConfig dataclass
    events.py           EventType enum + AgentExecutionEvent
    protocols.py        OrchestratorProtocol, ProviderProtocol, ToolsProtocol, etc.
    graph.py            StateGraph (LangGraph-compatible), CompiledGraph (~1500 LOC)
    state.py            Observable State wrapper over orchestrator
    capabilities/       StageBuilderCapability, GroundingRulesCapability

  core/
    verticals/
      base.py           VerticalBase (ABC), VerticalRegistry, VerticalConfig (~900 LOC)
      protocols/        15 @runtime_checkable Protocol interfaces (ISP-compliant)
      metadata.py       VerticalMetadataProvider mixin
      extension_loader.py  VerticalExtensionLoader mixin (1,897 LOC)
      extension_module_resolver.py  ExtensionModuleResolver (module resolution)
      extension_cache_manager.py    ExtensionCacheManager (cache lifecycle)
      capability_negotiator.py      CapabilityNegotiator (protocol negotiation)
      vertical_loader.py   load_vertical(), get_active_vertical()
    vertical_types.py   StageDefinition, TaskTypeHint, TieredToolConfig, StageBuilder
    tool_dependency_loader.py  YAMLToolDependencyProvider
    feature_flags.py    Phase-based rollout flags
    registry/base.py    BaseRegistry[K, V] generic template

  agent/
    orchestrator.py     AgentOrchestrator facade (3,940 LOC total, 21 coordinators + 8 runtime boundaries)
    orchestrator_properties.py  OrchestratorPropertyFacade
    callback_coordinator.py     CallbackCoordinator
    runtime/
      initialization_manager.py InitializationPhaseManager
    coordinators/       ConversationController, ToolPipeline, StreamingController, etc.
    conversation_state.py  ConversationStage enum (canonical stage source)
    shared_tool_registry.py  SharedToolRegistry singleton (double-checked locking)

  providers/
    base.py             BaseProvider ABC, Message, CompletionResponse, StreamChunk (~500 LOC)
    registry.py         ProviderRegistry with lazy materialization (~500 LOC)
    (24 provider modules, all lazy-registered)

  tools/
    base.py             BaseTool ABC, ToolResult, ToolConfig (~700 LOC)
    registry.py         ToolRegistry with schema caching + hooks (~625 LOC)
    enums.py            CostTier (FREE/LOW/MEDIUM/HIGH)
    metadata.py         ToolMetadata, ToolMetadataRegistry

  workflows/
    unified_executor.py StateGraphExecutor
    definition.py       Workflow DSL types

  state/                4-scope state managers (workflow, conversation, team, global)
  security/safety/      InfrastructureScanner, SecretScanner, SafetyPattern
  config/               Settings with 16 nested config groups (flat-access deprecated), provider_config_registry.py
  ui/                   CLI (Typer) + TUI (Textual)
```

### A.3 Vertical Loading Sequence

```
1. Framework import
   victor.framework.__init__ loads core exports eagerly;
   164 names deferred via __getattr__ + _NAME_TO_MODULE reverse map

2. Vertical discovery (lazy, on first access)
   VerticalRegistry._ensure_registration()
     -> _registration_done flag (module-level, one-shot)
     -> importlib.metadata.entry_points(group="victor.verticals")
     -> For each entry point: load class, validate, register
     -> _external_discovered = True

3. Vertical activation
   load_vertical(name) or get_active_vertical()
     -> VerticalRegistry.get(name) -> Type[VerticalBase]
     -> vertical.get_config(use_cache=True)
       -> Check _config_cache with TTL (300s default, RLock-protected)
       -> Cache miss: assemble VerticalConfig from get_tools(), get_system_prompt(),
          get_stages(), get_provider_hints(), get_evaluation_criteria()
       -> customize_config() hook for final adjustments

4. Extension loading (lazy, per-vertical)
   vertical.get_extensions(strict=False)
     -> Probes isinstance() against 15 @runtime_checkable protocols
     -> Populates VerticalExtensions dataclass
     -> Cached with state tracking, clearable via clear_extension_cache()

5. Multi-entry-point discovery (independent groups)
   "victor.safety_rules"        -> Safety patterns per vertical
   "victor.tool_dependencies"   -> YAML-based tool graphs
   "victor.framework.teams.providers" -> Team specs
   "victor.api_routers"         -> FastAPI route providers
   "victor.escape_hatches"      -> Workflow condition/transform functions
```

### A.4 Execution Path (Agent.run)

```
Agent.create(vertical="coding", provider="anthropic")
  -> AgentOrchestrator.__init__()
     -> ProviderManager: lazy-materialize provider from ProviderRegistry
     -> ToolRegistrar: register tools from VerticalConfig.tools
     -> ConversationController: initialize state machine
     -> StreamingController: session lifecycle

Agent.run(task)  /  Agent.stream(task)
  -> AgentOrchestrator.chat() or .stream()
     -> ConversationController.add_message()
     -> ToolPipeline.resolve_tools()
        -> SharedToolRegistry.get_tool_schemas(only_enabled=True)  [CACHED]
     -> Provider.chat() or .stream()
     -> ToolPipeline.execute(tool_name, args)
        -> Before-hooks -> Tool lookup O(1) -> Validate -> Execute -> After-hooks
     -> StreamingCoordinator: chunk aggregation, event emission
     -> Stage transition via ConversationStateMachine
```

### A.5 Data/Control Flow Across Repos

```
                    +-------------------+
                    |   External User   |
                    +--------+----------+
                             |
                    +--------v----------+
                    |  victor/ui/cli    |
                    |  victor/ui/tui    |
                    +--------+----------+
                             |
                    +--------v----------+
                    | victor/framework/ |  Agent, Task, State, Events
                    |   Agent.create()  |
                    +--------+----------+
                             |
              +--------------+--------------+
              |                             |
   +----------v-----------+    +-----------v-----------+
   | victor/agent/         |    | victor/core/verticals/ |
   | AgentOrchestrator     |<---| VerticalRegistry       |
   | ToolPipeline          |    | VerticalBase subclass  |
   | ConversationController|    | VerticalExtensions     |
   +----------+------------+    +-----------+-----------+
              |                             |
   +----------v-----------+    +-----------v-----------+
   | victor/providers/     |    | external vertical pkg  |
   | ProviderRegistry      |    | (victor-coding, etc.)  |
   | BaseProvider impls    |    | tools, workflows, RL   |
   +----------+------------+    | safety, teams, prompts |
              |                 +-----------+-----------+
   +----------v-----------+                |
   | victor/tools/         |<--- registers tools via entry points
   | ToolRegistry          |     + VerticalConfig.tools list
   | BaseTool impls        |
   +----------+------------+
              |
   +----------v-----------+
   | victor/workflows/     |
   | StateGraph            |
   | WorkflowEngine        |
   +-----------------------+
```

---

## Section B: Findings (Ordered by Severity)

### B.1 CRITICAL

#### B.1.1 No API Version Contract Between Core and Verticals

**Evidence:** Grep for `__version__` checks, `API_VERSION`, or compatibility guards in vertical loading path yields zero results. `VerticalRegistry.discover_external_verticals()` at `core/verticals/base.py:735-815` performs structural validation (name non-empty, abstract methods implemented) but no version compatibility check.

**Risk:** A core framework upgrade that changes a protocol signature (e.g., adding a required parameter to `SafetyExtensionProtocol.get_bash_patterns()`) will silently break all installed verticals at runtime, not at install time.

**Impact:** High. With 4+ external repos on loose pins (`victor-ai>=0.5.6`), a minor release can cause cascading failures.

**Fix:** Add `VERTICAL_API_VERSION: int` to `VerticalBase`. Core checks `vertical.VERTICAL_API_VERSION >= MINIMUM_SUPPORTED_API_VERSION` during discovery. Bump on breaking protocol changes. Emit deprecation warnings for `API_VERSION == current - 1`, hard-fail for `< current - 1`.

---

#### B.1.2 Entry Point Groups Re-queried Without Caching

**Evidence:** `victor/framework/entry_point_loader.py:49-200` — each call to `load_tool_dependency_provider_from_entry_points()`, `load_safety_rules_from_entry_points()`, etc. invokes `importlib.metadata.entry_points()` independently. Six separate groups are queried: `victor.verticals`, `victor.safety_rules`, `victor.tool_dependencies`, `victor.framework.teams.providers`, `victor.escape_hatches`, `victor.api_routers`.

**Risk:** On systems with many installed packages, `entry_points()` scans all `dist-info` directories. Six independent scans during startup multiplies I/O overhead. Not catastrophic for dev, but measurable in containerized cold-start scenarios.

**Fix:** Single cached call: `_cached_eps = importlib.metadata.entry_points()` at module level, filter by group downstream. Or use `functools.lru_cache` on a wrapper.

---

### B.2 HIGH

#### B.2.1 ProviderRegistry Lacks Explicit Thread Safety

**Evidence:** `victor/providers/registry.py:228-240` — `_registry_instance` singleton and `_lazy_provider_specs` module-level dict have no locking. `_materialize_lazy_provider()` at lines 149-177 performs import + instantiation without synchronization. Contrast with `SharedToolRegistry` at `agent/shared_tool_registry.py:120-129` which uses double-checked locking.

**Risk:** Concurrent provider creation (e.g., parallel tool calls requesting different providers) could race on materialization. Python's GIL prevents data corruption but not duplicate instantiation or partially-initialized state.

**Fix:** Add `threading.Lock` around `_materialize_lazy_provider()`, matching the pattern already used by `SharedToolRegistry`.

---

#### B.2.2 VerticalExtensions Uses `Optional[Any]` for Most Fields

**Evidence:** `core/verticals/protocols/__init__.py` — `VerticalExtensions` dataclass:
```python
workflow_provider: Optional[WorkflowProviderProtocol] = None
rl_config_provider: Optional[RLConfigProviderProtocol] = None
tool_selection_strategy: Optional[ToolSelectionStrategyProtocol] = None
tiered_tool_config: Optional[Any] = None  # <-- untyped
```

Multiple protocol methods use `-> Optional[Any]` return types (e.g., `CapabilityProviderProtocol.get_capabilities() -> Dict[str, Any]`).

**Risk:** Leaked abstractions. Verticals implement contracts that claim `Any`, which defeats static analysis and allows silent type mismatches. MyPy cannot catch a vertical returning a wrong shape.

**Fix:** Replace `Any` with concrete protocol types or TypedDict/dataclass contracts. `tiered_tool_config: Optional[TieredToolConfig]` is the correct type — it's already defined in `vertical_types.py`.

---

#### B.2.3 victor-invest Has Dual Package Structure (Legacy + Modern)

**Evidence:** `victor-invest` contains both `victor_invest/` (modern vertical) and `src/investigator/` (347 Python files, legacy DDD engine). Modern tools wrap legacy services. Two CLI entry points coexist (`investigator` and `victor-invest`).

**Risk:** Maintenance burden doubles. Legacy engine evolution is unconstrained by vertical contracts. Dependency surface is massive (PostgreSQL, Redis, 10+ API clients). A change in `src/investigator/infrastructure/` can break `victor_invest/tools/` with no contract enforcement.

**Fix:** Phase out `src/investigator/` by extracting reusable infrastructure into a `victor-invest-core` library with stable interfaces. Legacy CLI becomes a thin wrapper. Target: legacy code shrinks to data-access adapters behind clean ports.

---

### B.3 MEDIUM

#### B.3.1 TieredToolConfig Contains Deprecated Fields

**Evidence:** `core/vertical_types.py` — `TieredToolConfig` has:
```python
semantic_pool: Set[str]  # DEPRECATED
stage_tools: Dict[str, Set[str]]  # DEPRECATED
```
Both have active `get_effective_semantic_pool()` and `get_tools_for_stage()` methods that still reference them.

**Risk:** Verticals may depend on deprecated fields. No runtime deprecation warning emitted. Removal will be a silent breaking change.

**Fix:** Add `warnings.warn("semantic_pool is deprecated...", DeprecationWarning)` in property accessors. Remove in next major version with API version bump (ties to B.1.1).

---

#### B.3.2 Protocol Duplication: Provider vs Extension Protocols

**Evidence:** Two parallel protocol hierarchies exist:
1. `core/verticals/protocols/providers.py` — `MiddlewareProvider`, `SafetyProvider`, etc. (marker protocols on VerticalBase subclasses)
2. `core/verticals/protocols/*.py` — `MiddlewareProtocol`, `SafetyExtensionProtocol`, etc. (implementation protocols on extension objects)

Plus legacy aliases: `ModeConfigProviderProtocol`, `PromptContributorProtocol`, etc.

**Risk:** Cognitive overhead. Contributors must understand 3 layers (provider protocol on vertical class -> returns instance -> instance satisfies extension protocol). Legacy aliases add confusion.

**Fix:** Document the two-tier pattern explicitly in SDK docs. Deprecate legacy aliases with `__getattr__` warnings. Consider collapsing to a single protocol per concern where the vertical class itself implements the extension (for simple cases).

---

#### B.3.3 Framework __init__.py Is a God Module

**Evidence:** `victor/framework/__init__.py` — 680 lines of imports/exports, 164 lazy names, `__all__` with 100+ symbols. Acts as both public API surface and internal wiring hub.

**Risk:** Any addition to the framework touches this file. Import ordering is fragile. Difficult to identify which symbols are public API vs internal.

**Fix:** Split into:
- `victor.framework.api` — stable public API (Agent, Task, State, Events, ToolSet)
- `victor.framework.internal` — internal wiring (lazy imports, bridges)
- `victor.framework.__init__` — re-exports from `.api` only, with `__all__` guarding

---

#### B.3.4 Vertical Tool Registration Is Inconsistent Across Repos

**Evidence:**
- **victor-coding**: Lists tool *names* via `get_tools()`, relies on core registry having them. No custom tools.
- **victor-invest**: Defines 21 custom `BaseTool` subclasses, registers via `register_investment_tools(tool_registry)`. Entry point only declares the vertical, not the tools.
- **victor-devops/research**: Same pattern as coding (name-only, no custom tools).

**Risk:** victor-invest's custom tools require explicit registration call. If the framework doesn't call `register_investment_tools()`, the vertical fails silently (tools listed in `get_tools()` won't resolve).

**Fix:** Add a `"victor.tools"` entry point group. Verticals with custom tools register them there. Framework discovers and registers automatically during vertical activation. Or add `register_tools(registry: ToolRegistry)` to VerticalBase as an optional hook.

---

#### B.3.5 Contrib Verticals Now Emit DeprecationWarning

**Status (updated 2026-03-15):** Built-in contrib verticals (`coding`, `devops`, `rag`, `dataanalysis`, `research`) now emit `DeprecationWarning` on import, signaling migration to the external vertical package pattern. This is a step toward the contract-first architecture described in Section C.

---

### B.4 LOW

#### B.4.1 No Contract Testing Infrastructure

**Evidence:** No shared test fixtures or contract test suite exists for verifying that a vertical correctly implements required protocols. Each vertical has its own ad-hoc tests. No CI job runs core contract tests against installed verticals.

**Fix:** Create `victor-contract-tests` package (or pytest plugin) that verticals install as dev dependency. It verifies: VerticalBase subclass validity, protocol conformance, tool name resolution, stage graph connectivity.

---

#### B.4.2 Inconsistent Version Pinning

**Evidence:**
- victor-coding: `victor-ai>=0.5.6` (no upper bound)
- victor-research: `victor-ai>=0.5.6` (no upper bound)
- victor-devops: `victor-ai>=0.5.6` (no upper bound)
- victor-invest: `victor-ai>=0.5.0,<0.6.0` (pessimistic pin)

**Risk:** Inconsistent strategy. Open pins allow breaking changes; pessimistic pin blocks minor improvements.

**Fix:** Standardize on `victor-ai~=0.5.6` (compatible release, allows 0.5.x but not 0.6.0) across all verticals. Combine with API version contract (B.1.1).

---

#### B.4.3 Team Persona Registration Side Effects at Import Time

**Evidence:** `victor-devops/victor_devops/teams/personas.py` registers personas with `FrameworkPersonaProvider` at module import time. This means importing the teams module has a global side effect.

**Risk:** Import ordering sensitivity. Tests that import teams module pollute global state.

**Fix:** Move registration to explicit `register()` classmethod called during vertical activation, not at import time.

---

## Section C: Target Architecture

### C.1 Contract-First Plugin Architecture

```
+------------------------------------------------------------------+
|                     victor-sdk (new package)                      |
|                                                                   |
|  Stable Contracts:                                                |
|    VerticalBase, VerticalConfig, StageDefinition                  |
|    All 15 extension protocols                                     |
|    BaseTool, ToolResult, ToolConfig, CostTier                     |
|    TeamSpec, TeamMemberSpec, TeamFormation                         |
|    BaseRLConfig, LearnerType                                      |
|    BaseYAMLWorkflowProvider                                       |
|    SafetyPattern, SafetyLevel                                     |
|    AgentSpec (for multi-agent verticals)                          |
|                                                                   |
|  Versioned: VERTICAL_API_VERSION = 2                              |
|  Semver: Follows core framework version                           |
|  Dependencies: Minimal (no heavy framework internals)             |
+------------------------------------------------------------------+
        |                    |                    |
        v                    v                    v
  victor-coding       victor-research       victor-invest
  depends on:         depends on:           depends on:
  victor-sdk~=0.6     victor-sdk~=0.6       victor-sdk~=0.6

        |                    |                    |
        +--------------------+--------------------+
                             |
                    +--------v---------+
                    |  victor (core)   |
                    |  depends on:     |
                    |  victor-sdk~=0.6 |
                    +------------------+
```

### C.2 Extension SDK Shape

```python
# victor_sdk/vertical.py
class VerticalBase(ABC):
    VERTICAL_API_VERSION: int = 2  # Must match core's MINIMUM_SUPPORTED

    name: str
    description: str
    version: str = "1.0.0"

    @classmethod
    @abstractmethod
    def get_tools(cls) -> List[str]: ...

    @classmethod
    @abstractmethod
    def get_system_prompt(cls) -> str: ...

    # Optional hooks (all have safe defaults)
    @classmethod
    def get_stages(cls) -> Dict[str, StageDefinition]: ...
    @classmethod
    def get_extensions(cls) -> VerticalExtensions: ...
    @classmethod
    def register_tools(cls, registry: ToolRegistry) -> None: ...
    @classmethod
    def customize_config(cls, config: VerticalConfig) -> VerticalConfig: ...

# victor_sdk/protocols.py
# All 15 protocols with concrete types (no Any)
# All @runtime_checkable

# victor_sdk/types.py
# StageDefinition, TaskTypeHint, TieredToolConfig, VerticalConfig
# ToolResult, SafetyPattern, ModeConfig, TeamSpec
# All frozen dataclasses for immutability
```

### C.3 Lifecycle Hooks (Standardized)

```python
class VerticalLifecycle(Protocol):
    """Optional lifecycle hooks for verticals."""

    def on_activate(self, context: ActivationContext) -> None:
        """Called when vertical is selected for a session."""

    def on_deactivate(self) -> None:
        """Called when session ends or vertical switches."""

    def on_tool_register(self, registry: ToolRegistry) -> None:
        """Called to register custom tools (for verticals with custom tools)."""

    def health_check(self) -> HealthStatus:
        """Called by framework to verify vertical is operational."""
```

### C.4 Repo Organization and Ownership

```
Organization: vjsingh1984/

  victor           Core framework. Owns: agent, providers, tools, workflows,
                   state, events, UI, SDK contracts.
                   Release: semver, PyPI as victor-ai

  victor-sdk       Extracted contract package. Owns: VerticalBase, all protocols,
                   shared types. Lightweight, minimal deps.
                   Release: version-locked to core

  victor-coding    Vertical. Owns: coding tools, workflows, RL config, personas.
                   Release: independent semver, pins victor-sdk~=X.Y

  victor-research  Vertical. Owns: research workflows, citation, fact-check.
  victor-devops    Vertical. Owns: DevOps workflows, infra safety, IaC handlers.
  victor-invest    Vertical. Owns: investment tools, analysis workflows, RL backtest.

  victor-contract-tests   Pytest plugin for vertical conformance testing.
```

---

## Section D: Phased Roadmap

### Phase 0: Foundation (Current + 2 weeks)

**Goal:** Establish version contract and fix critical gaps without breaking changes.

| Action | Files | Exit Criteria |
|--------|-------|---------------|
| Add `VERTICAL_API_VERSION = 1` to `VerticalBase` | `core/verticals/base.py` | All verticals declare version |
| Add version check in `discover_external_verticals()` | `core/verticals/base.py:767` | Unknown versions emit warning |
| Cache `entry_points()` result module-level | `framework/entry_point_loader.py` | Single scan per process |
| Add `threading.Lock` to `ProviderRegistry._materialize` | `providers/registry.py:149` | No race on concurrent creation |
| Add deprecation warnings to `TieredToolConfig.semantic_pool` | `core/vertical_types.py` | `DeprecationWarning` emitted |
| Standardize version pins to `~=0.5.6` across verticals | All vertical `pyproject.toml` | Consistent pinning |
| Move persona registration from import-time to explicit call | `victor-devops/teams/personas.py` | No import side effects |

**Exit criteria:** All existing tests pass. No behavioral changes for users.

---

### Phase 1: SDK Extraction (4-6 weeks)

**Goal:** Extract stable contracts into `victor-sdk` package.

| Action | Details | Exit Criteria |
|--------|---------|---------------|
| Create `victor-sdk` package | Contains: `VerticalBase`, all protocols, shared types, `BaseTool`, `ToolResult`, `TeamSpec`, `BaseRLConfig`, `BaseYAMLWorkflowProvider` | Package installable, imports work |
| Core depends on `victor-sdk` | `victor-ai` adds `victor-sdk` as dependency | Core uses SDK types |
| Verticals depend on `victor-sdk` | Replace `victor-ai` dep with `victor-sdk` + optional `victor-ai` | Verticals build against SDK only |
| Split `framework/__init__.py` | Public API in `.api`, internal in `.internal` | `__all__` has <30 symbols |
| Replace `Optional[Any]` in protocols | Concrete types throughout | MyPy strict passes on SDK |
| Create `victor-contract-tests` | Pytest plugin validating protocol conformance | Plugin runs against all 4 verticals |
| Add `register_tools()` hook to `VerticalBase` | Optional method, called during activation | victor-invest uses it for 21 tools |

**Compatibility shim:** `victor-ai` re-exports all SDK symbols for backward compat. Verticals can depend on either during migration.

**Exit criteria:** All 4 verticals pass contract tests. SDK package on PyPI. `mypy --strict` passes on SDK.

---

### Phase 2: Contract Hardening (4 weeks)

**Goal:** Enforce contracts, add observability, clean up legacy.

| Action | Details | Exit Criteria |
|--------|---------|---------------|
| Enforce `VERTICAL_API_VERSION` | Hard-fail on `< MINIMUM - 1` | Old verticals get clear error |
| Remove legacy protocol aliases | `ModeConfigProviderProtocol` -> `ModeConfigProvider` | No legacy names in SDK |
| Remove `TieredToolConfig.semantic_pool` | Breaking change, guarded by API version bump | Clean dataclass |
| Add `"victor.tools"` entry point group | Auto-discover custom tools from verticals | victor-invest tools auto-register |
| Add health_check to vertical lifecycle | Optional, called on activation | Framework logs vertical health |
| Cross-repo CI | Core CI runs contract tests against latest verticals | Matrix build green |
| victor-invest legacy extraction | Move `src/investigator/` infra to `victor-invest-core` | Clean boundary |

**Exit criteria:** API version = 2. All deprecated symbols removed. Cross-repo CI green.

---

### Phase 3: Optimization & Scale (Ongoing)

**Goal:** Performance, observability, ecosystem growth.

| Action | Details | Exit Criteria |
|--------|---------|---------------|
| Startup profiling | Measure cold-start with 4+ verticals installed | <500ms to first prompt |
| Extension loading parallelism | Parallel entry point loading for independent groups | Measurable speedup |
| Vertical hot-reload | Leverage existing `DynamicModuleLoader` for dev mode | Vertical code change -> auto-reload |
| Observability SDK | Structured logging + OpenTelemetry spans for vertical execution | Cross-repo distributed traces |
| Vertical marketplace | Discovery registry (PyPI tags + victor.dev catalog) | 3+ community verticals listed |
| SDK documentation | Auto-generated from protocol docstrings | docs.victor.dev/sdk |

**Exit criteria:** Cold-start <500ms. At least 1 community-contributed vertical.

---

## Section E: Score Tables

### E.1 Decoupling Quality Scorecard

| Dimension | Score | Evidence |
|-----------|-------|----------|
| Import direction | 10/10 | Zero imports from verticals into core (verified via grep) |
| Contract definition | 7/10 | 15 protocols exist but some use `Any`; no version contract |
| Plugin discovery | 8/10 | Entry points work; multiple groups; no caching |
| State isolation | 7/10 | Registries are singletons; some import-time side effects |
| Type safety | 5/10 | `Optional[Any]` in VerticalExtensions; protocols lack concrete types |
| Test isolation | 4/10 | No shared contract test suite; ad-hoc vertical tests only |
| Version compatibility | 3/10 | No API versioning; inconsistent dependency pins |
| Failure isolation | 7/10 | Silent continue on entry point load failure; but no health checks |
| **Weighted Average** | **6.4/10** | Weights: contracts 2x, version compat 2x, rest 1x |

### E.2 SOLID Evaluation Summary

| Principle | Core | Verticals | Key Issue | Fix |
|-----------|------|-----------|-----------|-----|
| **SRP** | 7/10 | 8/10 | `framework/__init__.py` is god module; `handlers.py` in invest is 145KB | Split init; split handlers by domain |
| **OCP** | 9/10 | 8/10 | Extension via protocols is excellent; TieredToolConfig has deprecated fields | Clean deprecated fields behind version gate |
| **LSP** | 8/10 | 7/10 | Protocols are substitutable; but `Any` return types weaken guarantees | Concrete types in protocols |
| **ISP** | 9/10 | 9/10 | 15 fine-grained protocols; verticals implement only what they need | Near-ideal |
| **DIP** | 8/10 | 7/10 | Core depends on abstractions; verticals sometimes import concrete coordinators | Verticals should use SDK protocols only |

### E.3 Comparative Positioning

Scoring criteria: 1 = poor/missing, 5 = adequate, 10 = best-in-class.
Weights reflect importance for production agentic systems.

| Dimension (Weight) | Victor | LangGraph | CrewAI | LangChain | LlamaIndex | AutoGen |
|---------------------|--------|-----------|--------|-----------|------------|---------|
| **Plugin/Vertical System** (2.0) | 9 | 4 | 5 | 6 | 5 | 4 |
| Rationale | 15 protocols, entry-point discovery, 4 extracted verticals | Subgraphs only, no plugin system | Role-based, no plugin extraction | Tools/chains extensible, no vertical concept | Llamapacks exist but shallow | Agent types, no plugin arch |
| **Workflow Engine** (1.5) | 9 | 10 | 5 | 7 | 5 | 6 |
| Rationale | StateGraph + YAML DSL + checkpointing | Native graph engine, gold standard | Sequential/hierarchical only | LCEL chains, less graph-native | Limited pipeline DAGs | Conversation patterns only |
| **Multi-Provider** (1.5) | 10 | 6 | 7 | 8 | 7 | 6 |
| Rationale | 24 providers, lazy loading, circuit breakers, OAuth | OpenAI-centric, some others | Multiple but less integrated | Good provider support | Good LLM support | OpenAI-focused |
| **Tool System** (1.5) | 9 | 7 | 6 | 8 | 7 | 7 |
| Rationale | Cost tiers, schema caching, hooks, metadata, MCP | Good tool support | Basic tools | Extensive tool ecosystem | Tool abstractions | Function calling |
| **Multi-Agent** (1.0) | 8 | 7 | 9 | 5 | 4 | 9 |
| Rationale | 9+ formations, personas, team specs | Multi-agent via subgraphs | Native crew/agent model | Limited agent coordination | Single-agent focus | Native multi-agent |
| **Observability** (1.0) | 7 | 7 | 5 | 8 | 6 | 5 |
| Rationale | Events, CQRS bridge, metrics; cross-repo gaps | LangSmith integration | Basic logging | LangSmith/callbacks mature | Callbacks exist | Basic logging |
| **Ecosystem/Community** (0.5) | 4 | 8 | 7 | 10 | 8 | 7 |
| Rationale | Small but focused; 4 verticals, limited community | Growing rapidly | Active community | Largest ecosystem | Large ecosystem | Microsoft-backed |
| | | | | | | |
| **Weighted Total** (/90) | **74.5** | **63.5** | **56.0** | **66.5** | **54.5** | **57.0** |
| **Normalized** (0-10) | **8.3** | **7.1** | **6.2** | **7.4** | **6.1** | **6.3** |

**Victor's differentiators:** Plugin/vertical architecture (unique), provider breadth (24), and tool system sophistication (cost tiers, hooks, MCP). Main gap: ecosystem size and community adoption.

**Key insight:** Victor's post-extraction architecture is structurally more mature than competitors for building domain-specific AI verticals. The contract-first approach with protocol-based ISP is ahead of the field. The main risk is that without SDK formalization (Phase 1), this advantage erodes as contracts drift.

---

## Appendix: Key File Reference Index

| Concept | File | Lines |
|---------|------|-------|
| Public API | `victor/framework/__init__.py` | 1-680 |
| Agent class | `victor/framework/agent.py` | full |
| VerticalBase | `victor/core/verticals/base.py` | 1-900 |
| VerticalRegistry | `victor/core/verticals/base.py` | 633-902 |
| Extension protocols | `victor/core/verticals/protocols/` | all files |
| VerticalExtensions | `victor/core/verticals/protocols/__init__.py` | dataclass |
| StageDefinition | `victor/core/vertical_types.py` | top |
| AgentOrchestrator | `victor/agent/orchestrator.py` | 3,940 total (facade + extracted components) |
| BaseProvider | `victor/providers/base.py` | 1-500 |
| ProviderRegistry | `victor/providers/registry.py` | 1-500 |
| BaseTool | `victor/tools/base.py` | 1-700 |
| ToolRegistry | `victor/tools/registry.py` | 1-625 |
| StateGraph | `victor/framework/graph.py` | 1-1500 |
| Events | `victor/framework/events.py` | full |
| Protocols | `victor/framework/protocols.py` | 1-942 |
| Feature flags | `victor/core/feature_flags.py` | 15-100 |
| Entry point loader | `victor/framework/entry_point_loader.py` | 49-200 |
| SharedToolRegistry | `victor/agent/shared_tool_registry.py` | 67-144 |
| Config | `victor/framework/config.py` | full |
| Team schema | `victor/framework/team_schema.py` | full |
