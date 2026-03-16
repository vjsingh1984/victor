# Victor Framework + Vertical Architecture Analysis

Senior systems architect analysis of framework-vertical integration.

---

## 1. Architecture Map

### Data Flow: Framework ↔ Verticals

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER REQUEST                                       │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     AGENT (victor/framework/agent.py)                        │
│  • Agent.create(vertical=CodingAssistant)                                   │
│  • Delegates to AgentOrchestrator                                           │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              VERTICAL INTEGRATION PIPELINE (framework/step_handlers.py)      │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ Tool    │→│ Prompt  │→│ Config  │→│Extension│→│Framework│→│ Context │   │
│  │ (10)    │ │ (20)    │ │ (40)    │ │ (45)    │ │ (60)    │ │ (100)   │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VERTICAL (e.g., victor/coding/)                           │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐                   │
│  │ get_extensions │ │ get_config()   │ │ get_workflows()│                   │
│  │ • middleware   │ │ • tools        │ │ • YAML + escape│                   │
│  │ • safety       │ │ • prompt       │ │   hatches      │                   │
│  │ • prompts      │ │ • stages       │ │                │                   │
│  │ • tool_deps    │ │ • mode_config  │ │                │                   │
│  │ • teams        │ │                │ │                │                   │
│  │ • rl_config    │ │                │ │                │                   │
│  └────────────────┘ └────────────────┘ └────────────────┘                   │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR (victor/agent/orchestrator.py)              │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │ToolPipeline  │ │Conversation  │ │ Provider     │ │ StateGraph   │        │
│  │ • execute    │ │ Controller   │ │ Manager      │ │ • nodes      │        │
│  │ • cache      │ │ • history    │ │ • 24 LLMs    │ │ • edges      │        │
│  │ • analytics  │ │ • stages     │ │ • fallback   │ │ • checkpoints│        │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘        │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OBSERVABILITY (victor/observability/)                │
│  EventBus ←──── graph events ←──── tool events ←──── state events           │
│  │                                                                           │
│  └──→ Protocol Backend (InMemory | SQLite | Kafka | Redis | SQS)            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Modules

| Module | Location | Responsibility |
|--------|----------|----------------|
| **StateGraph DSL** | `framework/graph.py:1217-1457` | LangGraph-compatible workflow builder |
| **WorkflowEngine** | `framework/workflow_engine.py` | Facade over YAML/Graph/HITL coordinators |
| **Agent** | `framework/agent.py:42-1071` | High-level API, vertical integration entry |
| **ToolPipeline** | `agent/tool_pipeline.py` | Execution, caching, failed signature tracking |
| **VerticalBase** | `core/verticals/base.py:151-1299` | Template method for domain verticals |
| **VerticalRegistry** | `core/verticals/base.py:1042-1299` | Discovery via entry_points |
| **EventBus** | `observability/event_bus.py` | Pub/sub with backpressure, sampling |
| **IEventBackend** | `core/events/protocols.py` | Protocol for distributed backends |

---

## 2. Gaps: Generic Capabilities in Verticals

### Must Promote to Framework

| Capability | Current Location | Why Generic | Promotion Target |
|------------|------------------|-------------|------------------|
| **Tool dependency graphs** | `{vertical}/tool_dependencies.py` | All verticals use identical YAML structure | `framework/tool_graph.py` |
| **Escape hatch registry** | `{vertical}/escape_hatches.py` | Same CONDITIONS/TRANSFORMS pattern | `framework/escape_hatches.py` |
| **Handler registration** | `{vertical}/handlers.py` | Identical `@dataclass` handler pattern | `framework/handlers.py` |
| **Team spec factory** | `coding/teams/specs.py:314-617` | Same `TeamSpec` structure everywhere | `teams/spec_factory.py` |
| **Cost tracking** | Scattered in verticals | Every vertical needs it | `framework/cost_tracker.py` |
| **Progressive disclosure** | `coding/` only | Generic tool surfacing pattern | `framework/progressive_tools.py` |

### Concrete Fixes

**1. Tool Dependency Graph (HIGH PRIORITY)**
```python
# Current: Each vertical duplicates YAML loading
class CodingToolDependencyProvider(YAMLToolDependencyProvider): ...
class ResearchToolDependencyProvider(YAMLToolDependencyProvider): ...

# Fix: Framework factory
def create_tool_dependency_provider(vertical_name: str) -> ToolDependencyProvider:
    yaml_path = discover_tool_deps_yaml(vertical_name)
    return UnifiedToolDependencyProvider(yaml_path)
```

**2. Escape Hatch Registry (MEDIUM PRIORITY)**
```python
# Current: String module paths, runtime import
def _get_escape_hatches_module(self) -> str:
    return "victor.coding.escape_hatches"  # Runtime error if wrong

# Fix: Protocol-based registration
class EscapeHatchRegistry:
    _conditions: Dict[str, Callable] = {}
    _transforms: Dict[str, Callable] = {}

    @classmethod
    def register_condition(cls, name: str, fn: Callable): ...
```

**3. Handler Pattern (MEDIUM PRIORITY)**
```python
# Current: Each vertical defines handlers identically
@dataclass
class CodeValidationHandler:
    async def __call__(self, node, context, tool_registry): ...

# Fix: Framework base class + registry
class BaseComputeHandler(Protocol):
    async def execute(self, node: ComputeNode, ctx: WorkflowContext) -> NodeResult: ...

HANDLER_REGISTRY.register("code_validation", CodeValidationHandler)
```

---

## 3. SOLID Evaluation

### SRP Violations

| Component | Violation | Fix |
|-----------|-----------|-----|
| `VerticalBase` (1299 lines) | Config, extensions, caching, validation, registry all in one | Split into `VerticalConfig`, `ExtensionLoader`, `VerticalCache` |
| `WorkflowEngine` | Pre-split was 2000+ lines with YAML/Graph/HITL/Cache | **FIXED**: Now uses coordinators (`framework/coordinators/`) |
| `StateGraph.compile()` | Validation + optimization + compilation | Extract `GraphValidator`, `GraphOptimizer` |

### OCP Violations

| Component | Violation | Fix |
|-----------|-----------|-----|
| `BUILTIN_VERTICALS` dict | Adding vertical requires framework change | Use entry_points exclusively |
| `EventCategory` enum | New category requires framework change | Allow vertical-registered categories |
| `TieredToolTemplate` | Hardcoded vertical names | Registry-based template lookup |

### LSP Violations

| Component | Violation | Fix |
|-----------|-----------|-----|
| `WorkflowEngine.execute_graph()` | Was returning `Dict` not `ExecutionResult` | **FIXED** in `workflow_engine.py:432-456` |
| `CompiledGraph.invoke()` | Mixed return types (dict vs ExecutionResult) | Standardize to `ExecutionResult` |

### ISP Compliance (GOOD)

---

## 4. Capability Entry Points & Optional Vertical Packages

### Current Wiring

- `victor.core.capability_registry` exposes a singleton registry that lazily bootstraps via `bootstrap_capabilities()` (see `victor/core/bootstrap.py:450-520`). During bootstrap we register STUB implementations from `victor.contrib.*` and overlay anything exposed through the `victor.capabilities` entry-point group.
- `victor.core.verticals.vertical_loader.VerticalLoader` scans the `victor.verticals`, `victor.tools`, and related entry-point groups. Discovered classes are cached and registered in `VerticalRegistry`, which lets higher layers request a vertical (e.g., `coding`) without hard dependencies.
- `victor.core.verticals.extension_loader.VerticalExtensionLoader` still uses the implicit namespace import pattern `victor.{vertical_name}.<module>` for safety, prompts, workflows, etc. This works when an external wheel publishes a namespace package (e.g., `victor-coding` ships a `victor/coding/` module), but produces noisy warnings when the optional wheel is not installed.

### Observed Failure Mode

Running `victor chat` without installing `victor-coding` leads to repeated warnings such as:
```
service extension failed to load for vertical 'coding': No module named 'victor.coding'
```
The warnings originate from `_get_extension_factory()` inside the extension loader when it tries to import `victor.coding.safety`, `victor.coding.prompts`, `victor.coding.workflows`, etc. As soon as the module import fails, the loader logs the warning and the `coding` vertical degrades to a stubbed state (no workflow/services), which later cascades into tool-selection fallbacks and hallucinated file reads.

### Future-Proof Fix Plan

1. **Capability Health Check at Bootstrap**  
    - Add a bootstrap hook in `victor/core/bootstrap.py` that inspects `VerticalLoader.discover_verticals()` output.  
    - When a requested vertical (per CLI flag or default) is absent, emit a single actionable warning that mentions the missing wheel (`victor-coding`) and the entry-point (`victor.verticals`).  
    - Surface the same signal via `UsageAnalytics` and Observability events so upstream orchestration can treat “missing capability” as a structured health event instead of letting ImportErrors leak into logs.

2. **Namespace Import Compatibility Shim**  
   - Create `victor/coding/__init__.py` as a thin compatibility layer that re-exports from `victor_coding` when it exists.  
   - Update `_get_extension_factory()` to probe both `victor.{name}` and `{victor_name}_` style modules via `importlib.util.find_spec`, preventing repeated warnings every time a capability is accessed.  
   - Gate the shim behind the capability registry: only try to import when `capabilities.is_enhanced(...)` for the relevant protocol is `True`, otherwise skip straight to stub behavior.

3. **Entry-Point-Driven Extension Binding**  
   - Extend the `victor.capabilities` entry points so vertical packages can register *all* extension providers (safety, prompt, workflows, RL, etc.) explicitly rather than relying on magic module names.  
   - `VerticalExtensionLoader` then resolves extensions with a lookup like `capabilities.get(WorkflowProviderProtocol)` before doing the namespace import. This keeps existing packages working yet enables a zero-warning experience when a capability is absent.

4. **Test + Doc Coverage**  
    - Add an integration test in `tests/integration/verticals/` that runs `victor chat` with the `coding` vertical disabled, asserting that (a) the CLI exits cleanly and (b) logs contain the single structured “capability missing” warning, not raw ImportErrors.  
    - Document the required entry points (`victor.verticals`, `victor.capabilities`, `victor.tools`, etc.) in `docs/verticals/coding.md` (or a dedicated contributor guide) so plugin authors know what to implement.  
    - Include a troubleshooting subsection describing how to interpret the structured warnings and which pip install command resolves them.

**Implementation status**: `victor/core/bootstrap.py` now resolves the requested vertical once, feeds it into `_report_capability_health()`, emits a usage+observability metric, and logs a single actionable warning with the matching entry point + `pip install victor-coding` guidance. This ensures “missing capability” shows up as explicit telemetry instead of cascading ImportErrors while we continue rolling out the remaining steps above.

These steps align the new capability-based design with the legacy namespace-import expectations, reduce noise in production logs, and give us the evidence trail we need for future design reviews. Once the shim + entry-point bindings are in place we can proceed with the earlier “Next Steps” (tool availability, persistence adapters, HF token validation) without reintroducing tight coupling to `victor_coding`.

The vertical system demonstrates excellent ISP:
- 15+ focused protocols in `core/verticals/protocols/providers.py`
- Verticals implement only needed protocols
- Example: `MiddlewareProvider`, `SafetyProvider`, `WorkflowProvider` are separate

### DIP Violations

| Component | Violation | Fix |
|-----------|-----------|-----|
| `vertical_loader.py:61-67` | Hardcoded class paths | Inject via registry |
| `step_handlers.py` | Direct imports of vertical modules | Use protocol + factory |

---

## 4. Scalability & Performance Risks

### Hot Paths

| Path | Risk | Mitigation |
|------|------|------------|
| **Tool signature hashing** | Called per tool call | Native Rust impl (`tool_pipeline.py:66-73`), 10-20x faster |
| **YAML workflow parsing** | Per-execution without cache | `UnifiedWorkflowCompiler` caches parsed definitions |
| **Extension loading** | 11 extensions per vertical | `_extensions_cache` with composite keys |
| **Event emission** | Every node/tool emits | Sampling config, backpressure strategies |

### Caching Architecture

| Cache | Location | TTL | Invalidation |
|-------|----------|-----|--------------|
| Config cache | `VerticalBase._config_cache` | Infinite | Manual `clear_config_cache()` |
| Extension cache | `VerticalBase._extensions_cache` | Infinite | Manual clear |
| YAML definition | `UnifiedWorkflowCompiler._definition_cache` | Configurable | `invalidate_yaml_cache()` |
| Tool results | `ToolPipeline._idempotent_cache` | Session | Mtime-based |
| Embeddings | `EmbeddingService._cache` | Infinite | Manual |

### Missing Caches (ADD)

```python
# 1. Compiled graph cache (expensive compilation)
class GraphCache:
    _compiled: Dict[str, CompiledGraph] = {}

    def get_or_compile(self, graph: StateGraph, key: str) -> CompiledGraph:
        if key not in self._compiled:
            self._compiled[key] = graph.compile()
        return self._compiled[key]

# 2. Tool schema cache (JSON schema generation)
class ToolSchemaCache:
    _schemas: Dict[str, Dict] = {}

    def get_schema(self, tool: BaseTool) -> Dict:
        if tool.name not in self._schemas:
            self._schemas[tool.name] = tool.get_json_schema()
        return self._schemas[tool.name]
```

### Extension Loading Optimization

**Current**: Synchronous, blocking loads
```python
# base.py:786-925 - loads 11 extensions sequentially
extensions = cls._get_cached_extension("middleware", ...)
```

**Fix**: Parallel async loading
```python
async def load_extensions_async(cls) -> VerticalExtensions:
    tasks = [
        asyncio.create_task(cls._load_async("middleware")),
        asyncio.create_task(cls._load_async("safety")),
        ...
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return VerticalExtensions(*results)
```

---

## 5. Competitive Comparison

### Dimensions

| Dimension | Victor | LangGraph | CrewAI | LangChain | AutoGen |
|-----------|:------:|:---------:|:------:|:---------:|:-------:|
| **Graph DSL** | 9 | 10 | 5 | 7 | 6 |
| **Multi-Agent** | 8 | 7 | 9 | 6 | 10 |
| **Provider Agnostic** | 10 | 6 | 7 | 8 | 7 |
| **Observability** | 9 | 7 | 5 | 6 | 6 |
| **Extensibility** | 8 | 8 | 6 | 9 | 7 |

### Rationale

| Dimension | Victor Score | Rationale |
|-----------|:------------:|-----------|
| **Graph DSL** | 9 | LangGraph-compatible with copy-on-write optimization, cyclic support, checkpoints; lacks visual editor |
| **Multi-Agent** | 8 | 5 formations, team specs, personas; CrewAI has better role definitions, AutoGen has conversation patterns |
| **Provider Agnostic** | 10 | 24 providers, context preserved across switches; unique USP |
| **Observability** | 9 | Protocol-based backends, sampling, backpressure; LangSmith integration would push to 10 |
| **Extensibility** | 8 | 15+ ISP protocols, entry_points for plugins; LangChain's ecosystem is larger |

### Victor Unique Strengths

1. **Provider-agnostic context**: Switch models mid-conversation (no competitor does this)
2. **Vertical system**: Domain-specific assistants with ISP protocols
3. **Copy-on-write state**: O(1) reads for read-heavy workflows
4. **Native Rust acceleration**: 10-20x faster signature hashing

### Victor Gaps vs Competitors

| Gap | Competitor Advantage |
|-----|---------------------|
| Visual workflow editor | LangGraph Studio |
| Marketplace/hub | LangChain Hub |
| Conversation patterns | AutoGen's GroupChat |
| Memory systems | LangChain's memory abstractions |
| Tracing integration | LangSmith native |

---

## 6. Roadmap: Phased Improvements

### Phase 1: Foundation Cleanup (P0)

| Task | Impact | Effort | File |
|------|--------|--------|------|
| Remove `BUILTIN_VERTICALS` hardcoding | DIP compliance | Low | `vertical_loader.py:61-67` |
| Add YAML schema validation | Catch errors at load | Medium | `base_yaml_provider.py` |
| Escape hatch protocol registry | Type-safe registration | Medium | New: `framework/escape_hatches.py` |
| Graph cache | Avoid recompilation | Low | `framework/graph_cache.py` |

### Phase 2: Capability Promotion (P1)

| Task | Impact | Effort | File |
|------|--------|--------|------|
| Unified tool dependency factory | Reduce vertical boilerplate | Medium | `framework/tool_graph.py` |
| Handler base class + registry | Consistent compute handlers | Medium | `framework/handlers.py` |
| Cost tracking framework | Cross-vertical billing | Medium | `framework/cost_tracker.py` |
| Event category extensibility | Vertical-specific events | Low | `event_bus.py` |

### Phase 3: Performance (P1)

| Task | Impact | Effort | File |
|------|--------|--------|------|
| Async extension loading | Faster startup | Medium | `base.py:786-925` |
| Tool schema caching | Reduce JSON generation | Low | `tools/registry.py` |
| Lazy protocol backend init | Defer Kafka/Redis connect | Low | `core/events/backends.py` |

### Phase 4: Parity Features (P2)

| Task | Impact | Effort | Competitor Reference |
|------|--------|--------|---------------------|
| Visual workflow editor | UX for workflow design | High | LangGraph Studio |
| Memory abstraction | Persistent agent memory | Medium | LangChain Memory |
| Conversation patterns | Multi-agent chat | Medium | AutoGen GroupChat |
| Tracing export | APM integration | Medium | LangSmith |

### Phase 5: Differentiation (P2)

| Task | Impact | Effort |
|------|--------|--------|
| Cross-vertical workflows | Multi-domain orchestration | High |
| Provider-aware optimization | Route by model strengths | Medium |
| RL-guided tool selection | Learn optimal tool sequences | High |
| Distributed workflow execution | Scale beyond single instance | High |

---

## Summary

**Architecture Strengths:**
- Clean ISP protocol design (15+ focused interfaces)
- Coordinator pattern for SRP in WorkflowEngine
- Protocol-based event backends for scalability
- Copy-on-write state optimization

**Critical Fixes Needed:**
1. Remove hardcoded vertical mappings (DIP)
2. Promote tool dependencies, handlers, escape hatches to framework
3. Add YAML schema validation
4. Implement async extension loading

**Competitive Position:**
Victor excels in provider agnosticism and observability. To reach best-in-class:
- Add visual workflow editor (LangGraph parity)
- Build memory abstractions (LangChain parity)
- Implement conversation patterns (AutoGen parity)

**Estimated Timeline:**
- Phase 1-2: 2-3 weeks (foundation + promotion)
- Phase 3: 1-2 weeks (performance)
- Phase 4-5: 4-6 weeks (parity + differentiation)
