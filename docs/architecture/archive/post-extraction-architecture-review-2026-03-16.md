# Victor Ecosystem: Post-Extraction Architecture Review

**Date**: 2026-03-16
**Scope**: Core framework + 6 external verticals + SDK
**Method**: Code inspection across all repos + Gemini cross-validation

---

## Section A: Current-State Architecture Map

### A.1 Repository Topology

```
victor-ai (core)          victor-sdk (bundled)      External Verticals
┌─────────────────────┐   ┌──────────────────┐     ┌──────────────────┐
│ victor/              │   │ victor_sdk/       │     │ victor-coding    │ 41K LOC
│   agent/             │   │   core/types.py   │     │ victor-invest    │ 28K LOC
│     orchestrator.py  │   │   verticals/      │     │ victor-rag       │ 11K LOC
│     protocols/  (9)  │   │     base.py       │     │ victor-devops    │  6K LOC
│     factory/    (4)  │   │     manifest.py   │     │ victor-dataanalysis│ 6K LOC
│   framework/         │   │   discovery.py    │     │ victor-research  │  5K LOC
│     agent.py         │   │   constants/      │     └──────────────────┘
│     extensions.py    │   └──────────────────┘
│   core/              │         ▲
│     verticals/       │         │ depends on
│     plugins/         │         │ (zero-runtime)
│     bootstrap.py     │─────────┘
│   providers/ (24)    │
│   tools/ (34)        │
│   workflows/         │
└─────────────────────┘
```

### A.2 Loading Sequence (verified from code)

```
1. CLI/API entry
     │
2. ensure_bootstrapped(settings, vertical)     [core/bootstrap.py:928]
     │
3. bootstrap_container()                        [core/bootstrap.py:180]
     ├── Register 12+ service categories
     ├── PluginRegistry.discover()              [core/plugins/registry.py:61]
     │     └── scan entry_points("victor.plugins")
     │           └── CodingPlugin.register(context)
     │                 ├── context.register_vertical(CodingAssistant)
     │                 ├── context.register_chunker(CodeChunker)
     │                 └── context.register_tool(language_analyzer, ...)
     └── _register_vertical_services()          [core/bootstrap.py:838]
           ├── VerticalLoader.load(name)        [core/verticals/vertical_loader.py:123]
           │     ├── VerticalRegistry.get(name)  (built-in first)
           │     ├── _import_from_entrypoint()   (fallback)
           │     ├── _negotiate_manifest()       (API version + capability check)
           │     └── _activate()                 (set as active)
           └── activate_vertical_services()
                 └── VerticalExtensionLoader.get_extensions()
                       ├── get_middleware()
                       ├── get_safety_extension()
                       ├── get_prompt_contributor()
                       ├── get_mode_config_provider()
                       └── ... (10+ extension types)
```

### A.3 Extension Point Registry

| Entry Point Group | Purpose | Current Registrations |
|---|---|---|
| `victor.verticals` | Vertical discovery | benchmark (built-in) + 5 external |
| `victor.plugins` | Plugin system (tools, verticals, commands) | 5 contrib plugins |
| `victor.tool_dependencies` | Tool dependency providers | Per-vertical |
| `victor.safety_rules` | Safety rule providers | Per-vertical |
| `victor.framework.teams.providers` | Team spec providers | coding, invest |
| `victor.sdk.protocols` | SDK protocol implementations | coding (7), others (4 each) |
| `victor.sdk.capabilities` | Capability providers | coding-lsp, coding-git, etc. |
| `victor.api_routers` | FastAPI route providers | coding only |

---

## Section B: Findings (Ordered by Severity)

### B.1 CRITICAL: Dual Registration Path Collision

**Finding**: Verticals register via BOTH `victor.verticals` entry points AND `victor.plugins` entry points. Built-in contrib verticals also register via direct `VerticalRegistry.register()` in plugin `register()` methods. Three paths for the same vertical.

**Evidence**:
- `pyproject.toml:438-454` — contrib verticals commented out of `victor.verticals`, moved to `victor.plugins`
- `victor/verticals/contrib/coding/__init__.py:49` — `context.register_vertical(CodingAssistant)` in plugin
- External repos still use `victor.verticals` entry point

**Impact**: Name collision risk. Discovery order determines which class wins. No deterministic resolution.

**Fix**: Standardize on ONE path. Recommendation: `victor.plugins` as canonical, `victor.verticals` as legacy compat only. Add collision detection with clear error.

**Severity**: HIGH

---

### B.2 HIGH: Contrib Verticals Still Import Core Internals

**Finding**: Contrib verticals (bundled in core repo) import from `victor.core.verticals.base`, `victor.core.verticals.protocols`, `victor.core.tool_dependency_loader`, and other internal modules. External verticals do the same.

**Evidence** (from agent survey):
- victor-coding: 12 files import from `victor.core.*`
- victor-devops: 7 files import from `victor.core.*`
- victor-invest: 3 files import from `victor.core.*`

**Nuance**: These imports target `victor.core.verticals.protocols` which IS the intended contract surface. However, this module is NOT re-exported through `victor.framework.extensions` or `victor-sdk`, making it an implicit contract.

**Impact**: If `victor.core.verticals.protocols` is refactored, all 6 external repos break.

**Fix**: Promote vertical protocol types to either:
1. `victor-sdk` (preferred — zero-runtime-dependency contract), or
2. `victor.framework.extensions` (acceptable — stable re-export surface)

**Severity**: HIGH

---

### B.3 HIGH: Settings Injection Missing in Plugin Context

**Finding**: `HostPluginContext` (core/plugins/context.py) provides `register_tool()`, `register_vertical()`, `register_chunker()`, `register_command()`, `get_service()` — but does NOT expose `get_settings()`.

**Evidence**: `victor/core/plugins/context.py:100` — HostPluginContext has no settings accessor.

**Impact**: Plugins that need configuration (mode configs, tool budgets, API endpoints) must import `victor.config.settings` directly, creating a hidden dependency on the core config system.

**Fix**: Add `get_settings() -> Settings` to `PluginContext` protocol in SDK and implement in `HostPluginContext`.

**Severity**: HIGH

---

### B.4 MEDIUM: Extension Loading Has No Failure Isolation Per-Vertical

**Finding**: `VerticalExtensionLoader.get_extensions()` loads ALL extension types in a single call. If any extension fails in strict mode, the entire vertical activation fails.

**Evidence**: `extension_loader.py` `get_extensions()` method loads 10+ types sequentially.

**Impact**: A single broken extension (e.g., a faulty RL config) prevents the entire vertical from activating, even if the user only needs basic tool support.

**Fix**: Load extensions lazily on first access, not all-at-once during activation. Already partially implemented via `_load_cached_optional_extension` but not consistently applied.

**Severity**: MEDIUM

---

### B.5 MEDIUM: No Cross-Repo Contract Testing

**Finding**: No CI job runs external vertical tests against the latest core framework. Version compatibility is checked only at import time via `ExtensionManifest.min_framework_version`.

**Evidence**: `.github/workflows/` — no workflow triggers external vertical test suites. `roadmap.md` mentions this as a known gap.

**Impact**: Breaking changes in core can silently break external verticals until the vertical maintainer notices.

**Fix**: Add a CI matrix job that installs latest develop of victor-ai and runs `pytest` for each external vertical. GitHub Actions `workflow_dispatch` can trigger from the core repo.

**Severity**: MEDIUM

---

### B.6 MEDIUM: Startup Scan Overhead Scales Linearly

**Finding**: `PluginRegistry.discover()` scans ALL `importlib.metadata.entry_points(group="victor.plugins")` on every bootstrap. As the ecosystem grows, this linear scan adds latency.

**Evidence**: `core/plugins/registry.py:61` — `discover()` iterates all entry points.

**Impact**: With 50+ installed plugins, CLI startup could exceed 500ms.

**Fix**: Cache discovery results to disk with an invalidation hash (installed package versions). The `FND-004` work item already addressed entry-point/env-hash optimization but for `victor.verticals`, not `victor.plugins`.

**Severity**: MEDIUM

---

### B.7 LOW: Inconsistent Protocol Adoption Across Verticals

**Finding**: victor-coding has 100% protocol adoption. Other verticals lag significantly.

**Evidence** (from agent survey):

| Repo | VerticalBase | Protocols | Extensions | Mode Config | Teams | Capabilities | Score |
|------|-------------|-----------|-----------|------------|-------|-------------|-------|
| victor-coding | 100% | 100% | 100% | 100% | 100% | 100% | **100%** |
| victor-devops | 100% | 100% | 100% | 100% | 75% | 50% | **87%** |
| victor-invest | 100% | 100% | 75% | 0% | 100% | 0% | **63%** |
| victor-rag | 100% | 75% | 100% | 0% | 50% | 50% | **63%** |
| victor-dataanalysis | 100% | 75% | 100% | 0% | 50% | 0% | **54%** |
| victor-research | 100% | 75% | 100% | 0% | 50% | 0% | **54%** |

**Impact**: Users get inconsistent experience across verticals. Mode switching, team formation, and capability negotiation don't work for 4 of 6 verticals.

**Fix**: Create a vertical compliance checklist and CI check that validates protocol adoption level.

**Severity**: LOW (correctness not affected, only feature completeness)

---

### B.8 LOW: Event Fanout Uses Sync Path in Some Contexts

**Finding**: `VerticalLoader` uses `emit_event_sync()` with `use_background_loop=True` for observability events during discovery. If multiple verticals subscribe to discovery events, this could block.

**Evidence**: `vertical_loader.py:175` — `emit_event_sync(bus, topic, data, source="VerticalLoader", use_background_loop=True)`

**Impact**: Minimal in practice (discovery happens once), but architecturally incorrect for a system that claims async-first.

**Fix**: Use `await bus.emit()` in async discovery paths (`discover_verticals_async`). The sync path is acceptable for CLI startup.

**Severity**: LOW

---

## Section C: Target Architecture

### C.1 Contract-First Plugin Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      victor-sdk (PyPI)                   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ │
│  │ VerticalBase │ │ VictorPlugin │ │ ExtensionManifest│ │
│  │ (ABC)        │ │ (Protocol)   │ │ (Dataclass)      │ │
│  └──────────────┘ └──────────────┘ └──────────────────┘ │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ │
│  │ PluginContext │ │ 16 Extension │ │ ToolNames,       │ │
│  │ (Protocol)   │ │ Protocols    │ │ CapabilityIds    │ │
│  └──────────────┘ └──────────────┘ └──────────────────┘ │
└─────────────────────────────────────────────────────────┘
        ▲                                    ▲
        │ depends on                         │ depends on
        │ (zero runtime)                     │ (zero runtime)
┌───────┴──────────┐              ┌──────────┴─────────┐
│ victor-ai (Core) │              │ victor-coding, etc. │
│                  │◄─────────────│ (External Verticals)│
│ framework/       │  discovered  │                    │
│ core/            │  via entry   │ assistant.py       │
│ providers/       │  points      │ plugins.py         │
│ tools/           │              │ extensions/        │
└──────────────────┘              └────────────────────┘
```

### C.2 Stable Extension SDK Shape

The target SDK surface should include ALL types that external verticals need, eliminating imports from `victor.core.*`:

**Currently in SDK** (good):
- `VerticalBase`, `VictorPlugin`, `PluginContext`
- `ExtensionManifest`, `ExtensionType`
- `ToolNames`, `CapabilityIds`
- Normalization functions, type definitions

**Must move to SDK** (gap):
- 16 extension protocols (currently in `victor.core.verticals.protocols`)
- `VerticalExtensions` composite dataclass
- `SafetyPattern`, `TaskTypeHint`, `MiddlewarePriority`
- `VerticalConfig`, `StageDefinition` (already partially in SDK types)

**Must stay in framework** (runtime-dependent):
- `SafetyCoordinator`, `ConversationCoordinator` (via `victor.framework.extensions`)
- `ModeConfigRegistry`, `WorkflowExecutor`
- `ServiceContainer`, `ProviderRegistry`

### C.3 Lifecycle Hooks

```python
class VictorPlugin(Protocol):
    @property
    def name(self) -> str: ...

    def register(self, context: PluginContext) -> None:
        """Called during bootstrap. Register verticals, tools, chunkers."""
        ...

    def get_cli_app(self) -> Optional[typer.Typer]:
        """Return CLI subcommand app (optional)."""
        ...

    # NEW: Proposed lifecycle hooks
    def on_activate(self, context: PluginContext) -> None:
        """Called when this vertical becomes the active one."""
        ...

    def on_deactivate(self) -> None:
        """Called when switching away from this vertical."""
        ...

    def health_check(self) -> dict:
        """Return plugin health status for diagnostics."""
        ...
```

---

## Section D: Phased Implementation Roadmap

### Phase 0: Immediate Hardening (1 week)

| # | Action | Files | Exit Criteria |
|---|--------|-------|---------------|
| P0-1 | Add `get_settings()` to `PluginContext` protocol in SDK | `victor-sdk/victor_sdk/__init__.py`, `victor/core/plugins/context.py` | Plugins can access settings without direct import |
| P0-2 | Add collision detection to `VerticalLoader.load()` | `victor/core/verticals/vertical_loader.py` | Warning logged when same name registered via multiple paths |
| P0-3 | Promote 16 extension protocols to SDK | `victor-sdk/victor_sdk/verticals/protocols/` | External verticals import protocols from SDK, not `victor.core` |
| P0-4 | Add vertical compliance check CI step | `.github/workflows/vertical-validation.yml` | CI reports protocol adoption level per vertical |

**Exit criteria**: All external verticals can build against `victor-sdk` alone (zero `victor.core` imports for protocol types).

### Phase 1: Contract Stabilization (2 weeks)

| # | Action | Files | Exit Criteria |
|---|--------|-------|---------------|
| P1-1 | Cross-repo CI matrix | `.github/workflows/cross-repo-compat.yml` | Core develop push triggers vertical test suites |
| P1-2 | Startup discovery caching | `victor/core/plugins/registry.py` | Discovery cached to disk with package-hash invalidation |
| P1-3 | Lazy extension loading | `victor/core/verticals/extension_loader.py` | Extensions loaded on first access, not during activation |
| P1-4 | Remove `victor.verticals` entry point group | `pyproject.toml`, `vertical_loader.py` | Single discovery path via `victor.plugins` |

**Exit criteria**: Startup time stable at <200ms regardless of installed plugin count. Extensions load on demand.

### Phase 2: Protocol Graduation (2 weeks)

| # | Action | Files | Exit Criteria |
|---|--------|-------|---------------|
| P2-1 | Add `on_activate`/`on_deactivate`/`health_check` to VictorPlugin | `victor-sdk`, `core/plugins/` | Lifecycle hooks available for all plugins |
| P2-2 | Migrate 4 verticals to full protocol adoption | victor-rag, dataanalysis, research, invest | All at ≥80% compliance score |
| P2-3 | Add mode_config.py to verticals missing it | victor-rag, dataanalysis, research, invest | Mode switching works for all verticals |
| P2-4 | Version skew detection | `vertical_loader.py` | Clear error when SDK version mismatch detected |

**Exit criteria**: All 6 verticals at ≥80% protocol adoption. Lifecycle hooks documented.

### Phase 3: Ecosystem Scale (4 weeks)

| # | Action | Files | Exit Criteria |
|---|--------|-------|---------------|
| P3-1 | Pre-computed AOT manifest for fast discovery | `scripts/build_aot_manifest.py` | Zero entry-point scanning at startup for installed verticals |
| P3-2 | Vertical marketplace spec | `docs/guides/creating-verticals.md` | Third-party developers can create and publish verticals |
| P3-3 | Remove contrib verticals from core repo | `victor/verticals/contrib/` | Core repo has zero bundled verticals |
| P3-4 | SDK-only vertical template (`cookiecutter`) | `templates/vertical-template/` | `cookiecutter victor-vertical` scaffolds a complete vertical |

**Exit criteria**: Third-party vertical creation is documented and tooled. Core repo contains zero vertical code.

---

## Section E: Score Tables

### E.1 Decoupling Assessment

| Dimension | Score | Evidence |
|-----------|-------|---------|
| SDK Boundary | 9/10 | victor-sdk has zero runtime imports. Clean protocol layer. |
| Import Direction | 6/10 | External verticals import `victor.core.verticals.protocols` (should be SDK). |
| Plugin Interface | 8/10 | `VictorPlugin` + `PluginContext` well-defined. Missing `get_settings()`. |
| Discovery | 7/10 | Dual path (victor.verticals + victor.plugins) causes collision risk. |
| Extension Loading | 7/10 | 10+ extension types with caching. All-at-once loading, not lazy. |
| Config Isolation | 5/10 | Plugins import Settings directly. No injection via context. |
| Test Isolation | 4/10 | No cross-repo CI. Version skew untested. |
| Event Isolation | 8/10 | Async event bus with topic-prefix indexing. Minor sync leak. |
| **Weighted Average** | **6.8/10** | |

### E.2 Competitive Positioning (Optional)

Scoring weights: Architecture (20%), Extensibility (20%), Provider Breadth (15%), Production Readiness (15%), DX (15%), Multi-Agent (10%), Community (5%)

| Dimension | Victor | LangGraph | CrewAI | LangChain | AutoGen |
|-----------|--------|-----------|--------|-----------|---------|
| Architecture | 8 | 7 | 5 | 4 | 6 |
| Extensibility | 9 | 6 | 5 | 8 | 4 |
| Provider Breadth | 9 | 5 | 4 | 9 | 3 |
| Production Ready | 6 | 7 | 5 | 7 | 4 |
| Developer Experience | 6 | 7 | 8 | 6 | 5 |
| Multi-Agent | 7 | 8 | 9 | 5 | 8 |
| Community | 3 | 7 | 7 | 9 | 6 |
| **Weighted Total** | **7.3** | **6.7** | **5.9** | **6.6** | **5.1** |

Victor leads on architecture and extensibility (plugin system + 24 providers). Gaps are in production readiness (observability, cross-repo CI) and community (early-stage project).

---

## Appendix: Key File References

| Module | Path | LOC | Role |
|--------|------|-----|------|
| Agent API | `victor/framework/agent.py` | 1,000+ | Public facade |
| Extensions API | `victor/framework/extensions.py` | 197 | Stable re-export surface |
| SDK Base | `victor-sdk/victor_sdk/verticals/protocols/base.py` | 396 | VerticalBase ABC |
| Vertical Loader | `victor/core/verticals/vertical_loader.py` | 1,039 | Discovery + activation |
| Extension Loader | `victor/core/verticals/extension_loader.py` | 1,897 | Extension resolution + caching |
| Plugin Registry | `victor/core/plugins/registry.py` | 145 | Entry point scanning |
| Plugin Context | `victor/core/plugins/context.py` | 100 | SDK↔framework bridge |
| Bootstrap | `victor/core/bootstrap.py` | 1,008 | DI container setup |
| Orchestrator | `victor/agent/orchestrator.py` | 3,787 | Runtime coordination |
| Protocols (9 modules) | `victor/agent/protocols/` | 3,655 | Orchestrator contracts |
