# Changelog

All notable changes to Victor are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] (develop)

_No changes yet._

## [0.7.1] - 2026-06-21

A hardening and chat-UX release: the new `victor ui` chat surface gained controls and
resilience, per-turn cost/token accounting was finished end-to-end, and a sweep of
streaming, tool-selection, governance, and indexing bugs surfaced by live use were fixed.
`victor-contracts` stays at 0.7.0 (no protocol changes).

### Added
- **Chat UI controls & resilience** — stop/cancel an in-flight turn, close-guard so a
  mid-stream disconnect can't tear down the provider, and error recovery (#144)
- **Chat settings panel + session-restore seam** — provider/profile/model/approval controls
  and best-effort replay of a reconnected session's history (#146)
- **`victor ui --profile`** to select an agent profile for the chat surface
- **Informed tool approvals** — the policy ASK prompt now carries tool args and a human
  summary; per-call tool steps, accumulating reasoning, and first-token feedback in the UI (#135)
- **Tool telemetry in chat** — per-call duration, real output, output pruning, and follow-up
  suggestions surfaced as steps (#135)
- **Per-turn cost trace (C0)** — a cost/latency/token footer (L3) and cost-aware model
  routing that biases toward cheaper/faster providers on expensive turns (L4) (#139)
- **Reference-aware tool-result pruning** — context compaction that keeps referenced results,
  enabled by default (L1) (#137, #138)
- **`victor db maintain`** — checkpoint the WAL and reclaim space on the global and project
  databases (#153)
- RL learning-trace merge caching within a task (L2, flag-gated)

### Changed
- **CI split** — lightweight fast checks on `develop`, extensive gating reserved for
  `develop → main`; develop quick-tests scoped to changed-file unit tests (#136, #142)
- Scheduled security scan runs weekly instead of nightly (#145)
- Docs Pages deploy can be triggered on-demand via `workflow_dispatch` (#156)
- New project databases open with `auto_vacuum=INCREMENTAL`; `graph_module_metric_history`
  is capped per module to stop unbounded growth (#153)

### Fixed
- **Streaming cleanup** — the provider SSE stream is now closed on-task across the full
  consumer → resilience → provider decorator chain, eliminating "async generator ignored
  GeneratorExit" / "exit cancel scope in a different task" spam; also corrected the streaming
  `ToolExecutionResult` shape used for plateau/progress accounting (#152, #154, #155)
- **`web_search` reaches the model** — a deliberately-selected web tool is no longer dropped
  by stage pruning or the order-blind post-selection truncation on research-flavored tasks
  (#149, #157)
- **Governance approval in the chat UI** — the interactive ASK approval handler now survives
  the DI container bootstrap (and is registered via the correct API), so ASK-gated tools
  prompt instead of silently denying (#150)
- **No mutation tools forced onto read-only tasks** — write-intent no longer injects
  edit/write/shell on analyze/search/research turns; tool-call loops are detected earlier (#151)
- **`code_search` fails fast** when the semantic indexing provider is absent — falls back to
  literal search in <1s instead of a ~600s index-lock hang (#158)
- **Token accounting** — per-turn input/output tokens are measured end-to-end from provider
  usage and finalized on the service path; Ollama usage surfaced on the streaming done chunk
  (#134)
- **Graph tool** — stop advertising unimplemented modes (`dead_code`, `dynamic_imports`);
  suggest same-file symbols with an unambiguous basename fallback for `file:Symbol` (#125)
- Chainlit 2.x compatibility for chat-UI tool approval

### Security
- Remediated Semgrep SAST findings across workflows, CI, and Docker, and cleared the
  remaining Semgrep OSS findings blocking `develop → main` (#140, #148)
- Upgraded vulnerable dependencies (FastAPI/Starlette, transformers) to clear CVEs (#143)

### Removed
- Superseded standalone Vite UIs (`web/ui`, `ui/`) in favor of the Chainlit `victor ui` (#141)

## [0.7.0] - 2026-06-19

### Added
- **`victor ui`** — a pure-Python Chainlit web chat surface bound to `VictorClient` (streams tokens, renders reasoning and tool calls as steps); behind the optional `chat-ui` extra
- **Human-in-the-loop tool approval** for non-TTY surfaces — `SessionConfig.tool_approval` turns on the governance ASK path for named tools; `register_policy_approval_handler()` registers a surface's approval handler in the DI container
- **Single FastAPI HTTP server** — `/lsp/*` ported onto the core server and new `/credentials/{set,delete,status}` + `/tools/cancel` routes, so the full IDE/extension API surface lives on one server
- **VS Code extension ↔ FastAPI contract test** — guards endpoint drift between the extension client and backend routes
- **VS Code extension** unit harness — vitest + c8 coverage with a `vscode` mock (21 logic-module tests), wired into CI
- **ExtensionManifest** and **ExtensionType** in victor-contracts for vertical capability declaration
- **CapabilityNegotiator** for manifest validation during vertical activation
- **API versioning** (`CURRENT_API_VERSION=2`, `MIN_SUPPORTED_API_VERSION=1`)
- **CallbackCoordinator** for tool/streaming lifecycle callback delegation
- **OrchestratorPropertyFacade** — 37 properties extracted to separate module
- **InitializationPhaseManager** — structured 8-phase runtime initialization
- **ExtensionModuleResolver** — module path resolution extracted from extension loader
- **ExtensionCacheManager** — thread-safe namespaced cache for extensions
- SBOM generation (CycloneDX) in release pipeline
- Tests for `tool_pipeline`, `cqrs`, `workflows/executor`, `sqlite_lancedb` (46 new tests)
- `[embeddings]` optional extra for `sentence-transformers`, `lancedb`, `pyarrow`
- Deprecation warnings on all 5 contrib vertical imports
- TODO/FIXME triage document (81 markers categorized)

### Changed
- **Consolidated to a single HTTP server** — the legacy aiohttp `VictorAPIServer` was removed; `VictorFastAPIServer` is now the sole server and its routers own the full API surface
- **VS Code extension** upgraded to Node 22, TypeScript 5.9, and typescript-eslint 8
- **Orchestrator** reduced from 4,514 to 3,940 LOC via property/callback extraction
- **Extension loader** reduced from 2,049 to 1,897 LOC via resolver/cache extraction
- **protocols.py** decomposed from 3,703 LOC monolith into 9 domain modules with lazy `__getattr__` loading
- **ProviderPool** duplicate removed; wired with `use_provider_pooling` feature flag
- **Settings** flat access now emits `DeprecationWarning` (use nested groups instead)
- Victor-devops: 3 forbidden imports migrated to `victor.framework.extensions`
- 32 documentation files reconciled (provider count 22→24, tool count 33→34)
- Heavyweight dependencies (`sentence-transformers`, `lancedb`, `pyarrow`) moved to `[embeddings]` extra
- `victor chat` no longer exposes `--legacy`; the canonical framework client path is now the only chat path

### Deprecated
- **`FrameworkShim` compatibility surface**
  - Deprecated in: `0.7.0` on `2026-04-30`
  - To be removed in: `v1.0.0`
  - Target removal date: `2027-06-30`
  - Replacement: `Agent.create()` for public callers or `AgentFactory` / `AgentCreationFactory` for internal composition
  - Migration path: `docs/architecture/migration.md`
  - Compatibility shim status: warning-backed shim remains supported through `v1.0.0`
- **`TeamNode*` workflow compatibility aliases** (`TeamNode`, `TeamNodeConfig`, `TeamNodeWorkflow`, `TeamNodeExecutor`)
  - Deprecated in: `0.7.0` on `2026-04-30`
  - To be removed in: `v0.9.0`
  - Target removal date: `2027-03-31`
  - Replacement: `TeamStep*` workflow names
  - Migration path: `docs/architecture/migration.md`
  - Compatibility shim status: warning-backed aliases remain supported through `v0.9.0`
- **`WorkflowGraph` alias from `victor.workflows.graph`**
  - Deprecated in: `0.7.0` on `2026-04-30`
  - To be removed in: `v0.8.0`
  - Target removal date: `2026-12-31`
  - Replacement: `BasicWorkflowGraph` for the simple container or `victor.workflows.graph_dsl.WorkflowGraph` for the typed DSL
  - Migration path: `docs/architecture/migration.md`
  - Compatibility shim status: warning-backed alias remains supported through `v0.8.0`

### Fixed
- **Release pipeline** — the SBOM job installs the in-repo `victor-contracts` before victor-ai (it is not on PyPI), and the native-wheel build passes `--find-interpreter` so maturin finds a CPython under `manylinux: auto`
- Bare `except:` in `experiments.py` → `except (ValueError, TypeError)`
- `_send_rl_reward_signal` test updated for CallbackCoordinator delegation

### Security
- `eval()`/`exec()` audit: only 2 real calls, both sandboxed with `__builtins__: {}`
- SecretStr used for sensitive config fields (17 occurrences)

## [0.6.0] - 2026-03-31

### ⚠️ Breaking Changes
- **NONE** - This release maintains 100% backward compatibility

### Added

**Core Architecture** (Phases 1-7):
- **@register_vertical decorator** for declarative vertical registration with rich metadata
- **VerticalMetadata** dataclass for type-safe metadata extraction (replaces fragile `.replace()` patterns)
- **UnifiedEntryPointRegistry** for single-pass entry point scanning (31x faster: 500ms → 16ms)
- **VersionCompatibilityMatrix** with PEP 440 version constraint checking
- **ExtensionDependencyGraph** with topological sort and circular dependency detection
- **AsyncSafeCacheManager** with lock-per-key caching (eliminates race conditions)
- **OpenTelemetry integration** for production observability and monitoring
- **PluginNamespaceManager** for priority-based plugin namespace isolation
- **VerticalBehaviorConfigRegistry** for dynamic vertical configuration

**Testing & Validation** (Phase 8):
- 224 new tests with 100% pass rate
- 93% average code coverage across all new modules
- Performance benchmarks for entry point scanning and dependency resolution
- Backward compatibility tests validating legacy patterns

**Documentation** (Phase 9):
- Architecture documentation (~1,000 lines) with before/after comparison
- Migration guide (~900 lines) with step-by-step examples
- Complete API reference (~1,200 lines) for all new modules
- Best practices guide (~1,100 lines) for vertical development

**Operations** (Phase 10):
- Rollout plan (~1,500 lines) with 5-stage gradual deployment
- Monitoring dashboards (~1,100 lines) with 4 Grafana dashboards
- Deployment playbook (~1,000 lines) with blue-green procedures
- Legacy deprecation plan (~800 lines) with removal timeline

**Vertical Alignment** (Phase 11):
- All 6 external vertical packages aligned with SDK/framework
- victor-coding, victor-devops, victor-rag updated to v0.6.0
- victor-dataanalysis, victor-research, victor-invest updated to v0.6.0
- Validation script (`scripts/validate_verticals.py`) for automated checking

### Changed

**Performance**:
- Entry point scanning: 9+ independent calls → 1 unified scan (200-500ms startup improvement)
- Scan duration: 500ms → 16ms (31x faster)
- Dependency resolution: 2-5ms for complex graphs
- Cache contention: Single lock → Lock-per-key (enables parallel loading)

**Architecture**:
- Extension loader: 1,897 LOC with SOLID-compliant focused modules
- Vertical metadata: Type-safe extraction replaces fragile string manipulation
- Configuration: Externalized from hardcoded dict to dynamic registry
- Safety rules: Refactored from internal coordinators to framework capabilities

**Code Quality**:
- SOLID principles applied throughout new modules
- Protocol-based design (Interface Segregation Principle)
- Dependency Inversion - depends on abstractions, not concretions
- Single Responsibility - focused, cohesive modules

### Fixed

**Class Name Generation**:
- Fixed: Metadata extraction using pattern matching instead of `.replace()`
- Fixed: Type-safe `canonical_name` extraction preserves naming intent
- Fixed: Display name generation with proper title casing

**Import Boundaries**:
- Fixed: victor-dataanalysis safety rules now use `SafetyRulesCapabilityProvider` from framework
- Fixed: victor-rag safety rules now use `SafetyRulesCapabilityProvider` from framework
- Fixed: Removed forbidden imports from `victor.agent.coordinators.safety_coordinator`

### Deprecated

**Legacy Patterns** (with DeprecationWarnings):
- Class name extraction using `.replace("Assistant", "")` pattern
- Class name extraction using `.replace("Vertical", "")` pattern
- Hardcoded `_VERTICAL_CANONICALIZE_SETTINGS` dict
- Multiple independent `entry_points()` calls (use `UnifiedEntryPointRegistry`)
- Direct `victor.agent.coordinators` imports (use framework capabilities)

**Removal Timeline**: September 2026 (v0.7.0)

### Security

- Safety rule validation prevents dangerous operations
- PEP 440 version checking prevents incompatible framework versions
- OpenTelemetry integration enables security monitoring
- Secret masking in DevOps middleware

### Performance

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Entry Point Scans | 9+ | 1 | 200-500ms saved |
| Scan Duration | ~500ms | ~16ms | 31x faster |
| Startup Latency | Baseline + 500ms | Baseline | 200-500ms faster |
| Dependency Resolution (simple) | N/A | ~2ms | <5ms target ✅ |
| Dependency Resolution (complex) | N/A | ~3-5ms | <10ms target ✅ |
| Cycle Detection | N/A | ~1-2ms | <5ms target ✅ |

### Migration

**For Users**: No action required - all existing verticals work without modification

**For Vertical Developers**: Optional - add `@register_vertical` decorator:
```python
from victor.core.verticals.registration import register_vertical

@register_vertical(
    name="my_vertical",
    version="1.0.0",
    min_framework_version=">=0.6.0",
)
class MyVertical(VerticalBase):
    pass
```

**Documentation**: See `docs/verticals/migration_guide.md`

### Known Issues

1. **Class name construction** in contrib verticals uses lowercase `canonical_name`, causing incorrect camelCase instead of PascalCase (e.g., `ragRLConfig` vs `RAGRLConfig`). External verticals are not affected. Fix scheduled for v0.6.1.
2. **Conversation coordinator** usage in victor-dataanalysis and victor-rag uses internal API. Documented with TODO for future refactoring.

See `docs/verticals/KNOWN_ISSUES_v0.6.0.md` for details.

### Contributors

- Architecture design and implementation: Claude Code AI Assistant
- Testing: 224 tests with 93% coverage
- Documentation: 9,200+ lines across 13 files
- Validation: Automated alignment checking for all 6 vertical packages

---

## [0.5.7] - 2026-03-07

### Added
- Victor-SDK v0.5.7 with zero-runtime-dependency vertical definitions
- Capability negotiation and state externalization (Phase 4)
- Unified victor-ai / victor-contracts versioning with CI enforcement
- Processing/LSP re-export modules for external vertical imports
- CopyOnWriteState thread guard and entry point failure isolation
- Topic-prefix index for O(1) event dispatch
- Collision detection and public extension API for verticals
- Fast CI workflow for quicker feedback

### Changed
- Black formatting applied to 79 files
- FastAPI server hardened against injection, traversal, and data exposure
- SecretStr for API keys to prevent credential leakage

### Fixed
- VS Code extension: handle offline servers in `supportsCapability` check
- Build: victor-contracts built locally in release workflow

## [0.5.6] - 2026-03-01

### Fixed
- Added strawberry-graphql to dev dependencies for GraphQL integration tests

## [0.5.5] - 2026-02-28

### Added
- Initial public release with full framework, 22 providers, 33 tools
- Multi-agent team formations (sequential, parallel, hierarchical, pipeline)
- YAML workflow DSL with StateGraph execution engine
- 9 domain verticals (coding, devops, rag, dataanalysis, research, security, iac, classification, benchmark)
- CLI (Typer) and TUI (Textual) interfaces
- Docker support with multi-arch builds
