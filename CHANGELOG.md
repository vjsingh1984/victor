# Changelog

All notable changes to Victor are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] (develop)

### Added
- **ExtensionManifest** and **ExtensionType** in victor-sdk for vertical capability declaration
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
- **Orchestrator** reduced from 4,514 to 3,940 LOC via property/callback extraction
- **Extension loader** reduced from 2,049 to 1,897 LOC via resolver/cache extraction
- **protocols.py** decomposed from 3,703 LOC monolith into 9 domain modules with lazy `__getattr__` loading
- **ProviderPool** duplicate removed; wired with `use_provider_pooling` feature flag
- **Settings** flat access now emits `DeprecationWarning` (use nested groups instead)
- Victor-devops: 3 forbidden imports migrated to `victor.framework.extensions`
- 32 documentation files reconciled (provider count 22→24, tool count 33→34)
- Heavyweight dependencies (`sentence-transformers`, `lancedb`, `pyarrow`) moved to `[embeddings]` extra

### Deprecated
- **`TeamNode*` workflow compatibility aliases** (`TeamNode`, `TeamNodeConfig`, `TeamNodeWorkflow`, `TeamNodeExecutor`)
  - Deprecated in: `Unreleased` on `2026-04-30`
  - To be removed in: `v0.9.0`
  - Target removal date: `2027-03-31`
  - Replacement: `TeamStep*` workflow names
  - Migration path: `docs/architecture/migration.md`
  - Compatibility shim status: warning-backed aliases remain supported through `v0.9.0`

### Fixed
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
- Unified victor-ai / victor-sdk versioning with CI enforcement
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
- Build: victor-sdk built locally in release workflow

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
