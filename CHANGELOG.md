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

### Fixed
- Bare `except:` in `experiments.py` → `except (ValueError, TypeError)`
- `_send_rl_reward_signal` test updated for CallbackCoordinator delegation

### Security
- `eval()`/`exec()` audit: only 2 real calls, both sandboxed with `__builtins__: {}`
- SecretStr used for sensitive config fields (17 occurrences)

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
