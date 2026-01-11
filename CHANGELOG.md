# Changelog

All notable changes to Victor will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Presentation Abstraction Layer** - Decouples UI concerns from core logic with `PresentationAdapter` protocol
- **Registry Base Classes** - Centralized registry infrastructure in `victor/core/registry_base.py`
- **Tool Selector Protocol** - Unified `IToolSelector` protocol for tool selection strategies
- **SOLID Consolidation (Complete)** - 8-phase architectural refactoring for vertical configuration:
  - **Universal Registry System** - Type-safe generic registry with cache strategies (TTL, LRU, Manual, None)
  - **Data-Driven Mode Configuration** - YAML-based mode configs in `victor/config/modes/` for all verticals
  - **RL Configuration Consolidation** - Centralized RL configs in `victor/config/rl/` with task mappings and quality thresholds
  - **Capability System** - YAML-based capability registry in `victor/config/capabilities/` with 5 types (tool, workflow, middleware, validator, observer)
  - **Team Specifications** - YAML-based team configs in `victor/config/teams/` with 5 formations (pipeline, parallel, sequential, hierarchical, consensus)
  - **VerticalBase API** - New methods: `get_mode_config()`, `list_modes()`, `get_rl_config()`, `get_capabilities()`, `list_capability_names()`, `get_capability_provider()`, `get_team_provider()`, `get_team()`, `list_teams()`
  - **Core Modules** - `victor/core/capabilities/` with CapabilityLoader, `victor/core/teams/` with BaseYAMLTeamProvider
  - **Mode-Aware Loop Detection** - Context-aware LoopSignature class for mode-specific loop detection
  - **Database Migration System** - Version-based schema evolution with safe column addition

### Changed
- **Architecture Improvements** - Phase 1 and 2 refactoring for cleaner module boundaries
- **Workflow Compiler** - Infrastructure improvements for reliability
- **Tool Selection Defaults** - Centralized configuration in settings module
- **Deprecated Workflow Executor APIs** - Removed legacy execution engine stubs
- **Configuration System** - All vertical configurations now YAML-first with Python extensions
- **Grounding Verifier** - Enhanced path validation for nested project structures

### Fixed
- **Integration Test Blocking** - Resolved issues causing test hangs
- **PersonaProvider Deadlock** - Fixed RLock deadlock in persona provider
- **Import Errors** - Resolved import errors across integration and unit tests
- **Service Provider Types** - Correct handling of string service types in DI container
- **Nested Path Validation** - Fixed logic to properly validate suffix-based path references
- **Loop Detection** - Mode-aware loop detection now uses exploration multipliers correctly

### Removed
- **Dead Code Pruning** - Removed unused code from graph stores and chunker modules
- **Backward Compatibility Aliases** - Migrated to canonical names across codebase

## [0.4.0] - 2026-01-03

### Added
- **TD-002 Orchestrator Refactoring** - Dedicated coordinators for cleaner separation of concerns
- **YAML-First Workflow Architecture** - Workflows defined in YAML with Python escape hatches for complex logic
- **CLI Commands**:
  - `victor vertical create` - Scaffold new verticals
  - `victor workflow render` - Render workflow diagrams
  - `victor dashboard` - Interactive dashboard for monitoring
- **Bedrock Tool Calling Adapter** - AWS Bedrock provider with native tool support
- **Azure OpenAI Tool Calling Adapter** - Azure-specific tool calling implementation
- **Session Cost Tracking** - Display cumulative session costs in UI
- **Entry-Point Plugin System** - External verticals via `pyproject.toml` entry points

### Changed
- **Documentation Reorganization** - Restructured docs for OSS release
- **Positive-Only Messaging** - Removed competitor references from documentation
- **Updated SVG Diagrams** - All architecture diagrams updated for accuracy

### Fixed
- **CI/CD Pipeline** - Various fixes for reliable builds
- **Project Hash Isolation** - Embedding cache now isolated per project
- **Exception Handling** - Specific exception types throughout codebase
- **LSP Compliance** - Fixed LSP compliance issues in LMStudio provider

## [0.3.0] - 2025-12-31

### BREAKING CHANGES
- **Tool Selection API** - `use_semantic_tool_selection` setting replaced by `tool_selection_strategy`
- **ToolSelector.select_tools()** - `use_semantic` parameter removed; strategy configured globally
- **Protocols Renamed** - `ToolSelectorProtocol` and `SemanticToolSelectorProtocol` replaced by `IToolSelector`

**Migration Guide**:
```python
# Old (deprecated)
settings = Settings(use_semantic_tool_selection=True)
tools = await selector.select_tools(message, use_semantic=True)

# New (v0.3.0+)
settings = Settings(tool_selection_strategy="semantic")  # or "keyword", "hybrid", "auto"
tools = await selector.select_tools(message)
```

### Added

#### HIGH-002: Unified Tool Selection Architecture
- **IToolSelector Protocol** - Unified interface for all tool selection strategies with `select_tools()`, `get_supported_features()`, `record_tool_execution()`, and `close()` methods
- **KeywordToolSelector** - Fast registry-based selection (<1ms, no embeddings required) using tool metadata from @tool decorators
- **HybridToolSelector** - Blends semantic and keyword strategies with configurable weights (default: 70% semantic, 30% keyword)
- **Strategy Factory** - `create_tool_selector_strategy()` for creating selectors with auto-selection logic based on air-gapped mode and embedding availability
- **ToolSelectionContext** - Dataclass encapsulating context (conversation history, stage, vertical mode, etc.) for tool selection
- **Shared Utilities** - Extracted `selection_filters.py` (pure functions) and `selection_common.py` (stateful utilities) for reuse across strategies

#### Server & Extension Enhancements
- **FastAPI Server Backend** - New `victor serve --backend fastapi` option with OpenAPI docs at `/docs`
- **Server Backend Selection** - `--backend` flag for `victor serve` to choose between `aiohttp` (legacy) and `fastapi` (modern)
- **Profile CRUD Operations** - New `victor profiles create/edit/delete/set-default` commands for full profile management
- **Lightweight Tool Listing** - `victor tools list --lightweight` for fast tool discovery without agent initialization

#### P4.X Multi-Provider Excellence
- **Semantic Query Expansion** - Automatic query expansion with synonyms/related terms to fix false negatives in semantic search
- **Tool Deduplication Tracker** - Prevents redundant tool calls by tracking recent operations and detecting semantic overlap (integrated into ToolPipeline)
- **Provider-Specific Tool Guidance** - Each provider now boosts tools aligned with their strengths (Gemini -> code analysis, Claude -> reasoning, etc.)
- **RL-Based Semantic Threshold Learning** - Learns optimal similarity thresholds per (embedding_model, task_type, tool) context (integrated into code_search tool)
- **Hybrid Search (RRF)** - Combines semantic + keyword search using Reciprocal Rank Fusion for 50-80% better recall (integrated into code_search tool)
- **Comprehensive Unit Tests** - 66 new unit tests for query expansion and tool deduplication (100% coverage)
- **Integration Tests** - 16 new integration tests for P4 Multi-Provider Excellence features
- **Monitoring Script** - `scripts/show_semantic_threshold_rl.py` for viewing threshold learning status and exporting recommendations

### Changed

#### Tool Selection Architecture Refactor
- **Tool Selection Strategy** - New `tool_selection_strategy` setting (auto/keyword/semantic/hybrid) replaces `use_semantic_tool_selection`
- **ToolSelector API** - `select_tools()` now delegates to injected strategy instead of `use_semantic` parameter
- **Strategy Injection** - ToolSelector accepts optional `IToolSelector` strategy for clean dependency injection
- **Auto-Selection** - "auto" strategy intelligently picks keyword (air-gapped), semantic (embeddings available), or hybrid based on environment

#### Framework & Infrastructure
- **RL Framework Migration Complete** - Unified all 3 RL learners (continuation prompts, semantic threshold, model selector) into centralized framework with SQLite storage at `~/.victor/graph/graph.db`. Deprecated bespoke implementations moved to `archive/deprecated_rl_modules/`. All 19 import sites across API servers, UI, and scripts updated to use `RLCoordinator`.
- **VS Code Extension Server Discovery** - Auto-discovers existing servers on multiple ports before spawning new ones
- **VS Code Port Fallback** - Tries fallback ports (8765, 8766, 8767, 8768, 8000) if primary port is occupied
- **VS Code Exponential Backoff** - Improved reconnection with exponential backoff (100ms -> 30s max, 10 retries)
- **VS Code Multi-Window Sharing** - PID file at `~/.victor/server.pid` for server coordination across VS Code windows
- **VS Code Extension Configuration** - New settings: `victor.serverBackend`, `victor.fallbackPorts`
- **Semantic Similarity Threshold** - Lowered from 0.7 to 0.5 to reduce false negatives and improve recall
- **RL Continuation Bounds** - Expanded from [2, 12] to [1, 20] to give RL more liberty for provider-specific tuning
- **ToolPipeline** - Now supports optional deduplication tracker for preventing redundant calls
- **Code Search Tool** - Automatically records outcomes for RL threshold learning when enabled

### Removed
- **Settings**: `use_semantic_tool_selection` (use `tool_selection_strategy` instead)
- **ToolSelector.select_tools()**: `use_semantic` parameter (strategy now configured globally)
- **Protocols**: `ToolSelectorProtocol` and `SemanticToolSelectorProtocol` (use `IToolSelector` instead)

### Fixed
- Added missing `ServerStatus.Reconnecting` state to VS Code extension status bar configuration
- **Semantic Search False Negatives** - Query expansion now searches with multiple variations (e.g., "tool registration" -> ["register tool", "@tool decorator", "ToolRegistry"])
- **Google SDK Warning** - Suppressed cosmetic warning about non-text parts (Victor already handles multi-part responses correctly)

### Configuration
New settings for P4 Multi-Provider Excellence features (all disabled by default for backward compatibility):
```yaml
# Hybrid Search (Semantic + Keyword with RRF)
enable_hybrid_search: false
hybrid_search_semantic_weight: 0.6
hybrid_search_keyword_weight: 0.4

# RL-based threshold learning per (embedding_model, task_type, tool_context)
enable_semantic_threshold_rl_learning: false
semantic_threshold_overrides: {}  # Format: {"model:task:tool": threshold}

# Tool call deduplication
enable_tool_deduplication: false
tool_deduplication_window_size: 10

# Semantic search quality improvements
semantic_similarity_threshold: 0.5  # Lowered from 0.7
semantic_query_expansion_enabled: true
semantic_max_query_expansions: 5
```

## [0.2.3] - 2025-12-27

### Added
- **Homebrew Tap** - Install via `brew install vjsingh1984/tap/victor`
- **Auto-updating Homebrew Formula** - Polls PyPI every 6 hours for new releases

### Changed
- **Docker Hub Only** - Removed GHCR push to simplify deployment (Docker Hub: `vjsingh1984/victor-ai`)
- **Simplified Binary Builds** - macOS and Windows binaries only; Linux users should use `pip install victor-ai`

### Fixed
- **Docker Build Disk Space** - Added cleanup step to free ~25GB on GitHub Actions runners
- **macOS Runners** - Updated to `macos-15-intel` (macos-13 was retired)
- **CI/CD Reliability** - Streamlined release workflow for consistent builds

## [0.2.2] - 2025-12-27

### Fixed
- **Docker Build** - Added disk space cleanup for large PyTorch builds
- **GHCR Permissions** - Added packages:write permission (later removed in 0.2.3)

## [0.2.1] - 2025-12-27

### Fixed
- **Docker profiles.yaml** - Created `docker/profiles.yaml.example` for containerized deployments
- **Dockerfile** - Falls back to example file when profiles.yaml is gitignored

## [0.2.0] - 2025-12-27

### Added

#### Release Infrastructure
- **PyPI Trusted Publishing** - OIDC-based publishing without API tokens
- **GitHub Actions Release Pipeline** - Automated builds on tag push
- **Multi-Platform Binaries** - macOS ARM64/x64, Windows x64 via PyInstaller
- **Docker Images** - Pre-built containers with embedded models at `vjsingh1984/victor-ai`
- **Rust Native Extensions** - PyO3 bindings for SIMD-optimized operations
- **Homebrew Tap** - `vjsingh1984/homebrew-tap` with auto-update workflow

#### Provider Enhancements
- **Cerebras Provider** - Qwen-3 thinking filter and deduplication
- **vLLM Provider** - Local production serving with fallback tool parsing
- **llama.cpp Provider** - Direct GGUF model support
- **Keyring Support** - Secure API key storage for all cloud providers
- **TDD Tests** - Comprehensive provider integration tests

### Changed
- **Version** - Bumped to 0.2.0 for first public release

## [0.2.0-alpha] - 2025-12-02

### Added
- **Modern TUI** - Rich terminal interface powered by Textual with markdown rendering, syntax highlighting, and status bar showing provider/model/tokens
- **Stream Cancellation** - Press Ctrl+C to cancel streaming responses mid-generation
- **Debug CLI Mode** - Use `--renderer text` with `--log-level DEBUG` for plain console output with visible debug logs
- **Conceptual Query Detection** - Semantic tool selector now detects inheritance/pattern queries and automatically routes to `semantic_code_search` instead of keyword-based `code_search`
- **AST-Aware Code Chunking** - Semantic search chunker uses AST parsing for more intelligent code segmentation
- **Conversation State Machine** - Session restoration with `ConversationStateMachine` tracking conversation stages (INITIAL, EXPLORING, ANALYZING, IMPLEMENTING, REVIEWING)
- **Gitleaks Allowlist** - Test files with fake secrets are now properly excluded from secret scanning

### Changed
- **Semantic Tool Selection** - `semantic_code_search` added to search tool category alongside `code_search`
- **Embedding Registry** - Fixed `config.provider` -> `config.vector_store` bug in registry.py
- **Tool Descriptions** - Improved `code_search` and `semantic_code_search` descriptions to better guide LLM selection

### Fixed
- Gemini safety settings now include additional categories (HATE_SPEECH, DANGEROUS_CONTENT, CIVIC_INTEGRITY)
- Google provider logging added for debugging safety filter issues
- Resolved all ruff linting errors across codebase

### Documentation
- Consolidated MCP documentation (deleted redundant MCP_SETUP.md)
- Added cross-reference from AIRGAPPED_GUIDE.md to comprehensive AIRGAPPED.md
- Updated docs navigation structure

## [0.1.0-alpha] - 2025-02-27

### Added
- GitHub Actions CI matrix (3.10-3.12) running black, ruff, mypy, pytest (unit-only), and a CLI smoke test.
- Security workflow with gitleaks secret scan and pip-audit dependency check.
- Support policy (`SUPPORT.md`) and archive notice clarifying `archive/victor-legacy/` is frozen.

### Changed
- Scoped Ruff to active packages and cleaned up outstanding lint errors in core modules.
- Updated onboarding guidance to point at `docs/guides/QUICKSTART.md` and flagged aspirational docs with "planned" status notes.
- Expanded `.gitignore` for local artifacts (debug logs, demo workspace, .victor metadata).

### Fixed
- Addressed formatting/lint issues across tooling modules (imports, f-strings, abstract methods).
- Stabilized dynamic tool discovery imports and tightened legacy boundaries in documentation.

### Added
- Comprehensive community documentation (CONTRIBUTING.md, CODE_OF_CONDUCT.md)
- GitHub issue templates (bug report, feature request, question)
- Pull request template
- This CHANGELOG

## [0.1.0] - 2025-11-24

### Added

#### Core Features
- Universal provider abstraction supporting multiple LLM providers
- Ollama integration with full tool calling support
- Anthropic Claude integration (Sonnet 4.5, Opus, Haiku)
- OpenAI integration (GPT-4, GPT-4 Turbo, GPT-3.5)
- Google Gemini integration (1.5 Pro, 1.5 Flash)
- xAI Grok integration
- LMStudio and vLLM provider support
- Configuration management with YAML profiles
- Rich terminal UI with streaming responses
- Interactive REPL and one-shot command modes

#### Tools
- **File Operations**: Read, write, edit files
- **Multi-File Editor**: Transaction-based atomic edits with rollback
- **Bash Execution**: Safe command execution
- **Git Integration**: AI-powered commits, staging, branching, diff
- **Web Search**: Fetch documentation and resources
- **Semantic Search**: AI-powered codebase indexing and search
- **Database Tool**: SQLite, PostgreSQL, MySQL, SQL Server support
- **Docker Tool**: Container and image management
- **HTTP Tool**: API testing and HTTP requests

#### MCP Protocol
- Full Model Context Protocol server implementation
- MCP client for connecting to external servers
- JSON-RPC 2.0 compliance
- Stdio transport for Claude Desktop integration
- Tool and resource discovery
- Dual format parameter support (List[ToolParameter] and JSON Schema)

#### Architecture
- Provider abstraction layer for easy LLM integration
- Tool registry with dynamic registration
- Plugin architecture for extensibility
- Type-safe with Pydantic models
- Async/await throughout for performance
- Transaction manager for multi-file operations
- Context management and semantic search system

### Developer Experience
- Modern Python 3.10+ with async/await
- Comprehensive test suite
- Well-structured codebase following best practices
- Extensive documentation and examples
- Pre-commit hooks for code quality
- Black formatting, Ruff linting, MyPy type checking

### Documentation
- Comprehensive README with architecture diagrams
- Configuration guide
- Example scripts for all major features
- Complete session summary documenting implementation
- API documentation via docstrings

### Examples
- Basic usage demo
- MCP server and client demos
- Advanced tools demos (database, Docker, HTTP)
- Codebase indexing demo
- Git integration demo
- Multi-file editing demo

## [0.0.1] - 2025-11-01

### Added
- Initial project structure
- Basic provider abstraction
- Ollama provider implementation
- Simple file and bash tools
- Basic terminal interface

---

## Types of Changes

- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** for vulnerability fixes

---

[Unreleased]: https://github.com/vjsingh1984/victor/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/vjsingh1984/victor/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/vjsingh1984/victor/compare/v0.2.3...v0.3.0
[0.2.3]: https://github.com/vjsingh1984/victor/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/vjsingh1984/victor/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/vjsingh1984/victor/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/vjsingh1984/victor/compare/v0.2.0-alpha...v0.2.0
[0.2.0-alpha]: https://github.com/vjsingh1984/victor/compare/v0.1.0-alpha...v0.2.0-alpha
[0.1.0-alpha]: https://github.com/vjsingh1984/victor/compare/v0.1.0...v0.1.0-alpha
[0.1.0]: https://github.com/vjsingh1984/victor/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/vjsingh1984/victor/releases/tag/v0.0.1
