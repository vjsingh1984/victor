# Changelog

All notable changes to Victor will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0-alpha] - 2025-12-02

### Added
- **Modern TUI** - Rich terminal interface powered by Textual with markdown rendering, syntax highlighting, and status bar showing provider/model/tokens
- **Stream Cancellation** - Press Ctrl+C to cancel streaming responses mid-generation
- **Debug CLI Mode** - Use `--no-tui` or `--cli` flag for console output with visible debug logs
- **Conceptual Query Detection** - Semantic tool selector now detects inheritance/pattern queries and automatically routes to `semantic_code_search` instead of keyword-based `code_search`
- **AST-Aware Code Chunking** - Semantic search chunker uses AST parsing for more intelligent code segmentation
- **Conversation State Machine** - Session restoration with `ConversationStateMachine` tracking conversation stages (INITIAL, EXPLORING, ANALYZING, IMPLEMENTING, REVIEWING)
- **Gitleaks Allowlist** - Test files with fake secrets are now properly excluded from secret scanning

### Changed
- **Semantic Tool Selection** - `semantic_code_search` added to search tool category alongside `code_search`
- **Embedding Registry** - Fixed `config.provider` → `config.vector_store` bug in registry.py
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
- GitHub Actions CI matrix (3.10–3.12) running black, ruff, mypy, pytest (unit-only), and a CLI smoke test.
- Security workflow with gitleaks secret scan and pip-audit dependency check.
- Support policy (`SUPPORT.md`) and archive notice clarifying `archive/victor-legacy/` is frozen.

### Changed
- Scoped Ruff to active packages and cleaned up outstanding lint errors in core modules.
- Updated onboarding guidance to point at `docs/guides/QUICKSTART.md` and flagged aspirational docs with “planned” status notes.
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

## Version History

### [0.1.0] - The Foundation Release

This is the first major release of Victor, establishing it as a production-ready
AI coding assistant with enterprise-grade features.

**Key Highlights**:
- 15+ tools for comprehensive development workflows
- MCP protocol support for Claude Desktop integration
- Multi-file transaction system with atomic edits
- Advanced database, Docker, and HTTP capabilities
- Semantic search for intelligent codebase navigation
- Support for 6 LLM providers (frontier and local)

**Statistics**:
- ~8,000 lines of code written
- 26 files created
- 15+ tools implemented
- 6 LLM providers supported
- 90+ commits

**Community**:
- Open-sourced under MIT license
- Comprehensive contribution guidelines
- Professional documentation
- Ready for community contributions

---

## Types of Changes

- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** for vulnerability fixes

## Links

- [Repository](https://github.com/vjsingh1984/victor)
- [Issues](https://github.com/vjsingh1984/victor/issues)
- [Discussions](https://github.com/vjsingh1984/victor/discussions)
