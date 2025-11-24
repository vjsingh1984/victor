# Changelog

All notable changes to CodingAgent will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-11-24

### Added
- Initial release of CodingAgent
- Universal provider abstraction layer supporting multiple LLM providers
- Ollama provider integration for local model inference
- Tool calling framework with filesystem and bash tools
- Configuration management system with YAML profiles
- Terminal UI with Rich formatting and streaming support
- Interactive REPL mode for conversations
- Basic test suite with pytest
- Comprehensive documentation and examples
- CI/CD pipeline with GitHub Actions

### Features
- **Multi-Provider Support**: Foundation for Anthropic, OpenAI, Google, and local models
- **Ollama Integration**: Full support for local Ollama models with tool calling
- **Tool System**:
  - File operations (read, write, list directory)
  - Bash command execution with safety checks
  - Extensible tool registry
- **Configuration**: YAML-based profiles and provider settings
- **CLI**: Interactive and one-shot modes with typer and rich
- **Streaming**: Real-time streaming responses from models
- **Type Safety**: Full Pydantic models for data validation

### Developer Experience
- Modern Python 3.10+ with type hints
- Async/await throughout
- Comprehensive test coverage
- Easy to extend with new providers and tools
- Well-structured codebase following best practices

[Unreleased]: https://github.com/vijaysingh/victor/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/vijaysingh/victor/releases/tag/v0.1.0
