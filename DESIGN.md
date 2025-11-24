# Terminal Coding Agent - Design Document

## Overview

A universal terminal-based coding agent that works seamlessly with any LLM provider, supporting both frontier models (Claude, GPT, Gemini) and open-source models (via Ollama, LMStudio, vLLM).

## Core Architecture

### 1. Provider Abstraction Layer

```
┌─────────────────────────────────────────────┐
│           Terminal Interface                │
│         (Rich CLI + REPL)                  │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│         Agent Orchestrator                  │
│  - Conversation Management                  │
│  - Context Management                       │
│  - Tool Execution                          │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│      Unified LLM Provider Interface         │
│  - Standardized request/response format     │
│  - Streaming support                        │
│  - Tool calling normalization               │
└─────────────────┬───────────────────────────┘
                  │
    ┌─────────────┼─────────────┬──────────────┐
    │             │             │              │
┌───▼────┐  ┌────▼─────┐  ┌───▼────┐   ┌────▼─────┐
│Anthropic│  │ OpenAI   │  │ Google │   │ Local    │
│ Claude │  │   GPT    │  │ Gemini │   │ Models   │
└────────┘  └──────────┘  └────────┘   └─────┬────┘
                                              │
                              ┌───────────────┼───────────┐
                              │               │           │
                         ┌────▼────┐    ┌────▼────┐ ┌───▼────┐
                         │ Ollama  │    │LMStudio │ │ vLLM   │
                         └─────────┘    └─────────┘ └────────┘
```

### 2. Tool Calling System

**Universal Tool Definition:**
- MCP (Model Context Protocol) compatible
- JSON Schema based tool definitions
- Automatic conversion between provider-specific formats
- Built-in tools: file operations, bash commands, web search, code analysis

**Tool Execution Flow:**
1. Model requests tool call
2. Normalize tool call format
3. Execute with sandboxing/permissions
4. Return results in provider-expected format
5. Continue conversation

### 3. Key Components

#### A. Provider Manager (`providers/`)
- `base.py` - Abstract base class defining interface
- `anthropic_provider.py` - Claude integration
- `openai_provider.py` - OpenAI GPT integration
- `google_provider.py` - Gemini integration
- `ollama_provider.py` - Ollama local models
- `lmstudio_provider.py` - LMStudio integration
- `vllm_provider.py` - vLLM server integration
- `registry.py` - Provider discovery and registration

#### B. Tool System (`tools/`)
- `base.py` - Tool definition and execution framework
- `filesystem.py` - File read/write/edit operations
- `bash.py` - Command execution
- `web.py` - Web search and fetch
- `code_analysis.py` - AST parsing, grep, etc.
- `mcp_adapter.py` - MCP protocol support

#### C. Agent Core (`agent/`)
- `orchestrator.py` - Main agent loop
- `context.py` - Conversation and context management
- `memory.py` - Long-term memory and caching
- `planner.py` - Task planning and breakdown

#### D. Terminal UI (`ui/`)
- `cli.py` - Main CLI entry point (typer)
- `repl.py` - Interactive REPL mode
- `renderer.py` - Rich formatting and display
- `input.py` - Enhanced input handling

#### E. Configuration (`config/`)
- `settings.py` - Configuration management
- `profiles.py` - Model/provider profiles
- `credentials.py` - API key management

## Feature Set

### Phase 1: Core Functionality
- [x] Multi-provider support (Anthropic, OpenAI, Google, Ollama)
- [x] Streaming responses
- [x] Basic tool calling (file ops, bash)
- [x] Terminal REPL interface
- [x] Configuration management

### Phase 2: Advanced Features
- [ ] MCP server integration
- [ ] Context caching and optimization
- [ ] Multi-agent collaboration
- [ ] Custom tool plugin system
- [ ] Conversation branching
- [ ] Token usage tracking and cost estimation

### Phase 3: Developer Experience
- [ ] Interactive debugging mode
- [ ] Performance profiling
- [ ] Comprehensive testing suite
- [ ] Docker support
- [ ] CI/CD pipeline

## Technology Stack

### Core
- **Python 3.10+** - Modern Python with type hints
- **asyncio** - Async/await for concurrent operations
- **pydantic** - Data validation and settings management

### CLI/UI
- **typer** - CLI framework with auto-completion
- **rich** - Beautiful terminal formatting
- **prompt_toolkit** - Advanced input handling
- **textual** (optional) - TUI for complex interfaces

### LLM Providers
- **anthropic** - Official Anthropic SDK
- **openai** - Official OpenAI SDK
- **google-generativeai** - Official Google SDK
- **httpx** - HTTP client for custom providers (Ollama, vLLM)

### Tools/Utilities
- **tree-sitter** - Code parsing and analysis
- **gitpython** - Git operations
- **jsonschema** - Tool definition validation
- **tiktoken** - Token counting
- **python-dotenv** - Environment management

### Testing
- **pytest** - Testing framework
- **pytest-asyncio** - Async test support
- **pytest-mock** - Mocking utilities
- **respx** - HTTP mocking for API tests

## Configuration System

### Model Profiles (`~/.victor/profiles.yaml`)

```yaml
profiles:
  default:
    provider: ollama
    model: qwen2.5-coder:7b
    temperature: 0.7
    max_tokens: 4096

  claude-sonnet:
    provider: anthropic
    model: claude-sonnet-4-5
    temperature: 1.0
    max_tokens: 8192

  gpt4:
    provider: openai
    model: gpt-4-turbo
    temperature: 0.8

providers:
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    base_url: https://api.anthropic.com

  openai:
    api_key: ${OPENAI_API_KEY}

  ollama:
    base_url: http://localhost:11434

  lmstudio:
    base_url: http://localhost:1234
```

### Tool Configuration

```yaml
tools:
  filesystem:
    enabled: true
    allowed_paths:
      - /Users/vijaysingh/code
    restricted_paths:
      - /Users/vijaysingh/.ssh

  bash:
    enabled: true
    timeout: 60
    allowed_commands: []  # empty = all allowed

  web:
    enabled: true
    max_fetch_size: 10485760  # 10MB
```

## Security Considerations

1. **Sandboxing**: Tool execution in restricted environment
2. **Path validation**: Whitelist/blacklist for file operations
3. **Command filtering**: Dangerous command detection
4. **API key management**: Secure credential storage
5. **Permission system**: User confirmation for sensitive operations

## Distribution Strategy

1. **PyPI Package**: `pip install victor`
2. **GitHub Releases**: Binary distributions via PyInstaller
3. **Docker Image**: Pre-configured container
4. **Homebrew Formula**: macOS easy install
5. **Documentation**: Comprehensive docs site (MkDocs)

## Testing Strategy

### Unit Tests
- Provider implementations
- Tool execution
- Configuration parsing
- Format conversions

### Integration Tests
- End-to-end with Ollama (cost-free testing)
- Multi-turn conversations
- Tool calling workflows
- Error handling

### Performance Tests
- Streaming latency
- Context window management
- Concurrent request handling

## Development Roadmap

### Week 1: Foundation
- Repository setup
- Provider abstraction
- Ollama integration
- Basic tool system

### Week 2: Multi-Provider
- Anthropic, OpenAI, Google providers
- Tool calling normalization
- Configuration system

### Week 3: UI/UX
- Rich terminal interface
- REPL mode
- Streaming display
- Error handling

### Week 4: Polish
- Documentation
- Testing suite
- CI/CD setup
- Release preparation

## Success Metrics

1. **Compatibility**: Works with 6+ providers out of the box
2. **Performance**: <100ms overhead for provider abstraction
3. **Reliability**: >99% tool execution success rate
4. **UX**: <5 min from install to first successful interaction
5. **Community**: Active contributors and users

## References

- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/tool-use)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md)
