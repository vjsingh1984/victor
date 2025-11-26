# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Victor is a universal terminal-based AI coding assistant that provides a unified interface for working with multiple LLM providers (Anthropic Claude, OpenAI GPT, Google Gemini, xAI Grok, and local models via Ollama/LMStudio/vLLM). It features 25+ enterprise-grade tools including batch processing, code refactoring, test generation, CI/CD automation, and semantic search.

## Command Reference

### Development Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=victor --cov-report=term-missing --cov-report=html

# Run specific test file
pytest tests/unit/test_providers.py

# Run only unit tests
pytest tests/unit/

# Run integration tests (requires Ollama running)
pytest tests/integration/

# Run tests matching pattern
pytest -k "test_provider"
```

### Code Quality

```bash
# Lint with Ruff
ruff check victor tests

# Format with Black
black victor tests

# Format check (CI mode)
black --check victor tests

# Type check with mypy
mypy victor
```

### Running Victor

```bash
# Interactive mode (default profile)
victor

# With specific profile
victor --profile claude

# One-shot command
victor "Write a Python function to calculate Fibonacci numbers"

# List available providers
victor providers

# List configured profiles
victor profiles

# List available models (Ollama)
victor models

# Test a provider
victor test-provider ollama
```

## Architecture

### Core Abstractions

**Provider Abstraction (`victor/providers/`):**
- `base.py` defines `BaseProvider` abstract class with normalized `Message`, `ToolDefinition`, `CompletionResponse`, and `StreamChunk` models
- Each provider (Anthropic, OpenAI, Google, xAI, Ollama) implements `chat()`, `stream()`, `supports_tools()`, and `supports_streaming()` methods
- `registry.py` provides `ProviderRegistry` for dynamic provider registration and discovery
- All providers translate their native formats to/from the standard format defined in `base.py`

**Tool System (`victor/tools/`):**
- `base.py` defines `BaseTool` abstract class with `name`, `description`, `parameters` (JSON Schema), and `execute()` method
- `ToolRegistry` manages tool registration and lookup
- Tools return `ToolResult` with `success`, `output`, `error`, and `metadata` fields
- Tool parameters are validated using Pydantic models

**Agent Orchestrator (`victor/agent/orchestrator.py`):**
- Manages conversation history as a list of `Message` objects
- Coordinates between providers and tools
- Handles tool execution flow: model requests tool → normalize format → execute → return results → continue conversation
- Registers default tools at initialization: ReadFile, WriteFile, ListDirectory, Bash, FileEditor, Git, WebSearch

### Key Design Patterns

**Provider Registration:**
Providers self-register in their `__init__.py` files using `ProviderRegistry.register()`. This allows dynamic discovery without hardcoded imports.

**Tool Calling Flow:**
1. Provider-specific tool call format → normalized to standard format
2. `ToolRegistry` looks up tool by name
3. Tool `execute()` method called with parameters
4. Result formatted back to provider-specific format
5. Result added to conversation as tool response message

**Configuration Management:**
- `Settings` class (Pydantic BaseSettings) loads from `.env` files and environment variables
- `~/.victor/profiles.yaml` contains profile configurations with provider, model, temperature, max_tokens
- Profiles support environment variable substitution: `${ANTHROPIC_API_KEY}`

## Tool Development

### Tool Categories

**Core Tools** (always available):
- Filesystem: read_file, write_file, list_directory
- Bash: execute_bash (with timeout and error handling)
- File Editor: multi-file editing with transaction support

**Enterprise Tools** (25+ tools):
- Code Review: automated analysis with complexity metrics, security checks
- Security Scanner: secret detection (12+ patterns), vulnerability scanning
- Batch Processor: parallel multi-file operations (search, replace, analyze)
- Refactor: AST-based transformations (rename, extract, inline, organize imports)
- Testing: automated pytest test generation with fixtures
- CI/CD: generate and validate GitHub Actions, GitLab CI, CircleCI pipelines
- Documentation: auto-generate docstrings, API docs, README sections
- Dependency Management: package analysis, security auditing, requirements management
- Code Metrics: complexity analysis, maintainability index, technical debt
- Database: query SQLite, PostgreSQL, MySQL, SQL Server with safety controls
- Docker: container/image management (list, run, stop, logs, stats)
- HTTP: API testing and endpoint validation
- Git: AI-powered git with smart commits, staging, branching
- Web Search: fetch documentation and online resources
- Cache: tiered caching (memory + disk) for LLM responses and embeddings

### Creating Custom Tools

```python
from victor.tools.base import BaseTool, ToolResult
from typing import Any, Dict

class MyCustomTool(BaseTool):
    @property
    def name(self) -> str:
        return "my_custom_tool"

    @property
    def description(self) -> str:
        return "Description of what this tool does"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "param_name": {
                    "type": "string",
                    "description": "Parameter description"
                }
            },
            "required": ["param_name"]
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            # Tool implementation
            result = do_something(kwargs["param_name"])
            return ToolResult(success=True, output=result)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
```

Register in `orchestrator.py` via `self.tools.register(MyCustomTool())`.

## Provider Implementation

### Adding New Providers

1. Create `victor/providers/my_provider.py`:

```python
from victor.providers.base import BaseProvider, Message, CompletionResponse, StreamChunk, ToolDefinition
from typing import List, Optional, AsyncIterator, Any

class MyProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "my_provider"

    async def chat(self, messages: List[Message], *, model: str,
                   temperature: float = 0.7, max_tokens: int = 4096,
                   tools: Optional[List[ToolDefinition]] = None, **kwargs: Any) -> CompletionResponse:
        # Implement chat completion
        # Convert messages from standard format to provider format
        # Make API call
        # Convert response from provider format to standard CompletionResponse
        pass

    async def stream(self, messages: List[Message], *, model: str,
                     temperature: float = 0.7, max_tokens: int = 4096,
                     tools: Optional[List[ToolDefinition]] = None, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        # Implement streaming
        pass

    def supports_tools(self) -> bool:
        return True  # or False

    def supports_streaming(self) -> bool:
        return True  # or False
```

2. Register in `victor/providers/__init__.py`:

```python
from victor.providers.registry import ProviderRegistry
from victor.providers.my_provider import MyProvider

ProviderRegistry.register("my_provider", MyProvider)
```

### Tool Calling Translation

Each provider has different tool calling formats:
- **Anthropic**: `tools` array with `name`, `description`, `input_schema`
- **OpenAI**: `functions` array with `name`, `description`, `parameters`
- **Google**: `function_declarations` with specific schema format
- **Ollama**: Similar to OpenAI but with model-dependent support

The `normalize_tools()` method in each provider handles conversion from standard `ToolDefinition` format.

## Testing Strategy

### Test Structure

```
tests/
├── unit/                    # Unit tests (no external dependencies)
│   ├── test_providers.py    # Provider logic tests
│   └── test_tools.py        # Tool execution tests
└── integration/             # Integration tests (require external services)
    └── test_ollama_integration.py
```

### Pytest Configuration

- Tests marked with `@pytest.mark.integration` require external services (Ollama, etc.)
- Use `pytest -m "not integration"` to skip integration tests
- All async tests use `pytest-asyncio` with `asyncio_mode = "auto"`
- Coverage target: 90%+ (configured in pyproject.toml)

### Mocking Providers

```python
import pytest
from unittest.mock import AsyncMock
from victor.providers.base import CompletionResponse

@pytest.fixture
def mock_provider():
    provider = AsyncMock()
    provider.chat.return_value = CompletionResponse(
        content="Test response",
        role="assistant"
    )
    provider.supports_tools.return_value = True
    provider.supports_streaming.return_value = True
    return provider
```

## Configuration

### Profile Configuration (`~/.victor/profiles.yaml`)

```yaml
profiles:
  default:
    provider: ollama
    model: qwen2.5-coder:7b
    temperature: 0.7
    max_tokens: 4096

  claude:
    provider: anthropic
    model: claude-sonnet-4-5
    temperature: 1.0
    max_tokens: 8192

providers:
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}

  openai:
    api_key: ${OPENAI_API_KEY}
    organization: ${OPENAI_ORG_ID}

  ollama:
    base_url: http://localhost:11434
```

### Environment Variables

```bash
ANTHROPIC_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here
GOOGLE_API_KEY=your-key-here
XAI_API_KEY=your-key-here
```

## Important Implementation Details

### Message Format Normalization

The standard `Message` format includes:
- `role`: "system", "user", or "assistant"
- `content`: Message text
- `tool_calls`: Optional list of tool call dicts (for assistant messages)
- `tool_call_id`: Optional ID (for tool response messages)
- `name`: Optional name field (for tool responses)

Each provider must convert to/from this format in their `chat()` and `stream()` implementations.

### Async/Await Pattern

All I/O operations (provider calls, tool execution, file operations) use async/await:
- Providers implement `async def chat()` and `async def stream()`
- Tools implement `async def execute()`
- CLI uses `asyncio.run()` to execute async functions

### Error Handling

- `ProviderError`: Base exception for provider errors
- `ProviderNotFoundError`: Provider not in registry
- `ProviderAuthenticationError`: API key/auth issues
- `ProviderRateLimitError`: Rate limit exceeded
- `ProviderTimeoutError`: Request timeout
- All tool executions should return `ToolResult(success=False, error=...)` rather than raising exceptions

### Streaming Implementation

Streaming must yield `StreamChunk` objects:
- `content`: Incremental text
- `tool_calls`: Tool calls (if any)
- `stop_reason`: Why generation stopped
- `is_final`: Whether this is the last chunk

The final chunk should have `is_final=True` to signal completion.

## Package Structure

```
victor/
├── __init__.py              # Package version and exports
├── agent/                   # Agent orchestration
│   └── orchestrator.py      # Main agent loop and tool coordination
├── providers/               # LLM provider implementations
│   ├── base.py              # Abstract base class and standard formats
│   ├── registry.py          # Provider registration and discovery
│   ├── anthropic_provider.py
│   ├── openai_provider.py
│   ├── google_provider.py
│   ├── xai_provider.py
│   └── ollama_provider.py
├── tools/                   # Tool implementations (25+ tools)
│   ├── base.py              # Tool framework and registry
│   ├── filesystem.py        # File operations
│   ├── bash.py              # Command execution
│   ├── file_editor_tool.py  # Multi-file editing
│   ├── git_tool.py          # Git operations
│   ├── code_review_tool.py  # Code analysis
│   ├── security_scanner_tool.py
│   ├── batch_processor_tool.py
│   ├── refactor_tool.py
│   ├── testing_tool.py
│   ├── cicd_tool.py
│   ├── documentation_tool.py
│   ├── dependency_tool.py
│   ├── metrics_tool.py
│   └── [18 more tools...]
├── config/                  # Configuration management
│   └── settings.py          # Settings and profile loading
├── ui/                      # Terminal interface
│   └── cli.py               # Typer-based CLI
├── codebase/               # Semantic search and indexing
├── editing/                # Multi-file transaction management
├── mcp/                    # Model Context Protocol support
└── utils/                  # Utility functions
```

## CLI Entry Points

The package defines two CLI entry points in `pyproject.toml`:
- `victor`: Main command
- `vic`: Alias for convenience

Both point to `victor.ui.cli:app` (Typer app instance).

## Dependencies

### Core Runtime
- `anthropic>=0.34`: Official Anthropic SDK
- `openai>=1.40`: Official OpenAI SDK
- `google-generativeai>=0.7`: Official Google SDK
- `httpx>=0.27`: HTTP client for Ollama and custom providers
- `pydantic>=2.0`: Data validation and settings
- `typer>=0.12`: CLI framework
- `rich>=13.7`: Terminal formatting
- `gitpython>=3.1`: Git operations
- `tree-sitter>=0.21`: Code parsing
- `tiktoken>=0.7`: Token counting

### Development Only
- `pytest>=8.0`: Testing framework
- `pytest-asyncio>=0.23`: Async test support
- `pytest-cov>=4.1`: Coverage reporting
- `ruff>=0.5`: Fast linter
- `black>=24.0`: Code formatter
- `mypy>=1.10`: Static type checker

## Semantic Search & Embedding Configuration

Victor uses a plugin-based architecture for semantic code search, separating concerns between:
- **Embedding Model**: Converts text to vectors (Ollama, OpenAI, Sentence-Transformers)
- **Vector Store**: Stores and searches vectors (ChromaDB, FAISS, Pinecone)

### Production Configuration: Qwen3-Embedding:8b

**Default Setup (Maximum Accuracy):**
```yaml
# ~/.victor/config.yaml
codebase:
  # Vector Store
  vector_store: chromadb
  persist_directory: ~/.victor/embeddings/production
  distance_metric: cosine

  # Embedding Model: Qwen3-Embedding:8b (#1 on MTEB multilingual leaderboard)
  embedding_model_type: ollama
  embedding_model_name: qwen3-embedding:8b
  embedding_api_key: http://localhost:11434  # Ollama server URL

  # Configuration
  extra_config:
    collection_name: victor_codebase
    dimension: 4096  # Qwen3 embedding dimension
    batch_size: 8    # Adjust based on available RAM
```

**Why Qwen3-Embedding:8b:**
- **MTEB Score**: 70.58 (#1 on multilingual leaderboard, Jan 2025)
- **Context Window**: 40K tokens (excellent for large code files)
- **Dimension**: 4096 (high-quality semantic representations)
- **Languages**: 100+ (all programming languages, multilingual comments)
- **License**: Apache 2.0 (production-ready)
- **Cost**: Free (runs locally via Ollama)

**Installation:**
```bash
# 1. Install Ollama
curl https://ollama.ai/install.sh | sh

# 2. Pull Qwen3-Embedding model (4.7GB)
ollama pull qwen3-embedding:8b

# 3. Start Ollama server
ollama serve

# 4. Verify model is available
ollama list | grep qwen3-embedding
```

**Alternative Models:**

| Model | MTEB Score | Dimension | Context | Use Case |
|-------|------------|-----------|---------|----------|
| `qwen3-embedding:8b` | 70.58 | 4096 | 40K | Production (max accuracy) |
| `snowflake-arctic-embed2` | ~58-60 | 1024 | 8K | Balanced (fast + accurate) |
| `bge-m3` | 59.56 | 1024 | 8K | RAG (multi-functional retrieval) |
| `mxbai-embed-large` | SOTA* | 1024 | 512 | Small files only |
| `nomic-embed-text` | ~53 | 768 | 2K | Legacy/development |

*SOTA for Bert-large size class

**Python API Usage:**
```python
from victor.codebase.embeddings.base import EmbeddingConfig
from victor.codebase.embeddings.chromadb_provider import ChromaDBProvider

# Configure with Qwen3-Embedding:8b
config = EmbeddingConfig(
    vector_store="chromadb",
    persist_directory="~/.victor/embeddings/my_project",
    embedding_model_type="ollama",
    embedding_model_name="qwen3-embedding:8b",
    embedding_api_key="http://localhost:11434",
    extra_config={"dimension": 4096, "batch_size": 8}
)

# Initialize and index
provider = ChromaDBProvider(config)
await provider.initialize()
await provider.index_documents(documents)

# Semantic search
results = await provider.search_similar(
    "how to authenticate users with JWT tokens",
    limit=5
)
```

**Performance Tuning:**
- **RAM Requirements**: 8GB minimum, 16GB recommended for Qwen3:8b
- **Batch Size**: 4-16 (lower for large models, higher for small)
- **GPU**: Optional but recommended (CUDA/Metal/ROCm)
- **Context Truncation**: Qwen3's 40K context means rarely truncating code files

**Example Demo:**
See `examples/qwen3_embedding_demo.py` for a complete working example with semantic code search.

## Troubleshooting Common Issues

**Provider Authentication Errors:**
- Verify API keys are set in environment variables or `~/.victor/profiles.yaml`
- Check that keys are not wrapped in quotes in the YAML file
- Use `${ENV_VAR}` syntax for environment variable substitution

**Ollama Connection Issues:**
- Ensure Ollama is running: `ollama serve`
- Check base URL in profiles.yaml matches Ollama server
- Test with: `victor test-provider ollama`
- List available models: `victor models`

**Embedding Model Issues:**
- Verify model is pulled: `ollama list | grep qwen3-embedding`
- Check Ollama is running: `curl http://localhost:11434/api/tags`
- Test embedding generation: `curl http://localhost:11434/api/embeddings -d '{"model":"qwen3-embedding:8b","prompt":"test"}'`
- Reduce batch_size if running out of memory
- For large codebases (>100K files), consider using Qwen3:4b or snowflake-arctic-embed2

**Tool Calling Not Working:**
- Verify provider supports tool calling: check `provider.supports_tools()`
- Some Ollama models don't support tools (see `victor/config/tool_calling_models.yaml` for supported models)
- Check that tool parameters match the JSON Schema exactly

**Import Errors:**
- Run `pip install -e ".[dev]"` to install in development mode
- Verify you're in the virtual environment: `which python`
- Check that all providers are registered in `victor/providers/__init__.py`
- For embeddings, ensure httpx is installed: `pip install httpx`
