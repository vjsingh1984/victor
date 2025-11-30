# Getting Started with Victor

> Reality check: semantic tool selection is enabled by default; disable via `profiles.yaml` if desired. Stage pruning falls back to a small core tool set capped by `fallback_max_tools` (default 8) to avoid broadcasting everything. Allowlisted tools are cached (`tool_cache_*` settings) to avoid rerunning pure operations. `security_scan` is regex-only today (no CVE/IaC/package-audit yet). Docker users should prefer `docker/README.md` + `docker/QUICKREF.md`. The active package is `victor/`; the old copy now lives at `archive/victor-legacy/` and should be ignored.

Victor is a universal terminal-based AI coding assistant with 35 consolidated tools, supporting multiple LLM providers (Claude, GPT, Gemini, Ollama, vLLM, LMStudio) with 100% air-gapped capability.

## Quick Start (5 minutes)

### Installation

```bash
# Clone the repository
git clone https://github.com/vjsingh1984/victor.git
cd victor

# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[dev]"
```

### Basic Usage

#### 1. Interactive Mode (Recommended for beginners)

```bash
# Start Victor with default settings (Ollama + local embeddings)
victor main

# Victor will:
# - Use Ollama as the default provider
# - Use semantic tool selection when embeddings are available (falls back to keywords)
# - Load 35 tools automatically (editor, git, tests, docker, docs, refactors, web, etc.)
```

Example session:
```
$ victor main
🤖 Victor v0.1.0 - Universal AI Coding Assistant
Provider: ollama (qwen2.5-coder:7b)
Tools: 35 loaded (semantic selection enabled)

You: Write a Python function to validate email addresses

Victor: [Analyzes request, selects relevant tools semantically]
        [Uses code_executor tool to write function]
        [Returns validated code with tests]
```

#### 2. One-shot Command Mode

```bash
# Execute a single command and exit
victor main "Create a REST API endpoint for user registration"

# With specific provider
victor main --profile claude "Review this code for security issues"
```

#### 3. Specific Provider

```bash
# Use Claude Sonnet 4.5
victor main --profile claude

# Use OpenAI GPT-4
victor main --profile gpt4

# Use Google Gemini
victor main --profile gemini

# Use local Ollama (default)
victor main --profile ollama
```

## Configuration

### Profile Setup

Create `~/.victor/profiles.yaml`:

```yaml
profiles:
  # Default profile (Ollama + air-gapped)
  default:
    provider: ollama
    model: qwen2.5-coder:7b
    temperature: 0.7
    max_tokens: 4096

  # Claude profile (best for complex tasks)
  claude:
    provider: anthropic
    model: claude-sonnet-4-5
    temperature: 1.0
    max_tokens: 8192

  # GPT-4 profile
  gpt4:
    provider: openai
    model: gpt-4-turbo
    temperature: 0.8
    max_tokens: 4096

  # Gemini profile
  gemini:
    provider: google
    model: gemini-2.0-flash-exp
    temperature: 0.9
    max_tokens: 8192

providers:
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}

  openai:
    api_key: ${OPENAI_API_KEY}

  google:
    api_key: ${GOOGLE_API_KEY}

  ollama:
    base_url: http://localhost:11434
```

### Environment Variables

Create `.env` file:

```bash
# API Keys (optional - only if using cloud providers)
ANTHROPIC_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here
GOOGLE_API_KEY=your-key-here

# Embedding Configuration (optional - defaults work offline)
USE_SEMANTIC_TOOL_SELECTION=true
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=all-MiniLM-L12-v2

# Codebase Search (optional - defaults work offline)
CODEBASE_VECTOR_STORE=lancedb
CODEBASE_EMBEDDING_PROVIDER=sentence-transformers
```

## Reality check (current limits)

- `semantic_code_search` provides embedding-backed search (auto-reindexes when files change); `code_search` remains keyword-based. Install sentence-transformers + lancedb/chromadb for embeddings.
- `security_scan` currently checks for regex-based secrets and obvious config flags only; dependency/IaC/CVE scanning is not yet integrated.
- Tool surface: 35 tools ship by default (editor, git, tests, docker, docs, refactors, web). Coverage analysis and rich CI/CD generators are not implemented yet. See `docs/TOOL_CATALOG.md` (generated via `python scripts/generate_tool_catalog.py`) for the live list.
- Package layout: the `victor/` package is the active code path used by the CLI.

## Common Use Cases

### 1. Code Review & Security

```bash
# Interactive review
victor main
> Review the code in src/auth/login.py for security vulnerabilities

# One-shot command
victor main "Scan the entire codebase for SQL injection vulnerabilities"
```

### 2. Test Generation

```bash
victor main "Generate pytest tests for src/utils/validators.py with 100% coverage"
```

### 3. Documentation

```bash
victor main "Generate API documentation for all endpoints in src/api/"
```

### 4. Refactoring

```bash
victor main "Refactor the UserService class to use dependency injection"
```

### 5. Git Automation

```bash
victor main "Create a commit for the changes with a descriptive message"
victor main "Generate a pull request description for these changes"
```

### 6. CI/CD Pipeline

```bash
victor main "Generate a GitHub Actions workflow for Python testing and linting"
victor main "Validate the CircleCI config in .circleci/config.yml"
```

## Air-gapped / Offline Usage

Victor works 100% offline by default using:
- **Ollama** for LLM inference (local)
- **sentence-transformers** for embeddings (local, ~120MB)
- **LanceDB** for vector storage (local, disk-based)

### Setup for Offline Use

```bash
# 1. Install Ollama
curl https://ollama.ai/install.sh | sh

# 2. Pull a code model
ollama pull qwen2.5-coder:7b

# 3. Start Ollama
ollama serve

# 4. Use Victor (works offline)
victor main
```

### Docker (Pre-packaged Air-gapped Image)

```bash
# Build Docker image with embedded models
docker build -t victor:airgapped .

# Run completely offline
docker run --rm -it victor:airgapped bash

# Inside container
python3 /app/examples/airgapped_codebase_search.py
python3 /app/examples/tool_selection_demo.py
```

## Available Tools (31 Total)

### Core Tools (Always Available)
- **read_file** - Read file contents
- **write_file** - Write/create files
- **list_directory** - List directory contents
- **execute_bash** - Execute bash commands
- **file_editor** - Multi-file editing with transactions

### Enterprise Tools

**Code Quality:**
- **code_review** - Automated code review with complexity metrics
- **security_scanner** - Vulnerability and secret detection
- **refactor** - AST-based refactoring (rename, extract, inline)
- **metrics** - Code complexity and quality metrics

**Development:**
- **testing** - Generate pytest tests with fixtures
- **documentation** - Auto-generate docstrings and API docs
- **code_executor** - Execute and validate code
- **code_intelligence** - Code analysis and insights

**DevOps:**
- **git** - Git operations with AI-powered commits
- **cicd** - Generate and validate CI/CD pipelines
- **docker** - Container management
- **database** - SQL query execution with safety checks

**Automation:**
- **batch_processor** - Parallel multi-file operations
- **web_search** - Fetch documentation and resources
- **http** - API testing and validation
- **cache** - Tiered caching (memory + disk)
- **workflow** - Multi-step task orchestration

**Utilities:**
- **dependency** - Package analysis and security auditing
- **scaffold** - Project templates and boilerplate

## Examples & Demos

Run the demo scripts to see Victor in action:

```bash
# Tool selection demo (semantic matching)
python3 examples/tool_selection_demo.py

# Enterprise workflow (code review → tests → docs → git)
python3 examples/enterprise_workflow_demo.py

# Air-gapped codebase search
python3 examples/airgapped_codebase_search.py

# Multi-provider comparison
python3 examples/multi_provider_demo.py

# Advanced tools showcase
python3 examples/advanced_tools_demo.py
```

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# List available models
ollama list
```

### Embedding Model Issues

```bash
# Check if sentence-transformers is installed
python3 -c "from sentence_transformers import SentenceTransformer; print('OK')"

# Test model loading
python3 -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('all-MiniLM-L12-v2'); print(f'Dimension: {m.get_sentence_embedding_dimension()}')"
```

### Provider Authentication

```bash
# Verify API keys are set
echo $ANTHROPIC_API_KEY
echo $OPENAI_API_KEY
echo $GOOGLE_API_KEY

# Test provider
victor test-provider anthropic
victor test-provider openai
```

## Performance Tips

### 1. Use Semantic Tool Selection (Default)
- Faster than keyword matching
- More accurate tool selection
- Works offline (~8ms per query)

### 2. Use LanceDB for Large Codebases
- Disk-based (low memory)
- Scales to billions of vectors
- Sub-second search on millions of docs

### 3. Cache Embeddings
- Tool embeddings cached automatically
- Codebase embeddings persisted to disk
- Reused across sessions

### 4. Choose the Right Model
- **Ollama (qwen2.5-coder)**: Fast, offline, great for coding
- **Claude Sonnet 4.5**: Best for complex reasoning
- **GPT-4 Turbo**: Balanced performance
- **Gemini 2.0 Flash**: Fastest cloud option

## Next Steps

1. **Explore the examples**: Run demo scripts to see capabilities
2. **Configure your profile**: Set up providers and API keys
3. **Try air-gapped mode**: Test offline capability with Ollama
4. **Integrate into workflow**: Use Victor for daily coding tasks
5. **Read the docs**: Check `docs/` for detailed guides

## Additional Resources

- **Architecture**: `docs/EMBEDDING_ARCHITECTURE.md`
- **Tool Development**: `CLAUDE.md` (Tool Development section)
- **Provider Guide**: `PROVIDERS.md`
- **Air-gapped Setup**: `AIRGAPPED_GUIDE.md`
- **Docker**: `docker/README.md`

## Getting Help

```bash
# Show help
victor --help

# List providers
victor providers

# List profiles
victor profiles

# Test connection
victor test-provider ollama
```

For issues or feedback: https://github.com/vjsingh1984/victor/issues
