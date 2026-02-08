# Configuration Reference - Part 1

**Part 1 of 2:** Configuration Directory Structure, Configuration Priority, Quick Configuration, profiles.yaml, config.yaml, and mcp.yaml

---

## Navigation

- **[Part 1: Core Configuration](#)** (Current)
- [Part 2: Project & Environment](part-2-project-environment.md)
- [**Complete Guide](../index.md)**

---

# Configuration Reference

Complete reference for configuring Victor's behavior and integrations.

## Quick Links

| File | Purpose | Location | Reference |
|------|---------|----------|----------|
| **profiles.yaml** | Provider and model profiles | `~/.victor/profiles.yaml` | [#profilesyaml →](#profilesyaml) |
| **config.yaml** | Global settings and options | `~/.victor/config.yaml` | [#configyaml →](#configyaml) |
| **mcp.yaml** | MCP server configuration | `~/.victor/mcp.yaml` | [#mcpyaml →](#mcpyaml) |
| **.victor.md** | Project context | `<project>/.victor.md` | [.victor.md →](../../user-guide/index.md#5-project-context) |
| **CLAUDE.md** | AI instructions | `<project>/CLAUDE.md` | See `CLAUDE.md` in the repo root |

## Configuration Directory Structure

```
~/.victor/                      # User configuration (global)
├── profiles.yaml                # Provider/model profiles
├── config.yaml                  # Global settings
├── mcp.yaml                     # MCP server config
├── cache/                       # Cache storage
├── logs/                        # Log files
└── conversations/               # Conversation history

<project>/                       # Project-specific (local)
├── .victor.md                   # Project context
└── CLAUDE.md                    # AI instructions
```

## Configuration Priority

Victor loads configuration in this order (later overrides earlier):

1. **Default values** (built-in)
2. **Global config** (`~/.victor/config.yaml`)
3. **Profile config** (`~/.victor/profiles.yaml`)
4. **Project context** (`<project>/.victor.md`)
5. **Environment variables**
6. **CLI flags** (highest priority)

**Example**:
```bash
# Default: temperature=0.7
# config.yaml: temperature=0.5
# profiles.yaml: temperature=0.3
# Environment: VICTOR_TEMPERATURE=0.1
# CLI flag: --temperature 0.0
# Result: 0.0 (CLI flag wins)
```

## Quick Configuration

### Minimal Setup

**1. Local model** (no configuration needed):
```bash
# Victor auto-detects Ollama
victor chat "Hello!"
```

**2. Cloud provider** (environment variable):
```bash
# Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# Use Victor
victor chat --provider anthropic "Hello!"
```

**3. Profile-based** (recommended):
```yaml
# ~/.victor/profiles.yaml
profiles:
  default:
    provider: anthropic
    model: claude-sonnet-4-20250514
```

```bash
# Use profile
victor chat "Hello!"  # Automatically uses profile
```

---

## profiles.yaml

Provider and model profile configuration.

**Location**: `~/.victor/profiles.yaml`
**Required**: No (defaults will be used)
**Purpose**: Define reusable provider/model configurations

### Basic Structure

```yaml
profiles:
  # Profile name (can be anything)
  default:
    # Provider selection
    provider: anthropic  # openai, google, ollama, etc.

    # Model selection
    model: claude-sonnet-4-20250514

    # Optional: Model parameters
    temperature: 0.7
    max_tokens: 4096
    top_p: 0.9
    top_k: 40

    # Optional: API key
    api_key_env: ANTHROPIC_API_KEY

    # Optional: API base URL (for custom endpoints)
    base_url: https://api.anthropic.com

    # Optional: Other settings
    timeout: 30
    retry_attempts: 3
```

### Profile Examples

**Development Profile**:
```yaml
profiles:
  development:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.7
    max_tokens: 4096
    api_key_env: ANTHROPIC_API_KEY
```

**Production Profile**:
```yaml
profiles:
  production:
    provider: azure
    deployment: gpt-4o
    api_key_env: AZURE_OPENAI_API_KEY
    api_base_env: AZURE_OPENAI_ENDPOINT
    api_version: 2024-02-01
    max_tokens: 8192
    temperature: 0.3  # More deterministic
```

**Local Profile**:
```yaml
profiles:
  local:
    provider: ollama
    model: qwen2.5-coder:7b
    # No API key needed
    base_url: http://127.0.0.1:11434
```

**Cost-Optimized Profile**:
```yaml
profiles:
  cheap:
    provider: google
    model: gemini-2.0-flash-exp  # Free
    max_tokens: 4096
```

**Multi-Provider Profile** (with fallback):
```yaml
profiles:
  robust:
    provider: anthropic
    model: claude-sonnet-4-20250514
    api_key_env: ANTHROPIC_API_KEY
    # Fallback providers (if primary fails)
    fallback_providers:
      - provider: openai
        model: gpt-4o
      - provider: ollama
        model: qwen2.5-coder:7b
```

### Using Profiles

**Specify profile**:
```bash
victor --profile development chat "Hello"
```

**Set default profile**:
```yaml
# ~/.victor/profiles.yaml
default_profile: development
```

**List available profiles**:
```bash
victor config profiles
```

**Show current profile**:
```bash
victor config show
```

### Profile Reference

| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| **provider** | string | Yes | Provider name | `anthropic`, `openai`, `ollama` |
| **model** | string | Yes | Model name | `claude-sonnet-4-20250514` |
| **api_key_env** | string | No | Environment variable for API key | `ANTHROPIC_API_KEY` |
| **base_url** | string | No | Custom API base URL | `https://api.anthropic.com` |
| **temperature** | float | No | Sampling temperature (0.0-1.0) | `0.7` |
| **max_tokens** | int | No | Maximum tokens to generate | `4096` |
| **top_p** | float | No | Nucleus sampling (0.0-1.0) | `0.9` |
| **top_k** | int | No | Top-k sampling | `40` |
| **timeout** | int | No | Request timeout (seconds) | `30` |
| **retry_attempts** | int | No | Number of retries on failure | `3` |

[Full profiles.yaml Reference →](#profilesyaml)

---

## config.yaml

Global configuration for Victor's behavior.

**Location**: `~/.victor/config.yaml`
**Required**: No (defaults will be used)
**Purpose**: Global settings, caching, logging, etc.

### Basic Structure

```yaml
# Logging configuration
logging:
  level: INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: text  # text or json
  file: /var/log/victor.log
  rotation: daily
  retention: 30

# Cache configuration
cache:
  enabled: true
  ttl: 3600  # Time-to-live in seconds
  max_size: 1000  # Maximum number of cached items
  backend: memory  # memory, redis, disk

# Tool configuration
tools:
  lazy_load: true  # Load tools on-demand
  parallel_execution: true
  timeout: 30

# Conversation configuration
conversation:
  max_history: 100  # Maximum messages in history
  save_history: true
  history_file: ~/.victor/conversations/

# Provider configuration
providers:
  timeout: 30
  retry_attempts: 3
  circuit_breaker:
    enabled: true
    failure_threshold: 5
    recovery_timeout: 60

# Workflow configuration
workflows:
  directory: ~/.victor/workflows/
  auto_compile: true

# UI configuration
ui:
  theme: dark  # dark, light
  syntax_highlighting: true
  line_numbers: true
```

### Configuration Reference

#### Logging

```yaml
logging:
  level: INFO              # Log level
  format: text             # Output format (text/json)
  file: /var/log/victor.log # Log file path
  rotation: daily          # Log rotation (daily, weekly, size)
  retention: 30            # Days to keep logs
  console: true            # Also log to console
```

**Log Levels**:
- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages
- **WARNING**: Warning messages
- **ERROR**: Error messages
- **CRITICAL**: Critical failures

#### Cache

```yaml
cache:
  enabled: true            # Enable caching
  ttl: 3600               # Time-to-live (seconds)
  max_size: 1000          # Max cached items
  backend: memory         # Backend (memory, redis, disk)

  # Disk backend config
  disk:
    path: ~/.victor/cache/
    max_size_mb: 1024

  # Redis backend config
  redis:
    host: 127.0.0.1
    port: 6379
    db: 0
    password: null
```

**Cache Strategies**:
- **Memory**: Fastest, lost on restart
- **Disk**: Persistent, slower
- **Redis**: Shared across instances, persistent

#### Tools

```yaml
tools:
  lazy_load: true          # Load tools on-demand (faster startup)
  parallel_execution: true # Execute tools in parallel when possible
  timeout: 30              # Tool execution timeout (seconds)
  max_concurrent: 5        # Max parallel tool executions
```

#### Conversation

```yaml
conversation:
  max_history: 100         # Max messages in conversation history
  save_history: true       # Save conversation to disk
  history_file: ~/.victor/conversations/
  compress_history: false  # Compress old messages
```

#### Providers

```yaml
providers:
  timeout: 30              # Request timeout (seconds)
  retry_attempts: 3        # Retry attempts on failure
  circuit_breaker:
    enabled: true          # Enable circuit breaker
    failure_threshold: 5   # Failures before opening circuit
    recovery_timeout: 60   # Seconds before retry
```

**Circuit Breaker**: Prevents cascading failures by temporarily disabling failing providers.

#### Workflows

```yaml
workflows:
  directory: ~/.victor/workflows/  # Workflow directory
  auto_compile: true                # Auto-compile YAML workflows
  cache_compiled: true              # Cache compiled workflows
  max_parallel_nodes: 10            # Max parallel workflow nodes
```

#### UI

```yaml
ui:
  theme: dark              # Theme (dark, light)
  syntax_highlighting: true
  line_numbers: true
  font_size: 14
  editor: vim              # Default editor
  pager: less              # Default pager
```

[Full config.yaml Reference →](#configyaml)

---

## mcp.yaml

Model Context Protocol (MCP) server configuration.

**Location**: `~/.victor/mcp.yaml`
**Required**: No (only if using MCP)
**Purpose**: Configure MCP server settings

### Basic Structure

```yaml
# MCP server configuration
server:
  host: 127.0.0.1
  port: 8080
  transport: stdio  # stdio or sse

  # Authentication
  auth:
    enabled: false
    api_key: your-api-key

# Tool exposure
tools:
  expose_all: false  # Expose all tools to MCP clients
  allowed_tools:
    - read_file
    - write_file
    - search_files
    - run_tests

# Prompt templates
prompts:
  directory: ~/.victor/prompts/
  expose_prompts:
    - code_review
    - refactor

# Resources
resources:
  expose_all: false
  allowed_resources:
    - file:///
    - git:///
```

### MCP Reference

**Transport Modes**:
- **stdio**: Standard input/output (default)
- **sse**: Server-sent events (HTTP)

**Tool Exposure**: Control which tools are available to MCP clients

**Prompts**: Expose prompt templates to MCP clients

**Resources**: Expose file system and git resources

[Full mcp.yaml Reference →](#mcpyaml)

---

