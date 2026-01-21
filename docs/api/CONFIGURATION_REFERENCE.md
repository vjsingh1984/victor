# Victor AI 0.5.1 Configuration Reference

Complete reference for all configuration options in Victor AI.

**Table of Contents**
- [Overview](#overview)
- [Settings](#settings)
- [Environment Variables](#environment-variables)
- [Profiles](#profiles)
- [Path Configuration](#path-configuration)
- [Provider Configuration](#provider-configuration)
- [Tool Selection](#tool-selection)
- [Budget Management](#budget-management)
- [Logging and Observability](#logging-and-observability)
- [Security Settings](#security-settings)
- [Performance Tuning](#performance-tuning)

---

## Overview

Victor AI uses a hierarchical configuration system with multiple sources:

**Priority Order (highest to lowest):**
1. **Environment variables** - Runtime overrides
2. **.env file** - Project-specific settings
3. **~/.victor/profiles.yaml** - Global profile configuration
4. **Default values** - Built-in defaults

### Quick Start

```bash
# Set provider (environment variable)
export VICTOR_DEFAULT_PROVIDER="anthropic"
export ANTHROPIC_API_KEY="sk-ant-..."

# Or use .env file
cat > .env << EOF
VICTOR_DEFAULT_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
VICTOR_TOOL_CALL_BUDGET=50
EOF

# Or use profiles.yaml
cat > ~/.victor/profiles.yaml << EOF
default_provider: anthropic
anthropic_api_key: sk-ant-...
tool_call_budget: 50
EOF
```

---

## Settings

### Settings Class

Main settings class (`victor.config.settings.Settings`) with all configuration options.

### Core Settings

#### Default Provider

```python
# Settings attributes
default_provider: str = "ollama"  # Default provider to use
default_model: str = "qwen3-coder:30b"  # Default model
default_temperature: float = 0.7  # Default temperature (0.0-2.0)
default_max_tokens: int = 4096  # Default max tokens per response
```

**Environment Variables:**
```bash
VICTOR_DEFAULT_PROVIDER=anthropic
VICTOR_DEFAULT_MODEL=claude-sonnet-4-5
VICTOR_DEFAULT_TEMPERATURE=0.7
VICTOR_DEFAULT_MAX_TOKENS=8192
```

---

### API Keys

Configure API keys for cloud providers.

```python
# Settings attributes
anthropic_api_key: Optional[str] = None
openai_api_key: Optional[str] = None
google_api_key: Optional[str] = None
moonshot_api_key: Optional[str] = None
deepseek_api_key: Optional[str] = None
```

**Environment Variables:**
```bash
# Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI
export OPENAI_API_KEY="sk-..."

# Google Gemini
export GOOGLE_API_KEY="..."

# Moonshot AI (Kimi)
export MOONSHOT_API_KEY="..."

# DeepSeek
export DEEPSEEK_API_KEY="..."
```

**Profile Configuration:**
```yaml
# ~/.victor/profiles.yaml
anthropic_api_key: sk-ant-...
openai_api_key: sk-...
google_api_key: ...
```

---

### Local Provider URLs

Configure local provider endpoints.

```python
# Settings attributes
ollama_base_url: str = "http://localhost:11434"
lmstudio_base_urls: List[str] = ["http://127.0.0.1:1234"]
vllm_base_url: str = "http://localhost:8000"
```

**Environment Variables:**
```bash
# Ollama
export OLLAMA_BASE_URL="http://localhost:11434"

# LMStudio (supports multiple URLs for failover)
export LMSTUDIO_BASE_URLS="http://localhost:1234,http://192.168.1.100:1234"

# vLLM
export VLLM_BASE_URL="http://localhost:8000"
```

**Profile Configuration:**
```yaml
# ~/.victor/profiles.yaml
ollama_base_url: http://localhost:11434
lmstudio_base_urls:
  - http://127.0.0.1:1234
  - http://192.168.1.100:1234  # LAN server
vllm_base_url: http://localhost:8000
```

---

## Environment Variables

### Complete Reference

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `VICTOR_DEFAULT_PROVIDER` | str | `ollama` | Default LLM provider |
| `VICTOR_DEFAULT_MODEL` | str | `qwen3-coder:30b` | Default model identifier |
| `VICTOR_DEFAULT_TEMPERATURE` | float | `0.7` | Sampling temperature |
| `VICTOR_DEFAULT_MAX_TOKENS` | int | `4096` | Max tokens per response |
| `ANTHROPIC_API_KEY` | str | - | Anthropic API key |
| `OPENAI_API_KEY` | str | - | OpenAI API key |
| `GOOGLE_API_KEY` | str | - | Google API key |
| `OLLAMA_BASE_URL` | str | `http://localhost:11434` | Ollama endpoint |
| `LMSTUDIO_BASE_URLS` | str | `http://127.0.0.1:1234` | LMStudio endpoints (comma-separated) |
| `VLLM_BASE_URL` | str | `http://localhost:8000` | vLLM endpoint |
| `VICTOR_TOOL_CALL_BUDGET` | int | `100` | Max tool calls per session |
| `VICTOR_LOG_LEVEL` | str | `INFO` | Logging level |
| `VICTOR_AIRGAPPED_MODE` | bool | `false` | Enable air-gapped mode |
| `VICTOR_HEADLESS_MODE` | bool | `false` | Run without prompts |
| `VICTOR_DRY_RUN_MODE` | bool | `false` | Preview changes only |
| `VICTOR_ENABLE_PROVIDER_POOL` | bool | `false` | Enable provider pooling |
| `VICTOR_TOOL_SELECTION_STRATEGY` | str | `auto` | Tool selection strategy |

---

## Profiles

### Profile Configuration

Profiles allow pre-configured settings for different use cases.

**Profile Location:** `~/.victor/profiles.yaml`

### Example Profiles

```yaml
# ~/.victor/profiles.yaml

# Default settings
default_provider: anthropic
default_model: claude-sonnet-4-5
default_temperature: 0.7

# API keys
anthropic_api_key: sk-ant-...
openai_api_key: sk-...

# Local providers
ollama_base_url: http://localhost:11434

# Tool selection
tool_selection_strategy: auto
tool_call_budget: 100

# Logging
log_level: INFO
enable_observability_logging: false

# Vertical configuration
default_vertical: coding
vertical_loading_mode: eager

# Security
airgapped_mode: false
write_approval_mode: risky_only
```

### Profile Sections

#### Development Profile

```yaml
# ~/.victor/profiles.yaml (development)

default_provider: ollama
default_model: qwen2.5:32b
default_temperature: 0.8

# Local development settings
ollama_base_url: http://localhost:11434
tool_call_budget: 50

# Faster tool selection (no embeddings)
tool_selection_strategy: keyword

# Verbose logging
log_level: DEBUG
log_file: /tmp/victor-debug.log

# Enable all observability
enable_observability_logging: true
show_cost_metrics: true
```

#### Production Profile

```yaml
# ~/.victor/profiles.yaml (production)

default_provider: anthropic
default_model: claude-sonnet-4-5
default_temperature: 0.3

# High budget for complex tasks
tool_call_budget: 200

# Semantic tool selection for quality
tool_selection_strategy: semantic

# Minimal logging
log_level: WARNING

# Cost monitoring
show_cost_metrics: true

# Security
write_approval_mode: all_writes
server_api_key: your-secret-key
```

#### Air-Gapped Profile

```yaml
# ~/.victor/profiles.yaml (airgapped)

default_provider: ollama
default_model: qwen2.5:32b

# Disable all cloud features
airgapped_mode: true

# Local embeddings only
embedding_provider: sentence-transformers
embedding_model: BAAI/bge-small-en-v1.5

# Disable external tools
# (automatic when airgapped_mode=true)

# Local codebase search
codebase_vector_store: lancedb
codebase_embedding_provider: sentence-transformers
```

---

## Path Configuration

### ProjectPaths Class

Centralized path management for Victor AI.

```python
from victor.config.settings import ProjectPaths, get_project_paths

# Get paths for current project
paths = get_project_paths()

# Or specify project root
paths = ProjectPaths(project_root="/path/to/project")
```

### Project-Local Paths

Stored in `{project_root}/.victor/`:

```python
paths.project_victor_dir  # .victor directory
paths.project_context_file  # .victor/init.md
paths.conversation_db  # .victor/conversation.db
paths.embeddings_dir  # .victor/embeddings/
paths.graph_dir  # .victor/graph/
paths.index_metadata  # .victor/index_metadata.json
paths.backups_dir  # .victor/backups/
paths.changes_dir  # .victor/changes/
paths.sessions_dir  # .victor/sessions/
paths.conversations_export_dir  # .victor/conversations/
paths.mcp_config  # .victor/mcp.yaml
```

### Global Paths

Stored in `~/.victor/`:

```python
paths.global_victor_dir  # ~/.victor/
paths.global_profiles  # ~/.victor/profiles.yaml
paths.global_plugins_dir  # ~/.victor/plugins/
paths.global_cache_dir  # ~/.victor/cache/
paths.global_logs_dir  # ~/.victor/logs/
paths.global_embeddings_dir  # ~/.victor/embeddings/
paths.global_mcp_config  # ~/.victor/mcp.yaml
```

### Path Customization

```bash
# Custom directory name (instead of .victor)
export VICTOR_DIR_NAME=".ai-assistant"

# Custom context file (instead of init.md)
export VICTOR_CONTEXT_FILE="context.md"

# Custom global directory
export HOME="/custom/home"
# VICTOR_DIR will be $HOME/.victor (with security validation)
```

---

## Provider Configuration

### ProviderConfig

```python
class ProviderConfig(BaseSettings):
    api_key: Optional[str] = None
    base_url: Optional[Union[str, List[str]]] = None
    timeout: int = 300  # 5 minutes
    max_retries: int = 3
    organization: Optional[str] = None  # For OpenAI
```

### Provider-Specific Settings

#### Anthropic

```yaml
# profiles.yaml
anthropic_api_key: sk-ant-...
default_temperature: 0.7
default_max_tokens: 8192
```

#### OpenAI

```yaml
# profiles.yaml
openai_api_key: sk-...
openai_organization: org-...
default_temperature: 0.7
```

#### Local Providers (Ollama, LMStudio, vLLM)

```yaml
# profiles.yaml
ollama_base_url: http://localhost:11434
ollama_timeout: 300  # 5 minutes for CPU inference

lmstudio_base_urls:
  - http://127.0.0.1:1234
  - http://192.168.1.100:1234

vllm_base_url: http://localhost:8000
```

### Provider Pool Configuration

Load balancing across multiple provider instances.

```python
# Settings
enable_provider_pool: bool = False
pool_size: int = 3  # Max instances
pool_load_balancer: str = "adaptive"  # round_robin, least_connections, adaptive, random
pool_enable_warmup: bool = True
pool_warmup_concurrency: int = 3
pool_health_check_interval: int = 30  # seconds
pool_max_retries: int = 3
pool_min_instances: int = 1
```

**Environment Variables:**
```bash
VICTOR_ENABLE_PROVIDER_POOL=true
VICTOR_POOL_SIZE=3
VICTOR_POOL_LOAD_BALANCER=adaptive
VICTOR_POOL_ENABLE_WARMUP=true
VICTOR_POOL_HEALTH_CHECK_INTERVAL=30
```

**Profile Configuration:**
```yaml
# profiles.yaml
enable_provider_pool: true
pool_size: 3
pool_load_balancer: adaptive
pool_enable_warmup: true
pool_warmup_concurrency: 3
pool_health_check_interval: 30
pool_max_retries: 3
pool_min_instances: 1
```

---

## Tool Selection

### Strategy Selection

```python
# Settings
tool_selection_strategy: str = "auto"
# Options: "auto", "keyword", "semantic", "hybrid"
```

**Strategies:**
- `auto`: Automatically choose based on model size (default)
  - Large models (> 30B): semantic search
  - Small models (< 30B): keyword search
- `keyword`: Fast metadata-based (<1ms, no embeddings)
- `semantic`: ML-based similarity search (~50ms, high quality)
- `hybrid`: Blends semantic + keyword (~30ms)

**Environment Variable:**
```bash
VICTOR_TOOL_SELECTION_STRATEGY=hybrid
```

**Profile Configuration:**
```yaml
# profiles.yaml
tool_selection_strategy: hybrid
```

### Tool Selection Tuning

```python
# Settings
# DEPRECATED: Use tool_selection_strategy instead
use_semantic_tool_selection: bool = True
```

**Migration Guide:**
```python
# Old (deprecated)
use_semantic_tool_selection: true

# New
tool_selection_strategy: semantic
```

### Tool Budget Management

```python
# Settings
tool_call_budget: int = 100  # Max tool calls per session
tool_call_budget_warning_threshold: int = 80  # Warn at 80%
fallback_max_tools: int = 8  # Fallback when pruning removes all tools
```

**Environment Variables:**
```bash
VICTOR_TOOL_CALL_BUDGET=100
VICTOR_TOOL_CALL_BUDGET_WARNING_THRESHOLD=80
```

---

## Budget Management

### Tool Budget Configuration

```python
# Budget limits
BUDGET_LIMITS = BudgetLimits(
    min_session_budget=15,
    max_session_budget=100,
    warning_threshold_pct=0.8,
    planning_multiplier=2.5,
    explore_multiplier=3.0,
)
```

### Budget Types

```python
# Budget types tracked
class BudgetType(str, Enum):
    TOOL_CALLS = "tool_calls"  # Total tool calls
    ITERATIONS = "iterations"  # Total LLM iterations
    EXPLORATION = "exploration"  # Read/search operations
    ACTION = "action"  # Write/modify operations
```

### Budget Multipliers

```python
# Model-specific multipliers
# - GPT-4o: 1.0 (baseline)
# - Claude Opus: 1.2 (more capable)
# - DeepSeek: 1.3 (needs more exploration)
# - Ollama local: 1.5 (needs more attempts)

# Mode multipliers
# - BUILD: 2.0 (reading before writing)
# - PLAN: 2.5 (thorough analysis)
# - EXPLORE: 3.0 (exploration is primary goal)

# Productivity multipliers
# - High productivity: 0.8 (less budget needed)
# - Normal: 1.0
# - Low productivity: 1.2-2.0 (more attempts needed)
```

---

## Logging and Observability

### Logging Configuration

```python
# Settings
log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
log_file: Optional[str] = None  # Path to log file
```

**Environment Variables:**
```bash
VICTOR_LOG_LEVEL=DEBUG
VICTOR_LOG_FILE=/tmp/victor.log
```

**Profile Configuration:**
```yaml
# profiles.yaml
log_level: DEBUG
log_file: /tmp/victor-debug.log
```

### Observability Logging

JSONL export for dashboard integration.

```python
# Settings
enable_observability_logging: bool = False
observability_log_path: Optional[str] = None  # Defaults to ~/.victor/metrics/victor.jsonl
```

**Environment Variables:**
```bash
VICTOR_ENABLE_OBSERVABILITY_LOGGING=true
VICTOR_OBSERVABILITY_LOG_PATH=/custom/path/victor.jsonl
```

**CLI Flag:**
```bash
victor chat --log-events
```

### UI Settings

```python
# Settings
theme: str = "monokai"
show_token_count: bool = True
show_cost_metrics: bool = False
stream_responses: bool = True
use_emojis: bool = True  # Enable emoji indicators (✓, ✗, etc.)
```

**Environment Variables:**
```bash
VICTOR_THEME=monokai
VICTOR_SHOW_TOKEN_COUNT=true
VICTOR_SHOW_COST_METRICS=true
VICTOR_STREAM_RESPONSES=true
VICTOR_USE_EMOJIS=false
```

---

## Security Settings

### Air-Gapped Mode

```python
# Settings
airgapped_mode: bool = False
```

**Effects when enabled:**
- Only local providers (Ollama, LMStudio, vLLM)
- No web tools
- Local embeddings only
- No external API calls

**Environment Variable:**
```bash
VICTOR_AIRGAPPED_MODE=true
```

### Write Approval Mode

```python
# Settings
write_approval_mode: str = "risky_only"
# Options: "off", "risky_only", "all_writes"
```

**Modes:**
- `off`: Never require approval (dangerous, testing only)
- `risky_only`: Only for HIGH/CRITICAL risk operations (default)
- `all_writes`: Require for ALL write operations (recommended for task mode)

**Environment Variable:**
```bash
VICTOR_WRITE_APPROVAL_MODE=all_writes
```

### Headless Mode Settings

```python
# Settings
headless_mode: bool = False  # Run without prompts
dry_run_mode: bool = False  # Preview changes only
auto_approve_safe: bool = False  # Auto-approve safe operations
max_file_changes: Optional[int] = None  # Limit file changes
one_shot_mode: bool = False  # Exit after single request
```

**Environment Variables:**
```bash
VICTOR_HEADLESS_MODE=true
VICTOR_DRY_RUN_MODE=true
VICTOR_AUTO_APPROVE_SAFE=true
VICTOR_MAX_FILE_CHANGES=10
VICTOR_ONE_SHOT_MODE=true
```

### Server Security

```python
# Settings
server_api_key: Optional[str] = None  # Required for HTTP/WebSocket
server_session_secret: Optional[str] = None  # HMAC secret for tokens
server_max_sessions: int = 100
server_max_message_bytes: int = 32768
server_session_ttl_seconds: int = 86400  # 24 hours
```

**Environment Variables:**
```bash
VICTOR_SERVER_API_KEY=your-secret-key
VICTOR_SERVER_SESSION_SECRET=your-hmac-secret
VICTOR_SERVER_MAX_SESSIONS=100
VICTOR_SERVER_MAX_MESSAGE_BYTES=32768
VICTOR_SERVER_SESSION_TTL_SECONDS=86400
```

---

## Performance Tuning

### Embedding Configuration

```python
# Settings
unified_embedding_model: str = "BAAI/bge-small-en-v1.5"
embedding_provider: str = "sentence-transformers"  # sentence-transformers, ollama, vllm, lmstudio
embedding_model: str = unified_embedding_model  # Shared with codebase search
```

**Model Characteristics:**
- **Size:** 130MB
- **Dimensions:** 384
- **Speed:** ~6ms per embedding
- **MTEB Score:** 62.2
- **Use Case:** Code search, tool selection

**Environment Variables:**
```bash
VICTOR_EMBEDDING_PROVIDER=sentence-transformers
VICTOR_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

### Codebase Semantic Search

```python
# Settings
codebase_vector_store: str = "lancedb"  # lancedb, chromadb
codebase_embedding_provider: str = "sentence-transformers"
codebase_embedding_model: str = "BAAI/bge-small-en-v1.5"
codebase_persist_directory: Optional[str] = None  # Default: ~/.victor/embeddings/codebase
codebase_dimension: int = 384
codebase_batch_size: int = 32
codebase_graph_store: str = "sqlite"
```

**Environment Variables:**
```bash
VICTOR_CODEBASE_VECTOR_STORE=lancedb
VICTOR_CODEBASE_EMBEDDING_PROVIDER=sentence-transformers
VICTOR_CODEBASE_DIMENSION=384
VICTOR_CODEBASE_BATCH_SIZE=32
```

### Semantic Search Quality

```python
# Settings
semantic_similarity_threshold: float = 0.25  # Min score [0.1-0.9]
semantic_query_expansion_enabled: bool = True
semantic_max_query_expansions: int = 5
```

**Hybrid Search (Semantic + Keyword):**
```python
enable_hybrid_search: bool = False
hybrid_search_semantic_weight: float = 0.6  # Weight for semantic (0.0-1.0)
hybrid_search_keyword_weight: float = 0.4  # Weight for keyword (0.0-1.0)
```

**RL-based Threshold Learning:**
```python
enable_semantic_threshold_rl_learning: bool = False
semantic_threshold_overrides: dict = {}  # {"model:task:tool": threshold}
```

### Tool Optimization

```python
# Settings
enable_tool_deduplication: bool = True
tool_deduplication_window_size: int = 20  # Recent calls to track
tool_cache_enabled: bool = True
tool_cache_ttl: int = 600  # seconds
tool_cache_allowlist: List[str] = [
    "code_search",
    "semantic_code_search",
    "list_directory",
    "plan_files",
]
```

**Tool Retry Settings:**
```python
tool_retry_enabled: bool = True
tool_retry_max_attempts: int = 3
tool_retry_base_delay: float = 1.0  # seconds
tool_retry_max_delay: float = 10.0  # seconds
```

### Tool Selection Tuning (Per Profile)

Profile-specific tool selection overrides.

```yaml
# ~/.victor/profiles.yaml

profiles:
  local-qwen:
    provider: ollama
    model_name: qwen2.5:32b
    temperature: 0.7

    # Tool selection configuration
    tool_selection:
      model_size_tier: large  # tiny, small, medium, large, cloud
      base_threshold: 0.3
      base_max_tools: 15
      exploration_boost: 0.1
      execution_boost: 0.0
      context_aware: true

    # Provider tuning
    loop_repeat_threshold: 3
    max_continuation_prompts: 2
    quality_threshold: 0.6
    grounding_threshold: 0.65
    max_tool_calls_per_turn: 10
    tool_cache_enabled: true
    tool_deduplication_enabled: true
    session_idle_timeout: 300
    timeout: 120

  cloud-claude:
    provider: anthropic
    model_name: claude-sonnet-4-5
    temperature: 0.3

    tool_selection:
      model_size_tier: cloud
      base_threshold: 0.5
      base_max_tools: 25

    # Cloud defaults (no overrides needed)
```

**Model Size Tiers:**
```python
TOOL_SELECTION_PRESETS = {
    "tiny": {  # 0.5B-3B
        "base_threshold": 0.25,
        "base_max_tools": 8,
        "exploration_boost": 0.15,
        "execution_boost": 0.0,
        "context_aware": False,
    },
    "small": {  # 7B-8B
        "base_threshold": 0.30,
        "base_max_tools": 10,
        "exploration_boost": 0.10,
        "execution_boost": 0.0,
        "context_aware": True,
    },
    "medium": {  # 13B-15B
        "base_threshold": 0.35,
        "base_max_tools": 12,
        "exploration_boost": 0.05,
        "execution_boost": -0.05,
        "context_aware": True,
    },
    "large": {  # 30B+
        "base_threshold": 0.40,
        "base_max_tools": 15,
        "exploration_boost": 0.0,
        "execution_boost": -0.10,
        "context_aware": True,
    },
    "cloud": {  # Claude/GPT
        "base_threshold": 0.50,
        "base_max_tools": 25,
        "exploration_boost": 0.0,
        "execution_boost": -0.15,
        "context_aware": True,
    },
}
```

---

## Vertical Configuration

### Vertical Settings

```python
# Settings
default_vertical: str = "coding"  # coding, research, devops, etc.
auto_detect_vertical: bool = False  # Auto-detect from project (experimental)
vertical_loading_mode: str = "eager"  # eager, lazy, auto
```

**Vertical Modes:**
- `eager`: Load all extensions immediately (default, backward compatible)
- `lazy`: Load metadata only, defer heavy modules (faster startup)
- `auto`: Auto-choose based on environment (production=lazy, dev=eager)

**Environment Variables:**
```bash
VICTOR_DEFAULT_VERTICAL=coding
VICTOR_AUTO_DETECT_VERTICAL=true
VICTOR_VERTICAL_LOADING_MODE=lazy
```

### MCP Configuration

```python
# Settings
use_mcp_tools: bool = False
mcp_command: Optional[str] = None  # e.g., "python mcp_server.py"
mcp_prefix: str = "mcp"
```

**Project MCP Config (`.victor/mcp.yaml`):**
```yaml
servers:
  filesystem:
    command: python
    args: ["-m", "mcp_server_filesystem", "/path/to/allowed"]
  git:
    command: node
    args: ["mcp-server-git/dist/index.js", "/path/to/repo"]
```

---

## Code Execution Settings

### Sandbox Defaults

```python
# Settings
code_executor_network_disabled: bool = True
code_executor_memory_limit: Optional[str] = "512m"
code_executor_cpu_shares: Optional[int] = 256
```

**Environment Variables:**
```bash
VICTOR_CODE_EXECUTOR_NETWORK_DISABLED=true
VICTOR_CODE_EXECUTOR_MEMORY_LIMIT=512m
VICTOR_CODE_EXECUTOR_CPU_SHARES=256
```

---

## Complete Settings Reference

### Settings Class Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| **Core** ||||
| `default_provider` | str | `ollama` | Default LLM provider |
| `default_model` | str | `qwen3-coder:30b` | Default model |
| `default_temperature` | float | `0.7` | Sampling temperature |
| `default_max_tokens` | int | `4096` | Max tokens per response |
| **API Keys** ||||
| `anthropic_api_key` | str | `None` | Anthropic API key |
| `openai_api_key` | str | `None` | OpenAI API key |
| `google_api_key` | str | `None` | Google API key |
| `moonshot_api_key` | str | `None` | Moonshot API key |
| `deepseek_api_key` | str | `None` | DeepSeek API key |
| **Local Providers** ||||
| `ollama_base_url` | str | `http://localhost:11434` | Ollama endpoint |
| `lmstudio_base_urls` | list | `["http://127.0.0.1:1234"]` | LMStudio endpoints |
| `vllm_base_url` | str | `http://localhost:8000` | vLLM endpoint |
| **Logging** ||||
| `log_level` | str | `INFO` | Logging level |
| `log_file` | str | `None` | Log file path |
| `enable_observability_logging` | bool | `False` | Enable JSONL export |
| **Tool Selection** ||||
| `tool_selection_strategy` | str | `auto` | Tool selection strategy |
| `use_semantic_tool_selection` | bool | `True` | **DEPRECATED** |
| **Budget** ||||
| `tool_call_budget` | int | `100` | Max tool calls per session |
| `enable_provider_pool` | bool | `False` | Enable provider pooling |
| `pool_size` | int | `3` | Pool size |
| **Security** ||||
| `airgapped_mode` | bool | `False` | Air-gapped mode |
| `write_approval_mode` | str | `risky_only` | Write approval mode |
| `headless_mode` | bool | `False` | Headless mode |
| `dry_run_mode` | bool | `False` | Dry run mode |
| **Vertical** ||||
| `default_vertical` | str | `coding` | Default vertical |
| `vertical_loading_mode` | str | `eager` | Vertical loading mode |
| **Embeddings** ||||
| `unified_embedding_model` | str | `BAAI/bge-small-en-v1.5` | Embedding model |
| `embedding_provider` | str | `sentence-transformers` | Embedding provider |
| **Codebase Search** ||||
| `codebase_vector_store` | str | `lancedb` | Vector store backend |
| `semantic_similarity_threshold` | float | `0.25` | Similarity threshold |

---

**See Also:**
- [API Reference](API_REFERENCE.md) - Main API documentation
- [Protocol Reference](PROTOCOL_REFERENCE.md) - Protocol interfaces
- [Provider Reference](PROVIDER_REFERENCE.md) - Provider details
