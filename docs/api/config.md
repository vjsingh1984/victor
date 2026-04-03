# Configuration API Reference

Configuration management for Victor including settings, profiles, and path management.

## Overview

The Configuration API provides:
- **Settings** management with environment variable support
- **Profile templates** for different use cases
- **Project path management** for consistent file locations
- **Provider configuration** for 22+ LLM backends

## Quick Example

```python
from victor.config.settings import load_settings, Settings
from victor.config.profiles import get_profile, install_profile

# Load settings
settings = load_settings()
print(f"Default provider: {settings.default_provider}")

# Use profile
profile = get_profile("coding")
install_profile(profile)

# Access paths
from victor.config.settings import get_project_paths

paths = get_project_paths()
print(f"Context file: {paths.project_context_file}")
```

## Settings Class

Main application settings with environment variable support. Settings are stratified into 7 nested config groups while maintaining backward-compatible flat field access.

```python
class Settings(BaseSettings):
    """Main application settings."""
```

### Loading Settings

```python
from victor.config.settings import load_settings, Settings

# Load settings (reads .env, environment variables, defaults)
settings = load_settings()

# Create settings instance directly
settings = Settings(
    default_provider="anthropic",
    default_model="claude-sonnet-4-5-20250514"
)

# Flat access (backward compatible)
print(settings.default_provider)
print(settings.anthropic_api_key)

# Nested group access (preferred for typed grouping)
print(settings.provider.default_provider)   # "anthropic"
print(settings.provider.anthropic_api_key)  # API key
print(settings.tools.tool_retry_enabled)    # True
print(settings.resilience.circuit_breaker_failure_threshold)  # 5
print(settings.security.write_approval_mode)  # "risky_only"
print(settings.search.codebase_vector_store)  # "lancedb"
print(settings.events.event_backend_type)     # "in_memory"
print(settings.pipeline.intelligent_pipeline_enabled)  # True
```

### Nested Config Groups

| Group | Access | Fields | Purpose |
|-------|--------|--------|---------|
| `settings.provider` | `ProviderSettings` | 13 | Provider connection and model defaults |
| `settings.tools` | `ToolSettings` | 21 | Tool execution, selection, retry, caching |
| `settings.search` | `SearchSettings` | 18 | Codebase search and semantic config |
| `settings.resilience` | `ResilienceSettings` | 17 | Circuit breaker, retry, rate limiting |
| `settings.security` | `SecuritySettings` | 19 | Server security, sandboxing, approval |
| `settings.events` | `EventSettings` | 24 | Event system backend and config |
| `settings.pipeline` | `PipelineSettings` | 30+ | Intelligent pipeline, quality, recovery |

Flat and nested values are synced at construction time via `model_validator(mode="after")`. Flat fields are the source of truth; nested models use `exclude=True` to keep `model_dump()` clean.

### Environment Variables

Settings can be configured via environment variables:

```bash
# Provider selection
export VICTOR_DEFAULT_PROVIDER=anthropic
export VICTOR_DEFAULT_MODEL=claude-sonnet-4-5-20250514

# API keys
export ANTHROPIC_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here
export GOOGLE_API_KEY=your_key_here

# Local servers
export OLLAMA_BASE_URL=http://localhost:11434
export LMSTUDIO_BASE_URLS=http://localhost:1234

# Configuration directory
export VICTOR_DIR_NAME=.victor

# Context file
export VICTOR_CONTEXT_FILE=init.md

# Privacy
export VICTOR_AIRGAPPED_MODE=true
```

### Core Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `default_provider` | `str` | `"ollama"` | Default LLM provider |
| `default_model` | `str` | `"qwen3-coder:30b"` | Default model |
| `default_temperature` | `float` | `0.7` | Default temperature |
| `default_max_tokens` | `int` | `4096` | Default max tokens |
| `airgapped_mode` | `bool` | `False` | Disable network operations |

### API Key Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `anthropic_api_key` | `str \| None` | `None` | Anthropic API key |
| `openai_api_key` | `str \| None` | `None` | OpenAI API key |
| `google_api_key` | `str \| None` | `None` | Google API key |
| `moonshot_api_key` | `str \| None` | `None` | Moonshot AI API key |
| `deepseek_api_key` | `str \| None` | `None` | DeepSeek API key |

### Local Server Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `ollama_base_url` | `str` | `"http://localhost:11434"` | Ollama server URL |
| `lmstudio_base_urls` | `list[str]` | `["http://127.0.0.1:1234"]` | LMStudio endpoints |
| `vllm_base_url` | `str` | `"http://localhost:8000"` | vLLM server URL |

### Logging Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `log_level` | `str` | `"INFO"` | Logging level |
| `log_file` | `str \| None` | `None` | Log file path |
| `enable_observability_logging` | `bool` | `False` | Enable JSONL event logging |

### Tool Selection Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `use_semantic_tool_selection` | `bool` | `True` | Use embeddings for tool selection |
| `embedding_provider` | `str` | `"sentence-transformers"` | Embedding provider |
| `embedding_model` | `str` | `"BAAI/bge-small-en-v1.5"` | Embedding model |
| `tool_call_budget` | `int` | `200` | Max tool calls per session |
| `fallback_max_tools` | `int` | `8` | Cap tool list when pruning removes everything |

### Tool Result Caching

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `tool_cache_enabled` | `bool` | `True` | Enable tool result caching |
| `tool_cache_ttl` | `int` | `600` | Cache TTL (seconds) |
| `tool_cache_allowlist` | `list[str]` | See defaults | Tools to cache |

### Performance Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `tool_selection_cache_enabled` | `bool` | `True` | Cache tool selection results |
| `tool_selection_cache_ttl` | `int` | `300` | Selection cache TTL (seconds) |
| `framework_preload_enabled` | `bool` | `True` | Enable framework preloading |
| `http_connection_pool_enabled` | `bool` | `False` | Enable HTTP connection pooling |

### Headless Mode Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `headless_mode` | `bool` | `False` | Run without prompts |
| `dry_run_mode` | `bool` | `False` | Preview without changes |
| `auto_approve_safe` | `bool` | `False` | Auto-approve safe operations |
| `max_file_changes` | `int \| None` | `None` | Limit file modifications |
| `one_shot_mode` | `bool` | `False` | Exit after single request |

### Code Execution Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `code_executor_network_disabled` | `bool` | `True` | Disable network in code exec |
| `code_executor_memory_limit` | `str \| None` | `"512m"` | Memory limit |
| `code_executor_cpu_shares` | `int \| None` | `256` | CPU shares |

### UI Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `theme` | `str` | `"monokai"` | UI theme |
| `show_token_count` | `bool` | `True` | Show token usage |
| `show_cost_metrics` | `bool` | `False` | Show cost metrics |
| `stream_responses` | `bool` | `True` | Stream responses |
| `use_emojis` | `bool` | Auto | Enable emoji indicators |

### Write Approval Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `write_approval_mode` | `str` | `"risky_only"` | When to require approval |

Options:
- `"off"`: Never require approval (dangerous)
- `"risky_only"`: Only for HIGH/CRITICAL risk (default)
- `"all_writes"`: Require for ALL writes

### Tool Validation Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `tool_validation_mode` | `str` | `"lenient"` | Tool argument validation |

Options:
- `"strict"`: Block on validation errors
- `"lenient"`: Warn only
- `"off"`: Disable validation

### Semantic Search Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `semantic_similarity_threshold` | `float` | `0.25` | Min similarity score |
| `semantic_query_expansion_enabled` | `bool` | `True` | Enable query expansion |
| `semantic_max_query_expansions` | `int` | `5` | Max query variations |
| `enable_hybrid_search` | `bool` | `False` | Enable hybrid search |
| `hybrid_search_semantic_weight` | `float` | `0.6` | Semantic weight |
| `hybrid_search_keyword_weight` | `float` | `0.4` | Keyword weight |

## ProfileConfig Class

Configuration for a model profile.

```python
class ProfileConfig(BaseSettings):
    """Configuration for a model profile."""
```

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str` | Required | Provider name |
| `model` | `str` | Required | Model identifier |
| `temperature` | `float` | `0.7` | Temperature |
| `max_tokens` | `int` | `4096` | Max tokens |
| `description` | `str \| None` | `None` | Profile description |
| `tool_selection` | `dict \| None` | `None` | Tool selection config |
| `planning_provider` | `str \| None` | `None` | Override for planning |
| `planning_model` | `str \| None` | `None` | Override for planning |
| `timeout` | `int \| None` | `None` | Request timeout |
| `tool_cache_enabled` | `bool \| None` | `None` | Enable tool caching |

### Tool Selection Configuration

```python
tool_selection: {
    # Use predefined tier
    "model_size_tier": "small",  # tiny, small, medium, large, cloud

    # Or set explicit values
    "base_threshold": 0.3,
    "base_max_tools": 10,

    # Per-stage overrides
    "stages": {
        "initial": {"max_tools": 5},
        "execution": {"max_tools": 15}
    }
}
```

**Tier Presets**:

| Tier | Model Size | Max Tools | Threshold |
|------|-----------|-----------|-----------|
| `tiny` | 0.5B-3B | 5 | 0.6 |
| `small` | 7B-8B | 8 | 0.5 |
| `medium` | 13B-15B | 12 | 0.4 |
| `large` | 30B+ | 20 | 0.3 |
| `cloud` | Claude/GPT | 33 | 0.2 |

## ProjectPaths Class

Centralized path management for Victor.

```python
class ProjectPaths:
    """Centralized path management for Victor.

    Provides consistent paths for both project-local and global storage.
    """
```

### Directory Structure

```
{project_root}/.victor/
├── init.md              # Project context
├── conversation.db      # Conversation history
├── embeddings/          # Vector embeddings
├── graph/               # Graph data
├── backups/             # File edit backups
├── changes/             # Undo/redo history
├── sessions/            # Session snapshots
└── mcp.yaml             # MCP configuration

~/.victor/
├── profiles.yaml        # Global profiles
├── plugins/             # Plugins directory
├── cache/               # Global cache
├── logs/                # Log files
├── metrics/             # Observability metrics
└── embeddings/          # Global embeddings
```

### Project-Local Paths

| Property | Type | Path |
|----------|------|------|
| `project_victor_dir` | `Path` | `{project_root}/.victor/` |
| `project_context_file` | `Path` | `.victor/init.md` |
| `conversation_db` | `Path` | `.victor/conversation.db` |
| `embeddings_dir` | `Path` | `.victor/embeddings/` |
| `graph_dir` | `Path` | `.victor/graph/` |
| `backups_dir` | `Path` | `.victor/backups/` |
| `changes_dir` | `Path` | `.victor/changes/` |
| `sessions_dir` | `Path` | `.victor/sessions/` |
| `mcp_config` | `Path` | `.victor/mcp.yaml` |

### Global Paths

| Property | Type | Path |
|----------|------|------|
| `global_victor_dir` | `Path` | `~/.victor/` |
| `global_profiles` | `Path` | `~/.victor/profiles.yaml` |
| `global_plugins_dir` | `Path` | `~/.victor/plugins/` |
| `global_cache_dir` | `Path` | `~/.victor/cache/` |
| `global_logs_dir` | `Path` | `~/.victor/logs/` |
| `global_embeddings_dir` | `Path` | `~/.victor/embeddings/` |
| `global_mcp_config` | `Path` | `~/.victor/mcp.yaml` |

### Methods

```python
from victor.config.settings import get_project_paths

# Get paths for current project
paths = get_project_paths()

# Get paths for specific project
paths = get_project_paths(Path("/path/to/project"))

# Ensure directories exist
paths.ensure_project_dirs()
paths.ensure_global_dirs()

# Find context file
context = paths.find_context_file()
```

## Profile Management

### List Profiles

```python
from victor.config.profiles import list_profiles

profiles = list_profiles()
for profile in profiles:
    print(f"{profile.name}: {profile.display_name}")
```

### Get Profile

```python
from victor.config.profiles import get_profile

profile = get_profile("coding")
print(profile.description)
print(profile.settings)
```

### Install Profile

```python
from victor.config.profiles import install_profile, get_profile

profile = get_profile("coding")
config_path = install_profile(
    profile,
    config_dir=Path.home() / ".victor",
    provider_override="anthropic",
    model_override="claude-sonnet-4-5-20250514"
)
```

### Get Recommended Profile

```python
from victor.config.profiles import get_recommended_profile

profile = get_recommended_profile()
print(f"Recommended: {profile.display_name}")
```

## Built-in Profiles

### Basic Profile

```yaml
profiles:
  basic:
    provider: ollama
    model: qwen2.5-coder:7b
    temperature: 0.7
    max_tokens: 4096
```

**Use Case**: Simple defaults for beginners

### Advanced Profile

```yaml
profiles:
  advanced:
    provider: ollama
    model: qwen2.5-coder:14b
    temperature: 0.7
    max_tokens: 8192
    tool_selection:
      model_size_tier: medium
```

**Use Case**: More capable local models

### Expert Profile

```yaml
profiles:
  expert:
    provider: anthropic
    model: claude-sonnet-4-5-20250514
    temperature: 0.7
    max_tokens: 8192
    tool_selection:
      model_size_tier: cloud
```

**Use Case**: Full capabilities with cloud models

### Coding Profile

```yaml
profiles:
  coding:
    provider: ollama
    model: qwen3-coder:30b
    temperature: 0.3
    max_tokens: 8192
    tool_selection:
      model_size_tier: large
```

**Use Case**: Optimized for coding tasks

### Research Profile

```yaml
profiles:
  research:
    provider: anthropic
    model: claude-sonnet-4-5-20250514
    temperature: 0.7
    max_tokens: 8192
    tool_selection:
      model_size_tier: cloud
```

**Use Case**: Optimized for research and web search

## Configuration Precedence

Settings are loaded in order of precedence (highest first):

1. **Environment variables** (e.g., `VICTOR_DEFAULT_PROVIDER`)
2. **`.env` file** (in project root or home directory)
3. **`profiles.yaml`** (selected profile)
4. **Default values** (from Settings class)

```bash
# Environment variable overrides everything
export VICTOR_DEFAULT_PROVIDER=anthropic

# .env file overrides profiles.yaml
echo "VICTOR_DEFAULT_MODEL=claude-opus-4-5" > .env

# profiles.yaml overrides defaults
```

## Best Practices

### 1. Use Profiles for Common Configurations

```python
# Good - Use profile
from victor.framework import Agent

agent = Agent(profile="coding")

# Avoid - Hardcoded config
agent = Agent(
    provider="ollama",
    model="qwen3-coder:30b",
    temperature=0.3,
    max_tokens=8192,
    tools=[...]
)
```

### 2. Store Secrets in Environment Variables

```bash
# Good - Use environment variables
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...

# Avoid - Hardcoded in files
# profiles.yaml
# provider: anthropic
# api_key: sk-ant-xxx  # DON'T DO THIS
```

### 3. Use Project-Local Configuration

```python
# Good - Project-specific config
# .victor/init.md defines project context
agent = Agent()  # Automatically uses project context

# Or specify project root
paths = get_project_paths(Path("/my/project"))
```

### 4. Leverage Tool Selection Tiers

```yaml
# Good - Use tier presets
tool_selection:
  model_size_tier: small  # Automatically sets threshold and max_tools

# Avoid - Manual values (unless needed)
tool_selection:
  base_threshold: 0.5
  base_max_tools: 8
```

### 5. Use Write Approval in Production

```python
# Good - Enable approval for risky operations
# In settings or profiles.yaml
write_approval_mode: risky_only  # or all_writes

# Avoid - Unprotected writes in production
write_approval_mode: off
```

## See Also

- [Agent API](agent.md) - Agent configuration
- [Tools API](tools.md) - Tool configuration
- [Provider API](providers.md) - Provider-specific settings
- [Troubleshooting Guide](../guides/TROUBLESHOOTING.md) - Configuration issues
