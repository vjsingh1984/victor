# Configuration Options Reference

Complete reference for Victor configuration options from `victor/config/settings.py`.

## Configuration Files

| File | Location | Purpose |
|------|----------|---------|
| `profiles.yaml` | `~/.victor/profiles.yaml` | Model profiles and provider settings |
| `api_keys.yaml` | `~/.victor/api_keys.yaml` | API key storage (0600 permissions) |
| `init.md` | `.victor/init.md` | Project-specific context |
| `mcp.yaml` | `.victor/mcp.yaml` | MCP server configuration |

---

## Provider Settings

### Default Provider Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `default_provider` | `ollama` | Default LLM provider |
| `default_model` | `qwen3-coder:30b` | Default model identifier |
| `default_temperature` | `0.7` | Default sampling temperature (0.0-2.0) |
| `default_max_tokens` | `4096` | Default maximum output tokens |

### Local Provider URLs

| Setting | Default | Description |
|---------|---------|-------------|
| `ollama_base_url` | `http://localhost:11434` | Ollama server URL |
| `lmstudio_base_urls` | `["http://127.0.0.1:1234"]` | LMStudio server URLs (tries in order) |
| `vllm_base_url` | `http://localhost:8000` | vLLM server URL |

### API Keys

| Setting | Env Variable | Description |
|---------|--------------|-------------|
| `anthropic_api_key` | `ANTHROPIC_API_KEY` | Anthropic (Claude) API key |
| `openai_api_key` | `OPENAI_API_KEY` | OpenAI API key |
| `google_api_key` | `GOOGLE_API_KEY` | Google (Gemini) API key |
| `moonshot_api_key` | `MOONSHOT_API_KEY` | Moonshot AI (Kimi) API key |
| `deepseek_api_key` | `DEEPSEEK_API_KEY` | DeepSeek API key |

### Provider Resilience

| Setting | Default | Description |
|---------|---------|-------------|
| `resilience_enabled` | `True` | Enable circuit breaker and retry logic |
| `circuit_breaker_failure_threshold` | `5` | Failures before circuit opens |
| `circuit_breaker_success_threshold` | `2` | Successes before circuit closes |
| `circuit_breaker_timeout` | `60.0` | Seconds before half-open state |
| `circuit_breaker_half_open_max` | `3` | Max requests in half-open state |
| `retry_max_attempts` | `3` | Maximum retry attempts |
| `retry_base_delay` | `1.0` | Base delay in seconds |
| `retry_max_delay` | `60.0` | Maximum delay between retries |
| `retry_exponential_base` | `2.0` | Exponential backoff multiplier |

### Rate Limiting

| Setting | Default | Description |
|---------|---------|-------------|
| `rate_limiting_enabled` | `True` | Enable rate limiting |
| `rate_limit_requests_per_minute` | `50` | Requests per minute limit |
| `rate_limit_tokens_per_minute` | `50000` | Tokens per minute limit |
| `rate_limit_max_concurrent` | `5` | Maximum concurrent requests |
| `rate_limit_queue_size` | `100` | Maximum pending requests in queue |

---

## Tool Settings

### Tool Execution

| Setting | Default | Description |
|---------|---------|-------------|
| `tool_call_budget` | `500` | Maximum tool calls per session |
| `tool_call_budget_warning_threshold` | `400` | Warn when approaching budget |
| `fallback_max_tools` | `8` | Cap tool list when stage pruning removes all |

### Tool Retry

| Setting | Default | Description |
|---------|---------|-------------|
| `tool_retry_enabled` | `True` | Enable automatic retry for failed tools |
| `tool_retry_max_attempts` | `3` | Maximum retry attempts per tool |
| `tool_retry_base_delay` | `1.0` | Base delay in seconds |
| `tool_retry_max_delay` | `10.0` | Maximum delay between retries |

### Tool Caching

| Setting | Default | Description |
|---------|---------|-------------|
| `tool_cache_enabled` | `True` | Enable tool result caching |
| `tool_cache_ttl` | `600` | Cache TTL in seconds |
| `tool_cache_allowlist` | `["code_search", ...]` | Tools eligible for caching |

### Tool Validation

| Setting | Default | Description |
|---------|---------|-------------|
| `tool_validation_mode` | `lenient` | Validation mode: `strict`, `lenient`, `off` |

### Tool Deduplication

| Setting | Default | Description |
|---------|---------|-------------|
| `enable_tool_deduplication` | `True` | Prevent redundant tool calls |
| `tool_deduplication_window_size` | `20` | Number of recent calls to track |

### Tool Selection (Semantic)

| Setting | Default | Description |
|---------|---------|-------------|
| `use_semantic_tool_selection` | `True` | Use embeddings for tool selection |
| `embedding_provider` | `sentence-transformers` | Embedding provider |
| `embedding_model` | `BAAI/bge-small-en-v1.5` | Embedding model |

---

## Workflow Settings

### Workflow Definition Cache

| Setting | Default | Description |
|---------|---------|-------------|
| `workflow_definition_cache_enabled` | `True` | Cache parsed YAML workflows |
| `workflow_definition_cache_ttl` | `3600` | Cache TTL in seconds |
| `workflow_definition_cache_max_entries` | `100` | Maximum cached entries |

### StateGraph

| Setting | Default | Description |
|---------|---------|-------------|
| `stategraph_copy_on_write_enabled` | `True` | Enable copy-on-write for state |

### Checkpoints

| Setting | Default | Description |
|---------|---------|-------------|
| `checkpoint_enabled` | `True` | Enable checkpoint system |
| `checkpoint_auto_interval` | `5` | Tool calls between auto-checkpoints |
| `checkpoint_max_per_session` | `50` | Maximum checkpoints per session |
| `checkpoint_compression_enabled` | `True` | Compress checkpoint data |
| `checkpoint_compression_threshold` | `1024` | Min bytes before compression |

### Human-in-the-Loop (HITL)

| Setting | Default | Description |
|---------|---------|-------------|
| `hitl_default_timeout` | `300.0` | Default timeout in seconds |
| `hitl_default_fallback` | `abort` | Action on timeout: `abort`, `continue`, `skip` |
| `hitl_auto_approve_low_risk` | `False` | Auto-approve LOW risk actions |
| `hitl_keyboard_shortcuts_enabled` | `True` | Enable y/n shortcuts in TUI |

---

## Safety Settings

### Write Approval

| Setting | Default | Description |
|---------|---------|-------------|
| `write_approval_mode` | `risky_only` | Approval mode: `off`, `risky_only`, `all_writes` |

### Headless/Automation Mode

| Setting | Default | Description |
|---------|---------|-------------|
| `headless_mode` | `False` | Run without prompts |
| `dry_run_mode` | `False` | Preview changes without applying |
| `auto_approve_safe` | `False` | Auto-approve read-only and LOW risk ops |
| `max_file_changes` | `None` | Limit file modifications per session |
| `one_shot_mode` | `False` | Exit after completing single request |

### Air-Gapped Mode

| Setting | Default | Description |
|---------|---------|-------------|
| `airgapped_mode` | `False` | Only allow local providers, no web tools |

### Security Scans

| Setting | Default | Description |
|---------|---------|-------------|
| `security_dependency_scan` | `False` | Enable dependency vulnerability scanning |
| `security_iac_scan` | `False` | Enable infrastructure-as-code scanning |

---

## Performance Settings

### Context Management

| Setting | Default | Description |
|---------|---------|-------------|
| `max_context_tokens` | `100000` | Maximum tokens in context window |
| `response_token_reserve` | `4096` | Tokens reserved for model response |

### Context Compaction

| Setting | Default | Description |
|---------|---------|-------------|
| `context_compaction_strategy` | `tiered` | Strategy: `simple`, `tiered`, `semantic`, `hybrid` |
| `context_min_messages_to_keep` | `6` | Minimum messages after compaction |
| `context_tool_retention_weight` | `1.5` | Boost for tool result retention |
| `context_recency_weight` | `2.0` | Boost for recent messages |
| `context_semantic_threshold` | `0.3` | Min similarity for semantic retention |

### Exploration Limits

| Setting | Default | Description |
|---------|---------|-------------|
| `max_exploration_iterations` | `200` | Base iteration limit (multiplied by mode) |
| `max_exploration_iterations_action` | `500` | Limit for action tasks |
| `max_exploration_iterations_analysis` | `1000` | Limit for analysis tasks |
| `min_content_threshold` | `50` | Min chars for "substantial" output |
| `max_research_iterations` | `50` | Limit for web research |

### Recovery Thresholds

| Setting | Default | Description |
|---------|---------|-------------|
| `recovery_empty_response_threshold` | `5` | Force after N empty responses |
| `recovery_blocked_consecutive_threshold` | `6` | Force after N consecutive blocks |
| `recovery_blocked_total_threshold` | `9` | Force after N total blocked attempts |
| `max_continuation_prompts_analysis` | `6` | Continuation prompts for analysis |
| `max_continuation_prompts_action` | `5` | Continuation prompts for actions |
| `max_continuation_prompts_default` | `3` | Default continuation prompts |

### Session Timeout

| Setting | Default | Description |
|---------|---------|-------------|
| `session_idle_timeout` | `180` | Seconds of inactivity before forcing completion |

### Streaming Metrics

| Setting | Default | Description |
|---------|---------|-------------|
| `streaming_metrics_enabled` | `True` | Enable streaming performance metrics |
| `streaming_metrics_history_size` | `1000` | Samples to retain |

---

## Codebase Search Settings

### Vector Store

| Setting | Default | Description |
|---------|---------|-------------|
| `codebase_vector_store` | `lancedb` | Vector store: `lancedb`, `chromadb` |
| `codebase_embedding_provider` | `sentence-transformers` | Embedding provider |
| `codebase_embedding_model` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `codebase_persist_directory` | `None` | Custom persistence path |
| `codebase_dimension` | `384` | Embedding dimension |
| `codebase_batch_size` | `32` | Batch size for embedding generation |

### Graph Store

| Setting | Default | Description |
|---------|---------|-------------|
| `codebase_graph_store` | `sqlite` | Graph backend |
| `codebase_graph_path` | `None` | Custom graph db path |

### Semantic Search Quality

| Setting | Default | Description |
|---------|---------|-------------|
| `semantic_similarity_threshold` | `0.25` | Min score for results |
| `semantic_query_expansion_enabled` | `True` | Expand queries with synonyms |
| `semantic_max_query_expansions` | `5` | Max query variations |

### Hybrid Search

| Setting | Default | Description |
|---------|---------|-------------|
| `enable_hybrid_search` | `False` | Enable semantic + keyword search |
| `hybrid_search_semantic_weight` | `0.6` | Weight for semantic search |
| `hybrid_search_keyword_weight` | `0.4` | Weight for keyword search |

---

## Server Settings

### HTTP/WebSocket Server

| Setting | Default | Description |
|---------|---------|-------------|
| `server_api_key` | `None` | API key for HTTP/WebSocket auth |
| `server_session_secret` | `None` | HMAC secret for session tokens |
| `server_max_sessions` | `100` | Max simultaneous sessions |
| `server_max_message_bytes` | `32768` | Max WebSocket message size |
| `server_session_ttl_seconds` | `86400` | Session token TTL |

### Diagram Rendering

| Setting | Default | Description |
|---------|---------|-------------|
| `render_max_payload_bytes` | `20000` | Max diagram payload size |
| `render_timeout_seconds` | `10` | Render timeout |
| `render_max_concurrency` | `2` | Max concurrent renders |

### Code Execution Sandbox

| Setting | Default | Description |
|---------|---------|-------------|
| `code_executor_network_disabled` | `True` | Disable network in sandbox |
| `code_executor_memory_limit` | `512m` | Memory limit |
| `code_executor_cpu_shares` | `256` | CPU shares |

---

## UI Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `theme` | `monokai` | Color theme |
| `show_token_count` | `True` | Display token counts |
| `show_cost_metrics` | `False` | Show cost in metrics |
| `stream_responses` | `True` | Enable streaming output |
| `use_emojis` | `True` | Enable emoji indicators |

---

## Vertical Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `default_vertical` | `coding` | Default domain vertical |
| `auto_detect_vertical` | `False` | Auto-detect from project context |

---

## Conversation Memory

| Setting | Default | Description |
|---------|---------|-------------|
| `conversation_memory_enabled` | `True` | Enable SQLite persistence |
| `conversation_embeddings_enabled` | `True` | Enable LanceDB embeddings |

---

## Prompt Enrichment

| Setting | Default | Description |
|---------|---------|-------------|
| `prompt_enrichment_enabled` | `True` | Enable prompt enrichment |
| `prompt_enrichment_max_tokens` | `2000` | Max tokens to add |
| `prompt_enrichment_timeout_ms` | `500.0` | Timeout in milliseconds |
| `prompt_enrichment_cache_enabled` | `True` | Cache enrichments |
| `prompt_enrichment_cache_ttl` | `300` | Cache TTL in seconds |
| `prompt_enrichment_strategies` | `["knowledge_graph", ...]` | Enabled strategies |

---

## Intelligent Pipeline

| Setting | Default | Description |
|---------|---------|-------------|
| `intelligent_pipeline_enabled` | `True` | Master switch |
| `intelligent_quality_scoring` | `True` | Enable quality scoring |
| `intelligent_mode_learning` | `True` | Enable Q-learning for modes |
| `intelligent_prompt_optimization` | `True` | Enable prompt optimization |
| `intelligent_grounding_verification` | `True` | Enable hallucination detection |
| `intelligent_min_quality_threshold` | `0.5` | Minimum quality score |
| `intelligent_grounding_threshold` | `0.7` | Grounding confidence threshold |
| `intelligent_exploration_rate` | `0.3` | Q-learning exploration rate |
| `intelligent_learning_rate` | `0.1` | Q-learning alpha |
| `intelligent_discount_factor` | `0.9` | Q-learning gamma |

---

## Serialization

| Setting | Default | Description |
|---------|---------|-------------|
| `serialization_enabled` | `True` | Enable token-optimized serialization |
| `serialization_default_format` | `None` | Auto-select best format |
| `serialization_min_savings_threshold` | `0.15` | Min savings for alt format |
| `serialization_include_format_hint` | `True` | Include format description |
| `serialization_min_rows_for_tabular` | `3` | Min rows for tabular formats |
| `serialization_debug_mode` | `False` | Include data characteristics |

---

## Event System

| Setting | Default | Description |
|---------|---------|-------------|
| `event_backend_type` | `in_memory` | Backend: `in_memory`, `sqlite`, `redis`, etc. |
| `event_delivery_guarantee` | `at_most_once` | Delivery: `at_most_once`, `at_least_once`, `exactly_once` |
| `event_max_batch_size` | `100` | Max batch size |
| `event_flush_interval_ms` | `1000.0` | Flush interval in ms |

---

## Plugin System

| Setting | Default | Description |
|---------|---------|-------------|
| `plugin_enabled` | `True` | Enable plugin system |
| `plugin_packages` | `[]` | Python packages to load |
| `plugin_disabled` | `[]` | Plugin names to disable |
| `plugin_config` | `{}` | Per-plugin configuration |

---

## Observability

| Setting | Default | Description |
|---------|---------|-------------|
| `log_level` | `INFO` | Logging level |
| `log_file` | `None` | Log file path |
| `enable_observability_logging` | `False` | Write events to JSONL |
| `observability_log_path` | `None` | Custom JSONL path |
| `analytics_enabled` | `True` | Enable analytics |

---

## Profile Configuration (profiles.yaml)

Profiles in `~/.victor/profiles.yaml` can override many of these settings per-model:

```yaml
profiles:
  default:
    provider: ollama
    model: qwen3-coder:30b
    temperature: 0.7
    max_tokens: 4096
    description: "Default local model"

    # Provider tuning (per-profile overrides)
    loop_repeat_threshold: 3
    max_continuation_prompts: 5
    quality_threshold: 0.6
    grounding_threshold: 0.7
    max_tool_calls_per_turn: 10
    tool_cache_enabled: true
    tool_deduplication_enabled: true
    session_idle_timeout: 300
    timeout: 600

    # Tool selection tuning
    tool_selection:
      model_size_tier: medium  # tiny, small, medium, large, cloud
      base_threshold: 0.35
      base_max_tools: 12

  anthropic:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.5
    max_tokens: 8192
    tool_selection:
      model_size_tier: cloud

providers:
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
  openai:
    api_key: ${OPENAI_API_KEY}
```
